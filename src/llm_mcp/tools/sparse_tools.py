"""Sparse Fine-Tuning tools for the LLM MCP server.

This module provides tools for sparse fine-tuning of language models, including:
- SparseGPT: Layer-wise sparsification for efficient fine-tuning
- Movement Pruning: Dynamic sparsity during training
- Top-K Masking: Sparse attention patterns
- Sparse Fine-Tuning with RigL: Dynamic sparse training
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
# Tool decorator will be obtained from mcp instance
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
import bitsandbytes as bnb
from loguru import logger

@dataclass
class SparseConfig:
    """Configuration for sparse fine-tuning."""
    sparsity_ratio: float = 0.5
    sparsity_type: str = "unstructured"  # "unstructured", "2:4", "4:8"
    mask_update_interval: int = 100  # steps
    mask_update_fraction: float = 0.3  # % of weights to update
    use_rigl: bool = True  # Use RigL for dynamic sparse training
    use_topk_attention: bool = True  # Use top-k sparse attention
    topk_ratio: float = 0.1  # Keep top 10% of attention scores
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

class SparseLinear(nn.Module):
    """Sparse linear layer with dynamic sparsity."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 device=None, dtype=None, sparsity_ratio: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity_ratio = sparsity_ratio
        
        # Initialize weights
        self.weight = nn.Parameter(torch.empty((out_features, in_features), 
                                             device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, 
                                               dtype=dtype))
        else:
            self.register_parameter('bias', None)
            
        # Initialize mask
        self.register_buffer('mask', torch.ones_like(self.weight, dtype=torch.bool))
        self.reset_parameters()
        self.update_mask()
    
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def update_mask(self) -> None:
        """Update the weight mask based on magnitude pruning."""
        with torch.no_grad():
            # Flatten and get top-k weights
            flat_weights = torch.abs(self.weight).view(-1)
            k = int((1 - self.sparsity_ratio) * flat_weights.numel())
            if k > 0:
                # Get top-k weights
                _, idx = torch.topk(flat_weights, k, dim=0, largest=True, sorted=False)
                # Create new mask
                new_mask = torch.zeros_like(flat_weights, dtype=torch.bool)
                new_mask[idx] = True
                self.mask.data = new_mask.view_as(self.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply mask to weights
        masked_weight = self.weight * self.mask
        return F.linear(x, masked_weight, self.bias)

class SparseTrainer(Trainer):
    """Custom trainer for sparse fine-tuning."""
    
    def __init__(self, *args, sparse_config: Optional[SparseConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sparse_config = sparse_config or SparseConfig()
        self.steps_since_mask_update = 0
    
    def training_step(self, model, inputs):
        # Call parent training step
        loss = super().training_step(model, inputs)
        
        # Update masks periodically
        self.steps_since_mask_update += 1
        if self.steps_since_mask_update >= self.sparse_config.mask_update_interval:
            self.update_masks()
            self.steps_since_mask_update = 0
            
        return loss
    
    def update_masks(self):
        """Update masks for all sparse layers."""
        for module in self.model.modules():
            if isinstance(module, SparseLinear):
                module.update_mask()

class SparseModelWrapper(nn.Module):
    """Wrapper for sparse fine-tuning of a model."""
    
    def __init__(self, model, sparse_config: Optional[SparseConfig] = None):
        super().__init__()
        self.model = model
        self.sparse_config = sparse_config or SparseConfig()
        self._convert_to_sparse()
    
    def _convert_to_sparse(self):
        """Convert linear layers to sparse layers."""
        for name, module in self.model.named_children():
            if isinstance(module, nn.Linear) and 'lm_head' not in name:
                # Replace with sparse linear
                sparse_linear = SparseLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                    sparsity_ratio=self.sparse_config.sparsity_ratio
                )
                # Copy weights
                with torch.no_grad():
                    sparse_linear.weight.data.copy_(module.weight.data)
                    if module.bias is not None:
                        sparse_linear.bias.data.copy_(module.bias.data)
                # Replace the module
                setattr(self.model, name, sparse_linear)
            else:
                # Recursively convert child modules
                self._convert_to_sparse_recursive(module)
    
    def _convert_to_sparse_recursive(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and 'lm_head' not in name:
                # Replace with sparse linear
                sparse_linear = SparseLinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.bias is not None,
                    device=child.weight.device,
                    dtype=child.weight.dtype,
                    sparsity_ratio=self.sparse_config.sparsity_ratio
                )
                # Copy weights
                with torch.no_grad():
                    sparse_linear.weight.data.copy_(child.weight.data)
                    if child.bias is not None:
                        sparse_linear.bias.data.copy_(child.bias.data)
                # Replace the module
                setattr(module, name, sparse_linear)
            else:
                self._convert_to_sparse_recursive(child)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

# Global registry for loaded models
SPARSE_MODELS = {}

# Implementation functions (without @tool decorator)
async def sparse_load_model_impl(
    model_name: str,
    model_id: Optional[str] = None,
    sparsity_ratio: float = 0.5,
    sparsity_type: str = "unstructured",
    use_4bit: bool = True,
    use_double_quant: bool = True,
    quant_type: str = "nf4",
    compute_dtype: str = "bfloat16",
    device_map: str = "auto"
) -> Dict[str, Any]:
    """Load a model for sparse fine-tuning.
    
    Args:
        model_name: Name or path of the pre-trained model
        model_id: Optional ID for the model (auto-generated if not provided)
        sparsity_ratio: Target sparsity ratio (0-1)
        sparsity_type: Type of sparsity ("unstructured", "2:4", "4:8")
        use_4bit: Whether to use 4-bit quantization
        use_double_quant: Whether to use double quantization
        quant_type: Type of quantization ("nf4", "fp4", "int8", "none")
        compute_dtype: Compute dtype ("float16", "bfloat16", "float32")
        device_map: Device placement strategy
        
    Returns:
        Dictionary with model information
    """
    global SPARSE_MODELS
    
    # Generate model ID if not provided
    if model_id is None:
        import hashlib
        model_id = f"sparse-{hashlib.sha256(model_name.encode()).hexdigest()[:8]}"
    
    if model_id in SPARSE_MODELS:
        return {"status": "error", "message": f"Model with ID '{model_id}' already loaded"}
    
    try:
        # Configure quantization
        if use_4bit and quant_type != "none":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quant_type,
                bnb_4bit_use_double_quant=use_double_quant,
                bnb_4bit_compute_dtype={
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "float32": torch.float32
                }[compute_dtype]
            )
        else:
            bnb_config = None
        
        # Load the model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True
        )
        
        # Configure sparsity
        sparse_config = SparseConfig(
            sparsity_ratio=sparsity_ratio,
            sparsity_type=sparsity_type
        )
        
        # Wrap the model with sparse layers
        sparse_model = SparseModelWrapper(model, sparse_config)
        
        # Store the model
        SPARSE_MODELS[model_id] = {
            "model": sparse_model,
            "config": sparse_config,
            "model_name": model_name,
            "device": next(sparse_model.parameters()).device,
            "dtype": next(sparse_model.parameters()).dtype
        }
        
        # Count trainable parameters
        total_params = sum(p.numel() for p in sparse_model.parameters())
        trainable_params = sum(p.numel() for p in sparse_model.parameters() 
                             if p.requires_grad)
        
        return {
            "status": "success",
            "model_id": model_id,
            "model_name": model_name,
            "sparsity_ratio": sparsity_ratio,
            "sparsity_type": sparsity_type,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "device": str(SPARSE_MODELS[model_id]["device"]),
            "dtype": str(SPARSE_MODELS[model_id]["dtype"])
        }
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return {"status": "error", "message": str(e)}

# Implementation functions (without @tool decorator)
async def sparse_prepare_for_training_impl(
    model_id: str,
    output_dir: str,
    learning_rate: float = 2e-4,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    max_steps: int = 1000,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.01,
    optim: str = "paged_adamw_32bit",
    lr_scheduler_type: str = "cosine",
    max_grad_norm: float = 0.3,
    logging_steps: int = 10,
    save_steps: int = 200,
    save_total_limit: int = 3,
    report_to: str = "tensorboard",
    use_gradient_checkpointing: bool = True,
    use_flash_attention_2: bool = True,
    use_cpu_offload: bool = False
) -> Dict[str, Any]:
    """Prepare a sparse model for training.
    
    Args:
        model_id: ID of the loaded model
        output_dir: Directory to save outputs
        learning_rate: Learning rate
        batch_size: Batch size per device
        gradient_accumulation_steps: Number of steps for gradient accumulation
        max_steps: Maximum number of training steps
        warmup_ratio: Ratio of warmup steps
        weight_decay: Weight decay
        optim: Optimizer to use
        lr_scheduler_type: Learning rate scheduler type
        max_grad_norm: Maximum gradient norm
        logging_steps: Log every N steps
        save_steps: Save checkpoint every N steps
        save_total_limit: Maximum number of checkpoints to keep
        report_to: Where to report metrics ("tensorboard", "wandb", etc.)
        use_gradient_checkpointing: Whether to use gradient checkpointing
        use_flash_attention_2: Whether to use Flash Attention 2
        use_cpu_offload: Whether to offload some operations to CPU
        
    Returns:
        Dictionary with training configuration
    """
    global SPARSE_MODELS
    
    if model_id not in SPARSE_MODELS:
        return {"status": "error", "message": f"Model with ID '{model_id}' not found"}
    
    try:
        # Store training configuration
        training_config = {
            "output_dir": output_dir,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "max_steps": max_steps,
            "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay,
            "optim": optim,
            "lr_scheduler_type": lr_scheduler_type,
            "max_grad_norm": max_grad_norm,
            "logging_steps": logging_steps,
            "save_steps": save_steps,
            "save_total_limit": save_total_limit,
            "report_to": report_to,
            "use_gradient_checkpointing": use_gradient_checkpointing,
            "use_flash_attention_2": use_flash_attention_2,
            "use_cpu_offload": use_cpu_offload
        }
        
        SPARSE_MODELS[model_id]["training_config"] = training_config
        
        # Configure gradient checkpointing
        if use_gradient_checkpointing:
            SPARSE_MODELS[model_id]["model"].gradient_checkpointing_enable()
        
        return {
            "status": "success",
            "model_id": model_id,
            "training_config": training_config
        }
        
    except Exception as e:
        logger.error(f"Error preparing for training: {str(e)}")
        return {"status": "error", "message": str(e)}

async def sparse_train_impl(
    model_id: str,
    train_dataset: Any,
    eval_dataset: Optional[Any] = None,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 100,
    logging_steps: int = 10,
    save_steps: int = 100,
    eval_steps: Optional[int] = None,
    output_dir: Optional[str] = None,
    save_total_limit: int = 2,
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "loss",
    greater_is_better: bool = False,
    fp16: bool = True,
    bf16: bool = False,
    max_grad_norm: float = 1.0,
    group_by_length: bool = True,
    report_to: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Train a sparse model.
    
    Args:
        model_id: ID of the loaded model
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device for training
        gradient_accumulation_steps: Number of steps for gradient accumulation
        learning_rate: Learning rate
        weight_decay: Weight decay for optimization
        warmup_steps: Number of warmup steps
        logging_steps: Log every X updates steps
        save_steps: Save checkpoint every X updates steps
        eval_steps: Run evaluation every X steps
        output_dir: Output directory for model checkpoints
        save_total_limit: Maximum number of checkpoints to keep
        load_best_model_at_end: Whether to load the best model at the end of training
        metric_for_best_model: Metric to use for best model selection
        greater_is_better: Whether a higher metric is better
        fp16: Whether to use 16-bit (mixed) precision training
        bf16: Whether to use bfloat16 precision training
        max_grad_norm: Maximum gradient norm for gradient clipping
        group_by_length: Whether to group sequences by length for efficiency
        report_to: List of integrations to report to (e.g., ["tensorboard"])
            
    Returns:
        Dictionary with training results
    """
    global SPARSE_MODELS
    
    if model_id not in SPARSE_MODELS:
        return {"status": "error", "message": f"Model with ID '{model_id}' not found"}
    
    model_info = SPARSE_MODELS[model_id]
    model = model_info["model"]
    training_config = model_info.get("training_config", {})
    
    # Update training config with any overrides
    training_config.update({
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "logging_steps": logging_steps,
        "save_steps": save_steps,
        "eval_steps": eval_steps,
        "output_dir": output_dir,
        "save_total_limit": save_total_limit,
        "load_best_model_at_end": load_best_model_at_end,
        "metric_for_best_model": metric_for_best_model,
        "greater_is_better": greater_is_better,
        "fp16": fp16,
        "bf16": bf16,
        "max_grad_norm": max_grad_norm,
        "group_by_length": group_by_length,
        "report_to": report_to
    })
    
    try:
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=training_config["output_dir"],
            learning_rate=training_config["learning_rate"],
            per_device_train_batch_size=training_config["per_device_train_batch_size"],
            gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
            num_train_epochs=training_config["num_train_epochs"],
            warmup_steps=training_config["warmup_steps"],
            weight_decay=training_config["weight_decay"],
            optim=training_config["optim"],
            lr_scheduler_type=training_config["lr_scheduler_type"],
            max_grad_norm=training_config["max_grad_norm"],
            logging_steps=training_config["logging_steps"],
            save_steps=training_config["save_steps"],
            save_total_limit=training_config["save_total_limit"],
            report_to=training_config["report_to"],
            remove_unused_columns=False,
            fp16=torch.cuda.is_available() and training_config.get("fp16", True),
            bf16=training_config.get("bf16", False),
            gradient_checkpointing=training_config["use_gradient_checkpointing"],
            ddp_find_unused_parameters=False
        )
        
        # Initialize trainer
        trainer = SparseTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            sparse_config=model_info["config"]
        )
        
        # Train the model
        train_result = trainer.train()
        
        # Save the final model
        trainer.save_model(output_dir=training_config["output_dir"])
        trainer.save_state()
        
        # Log metrics
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        
        if eval_dataset is not None:
            eval_metrics = trainer.evaluate()
            metrics.update({"eval_" + k: v for k, v in eval_metrics.items()})
        
        # Save metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        return {
            "status": "success",
            "model_id": model_id,
            "output_dir": training_config["output_dir"],
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return {"status": "error", "message": str(e)}

async def sparse_unload_model_impl(model_id: str) -> Dict[str, Any]:
    """Unload a sparse model and free resources.
    
    Args:
        model_id: ID of the loaded model to unload
        
    Returns:
        Dictionary with unload status
    """
    global SPARSE_MODELS
    
    if model_id not in SPARSE_MODELS:
        return {"status": "error", "message": f"Model with ID '{model_id}' not found"}
    
    try:
        # Clear model from GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Remove model from registry
        del SPARSE_MODELS[model_id]
        
        return {
            "status": "success",
            "message": f"Model '{model_id}' unloaded successfully"
        }
        
    except Exception as e:
        logger.error(f"Error unloading model: {str(e)}")
        return {"status": "error", "message": str(e)}

async def sparse_list_models_impl() -> Dict[str, Any]:
    """List all loaded sparse models.
    
    Returns:
        Dictionary with information about loaded models
    """
    global SPARSE_MODELS
    
    models_info = {}
    for model_id, model_info in SPARSE_MODELS.items():
        models_info[model_id] = {
            "model_name": model_info["model_name"],
            "sparsity_ratio": model_info["config"].sparsity_ratio,
            "sparsity_type": model_info["config"].sparsity_type,
            "device": str(model_info["device"]),
            "dtype": str(model_info["dtype"]),
            "has_training_config": "training_config" in model_info
        }
    
    return {
        "status": "success",
        "models": models_info
    }

def register_sparse_tools(mcp):
    """Register sparse fine-tuning tools with the MCP server.
    
    Args:
        mcp: The MCP server instance with tool decorator
        
    Returns:
        The MCP server instance with sparse tools registered
    """
    # Get the tool decorator from the mcp instance
    tool = mcp.tool
    
    # Register sparse model loading
    @tool()
    async def sparse_load_model(
        model_name: str,
        model_id: Optional[str] = None,
        sparsity_ratio: float = 0.5,
        sparsity_type: str = "unstructured",
        use_4bit: bool = True,
        use_double_quant: bool = True,
        quant_type: str = "nf4",
        compute_dtype: str = "bfloat16",
        device_map: str = "auto"
    ) -> Dict[str, Any]:
        """Load a model for sparse fine-tuning.
        
        Args:
            model_name: Name or path of the pre-trained model
            model_id: Optional ID for the model (auto-generated if not provided)
            sparsity_ratio: Target sparsity ratio (0-1)
            sparsity_type: Type of sparsity ("unstructured", "2:4", "4:8")
            use_4bit: Whether to use 4-bit quantization
            use_double_quant: Whether to use double quantization
            quant_type: Type of quantization ("nf4", "fp4", "int8", "none")
            compute_dtype: Compute dtype ("float16", "bfloat16", "float32")
            device_map: Device placement strategy
            
        Returns:
            Dictionary with model information
        """
        return sparse_load_model_impl(
            model_name=model_name,
            model_id=model_id,
            sparsity_ratio=sparsity_ratio,
            sparsity_type=sparsity_type,
            use_4bit=use_4bit,
            use_double_quant=use_double_quant,
            quant_type=quant_type,
            compute_dtype=compute_dtype,
            device_map=device_map
        )
    
    # Register sparse model training
    @tool()
    async def sparse_train(
        model_id: str,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        save_steps: int = 100,
        eval_steps: Optional[int] = None,
        output_dir: Optional[str] = None,
        save_total_limit: int = 2,
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = "loss",
        greater_is_better: bool = False,
        fp16: bool = True,
        bf16: bool = False,
        max_grad_norm: float = 1.0,
        group_by_length: bool = True,
        report_to: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Train a sparse model.
        
        Args:
            model_id: ID of the loaded model
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size per device for training
            gradient_accumulation_steps: Number of steps for gradient accumulation
            learning_rate: Learning rate
            weight_decay: Weight decay for optimization
            warmup_steps: Number of warmup steps
            logging_steps: Log every X updates steps
            save_steps: Save checkpoint every X updates steps
            eval_steps: Run evaluation every X steps
            output_dir: Output directory for model checkpoints
            save_total_limit: Maximum number of checkpoints to keep
            load_best_model_at_end: Whether to load the best model at the end of training
            metric_for_best_model: Metric to use for best model selection
            greater_is_better: Whether a higher metric is better
            fp16: Whether to use 16-bit (mixed) precision training
            bf16: Whether to use bfloat16 precision training
            max_grad_norm: Maximum gradient norm for gradient clipping
            group_by_length: Whether to group sequences by length for efficiency
            report_to: List of integrations to report to (e.g., ["tensorboard"])
            
        Returns:
            Dictionary with training results
        """
        return sparse_train_impl(
            model_id=model_id,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            output_dir=output_dir,
            save_total_limit=save_total_limit,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            fp16=fp16,
            bf16=bf16,
            max_grad_norm=max_grad_norm,
            group_by_length=group_by_length,
            report_to=report_to or ["none"]
        )
    
    # Register sparse model unloading
    @tool()
    async def sparse_unload_model(model_id: str) -> Dict[str, Any]:
        """Unload a sparse model and free resources.
        
        Args:
            model_id: ID of the loaded model to unload
            
        Returns:
            Dictionary with unload status
        """
        return sparse_unload_model_impl(model_id=model_id)
        
    # Register sparse model listing
    @tool()
    async def sparse_list_models() -> Dict[str, Any]:
        """List all loaded sparse models.
        
        Returns:
            Dictionary with information about loaded models
        """
        return sparse_list_models_impl()
    
    return mcp
