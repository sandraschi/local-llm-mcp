"""
DoRA (Dropout LoRA) Tools for LLM Fine-Tuning

This module provides tools for fine-tuning large language models using DoRA (Dropout LoRA),
which enhances standard LoRA by adding dropout to the low-rank adaptation matrices.
This improves model robustness and can help prevent overfitting.

Key Features:
- Dropout on low-rank adaptation matrices
- Compatible with existing LoRA implementations
- Configurable dropout rates
- Support for various model architectures
- Seamless integration with existing training pipelines
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import is_torch_bf16_gpu_available, is_torch_tf32_available

# Get logger instance (configured in main.py)
logger = logging.getLogger(__name__)

# Global state for tracking loaded models
_dora_models: Dict[str, Any] = {}

# Background task for system metrics
async def collect_system_metrics():
    """Background task to collect system metrics."""
    while True:
        try:
            # Collect system metrics here
            await asyncio.sleep(60)  # Collect every 60 seconds
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            await asyncio.sleep(60)  # Wait before retrying

# Implementation functions (without @tool decorator)
async def dora_load_model_impl(
    model_name: str,
    model_id: Optional[str] = None,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
    use_double_quant: bool = True,
    quant_type: str = "nf4",
    compute_dtype: str = "bfloat16",
    device_map: str = "auto"
) -> Dict[str, Any]:
    """Load a model with DoRA (Dropout LoRA) for fine-tuning.
    
    Args:
        model_name: Name or path of the pre-trained model
        model_id: Optional ID for the model (auto-generated if not provided)
        lora_rank: Rank of the low-rank matrices
        lora_alpha: Scaling factor for the low-rank matrices
        lora_dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA to
        use_double_quant: Whether to use double quantization
        quant_type: Type of quantization to use ("nf4", "fp4", "int8", "none")
        compute_dtype: Compute dtype for training ("float16", "bfloat16", "float32")
        device_map: Device placement strategy ("auto", "cuda", "cpu", etc.)
        
    Returns:
        Dictionary containing model information
    """
    try:
        if model_id is None:
            model_id = f"dora-{os.path.basename(model_name).lower().replace('.', '-')}-{os.urandom(4).hex()}"
        
        # Set up quantization config if needed
        if quant_type != "none":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=quant_type in ["nf4", "fp4"],
                load_in_8bit=quant_type == "int8",
                bnb_4bit_quant_type=quant_type if quant_type in ["nf4", "fp4"] else None,
                bnb_4bit_use_double_quant=use_double_quant,
                bnb_4bit_compute_dtype={
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "float32": torch.float32
                }[compute_dtype]
            )
        else:
            bnb_config = None
        
        # Load the base model
        logger.info(f"Loading base model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True
        )
        
        # Set up DoRA config
        config = DoraConfig(
            model_name=model_name,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            use_double_quant=use_double_quant,
            quant_type=quant_type,
            compute_dtype=compute_dtype,
            device_map=device_map
        )
        
        # Apply DoRA
        logger.info("Applying DoRA to the model")
        model = DoraLoraModel(model, config)
        
        # Store the model
        _dora_models[model_id] = {
            "model": model,
            "config": config,
            "device": device_map if device_map != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
        }
        
        # Count trainable parameters
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        total_params = sum(
            p.numel() for p in model.parameters()
        )
        
        logger.info(f"Model loaded with ID: {model_id}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Total parameters: {total_params:,}")
        
        return {
            "status": "success",
            "model_id": model_id,
            "model_name": model_name,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "config": {
                "model_name": model_name,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "target_modules": config.target_modules,
                "use_double_quant": use_double_quant,
                "quant_type": quant_type,
                "compute_dtype": compute_dtype,
                "device_map": device_map
            }
        }
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Failed to load model: {str(e)}"
        }

async def dora_prepare_for_training_impl(
    model_id: str,
    output_dir: str = "./dora_output",
    learning_rate: float = 2e-4,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    num_train_epochs: int = 3,
    max_steps: int = -1,
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
    """Prepare a DoRA model for training.
    
    Args:
        model_id: ID of the loaded model
        output_dir: Directory to save the model (default: "./dora_output")
        learning_rate: Learning rate (default: 2e-4)
        batch_size: Batch size per device (default: 4)
        gradient_accumulation_steps: Number of steps to accumulate gradients (default: 4)
        num_train_epochs: Number of training epochs (default: 3)
        max_steps: Maximum number of training steps (-1 to use num_train_epochs) (default: -1)
        warmup_ratio: Ratio of warmup steps (default: 0.03)
        weight_decay: Weight decay (default: 0.01)
        optim: Optimizer to use (default: "paged_adamw_32bit")
        lr_scheduler_type: Learning rate scheduler type (default: "cosine")
        max_grad_norm: Maximum gradient norm (default: 0.3)
        logging_steps: Log every X updates steps (default: 10)
        save_steps: Save checkpoint every X updates steps (default: 200)
        save_total_limit: Maximum number of checkpoints to keep (default: 3)
        report_to: Comma-separated list of integrations to report to (default: "tensorboard")
        use_gradient_checkpointing: Whether to use gradient checkpointing (default: True)
        use_flash_attention_2: Whether to use Flash Attention 2.0 (default: True)
        use_cpu_offload: Whether to offload some operations to CPU (default: False)
        
    Returns:
        Dictionary with status and configuration
    """
    try:
        if model_id not in _dora_models:
            return {
                "status": "error",
                "message": f"Model with ID {model_id} not found"
            }
        
        model_info = _dora_models[model_id]
        
        # Set up training arguments
        training_args = {
            "output_dir": output_dir,
            "learning_rate": learning_rate,
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "num_train_epochs": num_train_epochs,
            "max_steps": max_steps,
            "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay,
            "optim": optim,
            "lr_scheduler_type": lr_scheduler_type,
            "gradient_checkpointing": use_gradient_checkpointing,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "max_grad_norm": max_grad_norm,
            "logging_steps": logging_steps,
            "save_strategy": "steps",
            "save_steps": save_steps,
            "save_total_limit": save_total_limit,
            "report_to": report_to.split(",") if report_to else [],
            "bf16": is_torch_bf16_gpu_available(),
            "tf32": is_torch_tf32_available(),
            "dataloader_pin_memory": True,
            "remove_unused_columns": False,
            "evaluation_strategy": "steps" if model_info.get("eval_dataset") else "no",
            "eval_steps": save_steps if model_info.get("eval_dataset") else None,
            "load_best_model_at_end": bool(model_info.get("eval_dataset")),
            "metric_for_best_model": "eval_loss" if model_info.get("eval_dataset") else None,
            "greater_is_better": False if model_info.get("eval_dataset") else None,
        }
        
        # Update model info with training configuration
        model_info["training_args"] = training_args
        model_info["output_dir"] = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Model {model_id} prepared for training")
        logger.info(f"Training configuration: {training_args}")
        
        return {
            "status": "success",
            "message": f"Model {model_id} prepared for training",
            "config": training_args
        }
        
    except Exception as e:
        logger.error(f"Error preparing model for training: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Failed to prepare model for training: {str(e)}"
        }

async def dora_train_impl(
    model_id: str,
    train_dataset: Any,
    eval_dataset: Optional[Any] = None,
    **training_kwargs
) -> Dict[str, Any]:
    """Train a DoRA model.
    
    Args:
        model_id: ID of the loaded model
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset
        **training_kwargs: Additional training arguments
        
    Returns:
        Dictionary with training results
    """
    try:
        if model_id not in _dora_models:
            return {
                "status": "error",
                "message": f"Model with ID {model_id} not found"
            }
        
        model_info = _dora_models[model_id]
        model = model_info["model"]
        
        # Update training arguments with any overrides
        training_args = model_info.get("training_args", {})
        training_args.update(training_kwargs)
        
        # Store datasets
        model_info["train_dataset"] = train_dataset
        if eval_dataset is not None:
            model_info["eval_dataset"] = eval_dataset
        
        # Set up training arguments
        training_args = TrainingArguments(**training_args)
        
        # Set up trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if eval_dataset is not None else None,
        )
        
        # Train the model
        logger.info(f"Starting training for model {model_id}")
        train_result = trainer.train()
        
        # Save the final model
        trainer.save_model(model_info["output_dir"])
        trainer.save_state()
        
        # Log training metrics
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        
        if eval_dataset is not None:
            eval_metrics = trainer.evaluate()
            metrics.update(eval_metrics)
        
        # Save metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        logger.info(f"Training completed for model {model_id}")
        
        return {
            "status": "success",
            "metrics": metrics,
            "output_dir": model_info["output_dir"]
        }
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Training failed: {str(e)}"
        }

async def dora_unload_model_impl(model_id: str) -> Dict[str, Any]:
    """Unload a DoRA model.
    
    Args:
        model_id: ID of the model to unload
        
    Returns:
        Dictionary with status message
    """
    try:
        if model_id not in _dora_models:
            return {
                "status": "error",
                "message": f"Model with ID {model_id} not found"
            }
        
        # Clean up model resources
        model_info = _dora_models.pop(model_id)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Model {model_id} unloaded")
        
        return {
            "status": "success",
            "message": f"Model {model_id} unloaded"
        }
        
    except Exception as e:
        logger.error(f"Error unloading model: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Failed to unload model: {str(e)}"
        }

async def dora_list_models_impl() -> Dict[str, Any]:
    """List all loaded DoRA models.
    
    Returns:
        Dictionary with loaded model information
    """
    try:
        models_info = {}
        for model_id, model_info in _dora_models.items():
            models_info[model_id] = {
                "config": model_info.get("config"),
                "device": model_info.get("device", "unknown"),
                "has_train_dataset": "train_dataset" in model_info,
                "has_eval_dataset": "eval_dataset" in model_info,
                "training_args": model_info.get("training_args")
            }
        
        return {
            "status": "success",
            "models": models_info
        }
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Failed to list models: {str(e)}"
        }

@dataclass
class DoraConfig:
    """Configuration for DoRA (Dropout LoRA) fine-tuning.
    
    Attributes:
        model_name: Name or path of the pre-trained model
        lora_rank: Rank of the low-rank matrices
        lora_alpha: Scaling factor for the low-rank matrices
        lora_dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA to
        use_double_quant: Whether to use double quantization
        quant_type: Type of quantization to use ("nf4", "fp4", "int8", "none")
        compute_dtype: Compute dtype for training ("float16", "bfloat16", "float32")
        device_map: Device placement strategy ("auto", "cuda", "cpu", etc.)
    """
    model_name: str
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    use_double_quant: bool = True
    quant_type: str = "nf4"  # "nf4", "fp4", "int8", "none"
    compute_dtype: str = "bfloat16"  # "float16", "bfloat16", "float32"
    device_map: str = "auto"

class DoraLoraModel(nn.Module):
    """A custom LoRA implementation with dropout for better robustness.
    
    This class extends the standard LoRA implementation by adding dropout to the
    low-rank adaptation matrices, which can help prevent overfitting.
    """
    
    def __init__(self, base_model: nn.Module, config: DoraConfig):
        """Initialize the DoRA model.
        
        Args:
            base_model: The base model to apply LoRA to
            config: Configuration for the DoRA model
        """
        super().__init__()
        self.base_model = base_model
        self.config = config
        self._setup_lora_layers()
    
    def _setup_lora_layers(self):
        """Set up LoRA layers with dropout."""
        for name, module in self.base_model.named_modules():
            if not any(target in name for target in self.config.target_modules):
                continue
                
            if isinstance(module, nn.Linear):
                # Replace the linear layer with our custom implementation
                new_module = DoraLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    lora_rank=self.config.lora_rank,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    device=next(self.base_model.parameters()).device
                )
                
                # Copy the original weights
                new_module.linear.weight = module.weight
                if module.bias is not None:
                    new_module.linear.bias = module.bias
                
                # Replace the module
                parent = self._get_parent_module(name)
                child_name = name.split('.')[-1]
                setattr(parent, child_name, new_module)
    
    def _get_parent_module(self, name: str) -> nn.Module:
        """Get the parent module of a given module.
        
        Args:
            name: Full name of the target module
            
        Returns:
            The parent module
        """
        module = self.base_model
        parts = name.split('.')
        for part in parts[:-1]:
            module = getattr(module, part)
        return module
    
    def forward(self, *args, **kwargs):
        """Forward pass through the model."""
        return self.base_model(*args, **kwargs)

class DoraLinear(nn.Module):
    """A linear layer with LoRA and dropout."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        device: Optional[torch.device] = None
    ):
        """Initialize the DoRA linear layer.
        
        Args:
            in_features: Size of each input sample
            out_features: Size of each output sample
            bias: Whether to include a bias term
            lora_rank: Rank of the low-rank matrices
            lora_alpha: Scaling factor for the low-rank matrices
            lora_dropout: Dropout probability for LoRA layers
            device: Device to place the layer on
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        
        # Base linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias, device=device)
        
        # LoRA A and B matrices with dropout
        self.lora_A = nn.Linear(in_features, lora_rank, bias=False, device=device)
        self.lora_B = nn.Linear(lora_rank, out_features, bias=False, device=device)
        
        # Dropout for LoRA
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        
        # Scaling factor
        self.scaling = lora_alpha / lora_rank
        
        # Initialize LoRA weights
        self._init_lora_weights()
    
    def _init_lora_weights(self):
        """Initialize LoRA weights."""
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, in_features)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, out_features)
        """
        # Base model output
        output = self.linear(x)
        
        # Apply LoRA with dropout
        lora_output = self.lora_B(self.lora_dropout(F.gelu(self.lora_A(x))))
        
        # Scale and add to base output
        output = output + lora_output * self.scaling
        
        return output

def register_dora_tools(mcp):
    """Register DoRA tools with the MCP server.
    
    Args:
        mcp: The MCP server instance with tool decorator
        
    Returns:
        The MCP server instance with DoRA tools registered
    """
    # Get the tool decorator from the mcp instance
    tool = mcp.tool
    
    @tool()
    async def dora_load_model(
        model_name: str,
        model_id: Optional[str] = None,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        use_double_quant: bool = True,
        quant_type: str = "nf4",
        compute_dtype: str = "bfloat16",
        device_map: str = "auto"
    ) -> Dict[str, Any]:
        """Load a model with DoRA (Dropout LoRA) for fine-tuning.
        
        Args:
            model_name: Name or path of the pre-trained model
            model_id: Optional ID for the model (auto-generated if not provided)
            lora_rank: Rank of the low-rank matrices
            lora_alpha: Scaling factor for the low-rank matrices
            lora_dropout: Dropout probability for LoRA layers
            target_modules: List of module names to apply LoRA to
            use_double_quant: Whether to use double quantization
            quant_type: Type of quantization to use ("nf4", "fp4", "int8", "none")
            compute_dtype: Compute dtype for training ("float16", "bfloat16", "float32")
            device_map: Device placement strategy ("auto", "cuda", "cpu", etc.)
            
        Returns:
            Dictionary containing model information
        """
        try:
            if model_id is None:
                model_id = f"dora-{os.path.basename(model_name).lower().replace('.', '-')}-{os.urandom(4).hex()}"
            
            # Set up quantization config if needed
            if quant_type != "none":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=quant_type in ["nf4", "fp4"],
                    load_in_8bit=quant_type == "int8",
                    bnb_4bit_quant_type=quant_type if quant_type in ["nf4", "fp4"] else None,
                    bnb_4bit_use_double_quant=use_double_quant,
                    bnb_4bit_compute_dtype={
                        "float16": torch.float16,
                        "bfloat16": torch.bfloat16,
                        "float32": torch.float32
                    }[compute_dtype]
                )
            else:
                bnb_config = None
            
            # Load the base model
            logger.info(f"Loading base model: {model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True
            )
            
            # Set up DoRA config
            config = DoraConfig(
                model_name=model_name,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                use_double_quant=use_double_quant,
                quant_type=quant_type,
                compute_dtype=compute_dtype,
                device_map=device_map
            )
            
            # Apply DoRA
            logger.info("Applying DoRA to the model")
            model = DoraLoraModel(model, config)
            
            # Store the model
            _dora_models[model_id] = {
                "model": model,
                "config": config,
                "device": device_map if device_map != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
            }
            
            # Count trainable parameters
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            total_params = sum(
                p.numel() for p in model.parameters()
            )
            
            logger.info(f"Model loaded with ID: {model_id}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            logger.info(f"Total parameters: {total_params:,}")
            
            return {
                "status": "success",
                "model_id": model_id,
                "model_name": model_name,
                "trainable_params": trainable_params,
                "total_params": total_params,
                "config": {
                    "model_name": model_name,
                    "lora_rank": lora_rank,
                    "lora_alpha": lora_alpha,
                    "lora_dropout": lora_dropout,
                    "target_modules": config.target_modules,
                    "use_double_quant": use_double_quant,
                    "quant_type": quant_type,
                    "compute_dtype": compute_dtype,
                    "device_map": device_map
                }
            }
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Failed to load model: {str(e)}"
            }
    
    @mcp.tool()
    async def dora_prepare_for_training(
        model_id: str,
        output_dir: str = "./dora_output",
        learning_rate: float = 2e-4,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        num_train_epochs: int = 3,
        max_steps: int = -1,
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
        """Prepare a DoRA model for training.
        
        Args:
            model_id: ID of the loaded model
            output_dir: Directory to save the model (default: "./dora_output")
            learning_rate: Learning rate (default: 2e-4)
            batch_size: Batch size per device (default: 4)
            gradient_accumulation_steps: Number of steps to accumulate gradients (default: 4)
            num_train_epochs: Number of training epochs (default: 3)
            max_steps: Maximum number of training steps (-1 to use num_train_epochs) (default: -1)
            warmup_ratio: Ratio of warmup steps (default: 0.03)
            weight_decay: Weight decay (default: 0.01)
            optim: Optimizer to use (default: "paged_adamw_32bit")
            lr_scheduler_type: Learning rate scheduler type (default: "cosine")
            max_grad_norm: Maximum gradient norm (default: 0.3)
            logging_steps: Log every X updates steps (default: 10)
            save_steps: Save checkpoint every X updates steps (default: 200)
            save_total_limit: Maximum number of checkpoints to keep (default: 3)
            report_to: Comma-separated list of integrations to report to (default: "tensorboard")
            use_gradient_checkpointing: Whether to use gradient checkpointing (default: True)
            use_flash_attention_2: Whether to use Flash Attention 2.0 (default: True)
            use_cpu_offload: Whether to offload some operations to CPU (default: False)
            
        Returns:
            Dictionary with status and configuration
        """
        try:
            if model_id not in _dora_models:
                return {
                    "status": "error",
                    "message": f"Model with ID {model_id} not found"
                }
            
            model_info = _dora_models[model_id]
            model = model_info["model"]
            
            # Set up training arguments
            training_args = {
                "output_dir": output_dir,
                "learning_rate": learning_rate,
                "per_device_train_batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "num_train_epochs": num_train_epochs,
                "max_steps": max_steps,
                "warmup_ratio": warmup_ratio,
                "weight_decay": weight_decay,
                "optim": optim,
                "lr_scheduler_type": lr_scheduler_type,
                "gradient_checkpointing": use_gradient_checkpointing,
                "gradient_checkpointing_kwargs": {"use_reentrant": False},
                "max_grad_norm": max_grad_norm,
                "logging_steps": logging_steps,
                "save_strategy": "steps",
                "save_steps": save_steps,
                "save_total_limit": save_total_limit,
                "report_to": report_to.split(",") if report_to else [],
                "bf16": is_torch_bf16_gpu_available(),
                "tf32": is_torch_tf32_available(),
                "dataloader_pin_memory": True,
                "remove_unused_columns": False,
                "evaluation_strategy": "steps" if model_info.get("eval_dataset") else "no",
                "eval_steps": save_steps if model_info.get("eval_dataset") else None,
                "load_best_model_at_end": bool(model_info.get("eval_dataset")),
                "metric_for_best_model": "eval_loss" if model_info.get("eval_dataset") else None,
                "greater_is_better": False if model_info.get("eval_dataset") else None,
            }
            
            # Update model info with training configuration
            model_info["training_args"] = training_args
            model_info["output_dir"] = output_dir
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            logger.info(f"Model {model_id} prepared for training")
            logger.info(f"Training configuration: {training_args}")
            
            return {
                "status": "success",
                "message": f"Model {model_id} prepared for training",
                "config": training_args
            }
            
        except Exception as e:
            logger.error(f"Error preparing model for training: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Failed to prepare model for training: {str(e)}"
            }
    
    @mcp.tool()
    async def dora_train(
        model_id: str,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        **training_kwargs
    ) -> Dict[str, Any]:
        """Train a DoRA model.
        
        Args:
            model_id: ID of the loaded model
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            **training_kwargs: Additional training arguments
            
        Returns:
            Dictionary with training results
        """
        try:
            if model_id not in _dora_models:
                return {
                    "status": "error",
                    "message": f"Model with ID {model_id} not found"
                }
            
            model_info = _dora_models[model_id]
            model = model_info["model"]
            
            # Update training arguments with any overrides
            training_args = model_info.get("training_args", {})
            training_args.update(training_kwargs)
            
            # Store datasets
            model_info["train_dataset"] = train_dataset
            if eval_dataset is not None:
                model_info["eval_dataset"] = eval_dataset
            
            # Set up training arguments
            training_args = TrainingArguments(**training_args)
            
            # Set up trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset if eval_dataset is not None else None,
            )
            
            # Train the model
            logger.info(f"Starting training for model {model_id}")
            train_result = trainer.train()
            
            # Save the final model
            trainer.save_model(model_info["output_dir"])
            trainer.save_state()
            
            # Log training metrics
            metrics = train_result.metrics
            metrics["train_samples"] = len(train_dataset)
            
            if eval_dataset is not None:
                eval_metrics = trainer.evaluate()
                metrics.update(eval_metrics)
            
            # Save metrics
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            
            logger.info(f"Training completed for model {model_id}")
            
            return {
                "status": "success",
                "metrics": metrics,
                "output_dir": model_info["output_dir"]
            }
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Training failed: {str(e)}"
            }
    
    @mcp.tool()
    async def dora_unload_model(model_id: str) -> Dict[str, Any]:
        """Unload a DoRA model.
        
        Args:
            model_id: ID of the model to unload
            
        Returns:
            Dictionary with status message
        """
        try:
            if model_id not in _dora_models:
                return {
                    "status": "error",
                    "message": f"Model with ID {model_id} not found"
                }
            
            # Clean up model resources
            model_info = _dora_models.pop(model_id)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Model {model_id} unloaded")
            
            return {
                "status": "success",
                "message": f"Model {model_id} unloaded"
            }
            
        except Exception as e:
            logger.error(f"Error unloading model: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Failed to unload model: {str(e)}"
            }
    
    @mcp.tool()
    async def dora_list_models() -> Dict[str, Any]:
        """List all loaded DoRA models.
        
        Returns:
            Dictionary with loaded model information
        """
        try:
            models_info = {}
            for model_id, model_info in _dora_models.items():
                models_info[model_id] = {
                    "config": model_info.get("config"),
                    "device": model_info.get("device", "unknown"),
                    "has_train_dataset": "train_dataset" in model_info,
                    "has_eval_dataset": "eval_dataset" in model_info,
                    "training_args": model_info.get("training_args")
                }
            
            return {
                "status": "success",
                "models": models_info
            }
            
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Failed to list models: {str(e)}"
            }
    
    # Start the background task for system metrics
    asyncio.create_task(collect_system_metrics())
    
    logger.info("DoRA tools registered")
    return mcp
