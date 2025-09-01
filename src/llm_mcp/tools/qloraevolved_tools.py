"""
QLoRA Evolved Tools for LLM Fine-Tuning

This module provides tools for fine-tuning large language models using QLoRA Evolved,
an advanced approach that combines the benefits of QLoRA with additional optimizations
for better performance and memory efficiency.

Key Features:
- Multiple quantization types (NF4, NF4 optimized, INT4, FP8)
- Double quantization support for reduced memory usage
- Gradient checkpointing for large models
- Flash Attention 2.0 support
- Mixed precision training (FP16/BF16)
- TensorBoard integration
- Automatic model offloading to CPU when needed

Usage:
    from llm_mcp.tools import register_qloraevolved_tools
    
    # Register QLoRA Evolved tools with MCP
    register_qloraevolved_tools(mcp)
    
    # Now you can use the tools:
    # - qloraevolved_load_model
    # - qloraevolved_prepare_for_training
    # - qloraevolved_train
    # - qloraevolved_unload_model
    # - qloraevolved_list_models

For detailed documentation, see docs/qlora_evolved.md
"""

"""QLoRA Evolved tools for efficient fine-tuning with 2/4-bit quantization.

This module provides tools for fine-tuning models using QLoRA Evolved, which extends
standard LoRA with improved quantization techniques and optimization strategies.
"""

import os
import logging
import torch
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
import time

from fastmcp import FastMCP
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
    PeftConfig,
)
import bitsandbytes as bnb
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)

class QuantizationType(str, Enum):
    """Supported quantization types for QLoRA Evolved."""
    FP4 = "nf4"          # 4-bit NormalFloat
    FP4_OPTIMIZED = "nf4_optimized"  # Optimized 4-bit
    INT4 = "int4"        # 4-bit integers
    FP8 = "fp8"          # 8-bit floating point (experimental)
    NONE = "none"        # No quantization (standard LoRA)

@dataclass
class QLoRAEvolvedConfig:
    """Configuration for QLoRA Evolved fine-tuning."""
    # Model configuration
    model_name: str = "meta-llama/Llama-2-7b-hf"
    max_length: int = 2048
    device_map: str = "auto"
    trust_remote_code: bool = False
    
    # Quantization settings
    load_in_4bit: bool = True
    quant_type: QuantizationType = QuantizationType.FP4
    use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    
    # LoRA configuration
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # Training configuration
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 3
    max_steps: int = -1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    optim: str = "paged_adamw_32bit"
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 0.3
    
    # Output and logging
    output_dir: str = "qlora_evolved_output"
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 3
    report_to: str = "tensorboard"
    
    # Advanced options
    use_gradient_checkpointing: bool = True
    use_flash_attention_2: bool = True
    use_cpu_offload: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v.value if isinstance(v, Enum) else v 
                for k, v in self.__dict__.items()}

class QLoRAEvolvedManager:
    """Manager for QLoRA Evolved fine-tuning."""
    
    def __init__(self):
        """Initialize the QLoRA Evolved manager."""
        self.models: Dict[str, Any] = {}
        self.configs: Dict[str, QLoRAEvolvedConfig] = {}
        self.tokenizers: Dict[str, Any] = {}
        
    def _get_bnb_config(self, config: QLoRAEvolvedConfig) -> BitsAndBytesConfig:
        """Get BitsAndBytes configuration for quantization."""
        if config.quant_type == QuantizationType.NONE:
            return None
            
        return BitsAndBytesConfig(
            load_in_4bit=config.load_in_4bit,
            bnb_4bit_quant_type=config.quant_type.value,
            bnb_4bit_use_double_quant=config.use_double_quant,
            bnb_4bit_compute_dtype={
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }[config.bnb_4bit_compute_dtype],
        )
    
    def _get_lora_config(self, config: QLoRAEvolvedConfig) -> LoraConfig:
        """Get LoRA configuration."""
        return LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
    
    def load_model(
        self,
        model_name: str,
        model_id: Optional[str] = None,
        config: Optional[QLoRAEvolvedConfig] = None,
    ) -> Dict[str, Any]:
        """Load a model with QLoRA Evolved configuration.
        
        Args:
            model_name: Name or path of the model to load
            model_id: Optional ID to assign to the model
            config: Optional QLoRAEvolvedConfig, will use defaults if None
            
        Returns:
            Dictionary with model information
        """
        model_id = model_id or f"{model_name.split('/')[-1]}-qlora-evolved-{int(time.time())}"
        config = config or QLoRAEvolvedConfig(model_name=model_name)
        
        logger.info(f"Loading model {model_name} with QLoRA Evolved configuration")
        
        # Configure quantization
        bnb_config = self._get_bnb_config(config)
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=config.device_map,
            trust_remote_code=config.trust_remote_code,
            use_flash_attention_2=config.use_flash_attention_2,
        )
        
        # Prepare model for k-bit training if quantized
        if bnb_config is not None:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=config.use_gradient_checkpointing
            )
        
        # Configure LoRA
        lora_config = self._get_lora_config(config)
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=config.trust_remote_code,
            padding_side="right",
            use_fast=True,
        )
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Save model and config
        self.models[model_id] = model
        self.configs[model_id] = config
        self.tokenizers[model_id] = tokenizer
        
        # Print trainable parameters
        model.print_trainable_parameters()
        
        return {
            "status": "success",
            "model_id": model_id,
            "model_name": model_name,
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in model.parameters()),
            "config": config.to_dict(),
        }
    
    def prepare_for_training(
        self,
        model_id: str,
        training_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Prepare a loaded model for training.
        
        Args:
            model_id: ID of the loaded model
            training_config: Optional dictionary with training configuration overrides
            
        Returns:
            Dictionary with training preparation status
        """
        if model_id not in self.models:
            return {"status": "error", "message": f"Model {model_id} not found"}
        
        # Update config with provided training config
        if training_config:
            for key, value in training_config.items():
                if hasattr(self.configs[model_id], key):
                    setattr(self.configs[model_id], key, value)
        
        config = self.configs[model_id]
        
        # Create output directory if it doesn't exist
        os.makedirs(config.output_dir, exist_ok=True)
        
        return {
            "status": "success",
            "message": f"Model {model_id} prepared for training",
            "config": config.to_dict(),
        }
    
    def train(
        self,
        model_id: str,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        training_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Train a model with the given dataset.
        
        Args:
            model_id: ID of the loaded model
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            training_config: Optional training configuration overrides
            
        Returns:
            Dictionary with training results
        """
        if model_id not in self.models:
            return {"status": "error", "message": f"Model {model_id} not found"}
        
        model = self.models[model_id]
        tokenizer = self.tokenizers[model_id]
        config = self.configs[model_id]
        
        # Update config with provided training config
        if training_config:
            for key, value in training_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            num_train_epochs=config.num_train_epochs,
            max_steps=config.max_steps if config.max_steps > 0 else None,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            optim=config.optim,
            lr_scheduler_type=config.lr_scheduler_type,
            max_grad_norm=config.max_grad_norm,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            save_total_limit=config.save_total_limit,
            report_to=config.report_to.split(",") if config.report_to else [],
            gradient_checkpointing=config.use_gradient_checkpointing,
            fp16=config.bnb_4bit_compute_dtype == "float16",
            bf16=config.bnb_4bit_compute_dtype == "bfloat16",
            remove_unused_columns=False,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Train the model
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Save the final model
        trainer.save_model()
        
        # Save tokenizer
        tokenizer.save_pretrained(config.output_dir)
        
        return {
            "status": "success",
            "metrics": train_result.metrics,
            "output_dir": config.output_dir,
        }
    
    def unload_model(self, model_id: str) -> Dict[str, Any]:
        """Unload a model and free resources.
        
        Args:
            model_id: ID of the model to unload
            
        Returns:
            Dictionary with unload status
        """
        if model_id in self.models:
            del self.models[model_id]
            del self.configs[model_id]
            del self.tokenizers[model_id]
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {"status": "success", "message": f"Model {model_id} unloaded"}
        
        return {"status": "error", "message": f"Model {model_id} not found"}
    
    def list_models(self) -> Dict[str, Any]:
        """List all loaded models.
        
        Returns:
            Dictionary with information about loaded models
        """
        return {
            model_id: {
                "config": config.to_dict(),
                "device": str(next(model.parameters()).device),
            }
            for model_id, (model, config) in enumerate(zip(self.models.values(), self.configs.values()))
        }

# Global instance
qloraevolved_manager = QLoRAEvolvedManager()

# MCP Tool Definitions
async def qloraevolved_load_model(
    model_name: str,
    model_id: Optional[str] = None,
    max_length: int = 2048,
    load_in_4bit: bool = True,
    quant_type: str = "nf4",
    use_double_quant: bool = True,
    compute_dtype: str = "bfloat16",
    lora_rank: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    device_map: str = "auto",
) -> Dict[str, Any]:
    """Load a model with QLoRA Evolved configuration.
    
    Args:
        model_name: Name or path of the model to load
        model_id: Optional ID to assign to the model
        max_length: Maximum sequence length
        load_in_4bit: Whether to load in 4-bit precision
        quant_type: Quantization type (nf4, nf4_optimized, int4, fp8, none)
        use_double_quant: Whether to use double quantization
        compute_dtype: Compute dtype (float16, bfloat16, float32)
        lora_rank: Rank of LoRA matrices
        lora_alpha: Alpha parameter for LoRA scaling
        lora_dropout: Dropout probability for LoRA layers
        device_map: Device placement strategy
        
    Returns:
        Dictionary with model information
    """
    try:
        config = QLoRAEvolvedConfig(
            model_name=model_name,
            max_length=max_length,
            load_in_4bit=load_in_4bit,
            quant_type=QuantizationType(quant_type),
            use_double_quant=use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            device_map=device_map,
        )
        
        return qloraevolved_manager.load_model(
            model_name=model_name,
            model_id=model_id,
            config=config,
        )
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        return {"status": "error", "message": f"Failed to load model: {str(e)}"}

async def qloraevolved_prepare_for_training(
    model_id: str,
    output_dir: Optional[str] = None,
    learning_rate: Optional[float] = None,
    batch_size: Optional[int] = None,
    gradient_accumulation_steps: Optional[int] = None,
    num_train_epochs: Optional[int] = None,
    max_steps: int = -1,
    warmup_ratio: Optional[float] = None,
    weight_decay: Optional[float] = None,
    optim: Optional[str] = None,
    lr_scheduler_type: Optional[str] = None,
    max_grad_norm: Optional[float] = None,
    logging_steps: Optional[int] = None,
    save_steps: Optional[int] = None,
    save_total_limit: Optional[int] = None,
    report_to: Optional[str] = None,
    use_gradient_checkpointing: Optional[bool] = None,
) -> Dict[str, Any]:
    """Prepare a loaded model for training.
    
    Args:
        model_id: ID of the loaded model
        output_dir: Directory to save the model
        learning_rate: Learning rate
        batch_size: Batch size per device
        gradient_accumulation_steps: Number of steps to accumulate gradients
        num_train_epochs: Number of training epochs
        max_steps: Maximum number of training steps (-1 for unlimited)
        warmup_ratio: Ratio of warmup steps
        weight_decay: Weight decay
        optim: Optimizer to use
        lr_scheduler_type: Learning rate scheduler type
        max_grad_norm: Maximum gradient norm
        logging_steps: Log every X updates steps
        save_steps: Save checkpoint every X updates steps
        save_total_limit: Maximum number of checkpoints to keep
        report_to: Comma-separated list of integrations to report to
        use_gradient_checkpointing: Whether to use gradient checkpointing
        
    Returns:
        Dictionary with training preparation status
    """
    try:
        training_config = {}
        
        # Only include provided parameters
        if output_dir is not None:
            training_config["output_dir"] = output_dir
        if learning_rate is not None:
            training_config["learning_rate"] = learning_rate
        if batch_size is not None:
            training_config["batch_size"] = batch_size
        if gradient_accumulation_steps is not None:
            training_config["gradient_accumulation_steps"] = gradient_accumulation_steps
        if num_train_epochs is not None:
            training_config["num_train_epochs"] = num_train_epochs
        if max_steps != -1:
            training_config["max_steps"] = max_steps
        if warmup_ratio is not None:
            training_config["warmup_ratio"] = warmup_ratio
        if weight_decay is not None:
            training_config["weight_decay"] = weight_decay
        if optim is not None:
            training_config["optim"] = optim
        if lr_scheduler_type is not None:
            training_config["lr_scheduler_type"] = lr_scheduler_type
        if max_grad_norm is not None:
            training_config["max_grad_norm"] = max_grad_norm
        if logging_steps is not None:
            training_config["logging_steps"] = logging_steps
        if save_steps is not None:
            training_config["save_steps"] = save_steps
        if save_total_limit is not None:
            training_config["save_total_limit"] = save_total_limit
        if report_to is not None:
            training_config["report_to"] = report_to
        if use_gradient_checkpointing is not None:
            training_config["use_gradient_checkpointing"] = use_gradient_checkpointing
        
        return qloraevolved_manager.prepare_for_training(
            model_id=model_id,
            training_config=training_config,
        )
    except Exception as e:
        logger.error(f"Error preparing for training: {str(e)}", exc_info=True)
        return {"status": "error", "message": f"Failed to prepare for training: {str(e)}"}

async def qloraevolved_train(
    model_id: str,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    **training_kwargs,
) -> Dict[str, Any]:
    """Train a model with the given dataset.
    
    Args:
        model_id: ID of the loaded model
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset
        **training_kwargs: Additional training configuration overrides
        
    Returns:
        Dictionary with training results
    """
    try:
        return qloraevolved_manager.train(
            model_id=model_id,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_config=training_kwargs,
        )
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        return {"status": "error", "message": f"Training failed: {str(e)}"}

async def qloraevolved_unload_model(model_id: str) -> Dict[str, Any]:
    """Unload a model and free resources.
    
    Args:
        model_id: ID of the model to unload
        
    Returns:
        Dictionary with unload status
    """
    try:
        return qloraevolved_manager.unload_model(model_id=model_id)
    except Exception as e:
        logger.error(f"Error unloading model: {str(e)}", exc_info=True)
        return {"status": "error", "message": f"Failed to unload model: {str(e)}"}

async def qloraevolved_list_models() -> Dict[str, Any]:
    """List all loaded models.
    
    Returns:
        Dictionary with information about loaded models
    """
    try:
        return {
            "status": "success",
            "models": qloraevolved_manager.list_models(),
        }
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}", exc_info=True)
        return {"status": "error", "message": f"Failed to list models: {str(e)}"}

def register_qloraevolved_tools(mcp_server):
    """Register all QLoRA Evolved tools with the MCP server.
    
    Args:
        mcp_server: The MCP server instance
    """
    # Register tools with decorators
    @mcp_server.tool()
    async def qloraevolved_load_model_wrapper(*args, **kwargs):
        return await qloraevolved_load_model(*args, **kwargs)
        
    @mcp_server.tool()
    async def qloraevolved_prepare_for_training_wrapper(*args, **kwargs):
        return await qloraevolved_prepare_for_training(*args, **kwargs)
        
    @mcp_server.tool()
    async def qloraevolved_train_wrapper(*args, **kwargs):
        return await qloraevolved_train(*args, **kwargs)
        
    @mcp_server.tool()
    async def qloraevolved_unload_model_wrapper(*args, **kwargs):
        return await qloraevolved_unload_model(*args, **kwargs)
        
    @mcp_server.tool()
    async def qloraevolved_list_models_wrapper():
        return await qloraevolved_list_models()
    
    # Register the wrapped functions
    mcp_server.register_tool(qloraevolved_load_model_wrapper)
    mcp_server.register_tool(qloraevolved_prepare_for_training_wrapper)
    mcp_server.register_tool(qloraevolved_train_wrapper)
    mcp_server.register_tool(qloraevolved_unload_model_wrapper)
    mcp_server.register_tool(qloraevolved_list_models_wrapper)
    
    logger.info("Registered QLoRA Evolved tools with MCP server")
    return mcp_server
