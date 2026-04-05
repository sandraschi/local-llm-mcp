"""
Unsloth Tools for LLM MCP

This module provides optimized fine-tuning capabilities using Unsloth,
a highly efficient framework for fine-tuning large language models.
"""

import os
import logging
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import torch
from dataclasses import dataclass, field

# Try to import Unsloth
try:
    import unsloth
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    UNLOTH_AVAILABLE = True
except ImportError:
    UNLOTH_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class UnslothTrainingConfig:
    """Configuration for Unsloth fine-tuning."""
    max_seq_length: int = 2048
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 10
    max_steps: int = 60
    learning_rate: float = 2e-4
    fp16: bool = not torch.cuda.is_bf16_supported()
    bf16: bool = torch.cuda.is_bf16_supported()
    logging_steps: int = 1
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    save_steps: int = 100
    output_dir: str = "unsloth_finetuned_model"
    eval_steps: Optional[int] = None
    eval_strategy: str = "steps"
    load_best_model_at_end: bool = True
    report_to: str = "tensorboard"
    seed: int = 42
    use_gradient_checkpointing: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('__')}

class UnslothManager:
    """Manages Unsloth models and fine-tuning."""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.configs: Dict[str, UnslothTrainingConfig] = {}
        self.tokenizer = None
        
    def load_model(
        self,
        model_name: str,
        max_seq_length: int = 2048,
        dtype: Optional[torch.dtype] = None,
        load_in_4bit: bool = True,
        token: Optional[str] = None,
        device_map: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """Load a model with Unsloth optimizations.
        
        Args:
            model_name: Name or path of the model to load
            max_seq_length: Maximum sequence length
            dtype: Data type for model weights
            load_in_4bit: Whether to load in 4-bit precision
            token: Hugging Face auth token
            device_map: Device placement strategy
            **kwargs: Additional arguments for Unsloth's FastLanguageModel
            
        Returns:
            Dictionary with model information
        """
        if not UNLOTH_AVAILABLE:
            raise ImportError(
                "Unsloth is not installed. Install with: "
                "pip install git+https://github.com/unslothai/unsloth.git"
            )
            
        logger.info(f"Loading model {model_name} with Unsloth optimizations...")
        
        # Set default dtype if not specified
        if dtype is None:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            
        # Load the model with Unsloth optimizations
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            token=token,
            device_map=device_map,
            **kwargs
        )
        
        # Store model and tokenizer
        model_id = f"{model_name.split('/')[-1]}-unsloth"
        self.models[model_id] = {
            'model': model,
            'tokenizer': tokenizer,
            'config': {
                'model_name': model_name,
                'max_seq_length': max_seq_length,
                'dtype': str(dtype),
                'load_in_4bit': load_in_4bit,
                'device_map': device_map,
                **kwargs
            }
        }
        
        # Initialize default config if not exists
        if model_id not in self.configs:
            self.configs[model_id] = UnslothTrainingConfig()
        
        return {
            'status': 'success',
            'model_id': model_id,
            'model_name': model_name,
            'max_seq_length': max_seq_length,
            'dtype': str(dtype),
            'device': str(model.device)
        }
    
    def prepare_for_training(
        self,
        model_id: str,
        training_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Prepare the model for training with the given configuration.
        
        Args:
            model_id: ID of the loaded model
            training_config: Training configuration overrides
            
        Returns:
            Dictionary with training preparation status
        """
        if model_id not in self.models:
            return {'status': 'error', 'message': f'Model {model_id} not found'}
            
        model_info = self.models[model_id]
        
        # Update config if provided
        if training_config:
            if model_id in self.configs:
                for key, value in training_config.items():
                    if hasattr(self.configs[model_id], key):
                        setattr(self.configs[model_id], key, value)
            else:
                self.configs[model_id] = UnslothTrainingConfig(**training_config)
        
        # Get the model and tokenizer
        model = model_info['model']
        tokenizer = model_info['tokenizer']
        
        # Enable gradient checkpointing if specified
        if self.configs[model_id].use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        
        # Prepare model for training
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,  # LoRA rank
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            use_gradient_checkpointing=self.configs[model_id].use_gradient_checkpointing,
            random_state=self.configs[model_id].seed,
        )
        
        # Update model in storage
        self.models[model_id]['model'] = model
        
        return {
            'status': 'success',
            'message': f'Model {model_id} prepared for training',
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'total_params': sum(p.numel() for p in model.parameters())
        }
    
    def train(
        self,
        model_id: str,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        training_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Train the model on the given dataset.
        
        Args:
            model_id: ID of the loaded model
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            training_config: Training configuration overrides
            
        Returns:
            Dictionary with training results
        """
        if model_id not in self.models:
            return {'status': 'error', 'message': f'Model {model_id} not found'}
            
        # Update config if provided
        if training_config:
            self.prepare_for_training(model_id, training_config)
            
        model_info = self.models[model_id]
        config = self.configs[model_id]
        
        # Prepare trainer
        from transformers import TrainingArguments, Trainer
        
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            learning_rate=config.learning_rate,
            fp16=config.fp16,
            bf16=config.bf16,
            logging_steps=config.logging_steps,
            optim=config.optim,
            weight_decay=config.weight_decay,
            lr_scheduler_type=config.lr_scheduler_type,
            save_steps=config.save_steps,
            eval_steps=config.eval_steps,
            evaluation_strategy=config.eval_strategy if eval_dataset is not None else "no",
            load_best_model_at_end=config.load_best_model_at_end if eval_dataset is not None else False,
            report_to=config.report_to,
            seed=config.seed,
        )
        
        # Create trainer
        trainer = Trainer(
            model=model_info['model'],
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            tokenizer=model_info['tokenizer'],
        )
        
        # Train the model
        train_result = trainer.train()
        
        # Save the model
        output_dir = os.path.join(config.output_dir, "final")
        trainer.save_model(output_dir)
        
        return {
            'status': 'success',
            'metrics': train_result.metrics,
            'output_dir': output_dir
        }
    
    def unload_model(self, model_id: str) -> Dict[str, Any]:
        """Unload a model and free memory.
        
        Args:
            model_id: ID of the model to unload
            
        Returns:
            Status dictionary
        """
        if model_id in self.models:
            del self.models[model_id]
            if model_id in self.configs:
                del self.configs[model_id]
            return {'status': 'success', 'message': f'Model {model_id} unloaded'}
        return {'status': 'error', 'message': f'Model {model_id} not found'}
    
    def list_models(self) -> Dict[str, Any]:
        """List all loaded models.
        
        Returns:
            Dictionary of loaded models
        """
        return {
            model_id: {
                'config': model_info['config'],
                'device': str(model_info['model'].device)
            }
            for model_id, model_info in self.models.items()
        }

# Global instance
unsloth_manager = UnslothManager()

def register_unsloth_tools(mcp):
    """Register Unsloth tools with the MCP server.
    
    Args:
        mcp: The MCP server instance
        
    Returns:
        The MCP server with Unsloth tools registered
    """
    if not UNLOTH_AVAILABLE:
        logger.warning(
            "Unsloth is not installed. Unsloth tools will not be available. "
            "Install with: pip install git+https://github.com/unslothai/unsloth.git"
        )
        return mcp
    
    @mcp.tool()
    async def unsloth_load_model(
        model_name: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        token: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Load a model with Unsloth optimizations.
        
        Args:
            model_name: Name or path of the model to load
            max_seq_length: Maximum sequence length
            load_in_4bit: Whether to load in 4-bit precision
            token: Hugging Face auth token
            **kwargs: Additional arguments for Unsloth's FastLanguageModel
            
        Returns:
            Dictionary with model information
        """
        return unsloth_manager.load_model(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            token=token,
            **kwargs
        )
    
    @mcp.tool()
    async def unsloth_prepare_for_training(
        model_id: str,
        training_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Prepare a model for training.
        
        Args:
            model_id: ID of the loaded model
            training_config: Training configuration overrides
            
        Returns:
            Dictionary with preparation status
        """
        return unsloth_manager.prepare_for_training(model_id, training_config)
    
    @mcp.tool()
    async def unsloth_train(
        model_id: str,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        training_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Train a model on the given dataset.
        
        Args:
            model_id: ID of the loaded model
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            training_config: Training configuration overrides
            
        Returns:
            Dictionary with training results
        """
        return unsloth_manager.train(model_id, train_dataset, eval_dataset, training_config)
    
    @mcp.tool()
    async def unsloth_unload_model(model_id: str) -> Dict[str, Any]:
        """Unload a model and free memory.
        
        Args:
            model_id: ID of the model to unload
            
        Returns:
            Status dictionary
        """
        return unsloth_manager.unload_model(model_id)
    
    @mcp.tool()
    async def unsloth_list_models() -> Dict[str, Any]:
        """List all loaded models.
        
        Returns:
            Dictionary of loaded models
        """
        return unsloth_manager.list_models()
    
    return mcp
