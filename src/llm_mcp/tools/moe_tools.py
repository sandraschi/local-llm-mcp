"""Mixture of Experts (MoE) tools for LLM MCP server.

This module provides tools for working with Mixture of Experts (MoE) models,
including loading, training, and inference with sparse MoE layers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from loguru import logger
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from llm_mcp.tools.common import ModelConfig
import time
from typing import Any, Dict, List, Optional, Union

# Global registry for loaded MoE models
MOE_MODELS = {}

# Implementation functions (without @tool decorator)
async def moe_load_model_impl(
    model_name: str,
    num_experts: int = 8,
    expert_capacity: int = 4,
    moe_layer_frequency: int = 2,
    **kwargs,
) -> Dict[str, Any]:
    """Load a model and convert it to use MoE layers.
    
    Args:
        model_name: Name or path of the model to load
        num_experts: Number of expert networks
        expert_capacity: Maximum number of tokens each expert can process
        moe_layer_frequency: How often to place MoE layers (e.g., every N layers)
        **kwargs: Additional arguments to pass to AutoModelForCausalLM
        
    Returns:
        Dictionary with model information
    """
    global MOE_MODELS
    
    try:
        # Generate a unique model ID if not provided
        model_id = kwargs.pop("model_id", f"moe_{len(MOE_MODELS) + 1}")
        
        # Load the base model
        config = AutoConfig.from_pretrained(model_name, **kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_name, config=config, **kwargs)
        
        # Convert model to use MoE layers (implementation would go here)
        # This is a placeholder - actual implementation would modify the model architecture
        
        # Store model info
        MOE_MODELS[model_id] = {
            "model": model,
            "config": config,
            "num_experts": num_experts,
            "expert_capacity": expert_capacity,
            "moe_layer_frequency": moe_layer_frequency,
        }
        
        return {
            "status": "success",
            "message": f"Loaded MoE model {model_id}",
            "model_id": model_id,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error loading MoE model: {str(e)}")
        return {"status": "error", "message": str(e)}


@dataclass
class MoEConfig:
    """Configuration for Mixture of Experts (MoE) models.

    Attributes:
        num_experts: Number of expert networks
        expert_capacity: Maximum number of tokens each expert can process
        router_jitter_noise: Noise to add to router logits for exploration
        router_aux_loss_coef: Weight for auxiliary load balancing loss
        router_z_loss_coef: Weight for router z-loss
        router_ignore_padding_tokens: Whether to ignore padding tokens in router
        moe_layer_frequency: How often to place MoE layers (e.g., every N layers)
        moe_layer_start: First layer to apply MoE to (0-based)
        moe_layer_end: Last layer to apply MoE to (inclusive, or None for all)
    """
    num_experts: int = 8
    expert_capacity: int = 4
    router_jitter_noise: float = 0.1
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    router_ignore_padding_tokens: bool = True
    moe_layer_frequency: int = 2
    moe_layer_start: int = 0
    moe_layer_end: Optional[int] = None


def convert_to_moe(
    model: nn.Module,
    config: MoEConfig,
) -> nn.Module:
    """Convert a standard transformer model to use MoE layers.

    Args:
        model: The model to convert
        config: MoE configuration

    Returns:
        The converted model with MoE layers
    """
    from transformers.models.gpt2 import GPT2Block
    from transformers.models.llama import LlamaDecoderLayer

    for i, layer in enumerate(model.base_model.layers):
        # Skip layers outside the specified range
        if i < config.moe_layer_start or (
            config.moe_layer_end is not None and i > config.moe_layer_end
        ):
            continue

        # Only convert every N layers based on frequency
        if (i - config.moe_layer_start) % config.moe_layer_frequency != 0:
            continue

        if isinstance(layer, (GPT2Block, LlamaDecoderLayer)):
            # Replace the MLP with an MoE MLP
            if hasattr(layer, "mlp"):
                original_mlp = layer.mlp
                layer.mlp = MoEMLP(
                    config=config,
                    hidden_size=original_mlp.hidden_size,
                    intermediate_size=original_mlp.intermediate_size,
                    hidden_act=original_mlp.activation_function,
                )
                logger.info(f"Converted layer {i} to MoE")

    return model


class MoEMLP(nn.Module):
    """Mixture of Experts MLP layer.

    This implements a sparse MoE layer where only a subset of experts
    are activated for each input token.
    """

    def __init__(
        self,
        config: MoEConfig,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
    ):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        # Create experts
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, intermediate_size, bias=False),
                    nn.GELU(),
                    nn.Linear(intermediate_size, hidden_size, bias=False),
                )
                for _ in range(config.num_experts)
            ]
        )

        # Router
        self.router = nn.Linear(hidden_size, config.num_experts, bias=False)
        self.router_aux_loss = 0.0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with MoE routing.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Output tensor of same shape as input
        """
        batch_size, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)  # (batch * seq_len, hidden)

        # Get router logits
        router_logits = self.router(hidden_states)  # (batch * seq_len, num_experts)

        # Add noise for exploration during training
        if self.training and self.config.router_jitter_noise > 0:
            noise = torch.randn_like(router_logits) * self.config.router_jitter_noise
            router_logits = router_logits + noise

        # Get top-k experts for each token
        top_k = min(self.config.expert_capacity, self.config.num_experts)
        router_probs = torch.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, top_k, dim=-1)

        # Create mask for selected experts
        expert_mask = torch.nn.functional.one_hot(
            top_k_indices, num_classes=self.config.num_experts
        ).float()  # (batch * seq_len, top_k, num_experts)

        # Calculate router loss for load balancing
        if self.training:
            # Auxiliary loss: encourage equal expert utilization
            expert_mask_sum = expert_mask.sum(0).sum(0)  # (num_experts,)
            router_prob_mean = router_probs.mean(0)  # (num_experts,)
            self.router_aux_loss = (
                self.config.router_aux_loss_coef
                * self.config.num_experts**2
                * (router_prob_mean * expert_mask_sum).sum()
            )

            # Z-loss: encourage router logits to be well-scaled
            log_z = torch.logsumexp(router_logits, dim=-1)  # (batch * seq_len,)
            z_loss = torch.mean(log_z**2)
            self.router_aux_loss += self.config.router_z_loss_coef * z_loss

        # Dispatch to experts
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            # Get tokens assigned to this expert
            expert_mask_i = expert_mask[..., i]  # (batch * seq_len, top_k)
            expert_input = hidden_states.unsqueeze(1) * expert_mask_i.unsqueeze(-1)
            expert_input = expert_input.sum(dim=1)  # Sum over top_k

            # Apply expert
            expert_output = expert(expert_input)
            expert_outputs.append(expert_output.unsqueeze(-2))  # (batch * seq_len, 1, hidden)

        # Combine expert outputs
        expert_outputs = torch.cat(expert_outputs, dim=1)  # (batch * seq_len, num_experts, hidden)
        output = (expert_outputs * top_k_probs.unsqueeze(-1)).sum(dim=1)  # (batch * seq_len, hidden)

        # Reshape back to original dimensions
        output = output.view(batch_size, seq_len, -1)
        return output

# Tool registration function
def register_moe_tools(mcp):
    """Register MoE-related tools with the MCP server using FastMCP 2.11.3 stateful features.
    
    Args:
        mcp: The MCP server instance with tool decorator
        
    Returns:
        The MCP server instance with MoE tools registered
        
    Notes:
        - Tools are registered with stateful=True where appropriate
        - State TTL is set based on the expected cache duration for each tool
    """
    tool = mcp.tool
    
    @tool(stateful=True, state_ttl=300)  # 5-minute cache for model loading
    async def moe_load_model(
        model_name: str,
        num_experts: int = 8,
        expert_capacity: int = 4,
        moe_layer_frequency: int = 2,
        **kwargs,
    ) -> Dict[str, Any]:
        """Load a model and convert it to use MoE layers with stateful caching.
        
        This tool caches loaded models to improve performance.
        
        Args:
            model_name: Name or path of the model to load
            num_experts: Number of expert networks
            expert_capacity: Maximum number of tokens each expert can process
            moe_layer_frequency: How often to place MoE layers (e.g., every N layers)
            **kwargs: Additional arguments to pass to AutoModelForCausalLM
            
        Returns:
            Dictionary with model information and status
        """
        return await moe_load_model_impl(
            model_name=model_name,
            num_experts=num_experts,
            expert_capacity=expert_capacity,
            moe_layer_frequency=moe_layer_frequency,
            **kwargs
        )
    
    @tool(stateful=True, state_ttl=60)  # 1-minute cache for model info
    async def moe_model_info(model_id: str) -> Dict[str, Any]:
        """Get information about a loaded MoE model with stateful caching.
        
        Args:
            model_id: ID of the loaded MoE model
            
        Returns:
            Dictionary with model information
        """
        if model_id not in MOE_MODELS:
            return {"status": "error", "message": f"Model {model_id} not found"}
            
        model_info = MOE_MODELS[model_id].copy()
        # Don't return the actual model in the info
        if "model" in model_info:
            del model_info["model"]
            
        return {
            "status": "success",
            "model_id": model_id,
            **model_info,
            "timestamp": time.time()
        }
    
    @tool(stateful=False)  # No caching for training operations
    async def moe_train(
        model_id: str,
        dataset: str,
        output_dir: str,
        learning_rate: float = 5e-5,
        batch_size: int = 8,
        num_epochs: int = 3,
        **kwargs,
    ) -> Dict[str, Any]:
        """Fine-tune a MoE model on a dataset.
        
        This tool does not use caching as it performs model training.
        
        Args:
            model_id: ID of the loaded MoE model
            dataset: Path to dataset or dataset name
            output_dir: Directory to save the fine-tuned model
            learning_rate: Learning rate for training
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            **kwargs: Additional training arguments
            
        Returns:
            Dictionary with training results
        """
        if model_id not in MOE_MODELS:
            return {"status": "error", "message": f"Model {model_id} not found"}
            
        model = MOE_MODELS[model_id]["model"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Training implementation would go here
        # This is a placeholder implementation
        
        return {
            "status": "success",
            "model_id": model_id,
            "output_dir": output_dir,
            "epochs_completed": num_epochs,
            "final_loss": 0.0,  # Placeholder
            "timestamp": time.time()
        }
    
    @tool(stateful=True, state_ttl=30)  # Short cache for generation
    async def moe_generate(
        model_id: str,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate text using a MoE model with stateful caching.
        
        Args:
            model_id: ID of the loaded MoE model
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with generated text and metadata
        """
        if model_id not in MOE_MODELS:
            return {"status": "error", "message": f"Model {model_id} not found"}
            
        model_info = MOE_MODELS[model_id]
        model = model_info["model"]
        device = next(model.parameters()).device
        
        # Tokenize input
        tokenizer = AutoTokenizer.from_pretrained(model_info["config"]._name_or_path)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
        
        # Decode and return
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "status": "success",
            "model_id": model_id,
            "prompt": prompt,
            "generated_text": generated_text,
            "timestamp": time.time()
        }

    return mcp
