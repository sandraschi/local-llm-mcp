# Mixture of Experts (MoE) Guide

This guide explains how to use the Mixture of Experts (MoE) implementation in the LLM MCP server. MoE models are designed to be more efficient than traditional dense models by activating only a subset of "expert" networks for each input.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration Options](#configuration-options)
- [Training with MoE](#training-with-moe)
- [Inference with MoE](#inference-with-moe)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## Overview

Mixture of Experts (MoE) is a neural network architecture that consists of multiple expert networks and a gating network that routes each input to the most relevant experts. This allows for larger model capacity without a proportional increase in computation.

Key papers:
- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)

## Key Features

- **Sparse Activation**: Only a subset of experts are activated for each input
- **Efficient Training**: Reduced computation compared to dense models of similar capacity
- **Flexible Configuration**: Customize number of experts, expert capacity, and routing strategy
- **Hugging Face Integration**: Works with any 🤗 Transformers model
- **Gradient Checkpointing**: Memory-efficient training
- **Mixed Precision Training**: Support for FP16/BF16

## Installation

MoE support requires PyTorch 1.12+ and the following dependencies:

```bash
pip install torch>=1.12.0
pip install transformers>=4.30.0
```

## Quick Start

### Loading a MoE Model

```python
from llm_mcp.tools.moe_tools import moe_load_model

# Load a base model and convert to MoE
model_info = await moe_load_model(
    model_name="meta-llama/Llama-2-7b-hf",
    num_experts=8,
    expert_capacity=4,
    moe_layer_frequency=2
)
```

### Training a MoE Model

```python
from llm_mcp.tools.moe_tools import moe_train

# Fine-tune the MoE model
training_result = await moe_train(
    model_id=model_info["model_id"],
    dataset="your-dataset",
    output_dir="./moe_model",
    learning_rate=5e-5,
    batch_size=8,
    num_epochs=3
)
```

### Generating Text

```python
from llm_mcp.tools.moe_tools import moe_generate

# Generate text with the MoE model
result = await moe_generate(
    model_id=model_info["model_id"],
    prompt="Once upon a time",
    max_length=100,
    temperature=0.7
)
print(result["generated_text"])
```

## Configuration Options

### MoE Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_experts` | int | 8 | Number of expert networks |
| `expert_capacity` | int | 4 | Maximum number of tokens each expert can process |
| `router_jitter_noise` | float | 0.1 | Noise to add to router logits for exploration |
| `router_aux_loss_coef` | float | 0.01 | Weight for auxiliary load balancing loss |
| `router_z_loss_coef` | float | 0.001 | Weight for router z-loss |
| `router_ignore_padding_tokens` | bool | True | Whether to ignore padding tokens in router |
| `moe_layer_frequency` | int | 2 | How often to place MoE layers (e.g., every N layers) |
| `moe_layer_start` | int | 0 | First layer to apply MoE to (0-based) |
| `moe_layer_end` | Optional[int] | None | Last layer to apply MoE to (inclusive, or None for all) |

## Training with MoE

### Data Preparation

MoE models work best with large, diverse datasets. The dataset should be in a format compatible with 🤗 Datasets:

```python
from datasets import load_dataset

dataset = load_dataset("your-dataset")
```

### Training Parameters

Key parameters for training MoE models:

```python
training_args = {
    "learning_rate": 5e-5,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 3,
    "fp16": True,  # Enable mixed precision training
    "gradient_checkpointing": True,  # Save memory
    "save_strategy": "epoch",
    "logging_steps": 10,
}
```

### Monitoring Training

During training, monitor these metrics:
- `loss`: Overall training loss
- `router_aux_loss`: Load balancing loss (should be minimized)
- `router_z_loss`: Router z-loss (should be minimized)
- `expert_utilization`: Percentage of experts being used

## Inference with MoE

### Batch Inference

For efficient inference, process multiple examples in a batch:

```python
results = []
prompts = ["First prompt", "Second prompt", "Third prompt"]

for prompt in prompts:
    result = await moe_generate(
        model_id=model_info["model_id"],
        prompt=prompt,
        max_length=100,
        temperature=0.7
    )
    results.append(result["generated_text"])
```

### Expert Activation Analysis

To analyze which experts are being used:

```python
# Get expert activations for a batch of inputs
with torch.no_grad():
    outputs = model(input_ids, output_router_logits=True)
    router_logits = outputs.router_logits  # (batch_size, seq_len, num_experts)
    expert_activations = (router_logits > 0).float().mean(dim=(0, 1))
    print("Expert utilization:", expert_activations.tolist())
```

## Best Practices

### Model Architecture
- Start with a smaller number of experts (4-8) and increase as needed
- Use expert capacity that's 1-2x your expected tokens per expert
- Place MoE layers in the middle layers of the network

### Training
- Use a higher learning rate than standard fine-tuning (2-5x)
- Enable gradient checkpointing to save memory
- Use mixed precision training (FP16/BF16) for faster training
- Monitor expert utilization to ensure all experts are being used

### Inference
- Use smaller batch sizes than with dense models
- Consider using top-k routing for more stable generations
- Cache expert outputs when possible for repeated sequences

## Troubleshooting

### Common Issues

1. **Low Expert Utilization**
   - Increase `router_aux_loss_coef`
   - Decrease number of experts
   - Increase batch size

2. **Training Instability**
   - Decrease learning rate
   - Increase `router_z_loss_coef`
   - Add gradient clipping

3. **Memory Issues**
   - Decrease batch size
   - Enable gradient checkpointing
   - Use smaller models or fewer experts

## References

1. [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
2. [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)
3. [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)
4. [GitHub: facebookresearch/fairseq](https://github.com/facebookresearch/fairseq/tree/main/examples/moe_lm)
5. [GitHub: laiguokun/moe](https://github.com/laiguokun/moe)
