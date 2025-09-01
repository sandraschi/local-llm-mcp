# Sparse Fine-Tuning

Sparse Fine-Tuning is an advanced technique that reduces the computational cost of fine-tuning by dynamically pruning and regrowing connections during training. This document provides a comprehensive guide to using sparse fine-tuning with the LLM MCP server.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration Options](#configuration-options)
- [Advanced Usage](#advanced-usage)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## Overview

Sparse Fine-Tuning works by maintaining a dynamic mask over the model's parameters, where only a subset of weights are updated during each training step. This approach can reduce training compute by up to 90% while maintaining model quality.

Key papers:
- [Rigging the Lottery: Making All Tickets Winners](https://arxiv.org/abs/2110.07634)
- [The State of Sparsity in Deep Neural Networks](https://arxiv.org/abs/1902.09574)

## Key Features

- **Dynamic Sparsity**: Automatically prunes and regrows connections during training
- **Multiple Sparsity Patterns**: Support for unstructured, 2:4, and 4:8 sparsity
- **Gradient Accumulation**: Efficient training with large effective batch sizes
- **Mixed Precision Training**: Reduced memory usage with FP16/BF16 support
- **Integration with Hugging Face**: Works with any 🤗 Transformers model

## Installation

Sparse fine-tuning requires PyTorch 1.12+ and the following additional dependencies:

```bash
pip install torch>=1.12.0
pip install transformers>=4.30.0
pip install bitsandbytes>=0.40.0
```

## Quick Start

Here's how to start sparse fine-tuning with a single command:

```python
from llm_mcp.tools.sparse_tools import sparse_finetune

# Load and fine-tune a model with default sparsity settings
model = sparse_finetune(
    model_name="meta-llama/Llama-2-7b-hf",
    dataset="your-dataset",
    sparsity_ratio=0.5,
    output_dir="./sparse_model"
)
```

## Configuration Options

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sparsity_ratio` | float | 0.5 | Target sparsity ratio (0.0 to 1.0) |
| `sparsity_type` | str | "unstructured" | Type of sparsity ("unstructured", "2:4", "4:8") |
| `mask_update_interval` | int | 100 | Steps between mask updates |
| `mask_update_fraction` | float | 0.3 | Fraction of weights to update each interval |
| `use_rigl` | bool | True | Use RigL algorithm for sparse training |
| `use_topk_attention` | bool | True | Use top-k attention for sparse attention |
| `topk_ratio` | float | 0.1 | Ratio of attention heads to keep |

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `learning_rate` | float | 5e-5 | Initial learning rate |
| `batch_size` | int | 8 | Training batch size |
| `num_epochs` | int | 3 | Number of training epochs |
| `warmup_steps` | int | 100 | Number of warmup steps |
| `gradient_accumulation_steps` | int | 4 | Number of steps for gradient accumulation |
| `fp16` | bool | True | Use mixed precision training |
| `gradient_checkpointing` | bool | True | Use gradient checkpointing |

## Advanced Usage

### Custom Sparsity Schedule

You can define a custom sparsity schedule using a function:

```python
def sparsity_schedule(step: int, total_steps: int) -> float:
    """Linear increase in sparsity from 0.1 to 0.9"""
    return 0.1 + 0.8 * min(1.0, step / total_steps)

model = sparse_finetune(
    model_name="meta-llama/Llama-2-7b-hf",
    dataset="your-dataset",
    sparsity_schedule=sparsity_schedule,
    output_dir="./sparse_model"
)
```

### Custom Pruning Criteria

You can implement custom pruning criteria by subclassing `BasePruningMethod`:

```python
from torch.nn.utils.prune import BasePruningMethod

class MagnitudePruning(BasePruningMethod):
    PRUNING_TYPE = 'unstructured'
    
    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        num_params = t.numel()
        k = int(num_params * (1 - self.amount))
        if k == 0:
            return mask
        
        # Keep the k largest magnitude weights
        topk = torch.topk(t.abs().view(-1), k=k, largest=True)
        mask.view(-1)[topk.indices] = 1
        return mask
```

## Best Practices

1. **Start with Pre-trained Weights**
   - Sparse fine-tuning works best when starting from a pre-trained model
   - Use the same architecture as your target task if possible

2. **Gradual Sparsity Increase**
   - Start with lower sparsity (10-30%) and gradually increase
   - Use a schedule to ramp up sparsity during training

3. **Learning Rate Scheduling**
   - Use a learning rate schedule with warmup
   - Consider higher learning rates than standard fine-tuning

4. **Regularization**
   - Use weight decay and dropout to prevent overfitting
   - Higher dropout rates (0.2-0.5) often work well with sparse training

## Troubleshooting

### Common Issues

1. **Training Instability**
   - Reduce learning rate
   - Decrease sparsity ratio
   - Increase warmup steps

2. **Memory Issues**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision training

3. **Slow Training**
   - Increase batch size if memory allows
   - Use a smaller model
   - Reduce sequence length

## References

1. [Rigging the Lottery: Making All Tickets Winners](https://arxiv.org/abs/2110.07634)
2. [The State of Sparsity in Deep Neural Networks](https://arxiv.org/abs/1902.09574)
3. [Sparse is Enough in Scaling Transformers](https://arxiv.org/abs/2111.12763)
4. [GitHub: facebookresearch/rigl](https://github.com/facebookresearch/rigl)
5. [GitHub: IST-DASLab/gptq](https://github.com/IST-DASLab/gptq)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
