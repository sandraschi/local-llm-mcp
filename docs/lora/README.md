# LoRA Integration

This document explains how to integrate and use LoRA (Low-Rank Adaptation) with the LLM MCP server for fine-tuning models with tool calling capabilities.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Training with Tool Data](#training-with-tool-data)
- [Saving and Loading Adapters](#saving-and-loading-adapters)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that significantly reduces the number of trainable parameters by learning low-rank updates to the model's weights.

## Installation

```bash
pip install peft torch
```

## Basic Usage

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Load your base model
model = AutoModelForCausalLM.from_pretrained("your-base-model")

# Define LoRA config
lora_config = LoraConfig(
    r=8,  # Rank
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
```

## Training with Tool Data

When fine-tuning with tool usage data, structure your dataset as follows:

```python
training_examples = [
    {
        "input": "What's the weather in Tokyo?",
        "output": "",
        "tool_calls": [{
            "name": "get_weather",
            "arguments": {"location": "Tokyo", "unit": "celsius"}
        }]
    },
    # More examples...
]
```

## Saving and Loading Adapters

### Saving
```python
model.save_pretrained("./lora-adapter")
```

### Loading
```python
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("your-base-model")
model = PeftModel.from_pretrained(model, "./lora-adapter")
```

## Best Practices

1. **Start Small**: Begin with a small rank (4-8) and increase if needed
2. **Target Layers**: Focus on attention layers (q_proj, v_proj) for best results
3. **Learning Rate**: Use a higher learning rate than full fine-tuning (1e-4 to 5e-4)
4. **Batch Size**: Use the largest batch size that fits in GPU memory
5. **Gradient Accumulation**: For larger effective batch sizes
6. **Checkpointing**: Save checkpoints during training

## Troubleshooting

### Common Issues
1. **OOM Errors**: Reduce batch size or use gradient accumulation
2. **Poor Performance**: Try increasing rank or adjust learning rate
3. **Training Instability**: Lower learning rate or increase batch size

### Debugging
```python
# Check trainable parameters
model.print_trainable_parameters()

# Check model structure
print(model)
```

## Advanced Topics

### QLoRA for Memory Efficiency
```python
from transformers import BitsAndBytesConfig
import torch

# 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "your-model",
    quantization_config=bnb_config,
    device_map="auto"
)
```

### Multi-GPU Training
```python
# Use accelerate for distributed training
from accelerate import Accelerator

accelerator = Accelerator()
model = accelerator.prepare(model)
```

## See Also
- [PEFT Documentation](https://huggingface.co/docs/peft/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Hugging Face PEFT Examples](https://github.com/huggingface/peft/tree/main/examples)
