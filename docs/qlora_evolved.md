# QLoRA Evolved Tools

QLoRA Evolved is an advanced fine-tuning approach that combines the benefits of QLoRA with additional optimizations for better performance and memory efficiency. This document provides a comprehensive guide to using QLoRA Evolved tools in the LLM MCP server.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
  - [Load a Model](#load-a-model)
  - [Prepare for Training](#prepare-for-training)
  - [Train a Model](#train-a-model)
  - [Unload a Model](#unload-a-model)
  - [List Loaded Models](#list-loaded-models)
- [Advanced Usage](#advanced-usage)
  - [Custom Training Configuration](#custom-training-configuration)
  - [Using Custom Datasets](#using-custom-datasets)
  - [Monitoring Training](#monitoring-training)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [FAQs](#faqs)

## Overview

QLoRA Evolved extends standard QLoRA with several improvements:

- **Enhanced Quantization**: Support for multiple 4-bit and 8-bit quantization types
- **Optimized Training**: Better memory efficiency and training speed
- **Flexible Configuration**: Fine-grained control over training parameters
- **Seamless Integration**: Works with existing Hugging Face models and datasets

## Features

- Multiple quantization types (NF4, NF4 optimized, INT4, FP8)
- Double quantization support for reduced memory usage
- Gradient checkpointing for large models
- Flash Attention 2.0 support
- Mixed precision training (FP16/BF16)
- TensorBoard integration
- Automatic model offloading to CPU when needed

## Installation

To use QLoRA Evolved, install the required dependencies:

```bash
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate bitsandbytes datasets peft

# For Flash Attention 2.0 (recommended for better performance)
pip install flash-attn --no-build-isolation

# For TensorBoard integration
pip install tensorboard
```

## Quick Start

Here's how to quickly start fine-tuning a model with QLoRA Evolved:

```python
# Load a model with QLoRA Evolved
response = await mcp.call(
    "qloraevolved_load_model",
    model_name="meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,
    quant_type="nf4_optimized",
    use_double_quant=True,
    compute_dtype="bfloat16",
    lora_rank=64,
    lora_alpha=16,
    lora_dropout=0.1
)
model_id = response["model_id"]

# Prepare for training
await mcp.call(
    "qloraevolved_prepare_for_training",
    model_id=model_id,
    output_dir="./qlora_evolved_output",
    learning_rate=2e-4,
    batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    warmup_ratio=0.03,
    weight_decay=0.01,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    max_grad_norm=0.3,
    logging_steps=10,
    save_steps=200,
    save_total_limit=3,
    report_to="tensorboard",
    use_gradient_checkpointing=True
)

# Train the model
train_result = await mcp.call(
    "qloraevolved_train",
    model_id=model_id,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# The model is automatically saved to the output directory
print(f"Model saved to: {train_result['output_dir']}")

# Unload the model when done
await mcp.call("qloraevolved_unload_model", model_id=model_id)
```

## API Reference

### Load a Model

Load a model with QLoRA Evolved configuration.

```python
response = await mcp.call(
    "qloraevolved_load_model",
    model_name="meta-llama/Llama-2-7b-hf",
    model_id=None,  # Optional custom ID
    max_length=2048,
    load_in_4bit=True,
    quant_type="nf4",  # "nf4", "nf4_optimized", "int4", "fp8", "none"
    use_double_quant=True,
    compute_dtype="bfloat16",  # "float16", "bfloat16", "float32"
    lora_rank=64,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=None,  # Default: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    device_map="auto",
    trust_remote_code=False,
)
```

**Parameters:**
- `model_name`: Name or path of the model to load (required)
- `model_id`: Optional ID to assign to the model (auto-generated if not provided)
- `max_length`: Maximum sequence length (default: 2048)
- `load_in_4bit`: Whether to load in 4-bit precision (default: True)
- `quant_type`: Quantization type (default: "nf4")
  - "nf4": 4-bit NormalFloat
  - "nf4_optimized": Optimized 4-bit
  - "int4": 4-bit integers
  - "fp8": 8-bit floating point (experimental)
  - "none": No quantization (standard LoRA)
- `use_double_quant`: Whether to use double quantization (default: True)
- `compute_dtype`: Compute dtype for training (default: "bfloat16")
- `lora_rank`: Rank of LoRA matrices (default: 64)
- `lora_alpha`: Alpha parameter for LoRA scaling (default: 16)
- `lora_dropout`: Dropout probability for LoRA layers (default: 0.1)
- `target_modules`: List of module names to apply LoRA to (default: common transformer modules)
- `device_map`: Device placement strategy (default: "auto")
- `trust_remote_code`: Whether to trust remote code (default: False)

**Returns:**
```json
{
  "status": "success",
  "model_id": "llama-2-7b-qlora-evolved-1234567890",
  "model_name": "meta-llama/Llama-2-7b-hf",
  "trainable_params": 4194304,
  "total_params": 6738415616,
  "config": {
    "model_name": "meta-llama/Llama-2-7b-hf",
    "max_length": 2048,
    "load_in_4bit": true,
    "quant_type": "nf4",
    "use_double_quant": true,
    "bnb_4bit_compute_dtype": "bfloat16",
    "lora_rank": 64,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "device_map": "auto",
    "trust_remote_code": false
  }
}
```

### Prepare for Training

Prepare a loaded model for training with the specified configuration.

```python
response = await mcp.call(
    "qloraevolved_prepare_for_training",
    model_id="llama-2-7b-qlora-evolved-1234567890",
    output_dir="./qlora_evolved_output",
    learning_rate=2e-4,
    batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    max_steps=-1,  # -1 to use num_train_epochs
    warmup_ratio=0.03,
    weight_decay=0.01,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    max_grad_norm=0.3,
    logging_steps=10,
    save_steps=200,
    save_total_limit=3,
    report_to="tensorboard",
    use_gradient_checkpointing=True,
    use_flash_attention_2=True,
    use_cpu_offload=False
)
```

**Parameters:**
- `model_id`: ID of the loaded model (required)
- `output_dir`: Directory to save the model (default: "./qlora_evolved_output")
- `learning_rate`: Learning rate (default: 2e-4)
- `batch_size`: Batch size per device (default: 4)
- `gradient_accumulation_steps`: Number of steps to accumulate gradients (default: 4)
- `num_train_epochs`: Number of training epochs (default: 3)
- `max_steps`: Maximum number of training steps (-1 to use num_train_epochs) (default: -1)
- `warmup_ratio`: Ratio of warmup steps (default: 0.03)
- `weight_decay`: Weight decay (default: 0.01)
- `optim`: Optimizer to use (default: "paged_adamw_32bit")
- `lr_scheduler_type`: Learning rate scheduler type (default: "cosine")
- `max_grad_norm`: Maximum gradient norm (default: 0.3)
- `logging_steps`: Log every X updates steps (default: 10)
- `save_steps`: Save checkpoint every X updates steps (default: 200)
- `save_total_limit`: Maximum number of checkpoints to keep (default: 3)
- `report_to`: Comma-separated list of integrations to report to (default: "tensorboard")
- `use_gradient_checkpointing`: Whether to use gradient checkpointing (default: True)
- `use_flash_attention_2`: Whether to use Flash Attention 2.0 (default: True)
- `use_cpu_offload`: Whether to offload some operations to CPU (default: False)

**Returns:**
```json
{
  "status": "success",
  "message": "Model llama-2-7b-qlora-evolved-1234567890 prepared for training",
  "config": {
    "output_dir": "./qlora_evolved_output",
    "learning_rate": 2e-4,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 3,
    "max_steps": -1,
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "optim": "paged_adamw_32bit",
    "lr_scheduler_type": "cosine",
    "max_grad_norm": 0.3,
    "logging_steps": 10,
    "save_steps": 200,
    "save_total_limit": 3,
    "report_to": "tensorboard",
    "use_gradient_checkpointing": true,
    "use_flash_attention_2": true,
    "use_cpu_offload": false
  }
}
```

### Train a Model

Train a prepared model on the given dataset.

```python
response = await mcp.call(
    "qloraevolved_train",
    model_id="llama-2-7b-qlora-evolved-1234567890",
    train_dataset=train_dataset,  # Hugging Face Dataset
    eval_dataset=eval_dataset,    # Optional evaluation dataset
    # Optional training configuration overrides
    learning_rate=1e-4,
    batch_size=8,
    max_steps=1000
)
```

**Parameters:**
- `model_id`: ID of the prepared model (required)
- `train_dataset`: Training dataset (required, Hugging Face Dataset)
- `eval_dataset`: Optional evaluation dataset (Hugging Face Dataset)
- Additional training configuration parameters can be passed to override the prepared configuration

**Returns:**
```json
{
  "status": "success",
  "metrics": {
    "train_runtime": 123.45,
    "train_samples_per_second": 12.34,
    "train_steps_per_second": 0.5,
    "train_loss": 1.234,
    "epoch": 3.0
  },
  "output_dir": "./qlora_evolved_output/final"
}
```

### Unload a Model

Unload a model and free resources.

```python
response = await mcp.call(
    "qloraevolved_unload_model",
    model_id="llama-2-7b-qlora-evolved-1234567890"
)
```

**Parameters:**
- `model_id`: ID of the model to unload (required)

**Returns:**
```json
{
  "status": "success",
  "message": "Model llama-2-7b-qlora-evolved-1234567890 unloaded"
}
```

### List Loaded Models

List all currently loaded models.

```python
response = await mcp.call("qloraevolved_list_models")
```

**Returns:**
```json
{
  "status": "success",
  "models": {
    "llama-2-7b-qlora-evolved-1234567890": {
      "config": {
        "model_name": "meta-llama/Llama-2-7b-hf",
        "max_length": 2048,
        "load_in_4bit": true,
        "quant_type": "nf4",
        "use_double_quant": true,
        "bnb_4bit_compute_dtype": "bfloat16",
        "lora_rank": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "device_map": "auto",
        "trust_remote_code": false
      },
      "device": "cuda:0"
    }
  }
}
```

## Advanced Usage

### Custom Training Configuration

You can customize the training configuration by passing a dictionary to the `training_config` parameter:

```python
training_config = {
    "learning_rate": 1e-4,
    "batch_size": 8,
    "gradient_accumulation_steps": 2,
    "num_train_epochs": 5,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "optim": "adamw_torch",
    "lr_scheduler_type": "cosine",
    "max_grad_norm": 0.5,
    "logging_steps": 50,
    "save_steps": 500,
    "save_total_limit": 5,
    "report_to": "tensorboard",
    "use_gradient_checkpointing": True,
    "use_flash_attention_2": True,
    "use_cpu_offload": False
}

await mcp.call(
    "qloraevolved_prepare_for_training",
    model_id=model_id,
    **training_config
)
```

### Using Custom Datasets

You can use any dataset that's compatible with the Hugging Face `datasets` library:

```python
from datasets import load_dataset

# Load a dataset
dataset = load_dataset("imdb", split="train")

def formatting_prompts_func(examples):
    return {
        "text": [f"### Review: {text}\n### Sentiment: {label}" 
                for text, label in zip(examples["text"], examples["label"])]
    }

# Format the dataset
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=["text", "label"]
)

# Train with the dataset
train_result = await mcp.call(
    "qloraevolved_train",
    model_id=model_id,
    train_dataset=dataset,
    training_config={"max_steps": 1000}
)
```

### Monitoring Training

Training progress is automatically logged to TensorBoard. To monitor training:

```bash
tensorboard --logdir=qlora_evolved_output/runs
```

## Best Practices

1. **Start Small**: Begin with a small model and dataset to test your training setup.
2. **Use Gradient Accumulation**: Increase `gradient_accumulation_steps` to effectively increase batch size without using more memory.
3. **Monitor Memory Usage**: Keep an eye on GPU memory usage and adjust batch size accordingly.
4. **Use Mixed Precision**: Enable `bf16` if your hardware supports it for better performance.
5. **Save Checkpoints**: Regularly save checkpoints to avoid losing progress.
6. **Clean Up**: Always unload models when done to free up GPU memory.

## Troubleshooting

### Out of Memory (OOM) Errors

- Reduce `batch_size` or increase `gradient_accumulation_steps`
- Enable `use_gradient_checkpointing`
- Use a smaller model or shorter sequences
- Enable `load_in_4bit` for 4-bit quantization
- Set `use_cpu_offload=True` to offload some operations to CPU

### Slow Training

- Increase `batch_size` if memory allows
- Use `bf16` if supported by your hardware
- Enable `use_flash_attention_2` for faster attention computation
- Use a more powerful GPU or multiple GPUs
- Reduce `gradient_accumulation_steps` if possible

## FAQs

### What models are supported?

QLoRA Evolved supports most Hugging Face models, including LLaMA, Mistral, and other architectures.

### Can I use custom models?

Yes, as long as they're compatible with the Hugging Face `transformers` library.

### How do I know if my hardware is compatible?

QLoRA Evolved requires a CUDA-compatible GPU with at least 12GB of VRAM for 7B models in 4-bit precision.

### Can I resume training from a checkpoint?

Yes, you can load a previously saved model and continue training.

### How do I export my fine-tuned model?

The model is automatically saved to the specified `output_dir`. You can load it like any other Hugging Face model.
