# Unsloth Integration

This document provides a comprehensive guide to using Unsloth for efficient fine-tuning of large language models within the LLM MCP server.

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

Unsloth is a highly optimized framework for fine-tuning large language models, offering significant speed improvements and memory efficiency compared to standard fine-tuning approaches. This integration brings Unsloth's capabilities to the LLM MCP server, allowing you to fine-tune models with minimal code and maximum performance.

## Features

- **2-4x Faster Training**: Optimized CUDA kernels for faster training
- **Memory Efficient**: 50-80% less memory usage than standard fine-tuning
- **Easy Integration**: Simple API that works with existing Hugging Face models
- **Gradient Checkpointing**: Reduced memory usage with minimal performance impact
- **Mixed Precision Training**: Automatic mixed precision (AMP) support
- **LoRA Support**: Built-in support for Low-Rank Adaptation
- **TensorBoard Integration**: Built-in support for training visualization

## Installation

To use Unsloth, install it with:

```bash
# Install with pip
pip install git+https://github.com/unslothai/unsloth.git

# Or with conda
conda install -c conda-forge cudatoolkit-dev -y
conda install -c conda-forge cudatoolkit -y
pip install --no-cache-dir git+https://github.com/unslothai/unsloth.git
```

## Quick Start

Here's how to quickly start fine-tuning a model with Unsloth:

```python
# Load a model
response = await mcp.call(
    "unsloth_load_model",
    model_name="meta-llama/Llama-2-7b-hf",
    load_in_4bit=True
)
model_id = response["model_id"]

# Prepare for training
await mcp.call("unsloth_prepare_for_training", model_id=model_id)

# Train the model
train_result = await mcp.call(
    "unsloth_train",
    model_id=model_id,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# The model is automatically saved to the output directory
print(f"Model saved to: {train_result['output_dir']}")
```

## API Reference

### Load a Model

Load a model with Unsloth optimizations.

```python
response = await mcp.call(
    "unsloth_load_model",
    model_name="meta-llama/Llama-2-7b-hf",
    max_seq_length=2048,
    load_in_4bit=True,
    token=None,  # Hugging Face auth token if needed
    device_map="auto"
)
```

**Parameters:**
- `model_name`: Name or path of the model to load (required)
- `max_seq_length`: Maximum sequence length (default: 2048)
- `load_in_4bit`: Whether to load in 4-bit precision (default: True)
- `token`: Hugging Face auth token (optional)
- `device_map`: Device placement strategy (default: "auto")

**Returns:**
```json
{
  "status": "success",
  "model_id": "llama-2-7b-unsloth",
  "model_name": "meta-llama/Llama-2-7b-hf",
  "max_seq_length": 2048,
  "dtype": "bfloat16",
  "device": "cuda:0"
}
```

### Prepare for Training

Prepare a loaded model for training with the specified configuration.

```python
response = await mcp.call(
    "unsloth_prepare_for_training",
    model_id="llama-2-7b-unsloth",
    training_config={
        "learning_rate": 2e-4,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 10,
        "max_steps": 100,
        "output_dir": "my_finetuned_model"
    }
)
```

**Parameters:**
- `model_id`: ID of the loaded model (required)
- `training_config`: Dictionary with training configuration (optional)
  - `learning_rate`: Learning rate (default: 2e-4)
  - `batch_size`: Batch size per device (default: 2)
  - `gradient_accumulation_steps`: Number of steps to accumulate gradients (default: 4)
  - `warmup_steps`: Number of warmup steps (default: 10)
  - `max_steps`: Maximum number of training steps (default: 100)
  - `output_dir`: Directory to save the model (default: "unsloth_finetuned_model")
  - `fp16`: Whether to use fp16 training (default: True if bf16 not supported)
  - `bf16`: Whether to use bf16 training (default: True if supported)
  - `logging_steps`: Log every X updates steps (default: 1)
  - `save_steps`: Save checkpoint every X updates steps (default: 100)
  - `optim`: Optimizer to use (default: "adamw_8bit")
  - `weight_decay`: Weight decay (default: 0.01)
  - `lr_scheduler_type`: Learning rate scheduler type (default: "cosine")
  - `seed`: Random seed (default: 42)
  - `use_gradient_checkpointing`: Whether to use gradient checkpointing (default: True)

**Returns:**
```json
{
  "status": "success",
  "message": "Model llama-2-7b-unsloth prepared for training",
  "trainable_params": 4194304,
  "total_params": 6738415616
}
```

### Train a Model

Train a prepared model on the given dataset.

```python
train_result = await mcp.call(
    "unsloth_train",
    model_id="llama-2-7b-unsloth",
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # optional
    training_config={
        "learning_rate": 2e-4,
        "max_steps": 1000,
        "output_dir": "my_finetuned_model"
    }
)
```

**Parameters:**
- `model_id`: ID of the prepared model (required)
- `train_dataset`: Training dataset (required)
- `eval_dataset`: Optional evaluation dataset
- `training_config`: Training configuration overrides (optional)

**Returns:**
```json
{
  "status": "success",
  "metrics": {
    "train_runtime": 123.45,
    "train_samples_per_second": 12.34,
    "train_steps_per_second": 0.5,
    "train_loss": 1.234,
    "epoch": 1.0
  },
  "output_dir": "./my_finetuned_model/final"
}
```

### Unload a Model

Unload a model and free memory.

```python
response = await mcp.call(
    "unsloth_unload_model",
    model_id="llama-2-7b-unsloth"
)
```

**Parameters:**
- `model_id`: ID of the model to unload (required)

**Returns:**
```json
{
  "status": "success",
  "message": "Model llama-2-7b-unsloth unloaded"
}
```

### List Loaded Models

List all currently loaded models.

```python
models = await mcp.call("unsloth_list_models")
```

**Returns:**
```json
{
  "llama-2-7b-unsloth": {
    "config": {
      "model_name": "meta-llama/Llama-2-7b-hf",
      "max_seq_length": 2048,
      "dtype": "bfloat16",
      "load_in_4bit": true,
      "device_map": "auto"
    },
    "device": "cuda:0"
  }
}
```

## Advanced Usage

### Custom Training Configuration

You can customize the training configuration by passing a dictionary to the `training_config` parameter:

```python
training_config = {
    "learning_rate": 1e-4,
    "batch_size": 4,
    "gradient_accumulation_steps": 2,
    "warmup_steps": 20,
    "max_steps": 1000,
    "output_dir": "custom_finetuned_model",
    "fp16": True,
    "bf16": False,
    "logging_steps": 10,
    "save_steps": 200,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "seed": 42,
    "use_gradient_checkpointing": True
}

await mcp.call(
    "unsloth_prepare_for_training",
    model_id=model_id,
    training_config=training_config
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
    "unsloth_train",
    model_id=model_id,
    train_dataset=dataset,
    training_config={"max_steps": 100}
)
```

### Monitoring Training

Training progress is automatically logged to TensorBoard. To monitor training:

```bash
tensorboard --logdir=unsloth_finetuned_model/runs
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

### Slow Training

- Increase `batch_size` if memory allows
- Use `bf16` if supported by your hardware
- Disable gradient checkpointing if memory allows
- Use a more powerful GPU or multiple GPUs

## FAQs

### What models are supported?

Unsloth supports most Hugging Face models, including LLaMA, Mistral, and other architectures.

### Can I use custom models?

Yes, as long as they're compatible with the Hugging Face `transformers` library.

### How do I know if my hardware is compatible?

Unsloth requires a CUDA-compatible GPU with at least 12GB of VRAM for 7B models in 4-bit precision.

### Can I resume training from a checkpoint?

Yes, you can load a previously saved model and continue training.

### How do I export my fine-tuned model?

The model is automatically saved to the specified `output_dir`. You can load it like any other Hugging Face model.
