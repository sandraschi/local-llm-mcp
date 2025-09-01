# Hardware Requirements and Performance Estimates

This document provides hardware requirements and performance estimates for different model sizes and GPU configurations when using the fine-tuning tools in the LLM MCP server.

## GPU Recommendations

| GPU Model | VRAM | Recommended For | Max Model Size (4-bit) |
|-----------|------|-----------------|------------------------|
| RTX 3090/4090 | 24GB | 7B models | 13B (with limitations) |
| A6000 | 48GB | 13B models | 30B (with limitations) |
| A100 40GB | 40GB | 13B-30B models | 30B |
| H100 80GB | 80GB | 30B+ models | 70B+ |

## Performance Estimates

### RTX 4090 (24GB)

| Model Size | Quantization | Batch Size | Tokens/sec | VRAM Usage | 1M Tokens | 1B Tokens |
|------------|--------------|------------|------------|------------|-----------|-----------|
| 7B | 4-bit | 4-8 | 15-20 | 18-22GB | 14-18h | 580-750d |
| 13B | 4-bit | 1-2 | 5-8 | 22-24GB | 35-55h | 1450-2300d |
| 30B+ | - | - | - | ❌ OOM | - | - |

### H100 80GB

| Model Size | Quantization | Batch Size | Tokens/sec | VRAM Usage | 1M Tokens | 1B Tokens |
|------------|--------------|------------|------------|------------|-----------|-----------|
| 7B | 4-bit | 16-32 | 45-60 | 30-35GB | 4.5-6h | 190-250d |
| 13B | 4-bit | 8-16 | 30-40 | 50-60GB | 7-9h | 290-380d |
| 30B | 4-bit | 4-8 | 15-25 | 70-75GB | 11-19h | 460-770d |
| 70B | 4-bit | 1-2 | 4-6 | 75-80GB | 46-70h | 1900-2900d |

## Cost-Effectiveness Analysis

| GPU | Price (USD) | Tokens/\$ (7B) | Tokens/\$ (13B) | Best For |
|-----|-------------|----------------|-----------------|----------|
| RTX 4090 | ~$1,600 | ~9,375 | ~3,125 | Budget 7B models |
| H100 80GB | ~$30,000 | ~4,167 | ~3,333 | Large models, production |
| A100 40GB | ~$15,000 | ~2,500 | ~2,083 | Balanced option |

## Recommendations

1. **For 7B models**:
   - RTX 4090 provides best value
   - Use 4-bit quantization
   - Expected speed: ~15-20 tokens/sec

2. **For 13B models**:
   - H100 recommended for production
   - RTX 4090 possible but slower
   - Use gradient checkpointing

3. **For 30B+ models**:
   - H100 required
   - Consider model parallelism
   - Use 4-bit quantization

## Memory Optimization Tips

1. **Gradient Checkpointing**:
   ```python
   training_args = {
       "gradient_checkpointing": True,
       "gradient_checkpointing_kwargs": {"use_reentrant": False}
   }
   ```

2. **Mixed Precision Training**:
   ```python
   training_args = {
       "bf16": True,  # For Ampere and newer GPUs
       "fp16": False  # For older GPUs
   }
   ```

3. **Gradient Accumulation**:
   ```python
   training_args = {
       "per_device_train_batch_size": 4,
       "gradient_accumulation_steps": 8  # Effective batch size = 32
   }
   ```

4. **Flash Attention**:
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       model_name,
       use_flash_attention_2=True,
       torch_dtype=torch.bfloat16
   )
   ```

## Cloud Options

| Provider | GPU | Hourly Cost (USD) | Best For |
|----------|-----|-------------------|----------|
| Lambda Labs | H100 | $1.10 | On-demand training |
| RunPod | A100 | $0.99 | Cost-effective |
| AWS | p4d.24xlarge | $32.77 | Enterprise |
| GCP | a2-ultragpu-1g | $3.67 | GCP users |

## Estimating Training Time

Use this formula to estimate training time:

```
Total Time (hours) = (Total Tokens × Epochs) / (Tokens per Second × 3600)
```

Example for 1M tokens on 4090 with 7B model:
```
(1,000,000 × 1) / (15 × 3600) = ~18.5 hours
```

## When to Upgrade

Consider upgrading when:
- Your models don't fit in VRAM
- Training takes longer than a week
- You need to train larger models
- Your time is worth more than the GPU cost
