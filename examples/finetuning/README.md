# Fine-Tuning Examples

This directory contains example scripts demonstrating various fine-tuning techniques available in the LLM MCP server.

## Available Examples

1. **QLoRA Evolved** (`qlora_evolved_example.py`)
   - Demonstrates 4-bit quantized fine-tuning with QLoRA Evolved
   - Uses efficient parameter updates with low-rank adaptation
   - Ideal for resource-constrained environments

2. **DoRA** (`dora_example.py`)
   - Shows Dropout LoRA fine-tuning
   - Includes adaptive dropout for improved robustness
   - Good for preventing overfitting

3. **Sparse Fine-Tuning** (`sparse_finetuning_example.py`)
   - Implements structured and unstructured sparsity
   - Reduces model size and improves inference speed
   - Configurable sparsity patterns and schedules

4. **Mixture of Experts (MoE)** (`moe_example.py`)
   - Demonstrates training with sparse expert networks
   - Shows expert utilization and routing
   - Efficiently scales model capacity

## Prerequisites

- Python 3.8+
- PyTorch 1.12+
- Transformers 4.30.0+
- LLM MCP server installed in development mode
- Access to LLM models (e.g., LLaMA-2)

## Installation

```bash
# Install required packages
pip install torch>=1.12.0 transformers>=4.30.0 datasets

# Install LLM MCP in development mode
cd /path/to/llm-mcp
pip install -e .
```

## Running Examples

Each example can be run directly:

```bash
# Run QLoRA Evolved example
python examples/finetuning/qlora_evolved_example.py

# Run DoRA example
python examples/finetuning/dora_example.py

# Run Sparse Fine-Tuning example
python examples/finetuning/sparse_finetuning_example.py

# Run MoE example
python examples/finetuning/moe_example.py
```

## Customization

Each script is designed to be easily customizable:

1. **Model Selection**: Change the `model_name` to use different base models
2. **Dataset**: Replace the example dataset with your own data
3. **Hyperparameters**: Adjust learning rate, batch size, and other training parameters
4. **Output Directory**: Specify where to save fine-tuned models

## Best Practices

1. **Start Small**: Test with smaller models before scaling up
2. **Monitor Resources**: Keep an eye on GPU memory usage
3. **Use Checkpoints**: Save model checkpoints during training
4. **Experiment**: Try different hyperparameters for optimal results

## Troubleshooting

- **CUDA Out of Memory**: Reduce batch size or model size
- **Slow Training**: Increase batch size or use gradient accumulation
- **Poor Results**: Adjust learning rate or try different hyperparameters

For more details, refer to the main documentation in the `docs/` directory.
