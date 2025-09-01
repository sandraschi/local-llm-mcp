"""
Sparse Fine-Tuning Example

This script demonstrates how to use the Sparse Fine-Tuning method
with the LLM MCP server.
"""
import asyncio
from llm_mcp.tools.sparse_tools import (
    sparse_load_model,
    sparse_train,
    sparse_generate,
    SparsityConfig
)

async def main():
    # Configure sparsity
    sparsity_config = SparsityConfig(
        sparsity_level=0.8,  # 80% sparsity
        sparse_attention_layers=[4, 8, 12],  # Apply to these layers
        sparsity_type="structured",  # or "unstructured"
        sparsity_scheduler="linear",  # or "cosine", "constant"
    )
    
    # Load a model with sparse fine-tuning
    print("Loading model with sparse fine-tuning...")
    model_info = await sparse_load_model(
        model_name="meta-llama/Llama-2-7b-hf",
        sparsity_config=sparsity_config,
        load_in_4bit=True,
    )
    
    print(f"Model loaded with ID: {model_info['model_id']}")
    
    # Fine-tune the model with sparsity
    print("\nStarting sparse fine-tuning...")
    try:
        training_result = await sparse_train(
            model_id=model_info["model_id"],
            dataset="imdb",  # Replace with your dataset
            output_dir="./sparse_model",
            learning_rate=5e-5,
            batch_size=8,
            num_epochs=3,
            gradient_accumulation_steps=4,
            logging_steps=10,
            sparsity_update_freq=100,  # Update sparsity every 100 steps
        )
        print(f"\nTraining completed! Model saved to: {training_result['output_dir']}")
    except Exception as e:
        print(f"Training failed: {str(e)}")
        return
    
    # Generate text with the fine-tuned model
    print("\nGenerating text with the sparse model...")
    generation_result = await sparse_generate(
        model_id=model_info["model_id"],
        prompt="The movie was",
        max_length=50,
        temperature=0.7,
    )
    print("\nGenerated text:")
    print(generation_result["generated_text"])
    
    # Get sparsity statistics
    print("\nSparsity statistics:")
    for layer, stats in generation_result.get("sparsity_stats", {}).items():
        print(f"{layer}: {stats['sparsity']*100:.1f}% sparsity")
    
    # Clean up
    print("\nCleaning up...")
    # Uncomment to delete the model when done
    # await sparse_unload_model(model_info["model_id"])

if __name__ == "__main__":
    asyncio.run(main())
