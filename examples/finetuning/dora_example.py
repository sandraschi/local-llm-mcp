"""
DoRA (Dropout LoRA) Fine-tuning Example

This script demonstrates how to use the DoRA fine-tuning method
with the LLM MCP server.
"""
import asyncio
from llm_mcp.tools.dora_tools import dora_load_model, dora_train, dora_generate

async def main():
    # Load a model with DoRA
    print("Loading model with DoRA...")
    model_info = await dora_load_model(
        model_name="meta-llama/Llama-2-7b-hf",
        load_in_4bit=True,
        lora_rank=64,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        use_dropout=True,
        dropout_rate=0.1,
    )
    
    print(f"Model loaded with ID: {model_info['model_id']}")
    
    # Fine-tune the model (example with dummy dataset)
    print("\nStarting fine-tuning...")
    try:
        training_result = await dora_train(
            model_id=model_info["model_id"],
            dataset="imdb",  # Replace with your dataset
            output_dir="./dora_model",
            learning_rate=2e-4,
            batch_size=8,
            num_epochs=3,
            gradient_accumulation_steps=4,
            logging_steps=10,
        )
        print(f"\nTraining completed! Model saved to: {training_result['output_dir']}")
    except Exception as e:
        print(f"Training failed: {str(e)}")
        return
    
    # Generate text with the fine-tuned model
    print("\nGenerating text with the fine-tuned model...")
    generation_result = await dora_generate(
        model_id=model_info["model_id"],
        prompt="The movie was",
        max_length=50,
        temperature=0.7,
    )
    print("\nGenerated text:")
    print(generation_result["generated_text"])
    
    # Clean up
    print("\nCleaning up...")
    # Uncomment to delete the model when done
    # await dora_unload_model(model_info["model_id"])

if __name__ == "__main__":
    asyncio.run(main())
