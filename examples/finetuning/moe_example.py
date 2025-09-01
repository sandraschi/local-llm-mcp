"""
Mixture of Experts (MoE) Example

This script demonstrates how to use the Mixture of Experts (MoE) implementation
with the LLM MCP server.
"""
import asyncio
from llm_mcp.tools.moe_tools import moe_load_model, moe_train, moe_generate, MoEConfig

async def main():
    # Configure MoE
    moe_config = MoEConfig(
        num_experts=8,                  # Number of expert networks
        expert_capacity=4,               # Tokens per expert
        router_jitter_noise=0.1,         # Noise for exploration
        router_aux_loss_coef=0.01,       # Load balancing loss weight
        moe_layer_frequency=2,           # Apply MoE every N layers
        moe_layer_start=4,               # Start MoE layers at this depth
    )
    
    # Load a model with MoE layers
    print("Loading model with MoE layers...")
    model_info = await moe_load_model(
        model_name="meta-llama/Llama-2-7b-hf",
        moe_config=moe_config,
        load_in_4bit=True,
    )
    
    print(f"Model loaded with ID: {model_info['model_id']}")
    print(f"Number of MoE layers: {model_info['num_moe_layers']}")
    
    # Fine-tune the MoE model
    print("\nStarting MoE fine-tuning...")
    try:
        training_result = await moe_train(
            model_id=model_info["model_id"],
            dataset="imdb",  # Replace with your dataset
            output_dir="./moe_model",
            learning_rate=5e-5,
            batch_size=4,  # Smaller batch size for MoE
            num_epochs=3,
            gradient_accumulation_steps=4,
            logging_steps=10,
            # MoE-specific training parameters
            expert_balance_loss_weight=0.1,
            expert_utilization_loss_weight=0.1,
        )
        print(f"\nTraining completed! Model saved to: {training_result['output_dir']}")
    except Exception as e:
        print(f"Training failed: {str(e)}")
        return
    
    # Generate text with the MoE model
    print("\nGenerating text with the MoE model...")
    generation_result = await moe_generate(
        model_id=model_info["model_id"],
        prompt="The movie was",
        max_length=50,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
    )
    
    print("\nGenerated text:")
    print(generation_result["generated_text"])
    
    # Print expert utilization
    if "expert_utilization" in generation_result:
        print("\nExpert utilization:")
        for layer, utilization in generation_result["expert_utilization"].items():
            print(f"{layer}: {utilization*100:.1f}%")
    
    # Clean up
    print("\nCleaning up...")
    # Uncomment to delete the model when done
    # await moe_unload_model(model_info["model_id"])

if __name__ == "__main__":
    asyncio.run(main())
