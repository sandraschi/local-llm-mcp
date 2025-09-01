"""Test script to verify manager functionality."""
import sys
import os
import asyncio

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

async def test_managers():
    """Test manager functionality."""
    try:
        # Import managers
        from llm_mcp.services.model_manager import ModelManager
        from llm_mcp.tools.model_tools import _model_manager
        from llm_mcp.tools.lora_tools import _lora_manager
        from llm_mcp.tools.gradio_tools import _gradio_manager
        
        print("✅ All managers imported successfully!")
        
        # Test model manager
        print("\n🔍 Testing ModelManager:")
        settings = {
            "providers": {
                "ollama": {"enabled": True, "base_url": "http://localhost:11434"},
                "lmstudio": {"enabled": True, "base_url": "http://localhost:1234"}
            }
        }
        model_manager = ModelManager(settings["providers"])
        await model_manager.initialize()
        print("✅ ModelManager initialized successfully")
        
        # List models
        models = await model_manager.list_models()
        print(f"Found {len(models)} models")
        for model in models[:3]:  # Print first 3 models
            print(f"- {model.id} ({model.provider.value})")
        
        # Test LoRA manager
        print("\n🔍 Testing LoRAManager:")
        print(f"LoRA adapters: {_lora_manager.list_available_adapters()}")
        
        # Test Gradio manager
        print("\n🔍 Testing GradioManager:")
        print(f"Gradio interfaces: {_gradio_manager.list_interfaces()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_managers())
    sys.exit(0 if success else 1)
