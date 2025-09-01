"""Test script to verify MCP server functionality."""
import sys
import os
import asyncio
import logging

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

async def test_mcp():
    """Test MCP server functionality."""
    try:
        # Import the MCP server
        from llm_mcp.main import main
        from llm_mcp.tools.model_management_tools import get_ollama, get_lmstudio
        
        print("✅ All imports successful!")
        
        # Test manager initialization
        print("\n🔍 Testing OllamaManager:")
        ollama = get_ollama()
        print(f"Ollama API base: {ollama.api_base}")
        
        print("\n🔍 Testing LMStudioManager:")
        lmstudio = get_lmstudio()
        print(f"LM Studio API base: {lmstudio.api_base}")
        
        # Try to list models (may fail if services are not running)
        try:
            print("\n🔍 Testing Ollama list_models():")
            models = await ollama.list_models()
            print(f"Found {len(models)} models")
        except Exception as e:
            print(f"⚠️ Failed to list Ollama models (Ollama may not be running): {e}")
        
        # Cleanup
        await ollama.close()
        await lmstudio.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_mcp())
    sys.exit(0 if success else 1)
