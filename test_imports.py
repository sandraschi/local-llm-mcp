"""Test script to verify imports are working correctly."""
import sys
import os
import traceback

# Add the src directory to the Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, src_path)
print(f"Added to path: {src_path}")

def test_imports():
    """Test that all required imports work."""
    imports = {
        'llm_mcp.tools.model_tools': ['_model_manager'],
        'llm_mcp.tools.lora_tools': ['_lora_manager'],
        'llm_mcp.tools.gradio_tools': ['_gradio_manager'],
        'llm_mcp.managers.model_manager': ['ModelManager']
    }
    
    all_success = True
    
    for module, attrs in imports.items():
        try:
            print(f"\n🔍 Testing import: from {module} import {', '.join(attrs)}")
            module_obj = __import__(module, fromlist=attrs)
            for attr in attrs:
                if hasattr(module_obj, attr):
                    print(f"✅ Found {attr} in {module}")
                    print(f"   {attr} type: {type(getattr(module_obj, attr))}")
                else:
                    print(f"❌ {attr} not found in {module}")
                    all_success = False
        except ImportError as e:
            print(f"❌ Failed to import {module}: {e}")
            print("Traceback:")
            traceback.print_exc()
            all_success = False
        except Exception as e:
            print(f"❌ Error checking {module}: {e}")
            traceback.print_exc()
            all_success = False
    
    # Try to list the directory structure to help with debugging
    try:
        print("\n📁 Directory structure:")
        for root, dirs, files in os.walk(os.path.join(src_path, 'llm_mcp')):
            level = root.replace(src_path, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                if f.endswith('.py') or f in ('__init__.py',):
                    print(f"{subindent}{f}")
    except Exception as e:
        print(f"⚠️ Could not list directory structure: {e}")
    
    return all_success

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
