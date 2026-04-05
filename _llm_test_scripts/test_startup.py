#!/usr/bin/env python3
"""Quick startup test for local-llm-mcp server.

Tests basic server startup without heavy dependencies.
"""
import asyncio
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_basic_imports():
    """Test that we can import the main modules."""
    try:
        from llm_mcp import config
        logger.info("✅ Config module imported successfully")
        
        from llm_mcp.tools import check_dependencies
        deps = check_dependencies()
        logger.info(f"✅ Dependencies checked: {sum(deps.values())}/{len(deps)} available")
        
        from llm_mcp.tools.vllm_tools import VLLM_AVAILABLE
        logger.info(f"✅ vLLM availability: {VLLM_AVAILABLE}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False

async def test_server_creation():
    """Test that we can create the FastMCP server instance."""
    try:
        from fastmcp import FastMCP
        
        mcp = FastMCP("test-llm-mcp")
        logger.info("✅ FastMCP server created successfully")
        
        # Test tool registration without loading heavy models
        from llm_mcp.tools import register_all_tools
        results = register_all_tools(mcp)
        
        successful = sum(1 for v in results.values() if v is True)
        total = len(results)
        logger.info(f"✅ Tool registration: {successful}/{total} tools registered")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Server creation failed: {e}")
        return False

async def main():
    """Run all startup tests."""
    logger.info("🚀 Starting local-llm-mcp startup tests...")
    
    tests = [
        ("Basic imports", test_basic_imports),
        ("Server creation", test_server_creation),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        logger.info(f"Running: {test_name}")
        if await test_func():
            passed += 1
        else:
            logger.error(f"Test failed: {test_name}")
    
    logger.info(f"🎯 Test results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        logger.info("✅ All tests passed! Server should start successfully.")
        return 0
    else:
        logger.error("❌ Some tests failed. Check errors above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
