#!/usr/bin/env python3
"""Run the local-llm-mcp server with enhanced startup and monitoring."""
import asyncio
import sys
import signal
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def setup_signal_handlers():
    """Setup graceful shutdown handlers."""
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """Main entry point for the server."""
    setup_signal_handlers()
    
    try:
        # Import after path setup
        from llm_mcp.main import main as server_main
        await server_main()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
