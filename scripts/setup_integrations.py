#!/usr/bin/env python3
"""
Setup script for LLM MCP integrations.

This script helps set up and configure integrations with:
- Ollama Web UI
- LM Studio MCP
"""
import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

# Paths
ROOT_DIR = Path(__file__).parent.parent
MCP_CONFIG_PATH = ROOT_DIR / "mcp.json"
DOCKER_COMPOSE_PATH = ROOT_DIR / "docker-compose.ollama-webui.yml"

def check_docker() -> bool:
    """Check if Docker is installed and running."""
    try:
        subprocess.run(
            ["docker", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def setup_ollama_webui() -> None:
    """Set up Ollama Web UI using Docker."""
    print("🚀 Setting up Ollama Web UI...")
    
    if not check_docker():
        print("❌ Docker is not installed or not running. Please install Docker first.")
        return
    
    if not DOCKER_COMPOSE_PATH.exists():
        print(f"❌ Docker Compose file not found at {DOCKER_COMPOSE_PATH}")
        return
    
    try:
        print("🐳 Starting Ollama Web UI container...")
        subprocess.run(
            ["docker", "compose", "-f", str(DOCKER_COMPOSE_PATH), "up", "-d"],
            check=True
        )
        print("✅ Ollama Web UI is now running at http://localhost:3000")
        print("   - Connect to your local Ollama instance at http://host.docker.internal:11434")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Ollama Web UI: {e}")

def setup_lmstudio_mcp() -> None:
    """Set up LM Studio MCP integration."""
    print("🔧 Setting up LM Studio MCP integration...")
    
    if not MCP_CONFIG_PATH.exists():
        print(f"❌ MCP configuration file not found at {MCP_CONFIG_PATH}")
        return
    
    try:
        # Load and validate the MCP config
        with open(MCP_CONFIG_PATH, 'r') as f:
            config = json.load(f)
        
        print("✅ MCP configuration is valid")
        print("\nTo use with LM Studio:")
        print("1. Open LM Studio")
        print("2. Go to the 'Program' tab in the right sidebar")
        print("3. Click 'Install' > 'Edit mcp.json'")
        print("4. Copy the contents of the following file:")
        print(f"   {MCP_CONFIG_PATH}")
        print("5. Save and restart LM Studio")
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid MCP configuration: {e}")
    except Exception as e:
        print(f"❌ Error setting up LM Studio MCP: {e}")

def main() -> None:
    """Main entry point for the setup script."""
    print("🔧 LLM MCP Integration Setup 🔧")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. Set up Ollama Web UI")
        print("2. Set up LM Studio MCP")
        print("3. Set up both")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            setup_ollama_webui()
        elif choice == "2":
            setup_lmstudio_mcp()
        elif choice == "3":
            setup_ollama_webui()
            print("\n" + "-" * 40 + "\n")
            setup_lmstudio_mcp()
        elif choice == "4":
            print("👋 Exiting setup")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
