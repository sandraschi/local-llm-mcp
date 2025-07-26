#!/usr/bin/env python3
"""
LLM MCP Chat Terminal

A command-line interface for interacting with the LLM MCP server.
Supports multiple LLM providers, conversation history, and more.
"""

import os
import sys
import json
import asyncio
import argparse
import readline
from typing import Dict, List, Optional, Any, AsyncGenerator
from pathlib import Path
from datetime import datetime
from enum import Enum

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import the LLM MCP components
from llm_mcp.services.model_service import model_service
from llm_mcp.config import get_settings
from llm_mcp.api.v1.models import GenerateRequest

class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class ChatMessage:
    """A single message in the conversation."""
    
    def __init__(
        self, 
        role: MessageRole, 
        content: str, 
        timestamp: Optional[datetime] = None
    ):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the message to a dictionary."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """Create a message from a dictionary."""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )
    
    def __str__(self) -> str:
        """Return a string representation of the message."""
        role_colors = {
            MessageRole.USER: Colors.BLUE,
            MessageRole.ASSISTANT: Colors.GREEN,
            MessageRole.SYSTEM: Colors.YELLOW
        }
        
        color = role_colors.get(self.role, Colors.ENDC)
        timestamp = self.timestamp.strftime("%H:%M:%S")
        
        return (
            f"{Colors.GRAY}[{timestamp}] {color}{self.role.upper()}:{Colors.ENDC} "
            f"{self.content}"
        )

class Conversation:
    """Manages conversation history and search functionality."""
    
    def __init__(self, max_history: int = 100):
        self.messages: List[ChatMessage] = []
        self.max_history = max_history
    
    def add_message(self, message: ChatMessage) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
        
        # Trim history if needed
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
    
    def get_recent(self, count: int = 10) -> List[ChatMessage]:
        """Get the most recent messages."""
        return self.messages[-count:]
    
    def clear(self) -> None:
        """Clear the conversation history."""
        self.messages = []
    
    def save_to_file(self, file_path: str) -> None:
        """Save the conversation to a file."""
        data = [msg.to_dict() for msg in self.messages]
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'Conversation':
        """Load a conversation from a file."""
        conv = cls()
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for msg_data in data:
                    conv.add_message(ChatMessage.from_dict(msg_data))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"{Colors.RED}Error loading conversation: {e}{Colors.ENDC}")
        
        return conv

class MCPChat:
    """Interactive chat interface for LLM MCP."""
    
    def __init__(self, config_path: Optional[str] = None):
        # Configuration paths
        self.config_path = config_path or os.path.expanduser("~/.config/llm-mcp/terminal.yaml")
        self.history_path = os.path.expanduser("~/.local/share/llm-mcp/chat_history.json")
        
        # Create necessary directories
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        
        # Initialize components
        self.settings = get_settings()
        self.conversation = Conversation()
        self.running = True
        self.current_provider = None
        self.current_model = None
        
        # Load history if it exists
        self._load_history()
        
        # Set up readline for better input handling
        self._setup_readline()
    
    def _setup_readline(self) -> None:
        """Set up readline for better command line interaction."""
        # Enable tab completion
        readline.parse_and_bind("tab: complete")
        
        # Set up history
        histfile = os.path.join(os.path.expanduser("~"), ".llm_mcp_history")
        try:
            readline.read_history_file(histfile)
        except FileNotFoundError:
            pass
        
        import atexit
        atexit.register(readline.write_history_file, histfile)
    
    def _load_history(self) -> None:
        """Load conversation history from file."""
        self.conversation = Conversation.load_from_file(self.history_path)
    
    def _save_history(self) -> None:
        """Save the current conversation history to a file."""
        self.conversation.save_to_file(self.history_path)
    
    async def initialize(self) -> None:
        """Initialize the chat interface."""
        print(f"{Colors.HEADER}LLM MCP Chat Terminal{Colors.ENDC}")
        print(f"{Colors.BLUE}Type '/help' for a list of commands{Colors.ENDC}\n")
        
        # Initialize the model service
        try:
            await model_service.initialize()
            print(f"{Colors.GREEN}✓ Model service initialized{Colors.ENDC}")
            
            # List available providers and models
            await self._list_providers()
            
            # Set default provider and model if available
            if not self.current_provider and model_service.providers:
                self.current_provider = next(iter(model_service.providers.keys()))
                print(f"\n{Colors.YELLOW}Using provider: {self.current_provider}{Colors.ENDC}")
                
                # List models for the current provider
                await self._list_models()
                
                # Set the first available model
                models = await model_service.list_models(self.current_provider)
                if models:
                    self.current_model = models[0]["name"]
                    print(f"{Colors.YELLOW}Using model: {self.current_model}{Colors.ENDC}")
            
            print("\n" + "=" * 50 + "\n")
            
        except Exception as e:
            print(f"{Colors.RED}Failed to initialize model service: {e}{Colors.ENDC}")
            self.running = False
    
    async def _list_providers(self) -> None:
        """List all available providers."""
        print(f"{Colors.HEADER}Available Providers:{Colors.ENDC}")
        
        for i, provider_name in enumerate(model_service.providers.keys(), 1):
            print(f"  {i}. {provider_name}")
    
    async def _list_models(self, provider_name: Optional[str] = None) -> None:
        """List all available models for a provider."""
        provider = provider_name or self.current_provider
        if not provider:
            print(f"{Colors.RED}No provider selected. Use /provider to select one.{Colors.ENDC}")
            return
        
        try:
            models = await model_service.list_models(provider)
            print(f"{Colors.HEADER}Available Models for {provider}:{Colors.ENDC}")
            
            for i, model in enumerate(models, 1):
                print(f"  {i}. {model['name']} ({model.get('id', 'N/A')})")
                
        except Exception as e:
            print(f"{Colors.RED}Failed to list models: {e}{Colors.ENDC}")
    
    async def _process_command(self, command: str) -> bool:
        """Process a chat command."""
        command = command.strip()
        
        if not command.startswith('/'):
            return False
        
        parts = command[1:].split(' ', 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if cmd in ['exit', 'quit']:
            self.running = False
            return True
            
        elif cmd == 'help':
            self._show_help()
            return True
            
        elif cmd == 'clear':
            self.conversation.clear()
            print(f"{Colors.GREEN}Conversation cleared.{Colors.ENDC}")
            return True
            
        elif cmd == 'providers':
            await self._list_providers()
            return True
            
        elif cmd == 'models':
            await self._list_models()
            return True
            
        elif cmd == 'provider':
            if not args:
                if self.current_provider:
                    print(f"Current provider: {self.current_provider}")
                else:
                    print("No provider selected. Use '/provider <name>' to select one.")
                return True
                
            provider_name = args.strip()
            if provider_name in model_service.providers:
                self.current_provider = provider_name
                print(f"{Colors.GREEN}Provider set to: {provider_name}{Colors.ENDC}")
                await self._list_models()
            else:
                print(f"{Colors.RED}Unknown provider: {provider_name}{Colors.ENDC}")
            return True
            
        elif cmd == 'model':
            if not self.current_provider:
                print(f"{Colors.RED}No provider selected. Use /provider first.{Colors.ENDC}")
                return True
                
            if not args:
                if self.current_model:
                    print(f"Current model: {self.current_model}")
                else:
                    print("No model selected. Use '/model <name>' to select one.")
                return True
                
            model_name = args.strip()
            self.current_model = model_name
            print(f"{Colors.GREEN}Model set to: {model_name}{Colors.ENDC}")
            return True
            
        elif cmd == 'pull':
            if not args:
                print("Usage: /pull <model_name> [provider]")
                return True
                
            parts = args.split(' ', 1)
            model_name = parts[0]
            provider = parts[1] if len(parts) > 1 else self.current_provider
            
            if not provider:
                print(f"{Colors.RED}No provider specified or selected.{Colors.ENDC}")
                return True
                
            try:
                print(f"Pulling model '{model_name}' from {provider}...")
                await model_service.pull_model(model_name, provider)
                print(f"{Colors.GREEN}Model pulled successfully!{Colors.ENDC}")
            except Exception as e:
                print(f"{Colors.RED}Failed to pull model: {e}{Colors.ENDC}")
            
            return True
            
        return False
    
    def _show_help(self) -> None:
        """Show the help message."""
        print(f"{Colors.HEADER}Available Commands:{Colors.ENDC}")
        print("  /help                 - Show this help message")
        print("  /exit, /quit         - Exit the chat")
        print("  /clear               - Clear the conversation")
        print("  /providers           - List available providers")
        print("  /provider <name>     - Set the current provider")
        print("  /models              - List available models for the current provider")
        print("  /model <name>        - Set the current model")
        print("  /pull <model> [prov] - Download a model from a provider")
    
    async def _stream_response(self, prompt: str) -> None:
        """Stream the response from the model."""
        if not self.current_provider or not self.current_model:
            print(f"{Colors.RED}No provider or model selected. Use /provider and /model to select one.{Colors.ENDC}")
            return
        
        # Add user message to conversation
        user_msg = ChatMessage(MessageRole.USER, prompt)
        self.conversation.add_message(user_msg)
        
        # Print assistant prefix
        print(f"{Colors.GREEN}Assistant:{Colors.ENDC} ", end="", flush=True)
        
        # Generate response
        response_text = ""
        try:
            async for chunk in model_service.generate(
                prompt=prompt,
                model=self.current_model,
                provider=self.current_provider,
                temperature=0.7,
                max_tokens=2000,
                stream=True
            ):
                print(chunk, end="", flush=True)
                response_text += chunk
                
            print()  # New line after response
            
            # Add assistant response to conversation
            assistant_msg = ChatMessage(MessageRole.ASSISTANT, response_text)
            self.conversation.add_message(assistant_msg)
            
            # Save conversation history
            self._save_history()
            
        except Exception as e:
            print(f"\n{Colors.RED}Error: {e}{Colors.ENDC}")
    
    async def run(self) -> None:
        """Run the chat interface."""
        await self.initialize()
        
        while self.running:
            try:
                # Get user input
                try:
                    user_input = input(f"{Colors.BLUE}You:{Colors.ENDC} ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Process commands
                if user_input.startswith('/'):
                    await self._process_command(user_input)
                    continue
                
                # Process regular message
                await self._stream_response(user_input)
                
            except Exception as e:
                print(f"\n{Colors.RED}Error: {e}{Colors.ENDC}")
                import traceback
                traceback.print_exc()
        
        # Clean up
        self._save_history()

async def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LLM MCP Chat Terminal")
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Create and run the chat interface
    chat = MCPChat(config_path=args.config)
    await chat.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"\n{Colors.RED}Fatal error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
