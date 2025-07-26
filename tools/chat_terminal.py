#!/usr/bin/env python3
"""
Advanced chat terminal for LLM MCP Server with features like:
- Multiple LLM provider support
- Conversation history with search
- Model parameter tuning
- Streaming responses
- Personas and rulebooks
- Context management
"""

import os
import re
import json
import yaml
import time
import asyncio
import argparse
import readline
import threading
from typing import Dict, List, Optional, Any, Union, AsyncGenerator, Tuple
from enum import Enum, auto
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor

# Import the LLM MCP components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from llm_mcp.services.provider_factory import ProviderFactory
from llm_mcp.models.base import ModelProvider

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

class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

@dataclass
class ChatConfig:
    """Configuration for chat interactions."""
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stream: bool = True
    timeout: int = 30
    model: Optional[str] = None
    provider: Optional[str] = None

@dataclass
class ChatMessage:
    """A single message in the conversation."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {})
        )

class Persona:
    """Represents a chat persona with system prompt, rules, and configuration."""
    
    def __init__(self, 
                 name: str, 
                 description: str = "", 
                 system_prompt: str = "", 
                 rules: List[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.rules = rules or []
        self.config = config or {}
        
        # Set default config if not provided
        if "temperature" not in self.config:
            self.config["temperature"] = 0.7
        if "max_tokens" not in self.config:
            self.config["max_tokens"] = 2000
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "rules": self.rules,
            "config": self.config
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Persona':
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            system_prompt=data.get("system_prompt", ""),
            rules=data.get("rules", []),
            config=data.get("config", {})
        )
    
    def get_system_messages(self) -> List[Dict[str, str]]:
        """Get all system messages including rules and system prompt."""
        messages = []
        
        # Add system prompt if available
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
            
        # Add rules if available
        if self.rules:
            rules_text = "\n".join(f"- {rule}" for rule in self.rules)
            messages.append({"role": "system", "content": f"Rules to follow:\n{rules_text}"})
            
        return messages


class Conversation:
    """Manages conversation history and search functionality."""
    
    def __init__(self, max_history: int = 1000):
        self.messages: List[ChatMessage] = []
        self.max_history = max_history
        self._search_index = {}
    
    def add_message(self, message: ChatMessage):
        """Add a message to the conversation."""
        self.messages.append(message)
        self._update_search_index(message, len(self.messages) - 1)
        
        # Trim history if needed
        if len(self.messages) > self.max_history:
            removed = self.messages.pop(0)
            self._remove_from_search_index(removed, 0)
    
    def search(self, query: str, limit: int = 5) -> List[Tuple[int, ChatMessage]]:
        """Search through conversation history."""
        if not query or not self.messages:
            return []
            
        query = query.lower()
        results = []
        
        # Simple substring search for now
        for idx, msg in enumerate(self.messages):
            if query in msg.content.lower():
                results.append((idx, msg))
                if len(results) >= limit:
                    break
                    
        return results
    
    def get_recent(self, count: int = 10) -> List[ChatMessage]:
        """Get the most recent messages."""
        return self.messages[-count:]
    
    def clear(self):
        """Clear the conversation history."""
        self.messages = []
        self._search_index = {}
    
    def _update_search_index(self, message: ChatMessage, index: int):
        """Update the search index with a new message."""
        # Simple word-based indexing
        words = re.findall(r'\b\w+\b', message.content.lower())
        for word in words:
            if word not in self._search_index:
                self._search_index[word] = []
            self._search_index[word].append(index)
    
    def _remove_from_search_index(self, message: ChatMessage, index: int):
        """Remove a message from the search index."""
        words = re.findall(r'\b\w+\b', message.content.lower())
        for word in words:
            if word in self._search_index and index in self._search_index[word]:
                self._search_index[word].remove(index)
                if not self._search_index[word]:
                    del self._search_index[word]

class ChatTerminal:
    """
    Advanced chat terminal with support for multiple LLM providers, personas,
    rulebooks, conversation history, and more.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        # Configuration paths
        self.config_path = config_path or os.path.expanduser("~/.config/llm-mcp/terminal.yaml")
        self.history_path = os.path.expanduser("~/.local/share/llm-mcp/history.json")
        self.personas_path = os.path.expanduser("~/.config/llm-mcp/personas/")
        self.rulebooks_path = os.path.expanduser("~/.config/llm-mcp/rulebooks/")
        self.cache_dir = os.path.expanduser("~/.cache/llm-mcp/")
        
        # Create necessary directories
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        os.makedirs(self.personas_path, exist_ok=True)
        os.makedirs(self.rulebooks_path, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize components
        self.config = self._load_config()
        self.provider_factory = ProviderFactory()
        self.provider = None
        self.model = None
        self.conversation = Conversation()
        self.personas: Dict[str, Persona] = {}
        self.active_persona: Optional[Persona] = None
        self.rules: List[str] = []
        self.chat_config = ChatConfig()
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._stop_event = threading.Event()
        
        # Load data
        self._load_personas()
        self._load_rulebooks()
        self._load_history()
        
        # Initialize provider if specified in config
        if self.config.get("provider") and self.config.get("model"):
            self.set_provider(self.config["provider"])
            self.set_model(self.config["model"])
        
        # Set up readline for better input handling
        self._setup_readline()
        
        # Debug mode flag
        self.debug = False
    
    def _setup_readline(self):
        """Set up readline for better command line interaction."""
        # Try to load readline history
        histfile = os.path.join(self.cache_dir, ".python_history")
        try:
            readline.read_history_file(histfile)
        except FileNotFoundError:
            pass
            
        # Set up tab completion
        readline.set_completer(self._completer)
        readline.parse_and_bind("tab: complete")
        
        # Save history on exit
        import atexit
        atexit.register(readline.write_history_file, histfile)
    
    def _completer(self, text: str, state: int) -> Optional[str]:
        """Tab completion for commands and history."""
        commands = [
            "/help", "/exit", "/status", "/provider", "/model", "/persona",
            "/personas", "/rulebook", "/rulebooks", "/rules", "/clear", "/save",
            "/search", "/config", "/stream", "/history"
        ]
        
        options = [cmd for cmd in commands if cmd.startswith(text)]
        if state < len(options):
            return options[state]
        return None
    
    async def stream_response(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream the response from the LLM."""
        if not self.provider or not self.model:
            yield "Error: No provider or model selected. Use /provider and /model commands first."
            return
        
        # Create a message for the prompt
        user_message = ChatMessage(
            role=MessageRole.USER,
            content=prompt,
            metadata={"provider": self.chat_config.provider or "unknown"}
        )
        self.conversation.add_message(user_message)
        
        # Prepare messages for the LLM
        messages = []
        
        # Add system messages from active persona
        if self.active_persona:
            messages.extend(self.active_persona.get_system_messages())
        
        # Add conversation history
        for msg in self.conversation.get_recent(10):  # Last 10 messages for context
            messages.append({"role": msg.role.value, "content": msg.content})
        
        # Stream the response
        response_text = ""
        try:
            async for chunk in self.provider.stream_chat(
                model_id=self.model,
                messages=messages,
                temperature=self.chat_config.temperature,
                max_tokens=self.chat_config.max_tokens,
                top_p=self.chat_config.top_p,
                frequency_penalty=self.chat_config.frequency_penalty,
                presence_penalty=self.chat_config.presence_penalty,
                timeout=self.chat_config.timeout
            ):
                if self._stop_event.is_set():
                    break
                    
                response_text += chunk
                yield chunk
                
        except Exception as e:
            error_msg = f"\n{Colors.RED}Error: {str(e)}{Colors.ENDC}"
            yield error_msg
            response_text += error_msg
            
        finally:
            # Save the assistant's response
            if response_text.strip():
                assistant_message = ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=response_text,
                    metadata={
                        "provider": self.chat_config.provider or "unknown",
                        "model": self.model,
                        "temperature": self.chat_config.temperature,
                        "tokens": len(response_text.split())  # Approximate
                    }
                )
                self.conversation.add_message(assistant_message)
                self._save_history()
    
    def search_history(self, query: str, limit: int = 5) -> List[Tuple[int, ChatMessage]]:
        """Search through conversation history."""
        return self.conversation.search(query, limit)
    
    def update_chat_config(self, **kwargs):
        """Update chat configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.chat_config, key):
                setattr(self.chat_config, key, value)
        
        # Save updated config
        self._save_config()
    
    def stop_streaming(self):
        """Stop any ongoing streaming response."""
        self._stop_event.set()
        self._stop_event.clear()  # Reset for next use
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        default_config = {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 2000,
            "default_persona": "assistant",
            "default_rulebook": "default",
            "history_size": 1000
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
                # Merge with defaults
                return {**default_config, **config}
            except Exception as e:
                print(f"Error loading config: {e}. Using defaults.")
        
        return default_config
    
    def save_config(self):
        """Save configuration to YAML file."""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def _save_history(self):
        """Save conversation history to file."""
        try:
            with open(self.history_path, 'w', encoding='utf-8') as f:
                json.dump([msg.to_dict() for msg in self.conversation.messages], f, indent=2, default=str)
        except Exception as e:
            if self.debug:
                print(f"Error saving history: {e}")
                import traceback
                traceback.print_exc()
    
    def _load_personas(self):
        """Load all personas from the personas directory."""
        self.personas = {}
        
        # Add default persona
        default_persona = Persona(
            name="assistant",
            description="Helpful AI assistant",
            system_prompt="You are a helpful AI assistant.",
            rules=["Be helpful and concise.", "Always respond in markdown format."]
        )
        self.personas[default_persona.name] = default_persona
        
        # Load custom personas
        if os.path.exists(self.personas_path):
            for filename in os.listdir(self.personas_path):
                if filename.endswith(('.yaml', '.yml')):
                    try:
                        with open(os.path.join(self.personas_path, filename), 'r', encoding='utf-8') as f:
                            data = yaml.safe_load(f)
                            if data and isinstance(data, dict):
                                persona = Persona.from_dict(data)
                                self.personas[persona.name] = persona
                    except Exception as e:
                        print(f"Error loading persona {filename}: {e}")
        
        # Set active persona
        default_persona_name = self.config.get("default_persona", "assistant")
        self.active_persona = self.personas.get(default_persona_name, default_persona)
    
    def _load_rulebook(self, name: str) -> List[str]:
        """Load a rulebook by name."""
        rulebook_path = os.path.join(self.rulebooks_path, f"{name}.yaml")
        if os.path.exists(rulebook_path):
            try:
                with open(rulebook_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict) and "rules" in data:
                        return data["rules"]
            except Exception as e:
                print(f"Error loading rulebook {name}: {e}")
        return []
    
    def _load_history(self):
        """Load conversation history from file."""
        try:
            if os.path.exists(self.history_path):
                with open(self.history_path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                    for msg in history:
                        try:
                            self.conversation.add_message(ChatMessage.from_dict(msg))
                        except Exception as e:
                            if self.debug:
                                print(f"Error loading message: {e}")
                                import traceback
                                traceback.print_exc()
        except Exception as e:
            if self.debug:
                print(f"Error loading history: {e}")
                import traceback
                traceback.print_exc()
    
    def _add_message(self, role: str, content: str):
        """Add a message to the history."""
        self.conversation.add_message(ChatMessage(role=role, content=content))
        self._save_history()
    
    def _get_chat_context(self) -> List[Dict[str, str]]:
        """Get the current chat context including system prompt and rules."""
        context = []
        
        # Add system prompt from active persona
        if self.active_persona:
            context.append({"role": "system", "content": self.active_persona.system_prompt})
        
        # Add rules
        if self.rules:
            rules_text = "\n".join(f"- {rule}" for rule in self.rules)
            context.append({"role": "system", "content": f"Rules to follow:\n{rules_text}"})
        
        # Add message history
        for msg in self.conversation.get_recent(10):  # Last 10 messages for context
            context.append({"role": msg.role.value, "content": msg.content})
        
        return context
    
    def set_provider(self, provider_name: str):
        """Set the current LLM provider."""
        self.provider = self.provider_factory.get_provider(provider_name)
        self.chat_config.provider = provider_name
        self._save_config()
        return True
    
    def set_model(self, model_name: str):
        """Set the current model."""
        if not self.provider:
            raise ValueError("No provider selected. Set a provider first.")
        
        available_models = self.provider.list_models()
        if model_name not in available_models:
            raise ValueError(f"Model not found: {model_name}. Available models: {', '.join(available_models)}")
        
        self.model = model_name
        self.chat_config.model = model_name
        self._save_config()
        return True
    
    def set_persona(self, persona_name: str) -> bool:
        """Set the active persona."""
        if persona_name in self.personas:
            self.active_persona = self.personas[persona_name]
            self.config["default_persona"] = persona_name
            return True
        return False
    
    def load_rulebook(self, rulebook_name: str) -> bool:
        """Load a rulebook by name."""
        rules = self._load_rulebook(rulebook_name)
        if rules:
            self.rules = rules
            self.config["default_rulebook"] = rulebook_name
            return True
        return False
    
    async def chat(self, message: str) -> str:
        """Send a message and get a response."""
        if not self.provider or not self.model:
            return "Error: No provider or model selected. Use /provider and /model commands first."
        
        # Add user message to history
        self._add_message("user", message)
        
        try:
            # Get chat context
            messages = self._get_chat_context()
            
            # Call the provider
            response = await self.provider.chat(
                model_id=self.model,
                messages=messages,
                temperature=self.config.get("temperature", 0.7),
                max_tokens=self.config.get("max_tokens", 2000)
            )
            
            # Add assistant response to history
            self._add_message("assistant", response)
            
            return response
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            return error_msg
    
    def list_personas(self) -> List[str]:
        """List all available personas."""
        return list(self.personas.keys())
    
    def list_rulebooks(self) -> List[str]:
        """List all available rulebooks."""
        rulebooks = []
        if os.path.exists(self.rulebooks_path):
            for filename in os.listdir(self.rulebooks_path):
                if filename.endswith(('.yaml', '.yml')):
                    rulebooks.append(os.path.splitext(filename)[0])
        return rulebooks
    
    def get_status(self) -> str:
        """Get current chat status."""
        return (
            f"Provider: {self.config.get('provider', 'None')}\n"
            f"Model: {self.config.get('model', 'None')}\n"
            f"Persona: {self.active_persona.name if self.active_persona else 'None'}\n"
            f"Rules: {len(self.rules)} active\n"
            f"History: {len(self.messages)} messages"
        )

async def main():
    parser = argparse.ArgumentParser(description="Minimal LLM Chat Terminal")
    parser.add_argument("--provider", help="Set the LLM provider")
    parser.add_argument("--model", help="Set the model to use")
    parser.add_argument("--persona", help="Set the persona to use")
    parser.add_argument("--rulebook", help="Load a rulebook")
    parser.add_argument("--config", help="Path to config file")
    
    args = parser.parse_args()
    
    # Initialize chat terminal
    terminal = ChatTerminal(config_path=args.config)
    
    # Apply command line arguments
    if args.provider:
        terminal.set_provider(args.provider)
    if args.model:
        terminal.set_model(args.model)
    if args.persona:
        if not terminal.set_persona(args.persona):
            print(f"Error: Persona '{args.persona}' not found.")
    if args.rulebook:
        if not terminal.load_rulebook(args.rulebook):
            print(f"Error: Rulebook '{args.rulebook}' not found.")
    
    print("""
╔══════════════════════════════════════════════╗
║           LLM Chat Terminal v1.0             ║
║  Type /help for available commands           ║
╚══════════════════════════════════════════════╝
""")
    
    print(terminal.get_status())
    print()
    
    while True:
        try:
            # Get user input with readline for better UX
            try:
                user_input = input("\033[94mYou:\033[0m ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            
            if not user_input:
                continue
                
            # Handle commands
            if user_input.startswith('/'):
                cmd = user_input[1:].lower().split()
                
                if not cmd:
                    continue
                    
                if cmd[0] == 'help':
                    print("""
Available commands:
  /help               - Show this help message
  /exit               - Exit the program
  /status             - Show current chat status
  /provider <name>    - Set the LLM provider
  /model <name>       - Set the model to use
  /persona <name>     - Set the active persona
  /personas           - List available personas
  /rulebook <name>    - Load a rulebook
  /rulebooks          - List available rulebooks
  /rules              - Show active rules
  /clear              - Clear chat history
  /save               - Save current configuration
                    """.strip())
                
                elif cmd[0] == 'exit':
                    print("Goodbye!")
                    break
                
                elif cmd[0] == 'status':
                    print("\n" + terminal.get_status())
                
                elif cmd[0] == 'provider':
                    if len(cmd) > 1:
                        provider = cmd[1]
                        if terminal.set_provider(provider):
                            print(f"Provider set to: {provider}")
                        else:
                            print(f"Unknown provider: {provider}")
                    else:
                        print(f"Current provider: {terminal.config.get('provider', 'None')}")
                
                elif cmd[0] == 'model':
                    if len(cmd) > 1:
                        model = ' '.join(cmd[1:])
                        terminal.set_model(model)
                        print(f"Model set to: {model}")
                    else:
                        print(f"Current model: {terminal.config.get('model', 'None')}")
                
                elif cmd[0] == 'persona':
                    if len(cmd) > 1:
                        persona = ' '.join(cmd[1:])
                        if terminal.set_persona(persona):
                            print(f"Persona set to: {persona}")
                        else:
                            print(f"Unknown persona: {persona}")
                    else:
                        print(f"Current persona: {terminal.active_persona.name if terminal.active_persona else 'None'}")
                
                elif cmd[0] == 'personas':
                    print("\nAvailable personas:")
                    for i, persona in enumerate(terminal.list_personas(), 1):
                        print(f"  {i}. {persona}")
                
                elif cmd[0] == 'rulebook':
                    if len(cmd) > 1:
                        rulebook = ' '.join(cmd[1:])
                        if terminal.load_rulebook(rulebook):
                            print(f"Loaded rulebook: {rulebook} ({len(terminal.rules)} rules)")
                        else:
                            print(f"Unknown rulebook: {rulebook}")
                    else:
                        print("Please specify a rulebook name")
                
                elif cmd[0] == 'rulebooks':
                    rulebooks = terminal.list_rulebooks()
                    if rulebooks:
                        print("\nAvailable rulebooks:")
                        for i, rulebook in enumerate(rulebooks, 1):
                            print(f"  {i}. {rulebook}")
                    else:
                        print("No rulebooks found.")
                
                elif cmd[0] == 'rules':
                    if terminal.rules:
                        print("\nActive rules:")
                        for i, rule in enumerate(terminal.rules, 1):
                            print(f"  {i}. {rule}")
                    else:
                        print("No active rules. Use /rulebook to load a rulebook.")
                
                elif cmd[0] == 'clear':
                    terminal.messages = []
                    print("Chat history cleared.")
                
                elif cmd[0] == 'save':
                    terminal.save_config()
                    print("Configuration saved.")
                
                else:
                    print(f"Unknown command: {cmd[0]}. Type /help for a list of commands.")
                
                continue
            
            # Process regular message
            print("\n\033[92mAssistant:\033[0m ", end='', flush=True)
            
            response = await terminal.chat(user_input)
            print(response)
            
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
