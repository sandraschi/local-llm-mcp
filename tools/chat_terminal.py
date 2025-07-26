#!/usr/bin/env python3
"""Minimal chat terminal with advanced features for testing LLM providers."""

import os
import json
import yaml
import argparse
import readline
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pathlib import Path
from datetime import datetime

# Import the LLM MCP components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from llm_mcp.services.provider_factory import ProviderFactory
from llm_mcp.models.base import ModelProvider

class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class Persona:
    """Represents a chat persona with system prompt and rules."""
    
    def __init__(self, name: str, description: str, system_prompt: str, rules: List[str]):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.rules = rules
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "rules": self.rules
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Persona':
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            system_prompt=data["system_prompt"],
            rules=data.get("rules", [])
        )

class ChatTerminal:
    """Minimal chat terminal with advanced features."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.path.expanduser("~/.config/llm-mcp/terminal.yaml")
        self.history_path = os.path.expanduser("~/.local/share/llm-mcp/history.txt")
        self.personas_path = os.path.expanduser("~/.config/llm-mcp/personas/")
        self.rulebooks_path = os.path.expanduser("~/.config/llm-mcp/rulebooks/")
        
        # Create necessary directories
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        os.makedirs(self.personas_path, exist_ok=True)
        os.makedirs(self.rulebooks_path, exist_ok=True)
        
        self.config = self._load_config()
        self.provider_factory = ProviderFactory()
        self.provider = None
        self.model = None
        self.messages = []
        self.personas: Dict[str, Persona] = {}
        self.active_persona: Optional[Persona] = None
        self.rules: List[str] = []
        
        self._load_personas()
        self._load_history()
    
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
        """Load chat history from file."""
        self.messages = []
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            msg = json.loads(line.strip())
                            self.messages.append((msg["role"], msg["content"]))
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"Error loading history: {e}")
    
    def _save_history(self):
        """Save chat history to file."""
        try:
            with open(self.history_path, 'w', encoding='utf-8') as f:
                for role, content in self.messages:
                    f.write(json.dumps({"role": role, "content": content}) + "\n")
        except Exception as e:
            print(f"Error saving history: {e}")
    
    def _add_message(self, role: str, content: str):
        """Add a message to the history."""
        self.messages.append((role, content))
        # Keep only the last N messages
        max_history = self.config.get("history_size", 1000)
        if len(self.messages) > max_history:
            self.messages = self.messages[-max_history:]
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
        for role, content in self.messages:
            context.append({"role": role, "content": content})
        
        return context
    
    def set_provider(self, provider_name: str):
        """Set the LLM provider."""
        try:
            provider_enum = getattr(ModelProvider, provider_name.upper())
            self.provider = self.provider_factory.create_provider(provider_enum, {})
            self.config["provider"] = provider_name
            return True
        except (AttributeError, ValueError) as e:
            print(f"Error setting provider: {e}")
            return False
    
    def set_model(self, model_name: str):
        """Set the model to use."""
        self.model = model_name
        self.config["model"] = model_name
    
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
