"""
Command-line interface commands for the LLM MCP chat terminal.
"""

import os
import re
import sys
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Callable, Awaitable
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from .chat_terminal import (
    ChatTerminal, ChatMessage, MessageRole, Colors, 
    Persona, Conversation, ChatConfig
)

class CommandCategory(Enum):
    GENERAL = "General"
    CONVERSATION = "Conversation"
    CONFIGURATION = "Configuration"
    PERSONAS = "Personas"
    RULEBOOKS = "Rulebooks"
    SYSTEM = "System"

@dataclass
class Command:
    """Represents a CLI command with handler and metadata."""
    name: str
    aliases: List[str]
    handler: Callable[..., Awaitable[None]]
    description: str
    usage: str
    category: CommandCategory
    requires_provider: bool = False
    requires_model: bool = False
    hidden: bool = False

class CommandProcessor:
    """Processes and executes chat terminal commands."""
    
    def __init__(self, terminal: ChatTerminal):
        self.terminal = terminal
        self.commands: Dict[str, Command] = {}
        self._register_commands()
    
    def _register_commands(self):
        """Register all available commands."""
        commands = [
            # General commands
            Command(
                name="help",
                aliases=["h", "?"],
                handler=self.cmd_help,
                description="Show this help message",
                usage="/help [command]",
                category=CommandCategory.GENERAL
            ),
            Command(
                name="exit",
                aliases=["quit", "q"],
                handler=self.cmd_exit,
                description="Exit the chat terminal",
                usage="/exit",
                category=CommandCategory.GENERAL
            ),
            Command(
                name="status",
                aliases=["info"],
                handler=self.cmd_status,
                description="Show current status and configuration",
                usage="/status",
                category=CommandCategory.GENERAL
            ),
            
            # Conversation commands
            Command(
                name="clear",
                aliases=["cls"],
                handler=self.cmd_clear,
                description="Clear the conversation history",
                usage="/clear",
                category=CommandCategory.CONVERSATION
            ),
            Command(
                name="search",
                aliases=["find"],
                handler=self.cmd_search,
                description="Search through conversation history",
                usage="/search <query>",
                category=CommandCategory.CONVERSATION
            ),
            
            # Configuration commands
            Command(
                name="config",
                aliases=["cfg"],
                handler=self.cmd_config,
                description="View or modify configuration",
                usage="/config [key] [value]",
                category=CommandCategory.CONFIGURATION
            ),
            Command(
                name="provider",
                aliases=["prov"],
                handler=self.cmd_provider,
                description="Set or show the current LLM provider",
                usage="/provider [name]",
                category=CommandCategory.CONFIGURATION
            ),
            Command(
                name="model",
                aliases=["mod"],
                handler=self.cmd_model,
                description="Set or show the current model",
                usage="/model [name]",
                category=CommandCategory.CONFIGURATION,
                requires_provider=True
            ),
            
            # Persona commands
            Command(
                name="persona",
                aliases=["p"],
                handler=self.cmd_persona,
                description="Set or show the current persona",
                usage="/persona [name]",
                category=CommandCategory.PERSONAS
            ),
            Command(
                name="personas",
                aliases=["plist"],
                handler=self.cmd_personas,
                description="List all available personas",
                usage="/personas",
                category=CommandCategory.PERSONAS
            ),
            
            # System commands
            Command(
                name="reload",
                aliases=["rl"],
                handler=self.cmd_reload,
                description="Reload configuration and personas",
                usage="/reload",
                category=CommandCategory.SYSTEM
            ),
            Command(
                name="debug",
                aliases=[],
                handler=self.cmd_debug,
                description="Toggle debug mode",
                usage="/debug",
                category=CommandCategory.SYSTEM,
                hidden=True
            )
        ]
        
        # Register all commands
        for cmd in commands:
            self.commands[cmd.name] = cmd
            for alias in cmd.aliases:
                self.commands[alias] = cmd
    
    async def process_command(self, input_text: str) -> bool:
        """Process a command and return True if it was a command."""
        if not input_text.startswith('/'):
            return False
            
        # Parse command and arguments
        parts = input_text[1:].split()
        if not parts:
            return False
            
        cmd_name = parts[0].lower()
        args = parts[1:]
        
        # Find the command
        cmd = self.commands.get(cmd_name)
        if not cmd or cmd.hidden:
            print(f"{Colors.RED}Unknown command: /{cmd_name}. Type /help for a list of commands.{Colors.ENDC}")
            return True
        
        # Check requirements
        if cmd.requires_provider and not self.terminal.provider:
            print(f"{Colors.RED}This command requires a provider to be set. Use /provider to set one.{Colors.ENDC}")
            return True
            
        if cmd.requires_model and not self.terminal.model:
            print(f"{Colors.RED}This command requires a model to be set. Use /model to set one.{Colors.ENDC}")
            return True
        
        # Execute the command
        try:
            await cmd.handler(args)
        except Exception as e:
            print(f"{Colors.RED}Error executing command: {e}{Colors.ENDC}")
            import traceback
            traceback.print_exc()
        
        return True
    
    # Command implementations
    async def cmd_help(self, args: List[str]) -> None:
        """Show help message."""
        if args:
            # Show help for specific command
            cmd = self.commands.get(args[0].lower())
            if not cmd or cmd.hidden:
                print(f"{Colors.RED}Unknown command: {args[0]}{Colors.ENDC}")
                return
                
            print(f"{Colors.HEADER}{Colors.BOLD}Command: /{cmd.name}{Colors.ENDC}")
            print(f"  {cmd.description}")
            print(f"  {Colors.BOLD}Usage:{Colors.ENDC} {cmd.usage}")
            if cmd.aliases:
                print(f"  {Colors.BOLD}Aliases:{Colors.ENDC} {', '.join(f'/{a}' for a in cmd.aliases)}")
            return
        
        # Show general help
        print(f"{Colors.HEADER}{Colors.BOLD}LLM MCP Chat Terminal - Available Commands{Colors.ENDC}\n")
        
        # Group commands by category
        categories: Dict[CommandCategory, List[Command]] = {}
        for cmd in set(self.commands.values()):
            if cmd.hidden:
                continue
            if cmd.category not in categories:
                categories[cmd.category] = []
            categories[cmd.category].append(cmd)
        
        # Print commands by category
        for category, cmds in categories.items():
            print(f"{Colors.BOLD}{category.value}:{Colors.ENDC}")
            for cmd in sorted(cmds, key=lambda c: c.name):
                print(f"  /{cmd.name:<10} {cmd.description}")
            print()
        
        print(f"Type {Colors.BOLD}/help <command>{Colors.ENDC} for more information about a command.")
    
    async def cmd_exit(self, args: List[str]) -> None:
        """Exit the chat terminal."""
        print("\nGoodbye!")
        sys.exit(0)
    
    async def cmd_status(self, args: List[str]) -> None:
        """Show current status and configuration."""
        print(f"{Colors.HEADER}{Colors.BOLD}Status{Colors.ENDC}")
        print(f"  {Colors.BOLD}Provider:{Colors.ENDC} {self.terminal.provider or 'None'}")
        print(f"  {Colors.BOLD}Model:{Colors.ENDC} {self.terminal.model or 'None'}")
        
        if self.terminal.active_persona:
            print(f"  {Colors.BOLD}Persona:{Colors.ENDC} {self.terminal.active_persona.name}")
        
        print(f"\n{Colors.BOLD}Configuration:{Colors.ENDC}")
        for field in self.terminal.chat_config.__dataclass_fields__:
            if field.startswith('_'):
                continue
            value = getattr(self.terminal.chat_config, field)
            print(f"  {field}: {value}")
    
    async def cmd_clear(self, args: List[str]) -> None:
        """Clear the conversation history."""
        self.terminal.conversation.clear()
        print(f"{Colors.GREEN}Conversation history cleared.{Colors.ENDC}")
    
    async def cmd_search(self, args: List[str]) -> None:
        """Search through conversation history."""
        if not args:
            print(f"{Colors.YELLOW}Usage: /search <query>{Colors.ENDC}")
            return
            
        query = ' '.join(args)
        results = self.terminal.search_history(query)
        
        if not results:
            print(f"{Colors.YELLOW}No matches found for '{query}'.{Colors.ENDC}")
            return
            
        print(f"{Colors.HEADER}Search results for '{query}':{Colors.ENDC}")
        for idx, (msg_idx, msg) in enumerate(results, 1):
            print(f"\n{Colors.BOLD}{idx}. Message {msg_idx+1}{Colors.ENDC}")
            print(f"{Colors.BLUE}{msg.role.upper()}:{Colors.ENDC} {msg.content[:200]}{'...' if len(msg.content) > 200 else ''}")
    
    async def cmd_config(self, args: List[str]) -> None:
        """View or modify configuration."""
        if not args:
            # Show all config
            print(f"{Colors.HEADER}Current Configuration:{Colors.ENDC}")
            for field in self.terminal.chat_config.__dataclass_fields__:
                if field.startswith('_'):
                    continue
                value = getattr(self.terminal.chat_config, field)
                print(f"  {field}: {value}")
            return
            
        if len(args) == 1:
            # Get value
            key = args[0]
            if not hasattr(self.terminal.chat_config, key):
                print(f"{Colors.RED}Unknown configuration key: {key}{Colors.ENDC}")
                return
                
            value = getattr(self.terminal.chat_config, key)
            print(f"{key}: {value}")
            return
            
        if len(args) >= 2:
            # Set value
            key, value_str = args[0], ' '.join(args[1:])
            if not hasattr(self.terminal.chat_config, key):
                print(f"{Colors.RED}Unknown configuration key: {key}{Colors.ENDC}")
                return
                
            # Try to convert to the correct type
            current_value = getattr(self.terminal.chat_config, key)
            try:
                if isinstance(current_value, bool):
                    value = value_str.lower() in ('true', 'yes', 'y', '1')
                elif isinstance(current_value, int):
                    value = int(value_str)
                elif isinstance(current_value, float):
                    value = float(value_str)
                else:
                    value = value_str
                    
                self.terminal.update_chat_config(**{key: value})
                print(f"{Colors.GREEN}Configuration updated: {key} = {value}{Colors.ENDC}")
            except ValueError as e:
                print(f"{Colors.RED}Invalid value for {key}: {e}{Colors.ENDC}")
    
    async def cmd_provider(self, args: List[str]) -> None:
        """Set or show the current LLM provider."""
        available_providers = self.terminal.provider_factory.list_providers()
        
        if not args:
            # Show current provider
            current = self.terminal.provider
            print(f"{Colors.HEADER}Available Providers:{Colors.ENDC}")
            for provider in available_providers:
                prefix = "* " if provider == current else "  "
                print(f"{prefix}{provider}")
            return
            
        # Set provider
        provider_name = args[0]
        if provider_name not in available_providers:
            print(f"{Colors.RED}Unknown provider: {provider_name}. Available providers: {', '.join(available_providers)}{Colors.ENDC}")
            return
            
        try:
            self.terminal.set_provider(provider_name)
            print(f"{Colors.GREEN}Provider set to: {provider_name}{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.RED}Failed to set provider: {e}{Colors.ENDC}")
    
    async def cmd_model(self, args: List[str]) -> None:
        """Set or show the current model."""
        if not self.terminal.provider:
            print(f"{Colors.RED}No provider selected. Use /provider first.{Colors.ENDC}")
            return
            
        available_models = self.terminal.provider.list_models()
        
        if not args:
            # Show available models
            current = self.terminal.model
            print(f"{Colors.HEADER}Available Models:{Colors.ENDC}")
            for model in available_models:
                prefix = "* " if model == current else "  "
                print(f"{prefix}{model}")
            return
            
        # Set model
        model_name = args[0]
        if model_name not in available_models:
            print(f"{Colors.RED}Unknown model: {model_name}. Available models: {', '.join(available_models)}{Colors.ENDC}")
            return
            
        try:
            self.terminal.set_model(model_name)
            print(f"{Colors.GREEN}Model set to: {model_name}{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.RED}Failed to set model: {e}{Colors.ENDC}")
    
    async def cmd_persona(self, args: List[str]) -> None:
        """Set or show the current persona."""
        if not args:
            # Show current persona
            if self.terminal.active_persona:
                persona = self.terminal.active_persona
                print(f"{Colors.HEADER}Current Persona:{Colors.ENDC}")
                print(f"  {Colors.BOLD}Name:{Colors.ENDC} {persona.name}")
                if persona.description:
                    print(f"  {Colors.BOLD}Description:{Colors.ENDC} {persona.description}")
                if persona.rules:
                    print(f"  {Colors.BOLD}Rules:{Colors.ENDC}")
                    for rule in persona.rules:
                        print(f"    - {rule}")
            else:
                print("No active persona. Use /persona <name> to set one.")
            return
            
        # Set persona
        persona_name = args[0]
        if persona_name not in self.terminal.personas:
            print(f"{Colors.RED}Unknown persona: {persona_name}. Use /personas to list available personas.{Colors.ENDC}")
            return
            
        self.terminal.active_persona = self.terminal.personas[persona_name]
        print(f"{Colors.GREEN}Persona set to: {persona_name}{Colors.ENDC}")
    
    async def cmd_personas(self, args: List[str]) -> None:
        """List all available personas."""
        if not self.terminal.personas:
            print("No personas found. Create some in the personas directory.")
            return
            
        print(f"{Colors.HEADER}Available Personas:{Colors.ENDC}")
        for name, persona in self.terminal.personas.items():
            active = " (active)" if self.terminal.active_persona == persona else ""
            print(f"\n{Colors.BOLD}{name}{active}{Colors.ENDC}")
            if persona.description:
                print(f"  {persona.description}")
    
    async def cmd_reload(self, args: List[str]) -> None:
        """Reload configuration and personas."""
        self.terminal._load_personas()
        self.terminal._load_rulebooks()
        print(f"{Colors.GREEN}Configuration and personas reloaded.{Colors.ENDC}")
    
    async def cmd_debug(self, args: List[str]) -> None:
        """Toggle debug mode."""
        # This is a hidden command for development
        self.terminal.debug = not getattr(self.terminal, 'debug', False)
        status = "enabled" if self.terminal.debug else "disabled"
        print(f"Debug mode {status}.")

# This module can be imported and used like this:
# processor = CommandProcessor(terminal)
# is_command = await processor.process_command("/help")
