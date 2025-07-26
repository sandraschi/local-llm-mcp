#!/usr/bin/env python3
"""
Main entry point for the LLM MCP Chat Terminal.

This script provides a user-friendly interface for interacting with the LLM MCP Server,
with support for multiple LLM providers, personas, and advanced features.
"""

import os
import sys
import asyncio
import signal
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add the parent directory to the path so we can import the LLM MCP modules
sys.path.append(str(Path(__file__).parent.parent))

# Import our terminal and command processor
from tools.chat_terminal import ChatTerminal, Colors
from tools.cli_commands import CommandProcessor

class ChatApplication:
    """Main chat application class that ties everything together."""
    
    def __init__(self):
        self.terminal = ChatTerminal()
        self.command_processor = CommandProcessor(self.terminal)
        self.running = False
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        """Handle signals for graceful shutdown."""
        print("\nShutting down...")
        self.running = False
    
    async def run(self):
        """Run the chat application."""
        self.running = True
        
        # Print welcome message
        self._print_welcome()
        
        # Main loop
        while self.running:
            try:
                # Get user input
                try:
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None, 
                        lambda: input(f"{Colors.BLUE}You:{Colors.ENDC} ")
                    )
                except (EOFError, KeyboardInterrupt):
                    print("\nUse /exit or press Ctrl+C again to quit.")
                    try:
                        user_input = await asyncio.get_event_loop().run_in_executor(
                            None, 
                            lambda: input(f"{Colors.BLUE}You:{Colors.ENDC} ")
                        )
                    except (EOFError, KeyboardInterrupt):
                        print("\nGoodbye!")
                        break
                
                # Process empty input
                if not user_input.strip():
                    continue
                
                # Check if it's a command
                is_command = await self.command_processor.process_command(user_input)
                if is_command:
                    continue
                
                # Process regular chat input
                await self._process_chat_input(user_input)
                
            except Exception as e:
                print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
                if getattr(self.terminal, 'debug', False):
                    import traceback
                    traceback.print_exc()
    
    def _print_typing_indicator(self, active: bool = True):
        """Show or hide a typing indicator."""
        if active:
            print(f"\r{Colors.YELLOW}Assistant is typing...{Colors.ENDC} ", end="", flush=True)
        else:
            print("\r" + " " * 30 + "\r", end="", flush=True)
    
    async def _process_chat_input(self, user_input: str):
        """Process a user's chat input and display the response."""
        if not self.terminal.provider or not self.terminal.model:
            print(f"{Colors.YELLOW}Please set a provider and model first using /provider and /model commands.{Colors.ENDC}")
            return
        
        # Show typing indicator in a separate thread
        typing_task = asyncio.create_task(self._typing_indicator())
        
        try:
            # Start streaming the response
            print(f"\n{Colors.GREEN}Assistant:{Colors.ENDC} ", end="", flush=True)
            response_text = ""
            chunk_count = 0
            start_time = asyncio.get_event_loop().time()
            
            async for chunk in self.terminal.stream_response(user_input):
                if chunk_count == 0:
                    # First chunk received, stop the typing indicator
                    typing_task.cancel()
                    print("\r" + " " * 30 + "\r", end="", flush=True)
                    print(f"{Colors.GREEN}Assistant:{Colors.ENDC} ", end="", flush=True)
                
                print(chunk, end="", flush=True)
                response_text += chunk
                chunk_count += 1
                
                # Add a small delay to make streaming more visible
                await asyncio.sleep(0.01)
            
            # Calculate and display token stats
            end_time = asyncio.get_event_loop().time()
            duration = max(0.1, end_time - start_time)  # Avoid division by zero
            token_count = len(response_text.split())  # Approximate
            tokens_per_second = token_count / duration
            
            # Print token stats in a subtle way
            print(f"\n{Colors.CYAN}Generated {token_count} tokens in {duration:.1f}s ({tokens_per_second:.1f} tokens/s){Colors.ENDC}")
            
        except asyncio.CancelledError:
            # Typing indicator was cancelled, which is expected
            pass
            
        except Exception as e:
            # Ensure typing indicator is hidden on error
            typing_task.cancel()
            print("\r" + " " * 30 + "\r", end="", flush=True)
            
            # Print error message
            error_type = type(e).__name__
            error_msg = f"{Colors.RED}Error ({error_type}): {str(e)}{Colors.ENDC}"
            
            # Add more specific error handling for common cases
            if "API key" in str(e):
                error_msg += f"\n{Colors.YELLOW}Please check your API key in the configuration.{Colors.ENDC}"
            elif "connection" in str(e).lower():
                error_msg += f"\n{Colors.YELLOW}Please check your internet connection and try again.{Colors.ENDC}"
                
            print(error_msg)
            
            if self.terminal.debug:
                import traceback
                traceback.print_exc()
        
        finally:
            # Ensure typing indicator is always stopped
            if not typing_task.done():
                typing_task.cancel()
    
    async def _typing_indicator(self):
        """Show a typing indicator while waiting for a response."""
        try:
            while True:
                self._print_typing_indicator(True)
                await asyncio.sleep(0.5)
                self._print_typing_indicator(False)
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            self._print_typing_indicator(False)
            raise
    
    def _print_welcome(self):
        """Print the welcome message and help."""
        print(f"""
{Colors.HEADER}{Colors.BOLD}LLM MCP Chat Terminal{Colors.ENDC}

Welcome to the LLM MCP Chat Terminal! This tool allows you to interact with various
LLM providers, manage personas, and customize your chat experience.

{Colors.BOLD}Quick Start:{Colors.ENDC}
  1. List available providers: {Colors.CYAN}/provider{Colors.ENDC}
  2. Set a provider: {Colors.CYAN}/provider <name>{Colors.ENDC}
  3. List available models: {Colors.CYAN}/model{Colors.ENDC}
  4. Set a model: {Colors.CYAN}/model <name>{Colors.ENDC}
  5. Start chatting!

Type {Colors.CYAN}/help{Colors.ENDC} for a list of all commands.
""")


def main():
    """Entry point for the chat terminal."""
    # Set up asyncio event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Create and run the application
    app = ChatApplication()
    
    try:
        loop.run_until_complete(app.run())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        # Clean up
        loop.close()


if __name__ == "__main__":
    main()
