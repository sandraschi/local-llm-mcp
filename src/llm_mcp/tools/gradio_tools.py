"""
Gradio Tools for LLM MCP

This module provides tools for creating and managing Gradio interfaces
for interacting with language models and their tools.
"""

import os
import logging
import asyncio
import threading
import webbrowser
from typing import Dict, List, Optional, Any, Callable, Union
from pathlib import Path

# Try to import Gradio
try:
    import gradio as gr
    from gradio.routes import App
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

logger = logging.getLogger(__name__)

class GradioManager:
    """Manages Gradio interfaces and their lifecycle."""
    
    def __init__(self):
        self.interfaces: Dict[str, Dict[str, Any]] = {}
        self._servers: Dict[str, Any] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._running = False
    
    def create_chat_interface(
        self,
        name: str,
        generate_fn: Callable,
        tools: Optional[List[Dict[str, Any]]] = None,
        title: str = "LLM Chat",
        description: str = "Chat with an LLM",
        theme: str = "default",
        share: bool = False,
        server_name: str = "0.0.0.0",
        server_port: int = 7860,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a chat interface.
        
        Args:
            name: Unique name for this interface
            generate_fn: Function to call for generating responses
            tools: List of available tools
            title: Interface title
            description: Interface description
            theme: Gradio theme
            share: Whether to create a public share link
            server_name: Interface address
            server_port: Port to run the interface on
            **kwargs: Additional arguments for Gradio interface
            
        Returns:
            Interface configuration
        """
        if not GRADIO_AVAILABLE:
            raise ImportError("Gradio is not installed. Install with: pip install gradio")
        
        # Store interface config
        self.interfaces[name] = {
            "type": "chat",
            "generate_fn": generate_fn,
            "tools": tools or [],
            "title": title,
            "description": description,
            "theme": theme,
            "share": share,
            "server_name": server_name,
            "server_port": server_port,
            "config": kwargs
        }
        
        return self.interfaces[name]
    
    def create_custom_interface(
        self,
        name: str,
        interface_fn: Callable,
        inputs: Optional[Any] = None,
        outputs: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a custom Gradio interface.
        
        Args:
            name: Unique name for this interface
            interface_fn: Function that creates the interface
            inputs: Input components
            outputs: Output components
            **kwargs: Additional arguments for Gradio interface
            
        Returns:
            Interface configuration
        """
        if not GRADIO_AVAILABLE:
            raise ImportError("Gradio is not installed. Install with: pip install gradio")
        
        # Store interface config
        self.interfaces[name] = {
            "type": "custom",
            "interface_fn": interface_fn,
            "inputs": inputs,
            "outputs": outputs,
            "config": kwargs
        }
        
        return self.interfaces[name]
    
    def launch_interface(
        self,
        name: str,
        server_name: Optional[str] = None,
        server_port: Optional[int] = None,
        share: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Launch a Gradio interface.
        
        Args:
            name: Name of the interface to launch
            server_name: Interface address (overrides config)
            server_port: Port to run the interface on (overrides config)
            share: Whether to create a public share link (overrides config)
            **kwargs: Additional arguments for Gradio launch
            
        Returns:
            Interface launch information
        """
        if not GRADIO_AVAILABLE:
            return {
                "status": "error",
                "message": "Gradio is not installed. Install with: pip install gradio"
            }
            
        if name not in self.interfaces:
            return {
                "status": "error",
                "message": f"Interface '{name}' not found"
            }
            
        config = self.interfaces[name].copy()
        
        # Override config with provided values
        if server_name is not None:
            config["server_name"] = server_name
        if server_port is not None:
            config["server_port"] = server_port
        if share is not None:
            config["share"] = share
            
        # Update with any additional kwargs
        config["config"].update(kwargs)
        
        # Create the interface in a separate thread
        def run_interface():
            try:
                if config["type"] == "chat":
                    # Create chat interface
                    with gr.Blocks(title=config["title"], theme=config["theme"]) as demo:
                        gr.Markdown(f"## {config['title']}")
                        if config["description"]:
                            gr.Markdown(config["description"])
                        
                        # Chat interface
                        chatbot = gr.Chatbot()
                        msg = gr.Textbox()
                        clear = gr.Button("Clear")
                        
                        # Tool visualization
                        with gr.Accordion("Tool Usage", open=False):
                            tool_output = gr.JSON(label="Tool Calls")
                        
                        # Chat function
                        async def respond(message, chat_history):
                            # Call the generate function
                            response = await config["generate_fn"](message, chat_history)
                            
                            # Update chat history
                            chat_history.append((message, response.get("content", "")))
                            
                            # Get tool calls if any
                            tool_calls = response.get("tool_calls", [])
                            
                            return {
                                chatbot: chat_history,
                                tool_output: tool_calls
                            }
                        
                        # Connect components
                        msg.submit(
                            respond,
                            [msg, chatbot],
                            [chatbot, tool_output]
                        )
                        
                        # Clear chat
                        def clear_chat():
                            return [], []
                        
                        clear.click(clear_chat, None, [chatbot, tool_output], queue=False)
                        
                elif config["type"] == "custom":
                    # Create custom interface
                    demo = gr.Interface(
                        fn=config["interface_fn"],
                        inputs=config["inputs"],
                        outputs=config["outputs"],
                        **config["config"]
                    )
                
                # Launch the interface
                server = demo.launch(
                    server_name=config.get("server_name", "0.0.0.0"),
                    server_port=config.get("server_port", 7860),
                    share=config.get("share", False),
                    **config["config"]
                )
                
                # Store the server
                self._servers[name] = server
                
                # Keep the interface running
                if hasattr(demo, 'close'):
                    demo.close()
                    
            except Exception as e:
                logger.error(f"Error in Gradio interface {name}: {e}")
        
        # Start the interface in a new thread
        thread = threading.Thread(
            target=run_interface,
            daemon=True,
            name=f"gradio-{name}"
        )
        thread.start()
        
        # Store the thread
        self._threads[name] = thread
        
        # Get the URL
        url = f"http://{config.get('server_name', 'localhost')}:{config.get('server_port', 7860)}"
        
        # Open in browser if local
        if config.get("server_name") in ["0.0.0.0", "localhost", "127.0.0.1"]:
            try:
                webbrowser.open(url)
            except Exception as e:
                logger.warning(f"Could not open browser: {e}")
        
        return {
            "status": "success",
            "name": name,
            "url": url,
            "server_name": config.get("server_name"),
            "server_port": config.get("server_port"),
            "share": config.get("share", False)
        }
    
    def close_interface(self, name: str) -> Dict[str, Any]:
        """Close a running Gradio interface.
        
        Args:
            name: Name of the interface to close
            
        Returns:
            Status dictionary
        """
        if name in self._servers:
            try:
                if hasattr(self._servers[name], 'close'):
                    self._servers[name].close()
                del self._servers[name]
                
                # Stop the thread if it's still running
                if name in self._threads and self._threads[name].is_alive():
                    self._threads[name].join(timeout=5)
                    
                return {"status": "success", "message": f"Interface '{name}' closed"}
                
            except Exception as e:
                return {"status": "error", "message": f"Error closing interface: {e}"}
        
        return {"status": "error", "message": f"Interface '{name}' not found or not running"}
    
    def list_interfaces(self) -> Dict[str, Any]:
        """List all interfaces and their status.
        
        Returns:
            Dictionary of interfaces and their status
        """
        return {
            name: {
                "type": config["type"],
                "running": name in self._servers,
                "config": {
                    k: v for k, v in config.items() 
                    if k not in ["generate_fn", "interface_fn"]
                }
            }
            for name, config in self.interfaces.items()
        }

# Global instance (internal use only)
_gradio_manager = GradioManager()

async def _create_chat_interface_impl(
    mcp: Any,
    name: str,
    title: str = "LLM Chat",
    description: str = "Chat with an LLM",
    theme: str = "default",
    tools: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Implementation for creating a chat interface.
    
    Args:
        mcp: The MCP server instance
        name: Unique name for this interface
        title: Interface title
        description: Interface description
        theme: Gradio theme
        tools: List of available tools
        **kwargs: Additional arguments for Gradio interface
        
    Returns:
        Interface configuration
    """
    async def generate_fn(message: str, history: List[tuple] = None) -> Dict[str, Any]:
        history = history or []
        response = await mcp.generate_with_tools(
            [{"role": "user", "content": message}],
            tools=tools
        )
        
        tool_calls = response.get("tool_calls", [])
        if tool_calls:
            tool_results = []
            for call in tool_calls:
                result = await mcp.execute_tool_call(call)
                tool_results.append({
                    "tool_call_id": call.get("id"),
                    "name": call.get("name"),
                    "arguments": call.get("arguments"),
                    "result": result
                })
            response["tool_calls"] = tool_results
        
        return response
    
    return _gradio_manager.create_chat_interface(
        name=name,
        generate_fn=generate_fn,
        title=title,
        description=description,
        theme=theme,
        tools=tools,
        **kwargs
    )


def register_gradio_tools(mcp):
    """Register Gradio tools with the MCP server.
    
    Args:
        mcp: The MCP server instance
        
    Returns:
        The MCP server with Gradio tools registered
    """
    if not GRADIO_AVAILABLE:
        logger.warning("Gradio is not installed. Gradio tools will not be available.")
        return mcp
    
    @mcp.tool()
    async def gradio_create_chat_interface(
        name: str,
        title: str = "LLM Chat",
        description: str = "Chat with an LLM",
        theme: str = "default",
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a chat interface.
        
        Args:
            name: Unique name for this interface
            title: Interface title
            description: Interface description
            theme: Gradio theme
            tools: List of available tools
            **kwargs: Additional arguments for Gradio interface
            
        Returns:
            Interface configuration
        """
        return await _create_chat_interface_impl(
            mcp=mcp,
            name=name,
            title=title,
            description=description,
            theme=theme,
            tools=tools,
            **kwargs
        )
    
    @mcp.tool()
    async def gradio_launch_interface(
        name: str,
        server_name: str = "0.0.0.0",
        server_port: int = 7860,
        share: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Launch a Gradio interface.
        
        Args:
            name: Name of the interface to launch
            server_name: Interface address
            server_port: Port to run the interface on
            share: Whether to create a public share link
            **kwargs: Additional arguments for Gradio launch
            
        Returns:
            Interface launch information
        """
        return _gradio_manager.launch_interface(
            name=name,
            server_name=server_name,
            server_port=server_port,
            share=share,
            **kwargs
        )
    
    @mcp.tool()
    async def gradio_close_interface(name: str) -> Dict[str, Any]:
        """Close a running Gradio interface.
        
        Args:
            name: Name of the interface to close
            
        Returns:
            Status dictionary
        """
        return _gradio_manager.close_interface(name)
    
    @mcp.tool()
    async def gradio_list_interfaces() -> Dict[str, Any]:
        """List all available Gradio interfaces.
        
        Returns:
            Dictionary of interfaces and their status
        """
        return _gradio_manager.list_interfaces()
    
    return mcp
