"""Tool calling functionality for LLMs.

This module provides utilities for working with LLMs that support tool/function calling,
such as OpenAI's GPT-4, Anthropic's Claude, and others.
"""
import inspect
import json
import logging
from typing import Dict, List, Any, Callable, Optional, Type, get_type_hints, Union
from pydantic import BaseModel, create_model

logger = logging.getLogger(__name__)

class ToolCall(BaseModel):
    """Represents a tool/function call from an LLM."""
    name: str
    arguments: Dict[str, Any]
    id: Optional[str] = None
    type: str = "function"

class ToolCallResult(BaseModel):
    """Result of a tool call execution."""
    call_id: str
    name: str
    content: Any
    is_error: bool = False

def tool_to_schema(tool_func: Callable) -> Dict[str, Any]:
    """Convert a Python function to an OpenAI-style function schema.
    
    Args:
        tool_func: The function to convert
        
    Returns:
        Dictionary containing the function schema
    """
    # Get function signature and docstring
    sig = inspect.signature(tool_func)
    docstring = inspect.getdoc(tool_func) or ""
    
    # Extract parameter descriptions from docstring
    param_descriptions = {}
    for line in docstring.split('\n'):
        line = line.strip()
        if line.startswith('Args:'):
            continue
        if ':' in line:
            param_name = line.split(':')[0].strip()
            param_desc = ':'.join(line.split(':')[1:]).strip()
            if param_name and param_desc:
                param_descriptions[param_name] = param_desc
    
    # Build parameter schema
    properties = {}
    required = []
    type_map = {
        'str': 'string',
        'int': 'integer',
        'float': 'number',
        'bool': 'boolean',
        'list': 'array',
        'dict': 'object'
    }
    
    for name, param in sig.parameters.items():
        if name == 'self':
            continue
            
        param_type = 'string'  # Default type
        if param.annotation != inspect.Parameter.empty:
            type_name = param.annotation.__name__
            param_type = type_map.get(type_name.lower(), 'string')
        
        param_schema = {
            'type': param_type,
            'description': param_descriptions.get(name, '')
        }
        
        # Handle default values
        if param.default != inspect.Parameter.empty:
            param_schema['default'] = param.default
        else:
            required.append(name)
            
        properties[name] = param_schema
    
    # Create function schema
    function_schema = {
        'name': tool_func.__name__,
        'description': docstring.split('\n')[0] if docstring else '',
        'parameters': {
            'type': 'object',
            'properties': properties,
            'required': required
        }
    }
    
    return function_schema

class ToolCallingMixin:
    """Mixin for adding tool calling support to LLM clients."""
    
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._tool_schemas: List[Dict[str, Any]] = []
    
    def register_tool(self, func: Callable) -> None:
        """Register a tool/function that the LLM can call.
        
        Args:
            func: The function to register as a callable tool
        """
        if not callable(func):
            raise ValueError(f"Expected callable, got {type(func)}")
            
        tool_name = func.__name__
        if tool_name in self._tools:
            logger.warning(f"Tool '{tool_name}' is already registered, overwriting")
            
        self._tools[tool_name] = func
        
        # Convert to schema
        schema = tool_to_schema(func)
        self._tool_schemas = [s for s in self._tool_schemas if s['name'] != tool_name]
        self._tool_schemas.append(schema)
        
        logger.info(f"Registered tool: {tool_name}")
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get the schemas for all registered tools."""
        return self._tool_schemas.copy()
    
    async def execute_tool_call(self, tool_call: Union[ToolCall, Dict]) -> ToolCallResult:
        """Execute a tool call and return the result.
        
        Args:
            tool_call: The tool call to execute, either as a ToolCall or dict
            
        Returns:
            ToolCallResult with the execution result
        """
        if isinstance(tool_call, dict):
            tool_call = ToolCall(**tool_call)
            
        if not isinstance(tool_call, ToolCall):
            raise ValueError(f"Expected ToolCall or dict, got {type(tool_call)}")
            
        if tool_call.name not in self._tools:
            return ToolCallResult(
                call_id=tool_call.id or "",
                name=tool_call.name,
                content=f"Error: Tool '{tool_call.name}' not found",
                is_error=True
            )
            
        try:
            # Get the tool function
            func = self._tools[tool_call.name]
            
            # Call the function with the provided arguments
            if inspect.iscoroutinefunction(func):
                result = await func(**tool_call.arguments)
            else:
                result = func(**tool_call.arguments)
                
            return ToolCallResult(
                call_id=tool_call.id or "",
                name=tool_call.name,
                content=result
            )
            
        except Exception as e:
            logger.exception(f"Error executing tool {tool_call.name}")
            return ToolCallResult(
                call_id=tool_call.id or "",
                name=tool_call.name,
                content=f"Error: {str(e)}",
                is_error=True
            )
    
    def tool(self, func: Callable = None, **kwargs):
        """Decorator to register a tool/function.
        
        Can be used as @tool or @tool(name="custom_name")
        """
        def decorator(f):
            # If a name is provided in kwargs, update the function's __name__
            if 'name' in kwargs:
                f.__name__ = kwargs['name']
            self.register_tool(f)
            return f
            
        if func is None:
            # Called with arguments: @tool(name="custom_name")
            return decorator
        else:
            # Called without arguments: @tool
            return decorator(func)

def create_tool_calling_llm(llm_client: Any) -> 'ToolCallingLLM':
    """Create a tool-calling enabled LLM client.
    
    Args:
        llm_client: The base LLM client to add tool calling to
        
    Returns:
        ToolCallingLLM instance with tool calling support
    """
    class ToolCallingLLM(ToolCallingMixin):
        def __init__(self, client):
            super().__init__()
            self._client = client
            
        async def generate_with_tools(
            self,
            messages: List[Dict[str, Any]],
            tools: Optional[List[Dict[str, Any]]] = None,
            **kwargs
        ) -> Dict[str, Any]:
            """Generate a response with tool calling support.
            
            Args:
                messages: List of message dictionaries
                tools: List of tool schemas to use (defaults to registered tools)
                **kwargs: Additional arguments to pass to the underlying LLM
                
            Returns:
                Dictionary with the generated response and tool calls
            """
            tools_to_use = tools or self.get_tool_schemas()
            
            # Add tool schemas to the request if any tools are registered
            if tools_to_use:
                kwargs['tools'] = tools_to_use
                kwargs['tool_choice'] = 'auto'
            
            # Call the underlying LLM
            response = await self._client.generate(messages, **kwargs)
            
            # Process tool calls in the response
            if 'tool_calls' in response:
                tool_calls = response['tool_calls']
                tool_results = []
                
                for call in tool_calls:
                    result = await self.execute_tool_call(call)
                    tool_results.append({
                        'tool_call_id': result.call_id,
                        'role': 'tool',
                        'name': result.name,
                        'content': str(result.content),
                        'is_error': result.is_error
                    })
                
                # If there are tool results, we can either return them or make another call
                if tool_results:
                    # Add tool results to messages
                    messages.extend(tool_results)
                    
                    # Option 1: Return the tool results
                    # return {"tool_results": tool_results, "messages": messages}
                    
                    # Option 2: Make another call with the tool results
                    return await self.generate_with_tools(messages, tools=tools, **kwargs)
            
            return response
    
    return ToolCallingLLM(llm_client)
