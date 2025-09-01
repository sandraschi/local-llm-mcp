# Tool Calling with LLMs

This document explains how to use the tool calling functionality in the LLM MCP server, which allows LLMs to interact with external tools and functions.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Defining Tools](#defining-tools)
- [Registering Tools](#registering-tools)
- [Using Tools with LLMs](#using-tools-with-llms)
- [Advanced Usage](#advanced-usage)
- [Best Practices](#best-practices)
- [Related Guides](#related-guides)

## Overview

Tool calling enables LLMs to execute functions with structured inputs and outputs. This is particularly useful for:
- Accessing external APIs and services
- Performing computations
- Retrieving real-time data
- Interacting with databases
- And more

The tool calling system supports:
- Both synchronous and asynchronous functions
- Automatic schema generation from Python functions
- Error handling and validation
- Integration with any LLM that supports function calling

## Quick Start

1. **Define a tool**:

```python
from llm_mcp.tools.tool_calling import tool

@tool
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a location.
    
    Args:
        location: The city and state, e.g., "San Francisco, CA"
        unit: The unit of temperature ("celsius" or "fahrenheit")
        
    Returns:
        A string describing the current weather
    """
    # Implementation here
    return f"The weather in {location} is 22°{unit[0].upper()}"
```

2. **Use the tool with an LLM**:

```python
from llm_mcp.tools.tool_calling import create_tool_calling_llm

# Assuming you have an LLM client (e.g., OpenAI, Anthropic, etc.)
llm = create_tool_calling_llm(your_llm_client)

# Register the tool
llm.register_tool(get_weather)

# Use the LLM with tool calling
response = await llm.generate_with_tools(
    [{"role": "user", "content": "What's the weather in Paris?"}],
    model="your-model-name"
)
```

## Defining Tools

Tools are just Python functions with type hints and docstrings. The system automatically generates the appropriate schema for the LLM.

### Required Elements

1. **Type Hints**: All parameters and return types should have type hints.
2. **Docstrings**: Use Google-style docstrings with Args and Returns sections.
3. **Descriptions**: Provide clear descriptions for parameters and the function itself.

### Example Tool

```python
from typing import List, Dict

@tool
def search_web(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """Search the web for information.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return (1-10)
        
    Returns:
        List of search results with 'title' and 'url' keys
    """
    # Implementation here
    pass
```

## Registering Tools

You can register tools in several ways:

### 1. Using the Decorator (Recommended)

```python
from llm_mcp.tools.tool_calling import tool

@tool
def get_time(timezone: str = "UTC") -> str:
    """Get the current time in the specified timezone."""
    # Implementation
    pass
```

### 2. Using register_tool

```python
def get_stock_price(symbol: str) -> float:
    """Get the current stock price for a symbol."""
    # Implementation
    pass

llm.register_tool(get_stock_price)
```

### 3. With Custom Names

```python
@tool(name="get_forecast")
async def get_weather_forecast(location: str, days: int = 1) -> Dict:
    """Get weather forecast for a location."""
    # Implementation
    pass
```

## Using Tools with LLMs

### Basic Usage

```python
# Initialize your LLM client (e.g., OpenAI, Anthropic, etc.)
llm = create_tool_calling_llm(your_llm_client)

# Register tools
llm.register_tool(get_weather)
llm.register_tool(search_web)

# Generate a response with tool calling
response = await llm.generate_with_tools(
    [{"role": "user", "content": "What's the weather in Tokyo and find me some local news?"}],
    model="your-model-name"
)
```

### Handling Tool Results

The LLM will automatically receive the results of any tool calls and can use them to generate a final response.

## Advanced Usage

### Custom Tool Schemas

For more control, you can define a custom schema:

```python
custom_tool = {
    "name": "custom_tool",
    "description": "A custom tool with a specific schema",
    "parameters": {
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "First parameter"},
            "param2": {"type": "number", "description": "Second parameter"}
        },
        "required": ["param1"]
    }
}

# Register the custom tool
@llm.tool(**custom_tool)
async def custom_implementation(param1: str, param2: int = 1) -> str:
    # Implementation
    pass
```

### Error Handling

Handle errors in your tool implementations:

```python
@tool
def get_weather(location: str) -> str:
    try:
        # Implementation that might fail
        return f"Weather in {location}: Sunny"
    except Exception as e:
        return f"Error getting weather: {str(e)}"
```

## Best Practices

1. **Keep Tools Focused**: Each tool should do one thing well.
2. **Handle Errors Gracefully**: Always include error handling in your tools.
3. **Use Descriptive Names**: Make tool and parameter names clear and descriptive.
4. **Document Thoroughly**: Include comprehensive docstrings with examples.
5. **Validate Inputs**: Validate all inputs before processing.
6. **Limit Tool Scope**: Don't give tools more access than necessary.
7. **Monitor Usage**: Keep track of tool usage for debugging and optimization.

## Related Guides

- [LoRA Integration](../lora/README.md) - Learn how to fine-tune models with LoRA for better tool usage
- [Gradio Interface](../gradio/README.md) - Create interactive web interfaces for your tool-calling LLM

## Example: Complete Tool Implementation

```python
from typing import List, Dict
from datetime import datetime
import pytz

@tool
def get_time(timezone: str = "UTC") -> str:
    """Get the current time in the specified timezone.
    
    Args:
        timezone: IANA timezone name (e.g., 'America/New_York', 'Asia/Tokyo')
        
    Returns:
        Formatted time string
        
    Raises:
        ValueError: If the timezone is invalid
    """
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz)
        return now.strftime(f"The current time in {timezone} is %Y-%m-%d %H:%M:%S %Z")
    except pytz.exceptions.UnknownTimeZoneError:
        raise ValueError(f"Unknown timezone: {timezone}")
```

## Integration with Different LLMs

The tool calling system is designed to work with any LLM that supports function calling. Here's how to integrate with popular providers:

### OpenAI

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key="your-api-key")
llm = create_tool_calling_llm(client)

# Register tools and use as shown above
```

### Anthropic

```python
import anthropic

client = anthropic.AsyncAnthropic(api_key="your-api-key")
llm = create_tool_calling_llm(client)
```

## Troubleshooting

### Common Issues

1. **Tool Not Found**: Make sure the tool is registered before use.
2. **Schema Generation Issues**: Ensure your function has proper type hints and docstrings.
3. **Authentication Errors**: Check your API keys and permissions.
4. **Rate Limiting**: Implement retry logic for rate-limited APIs.

### Debugging

Enable debug logging to see detailed information about tool calls:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Conclusion

Tool calling enables powerful interactions between LLMs and external systems. By following this guide, you can create robust, maintainable tools that extend the capabilities of your LLM applications.
