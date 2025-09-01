# Gradio Interface for LLM MCP

This document explains how to create interactive web interfaces for the LLM MCP server using Gradio.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Basic Chat Interface](#basic-chat-interface)
- [Tool Visualization](#tool-visualization)
- [Advanced Features](#advanced-features)
- [Deployment](#deployment)
- [Best Practices](#best-practices)

## Overview

Gradio provides an easy way to create web interfaces for your LLM applications. This guide shows how to integrate it with the LLM MCP server's tool calling functionality.

## Installation

```bash
pip install gradio>=4.0.0
```

## Basic Chat Interface

```python
import gradio as gr
from llm_mcp.tools.tool_calling import create_tool_calling_llm

# Initialize your LLM with tools
llm = create_tool_calling_llm(your_llm_client)
llm.register_tool(get_weather)
llm.register_tool(search_web)

# Store conversation history
conversation_history = []

async def chat(message: str, history):
    # Add user message to history
    conversation_history.append({"role": "user", "content": message})
    
    try:
        # Get response with tool calling
        response = await llm.generate_with_tools(
            conversation_history,
            model="your-model-name"
        )
        
        # Add assistant's response to history
        if "content" in response:
            conversation_history.append({"role": "assistant", "content": response["content"]})
            return response["content"]
            
        return "I don't have a response for that."
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Create the interface
demo = gr.ChatInterface(
    fn=chat,
    title="LLM Tool Calling Demo",
    description="Chat with an LLM that can use tools"
)

if __name__ == "__main__":
    demo.launch()
```

## Tool Visualization

Add visual feedback for tool usage:

```python
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="Your message")
            submit_btn = gr.Button("Send")
        
        with gr.Column(scale=1):
            gr.Markdown("### Tool Usage")
            tool_calls = gr.JSON(label="Recent Tool Calls")
    
    def update_tool_display():
        # Get recent tool calls from conversation history
        recent_tools = [
            {"tool": msg.get("name", ""), "args": msg.get("arguments", {})}
            for msg in conversation_history[-5:]
            if "tool_calls" in msg
        ]
        return recent_tools
    
    # Update tool display on message submission
    submit_btn.click(
        lambda x: ("", update_tool_display()),
        inputs=msg,
        outputs=[msg, tool_calls]
    )
```

## Advanced Features

### Tool Configuration Panel

```python
with gr.Blocks() as demo:
    with gr.Tab("Chat"):
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
    
    with gr.Tab("Tools"):
        gr.Markdown("## Tool Configuration")
        
        # Tool toggles
        with gr.Row():
            weather_toggle = gr.Checkbox(label="Enable Weather Tool", value=True)
            search_toggle = gr.Checkbox(label="Enable Web Search", value=True)
        
        # Tool-specific settings
        with gr.Group(visible=True) as weather_settings:
            gr.Markdown("### Weather Tool Settings")
            default_location = gr.Textbox(label="Default Location", value="New York")
            temp_unit = gr.Radio(
                ["celsius", "fahrenheit"], 
                label="Temperature Unit",
                value="celsius"
            )
```

### Asynchronous Tool Execution

```python
import asyncio
from typing import List, Dict, Any

async def process_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Execute tool calls asynchronously."""
    tasks = [llm.execute_tool_call(call) for call in tool_calls]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

### Streaming Responses

```python
async def chat_stream(message: str, history):
    # Initialize response
    full_response = ""
    
    # Get streaming response
    async for chunk in llm.stream_with_tools(
        [{"role": "user", "content": message}],
        model="your-model-name"
    ):
        # Process the chunk
        if "content" in chunk:
            full_response += chunk["content"]
            yield full_response
```

## Deployment

### Local Development
```bash
python app.py
```

### Public Sharing
```python
if __name__ == "__main__":
    demo.launch(share=True)  # Generates a public URL
```

### Deploying with Docker

1. Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

2. Build and run:
```bash
docker build -t llm-mcp-gradio .
docker run -p 7860:7860 llm-mcp-gradio
```

## Best Practices

### UI/UX
1. **Loading States**: Show loading indicators during tool execution
2. **Error Handling**: Display user-friendly error messages
3. **Responsive Design**: Ensure the interface works on different screen sizes
4. **Accessibility**: Use proper contrast and ARIA labels

### Performance
1. **Caching**: Cache tool results when appropriate
2. **Batching**: Process multiple tool calls in parallel
3. **Resource Management**: Clean up resources when done

### Security
1. **Input Validation**: Sanitize all user inputs
2. **Rate Limiting**: Prevent abuse of your API
3. **Authentication**: Add login if needed

## Examples

### Complete App with Tool Visualization

```python
import gradio as gr
import asyncio
from typing import List, Dict, Any

# Initialize LLM and tools
llm = create_tool_calling_llm(your_llm_client)
llm.register_tool(get_weather)
llm.register_tool(search_web)

# Store conversation history
conversation_history = []

async def chat(message: str, history: List[List[str]]) -> str:
    # Add user message to history
    conversation_history.append({"role": "user", "content": message})
    
    try:
        # Get response with tool calling
        response = await llm.generate_with_tools(
            conversation_history,
            model="your-model-name"
        )
        
        # Process tool calls if any
        while "tool_calls" in response and response["tool_calls"]:
            # Execute tool calls
            tool_results = []
            for call in response["tool_calls"]:
                result = await llm.execute_tool_call(call)
                tool_results.append({
                    "role": "tool",
                    "name": call.name,
                    "content": str(result.content)
                })
            
            # Add tool results to conversation
            conversation_history.extend(tool_results)
            
            # Get next response
            response = await llm.generate_with_tools(
                conversation_history,
                model="your-model-name"
            )
        
        # Add assistant's response to history
        if "content" in response:
            conversation_history.append({
                "role": "assistant",
                "content": response["content"]
            })
            return response["content"]
            
        return "I don't have a response for that."
        
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Create the interface
with gr.Blocks() as demo:
    gr.Markdown("# LLM Tool Calling Demo")
    
    with gr.Row():
        # Chat interface
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500)
            msg = gr.Textbox(label="Your message")
            submit_btn = gr.Button("Send")
        
        # Tool visualization
        with gr.Column(scale=1):
            gr.Markdown("## Tool Usage")
            tool_calls = gr.JSON(label="Recent Tool Calls")
    
    # Handle message submission
    submit_btn.click(
        chat, 
        inputs=[msg, chatbot], 
        outputs=[msg, chatbot]
    )
    
    # Update tool visualization
    def update_tool_display():
        recent_tools = [
            {"tool": msg.get("name", ""), "args": msg.get("arguments", {})}
            for msg in conversation_history[-5:]
            if "tool_calls" in msg
        ]
        return recent_tools
    
    submit_btn.click(
        update_tool_display,
        outputs=[tool_calls]
    )

if __name__ == "__main__":
    demo.launch(debug=True)
```

## Troubleshooting

### Common Issues
1. **Tool Calls Not Working**: Ensure tools are properly registered
2. **Performance Issues**: Check for unnecessary re-renders
3. **Connection Errors**: Verify your LLM server is running

### Debugging
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use Gradio's built-in debug mode
demo.launch(debug=True)
```

## See Also
- [Gradio Documentation](https://gradio.app/docs/)
- [Gradio Examples](https://gradio.app/guides/quickstart)
