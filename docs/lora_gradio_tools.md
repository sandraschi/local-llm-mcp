# LoRA and Gradio Tools

This document provides an overview of the LoRA (Low-Rank Adaptation) and Gradio tools available in the LLM MCP server.

## Table of Contents
- [LoRA Tools](#lora-tools)
  - [List Available LoRA Adapters](#list-available-lora-adapters)
  - [Load LoRA Adapter](#load-lora-adapter)
  - [Unload LoRA Adapter](#unload-lora-adapter)
  - [List Loaded LoRA Adapters](#list-loaded-lora-adapters)
- [Gradio Tools](#gradio-tools)
  - [Create Chat Interface](#create-chat-interface)
  - [Launch Interface](#launch-interface)
  - [Close Interface](#close-interface)
  - [List Interfaces](#list-interfaces)
- [Example Usage](#example-usage)
- [Dependencies](#dependencies)

## LoRA Tools

### List Available LoRA Adapters

List all available LoRA adapters in the configured directory.

**Endpoint**: `lora_list_adapters`

**Parameters**:
- `directory` (str, optional): Directory containing LoRA adapters. Defaults to `~/.cache/llm-mcp/loras`

**Returns**:
```json
{
  "status": "success",
  "adapters": [
    {
      "name": "adapter_name",
      "path": "/path/to/adapter",
      "config": { ... },
      "size_mb": 123.45
    }
  ]
}
```

### Load LoRA Adapter

Load a LoRA adapter onto a base model.

**Endpoint**: `lora_load_adapter`

**Parameters**:
- `adapter_name` (str): Name of the adapter to load
- `model_name` (str): Name of the base model to load the adapter onto
- `adapter_dir` (str, optional): Directory containing the adapter. Defaults to `~/.cache/llm-mcp/loras`
- `device_map` (str, optional): Device to load the adapter on. Defaults to "auto"
- `dtype` (str, optional): Data type for the adapter. Defaults to "float16"
- `trust_remote_code` (bool, optional): Whether to trust remote code. Defaults to False
- **kwargs: Additional arguments passed to `peft.PeftModel.from_pretrained`

**Returns**:
```json
{
  "status": "success",
  "message": "Adapter loaded successfully",
  "adapter": {
    "name": "adapter_name",
    "model_name": "base_model_name",
    "device": "cuda:0",
    "dtype": "float16"
  }
}
```

### Unload LoRA Adapter

Unload a currently loaded LoRA adapter.

**Endpoint**: `lora_unload_adapter`

**Parameters**:
- `adapter_name` (str): Name of the adapter to unload

**Returns**:
```json
{
  "status": "success",
  "message": "Adapter unloaded successfully"
}
```

### List Loaded LoRA Adapters

List all currently loaded LoRA adapters.

**Endpoint**: `lora_list_loaded`

**Returns**:
```json
{
  "status": "success",
  "adapters": [
    {
      "name": "adapter_name",
      "model_name": "base_model_name",
      "device": "cuda:0",
      "dtype": "float16"
    }
  ]
}
```

## Gradio Tools

### Create Chat Interface

Create a new Gradio chat interface.

**Endpoint**: `gradio_create_chat_interface`

**Parameters**:
- `name` (str): Unique name for this interface
- `title` (str, optional): Interface title. Defaults to "LLM Chat"
- `description` (str, optional): Interface description. Defaults to "Chat with an LLM"
- `theme` (str, optional): Gradio theme. Defaults to "default"
- `tools` (List[Dict], optional): List of available tools
- **kwargs: Additional arguments for Gradio interface

**Returns**:
```json
{
  "status": "success",
  "name": "interface_name",
  "type": "chat",
  "config": { ... }
}
```

### Launch Interface

Launch a Gradio interface.

**Endpoint**: `gradio_launch_interface`

**Parameters**:
- `name` (str): Name of the interface to launch
- `server_name` (str, optional): Interface address. Defaults to "0.0.0.0"
- `server_port` (int, optional): Port to run the interface on. Defaults to 7860
- `share` (bool, optional): Whether to create a public share link. Defaults to False
- **kwargs: Additional arguments for Gradio launch

**Returns**:
```json
{
  "status": "success",
  "name": "interface_name",
  "url": "http://localhost:7860",
  "server_name": "0.0.0.0",
  "server_port": 7860,
  "share": false
}
```

### Close Interface

Close a running Gradio interface.

**Endpoint**: `gradio_close_interface`

**Parameters**:
- `name` (str): Name of the interface to close

**Returns**:
```json
{
  "status": "success",
  "message": "Interface 'interface_name' closed"
}
```

### List Interfaces

List all available Gradio interfaces.

**Endpoint**: `gradio_list_interfaces`

**Returns**:
```json
{
  "interface_name": {
    "type": "chat",
    "running": true,
    "config": {
      "title": "LLM Chat",
      "description": "Chat with an LLM",
      "theme": "default",
      "server_name": "0.0.0.0",
      "server_port": 7860,
      "share": false
    }
  }
}
```

## Example Usage

### Using LoRA Adapters

1. List available adapters:
   ```python
   response = await mcp.call("lora_list_adapters")
   print(response["adapters"])
   ```

2. Load an adapter:
   ```python
   response = await mcp.call(
       "lora_load_adapter",
       adapter_name="my_lora_adapter",
       model_name="meta-llama/Llama-2-7b-hf"
   )
   print(response["message"])
   ```

3. Unload an adapter:
   ```python
   response = await mcp.call("lora_unload_adapter", adapter_name="my_lora_adapter")
   print(response["message"])
   ```

### Using Gradio Interfaces

1. Create a chat interface:
   ```python
   response = await mcp.call(
       "gradio_create_chat_interface",
       name="my_chat",
       title="My LLM Chat",
       description="Chat with a fine-tuned LLM"
   )
   ```

2. Launch the interface:
   ```python
   response = await mcp.call(
       "gradio_launch_interface",
       name="my_chat",
       server_port=9000
   )
   print(f"Interface available at: {response['url']}")
   ```

3. Close the interface:
   ```python
   response = await mcp.call("gradio_close_interface", name="my_chat")
   print(response["message"])
   ```

## Dependencies

- `peft>=0.4.0` - For LoRA adapter support
- `gradio>=3.0.0` - For Gradio interfaces
- `torch` - Required for PEFT and model loading
- `transformers` - Required for model loading and inference

Install with:
```bash
pip install peft gradio torch transformers
```
