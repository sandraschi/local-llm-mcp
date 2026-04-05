# Local LLM MCP Server - User Interaction Guide

Welcome to the **Local LLM MCP Server**! This guide helps you effectively use the 20+ tools for comprehensive LLM management.

## 🚀 **Quick Start Examples**

### **Basic Text Generation**
```python
# Simple text generation
result = await generate_text(
    model="llama3",
    prompt="Explain machine learning in simple terms"
)
print(result["data"])  # Generated text
```

### **Chat Completion**
```python
# Multi-turn conversation
messages = [
    {"role": "user", "content": "What is quantum computing?"}
]
result = await chat_completion(
    model="llama3",
    messages=messages,
    temperature=0.7
)
```

### **Model Management**
```python
# Check available models
models = await list_models()
print("Available models:", [m["name"] for m in models["data"]])

# Get detailed info
info = await get_model_info(model_name="llama3")
print(f"Model size: {info['size_gb']}GB")
```

### **Health Monitoring**
```python
# System health check
health = await llm_health_tool("health_check")
if health["health_score"] < 80:
    print("Warning: System health degraded")
    print("Issues:", health["issues"])
```

## 🛠️ **Portmanteau Tool Usage**

### **Health Operations**
```python
# Comprehensive health check
health = await llm_health_tool("health_check")

# List all available tools
tools = await llm_health_tool("list_tools", detail=2)

# Get help for specific tool
help_info = await llm_health_tool("tool_help", tool_name="generate_text")

# System information
sys_info = await llm_health_tool("system_info")

# Hardware requirements for fine-tuning
reqs = await llm_health_tool("hardware_requirements")
```

### **Model Operations**
```python
# List models across all providers
models = await llm_models_tool("list_models")

# Register new model
await llm_models_tool("register_model",
    name="my-custom-model",
    provider="ollama",
    model_path="/path/to/model"
)

# Load/unload models
await llm_models_tool("load_model", model_name="llama3")
await llm_models_tool("unload_model", model_name="llama3")
```

### **Generation Operations**
```python
# Text generation with full control
result = await llm_generation_tool("generate_text",
    model="llama3",
    prompt="Write a Python function to calculate fibonacci numbers",
    temperature=0.3,
    max_tokens=500,
    frequency_penalty=0.1,
    presence_penalty=0.1
)

# Chat completion
result = await llm_generation_tool("chat_completion",
    model="llama3",
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant"},
        {"role": "user", "content": "How do I optimize this SQL query?"}
    ]
)

# Text embeddings
embeddings = await llm_generation_tool("embed_text",
    model="text-embedding-ada-002",
    text=["Hello world", "Machine learning is awesome"]
)
```

### **Multimodal Operations**
```python
# Image analysis
analysis = await llm_multimodal_tool("analyze_image",
    image="/path/to/image.jpg",
    model_name="clip-ViT-B-32"
)
print(f"Image contains: {analysis['description']}")

# Generate image from text
image_result = await llm_multimodal_tool("generate_image",
    prompt="A serene mountain landscape at sunset",
    width=1024,
    height=768,
    num_inference_steps=50
)

# Compare images
similarity = await llm_multimodal_tool("compare_images",
    image1="/path/to/image1.jpg",
    image2="/path/to/image2.jpg"
)
print(f"Similarity score: {similarity['score']}")
```

### **Fine-tuning Operations**
```python
# Prepare dataset
dataset = await llm_finetuning_tool("prepare_dataset",
    dataset_path="/path/to/training/data.json",
    validation_split=0.1
)

# LoRA fine-tuning
training = await llm_finetuning_tool("lora_train",
    model_name="llama3",
    dataset=dataset,
    lora_rank=8,
    lora_alpha=16,
    output_dir="./lora-output",
    num_train_epochs=3
)

# Sparse fine-tuning
sparse_training = await llm_finetuning_tool("sparse_train",
    model_name="llama3",
    sparsity_ratio=0.5,
    sparsity_type="unstructured"
)

# DoRA fine-tuning
dora_training = await llm_finetuning_tool("dora_train",
    model_name="llama3",
    dropout_rate=0.1
)
```

## 🔧 **Provider-Specific Operations**

### **Ollama Integration**
```python
# List available models
ollama_models = await ollama_list_models()

# Pull a model
await ollama_pull_model(model="llama3:8b")

# Load for inference
await ollama_load_model(model="llama3:8b")

# Use in generation
result = await generate_text(model="llama3:8b", prompt="Hello!")

# Cleanup
await ollama_unload_model(model="llama3:8b")
await ollama_delete_model(model="llama3:8b")
```

### **LM Studio Integration**
```python
# List loaded models
lm_models = await lmstudio_list_models()

# Load a model
await lmstudio_load_model(model_path="/path/to/model.gguf")

# Use for generation
result = await generate_text(model="loaded-model", prompt="Generate text")

# Unload when done
await lmstudio_unload_model(model="loaded-model")
```

## 📊 **Understanding Response Formats**

All tools return structured responses:

```python
{
    "success": true,           # Operation success
    "data": {...},            # Primary results
    "metadata": {...},        # Additional context
    "summary": "Human-readable summary",
    "next_steps": [           # Suggested actions
        "Refine prompt for better results",
        "Try different model parameters"
    ],
    "error": null             # Error details if any
}
```

## 🎯 **Common Workflows**

### **Content Creation Pipeline**
```python
# 1. Check model availability
models = await list_models()
available_llms = [m for m in models["data"] if "llama" in m["name"]]

# 2. Generate initial content
draft = await generate_text(
    model="llama3",
    prompt="Write an article about AI ethics",
    max_tokens=1000
)

# 3. Refine with chat completion
refined = await chat_completion(
    model="llama3",
    messages=[
        {"role": "user", "content": f"Improve this draft: {draft['data']}"}
    ]
)
```

### **Model Evaluation Pipeline**
```python
# 1. Test multiple models
models_to_test = ["llama3", "mistral", "phi3"]
results = {}

for model in models_to_test:
    try:
        result = await generate_text(
            model=model,
            prompt="Explain recursion in programming",
            max_tokens=300
        )
        results[model] = result["data"]
    except:
        results[model] = "Failed to generate"

# 2. Compare embeddings
embeddings = await embed_text(text=list(results.values()))
# Analyze similarity scores...
```

### **Fine-tuning Pipeline**
```python
# 1. Assess system capabilities
health = await llm_health_tool("health_check")
if health["health_score"] < 70:
    print("Warning: System may not be suitable for training")

# 2. Prepare training data
dataset = await llm_finetuning_tool("prepare_dataset",
    data_path="./training_data.json",
    validation_split=0.2
)

# 3. Configure and run training
config = await llm_finetuning_tool("lora_prepare",
    model_name="llama3",
    dataset=dataset,
    lora_rank=16,
    learning_rate=2e-5
)

training = await llm_finetuning_tool("lora_train",
    config=config,
    output_dir="./fine-tuned-model"
)

# 4. Evaluate results
evaluation = await llm_finetuning_tool("evaluate_model",
    model_path=training["model_path"],
    test_dataset="./test_data.json"
)
```

## ⚠️ **Error Handling**

```python
# Always check for errors
result = await generate_text(model="nonexistent", prompt="test")
if not result.get("success", True):
    error_msg = result.get("error", "Unknown error")
    print(f"Operation failed: {error_msg}")

    # Try recovery suggestions
    if "next_steps" in result:
        print("Suggestions:", result["next_steps"])
```

## 🎨 **Best Practices**

1. **Start Simple**: Use basic tools before advanced features
2. **Check Health**: Monitor system resources during intensive operations
3. **Manage Memory**: Load/unload models as needed
4. **Use Appropriate Models**: Match model capabilities to task complexity
5. **Handle Errors Gracefully**: Check response status before using results
6. **Progressive Refinement**: Use initial results to guide next steps

## 📚 **Getting Help**

```python
# Get comprehensive tool help
help_info = await llm_health_tool("tool_help", tool_name="llm_generation_tool")

# Search for tools by functionality
search_results = await llm_health_tool("search_tools", query="generation")

# List all available operations
all_tools = await llm_health_tool("list_tools", detail=1)
```

This MCP server provides comprehensive LLM management capabilities with 20+ specialized tools for all your AI workflow needs!
