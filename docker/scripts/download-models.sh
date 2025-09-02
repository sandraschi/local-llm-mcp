#!/bin/bash
# Model Download and Management Script
# Downloads and manages models for vLLM v0.10.1.1

set -e

MODELS_DIR="./models"
HF_CACHE_DIR="$HOME/.cache/huggingface"

# Create directories if they don't exist
mkdir -p "$MODELS_DIR"
mkdir -p "$HF_CACHE_DIR"

function download_model() {
    local model_name="$1"
    local hf_token="$2"
    
    echo "📥 Downloading model: $model_name"
    
    if [ -n "$hf_token" ]; then
        export HUGGINGFACE_HUB_TOKEN="$hf_token"
    fi
    
    python3 -c "
from huggingface_hub import snapshot_download
import os

model_name = '$model_name'
cache_dir = '$HF_CACHE_DIR'

try:
    snapshot_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        resume_download=True,
        local_files_only=False
    )
    print(f'✅ Successfully downloaded {model_name}')
except Exception as e:
    print(f'❌ Failed to download {model_name}: {e}')
    exit(1)
"
}

function list_models() {
    echo "📋 Available models in cache:"
    find "$HF_CACHE_DIR" -name "*.safetensors" -o -name "*.bin" | head -20
    echo ""
    echo "💾 Cache size:"
    du -sh "$HF_CACHE_DIR" 2>/dev/null || echo "Cache directory not found"
}

function clean_cache() {
    echo "🧹 Cleaning model cache..."
    read -p "This will delete all cached models. Continue? (y/N): " confirm
    if [[ $confirm == [yY] ]]; then
        rm -rf "$HF_CACHE_DIR"
        mkdir -p "$HF_CACHE_DIR"
        echo "✅ Cache cleaned"
    else
        echo "❌ Cancelled"
    fi
}

case "$1" in
    "download")
        if [ -z "$2" ]; then
            echo "Usage: $0 download <model_name> [hf_token]"
            echo "Examples:"
            echo "  $0 download meta-llama/Meta-Llama-3.1-8B-Instruct"
            echo "  $0 download microsoft/DialoGPT-medium"
            echo "  $0 download mistralai/Mistral-7B-Instruct-v0.3"
            exit 1
        fi
        download_model "$2" "$3"
        ;;
    
    "list")
        list_models
        ;;
    
    "clean")
        clean_cache
        ;;
    
    "popular")
        echo "📚 Popular models for vLLM v0.10.1.1:"
        echo "  meta-llama/Meta-Llama-3.1-8B-Instruct    (8B, good balance)"
        echo "  meta-llama/Meta-Llama-3.1-70B-Instruct   (70B, very capable)"
        echo "  mistralai/Mistral-7B-Instruct-v0.3       (7B, fast)"
        echo "  microsoft/DialoGPT-medium                 (117M, conversational)"
        echo "  codellama/CodeLlama-7b-Instruct-hf        (7B, coding)"
        echo ""
        echo "Use: $0 download <model_name>"
        ;;
    
    *)
        echo "Usage: $0 {download|list|clean|popular}"
        echo ""
        echo "Commands:"
        echo "  download <model> [token]  - Download a model from HuggingFace"
        echo "  list                      - List cached models"
        echo "  clean                     - Clean model cache"
        echo "  popular                   - Show popular model suggestions"
        exit 1
        ;;
esac
