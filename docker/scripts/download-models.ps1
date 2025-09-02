# Model Download and Management Script (PowerShell)
# Downloads and manages models for vLLM v0.10.1.1

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("download", "list", "clean", "popular")]
    [string]$Action,
    
    [string]$ModelName = "",
    [string]$HfToken = ""
)

$ModelsDir = ".\models"
$HfCacheDir = "$env:USERPROFILE\.cache\huggingface"

function Write-ColorOutput($Message, $Color = "White") {
    Write-Host $Message -ForegroundColor $Color
}

function Download-Model($ModelName, $HfToken) {
    Write-ColorOutput "📥 Downloading model: $ModelName" "Cyan"
    
    if ($HfToken) {
        $env:HUGGINGFACE_HUB_TOKEN = $HfToken
    }
    
    $tempScript = "C:\temp\download_model_$(Get-Date -Format 'HHmmss').py"
    
    @"
from huggingface_hub import snapshot_download
import sys

try:
    snapshot_download(
        repo_id='$ModelName',
        cache_dir='$HfCacheDir',
        resume_download=True,
        local_files_only=False
    )
    print('SUCCESS: Downloaded $ModelName')
except Exception as e:
    print(f'ERROR: Failed to download $ModelName: {e}')
    sys.exit(1)
"@ | Out-File -FilePath $tempScript -Encoding UTF8
    
    python $tempScript
    Remove-Item $tempScript -ErrorAction SilentlyContinue
}

function List-Models() {
    Write-ColorOutput "📋 Available models in cache:" "Cyan"
    
    if (Test-Path $HfCacheDir) {
        Get-ChildItem -Path $HfCacheDir -Recurse -Filter "*.safetensors" -ErrorAction SilentlyContinue | Select-Object -First 10 | ForEach-Object { $_.FullName }
        
        Write-Host ""
        Write-ColorOutput "💾 Cache size:" "Cyan"
        try {
            $size = (Get-ChildItem -Path $HfCacheDir -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1GB
            Write-Host ("{0:N2} GB" -f $size)
        } catch {
            Write-Host "Could not calculate cache size"
        }
    } else {
        Write-ColorOutput "❌ Cache directory not found: $HfCacheDir" "Red"
    }
}

switch ($Action) {
    "download" {
        if (-not $ModelName) {
            Write-ColorOutput "Usage: .\download-models.ps1 download <model_name> [hf_token]" "Red"
            Write-Host "Examples:"
            Write-Host "  .\download-models.ps1 download meta-llama/Meta-Llama-3.1-8B-Instruct"
            Write-Host "  .\download-models.ps1 download mistralai/Mistral-7B-Instruct-v0.3"
            exit 1
        }
        Download-Model $ModelName $HfToken
    }
    
    "list" {
        List-Models
    }
    
    "clean" {
        Write-ColorOutput "🧹 Cleaning model cache..." "Yellow"
        $confirm = Read-Host "This will delete all cached models. Continue? (y/N)"
        
        if ($confirm -eq "y" -or $confirm -eq "Y") {
            if (Test-Path $HfCacheDir) {
                Remove-Item -Path $HfCacheDir -Recurse -Force -ErrorAction SilentlyContinue
                New-Item -Path $HfCacheDir -ItemType Directory -Force
                Write-ColorOutput "✅ Cache cleaned" "Green"
            }
        } else {
            Write-ColorOutput "❌ Cancelled" "Red"
        }
    }
    
    "popular" {
        Write-ColorOutput "📚 Popular models for vLLM v0.10.1.1:" "Cyan"
        Write-Host "  meta-llama/Meta-Llama-3.1-8B-Instruct    (8B, good balance)"
        Write-Host "  meta-llama/Meta-Llama-3.1-70B-Instruct   (70B, very capable)"
        Write-Host "  mistralai/Mistral-7B-Instruct-v0.3       (7B, fast)"
        Write-Host "  codellama/CodeLlama-7b-Instruct-hf        (7B, coding)"
        Write-Host ""
        Write-Host "Use: .\download-models.ps1 download <model_name>"
    }
}
