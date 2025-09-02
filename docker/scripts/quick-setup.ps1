# Quick Setup Script for vLLM v0.10.1.1 Docker

param(
    [switch]$SkipDownload,
    [string]$Model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
)

function Write-ColorOutput($Message, $Color = "White") {
    Write-Host $Message -ForegroundColor $Color
}

Write-ColorOutput "🚀 Setting up Local LLM MCP with vLLM v0.10.1.1" "Green"

# Check prerequisites
Write-ColorOutput "🔍 Checking prerequisites..." "Cyan"

try {
    docker --version | Out-Null
    Write-ColorOutput "✅ Docker found" "Green"
} catch {
    Write-ColorOutput "❌ Docker not found or not running" "Red"
    exit 1
}

try {
    nvidia-smi | Out-Null
    Write-ColorOutput "✅ NVIDIA GPU detected" "Green"
} catch {
    Write-ColorOutput "❌ NVIDIA GPU or drivers not found" "Red"
    exit 1
}

# Create .env file if it doesn't exist
if (-not (Test-Path "docker\.env")) {
    Write-ColorOutput "📝 Creating environment configuration..." "Cyan"
    Copy-Item -Path "docker\.env.example" -Destination "docker\.env" -Force
    Write-ColorOutput "✅ Environment file created. Edit docker\.env if needed." "Green"
}

# Create required directories
Write-ColorOutput "📁 Creating directories..." "Cyan"
New-Item -Path "models" -ItemType Directory -Force | Out-Null
New-Item -Path "logs" -ItemType Directory -Force | Out-Null
New-Item -Path "docker\backups" -ItemType Directory -Force | Out-Null
Write-ColorOutput "✅ Directories created" "Green"

# Download model if requested
if (-not $SkipDownload) {
    Write-ColorOutput "📥 Downloading model: $Model" "Cyan"
    Write-ColorOutput "This may take several minutes..." "Yellow"
    
    try {
        & ".\docker\scripts\download-models.ps1" download $Model
        Write-ColorOutput "✅ Model downloaded" "Green"
    } catch {
        Write-ColorOutput "⚠️ Model download failed, continuing setup..." "Yellow"
    }
}

# Build and start services
Write-ColorOutput "🔨 Building Docker images..." "Cyan"
docker-compose -f docker-compose.vllm-v10.yml build

Write-ColorOutput "🚀 Starting services..." "Cyan"
docker-compose -f docker-compose.vllm-v10.yml up -d

# Wait for services to be ready
Write-ColorOutput "⏳ Waiting for services to start..." "Cyan"
Start-Sleep -Seconds 30

# Check status
Write-ColorOutput "🔍 Checking service status..." "Cyan"
& ".\docker\scripts\docker-manage.ps1" status

Write-ColorOutput "🎉 Setup complete!" "Green"
Write-Host ""
Write-ColorOutput "Next steps:" "Cyan"
Write-Host "1. Test vLLM API: curl http://localhost:8000/v1/models"
Write-Host "2. Test MCP server: curl http://localhost:3001/health"
Write-Host "3. Use management script: .\docker\scripts\docker-manage.ps1 status"
Write-Host "4. View logs: .\docker\scripts\docker-manage.ps1 logs"
