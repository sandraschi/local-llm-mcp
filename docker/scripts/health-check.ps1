# Docker Health Check Script

param(
    [int]$IntervalSeconds = 30,
    [switch]$Continuous
)

$ProjectName = "local-llm-mcp"
$ComposeFile = "docker-compose.vllm-v10.yml"

function Write-ColorOutput($Message, $Color = "White") {
    Write-Host $Message -ForegroundColor $Color
}

function Check-ServiceHealth() {
    Write-ColorOutput "=== Docker Health Check - $(Get-Date) ===" "Yellow"
    
    # Container status
    Write-ColorOutput "🐳 Container Status:" "Cyan"
    $containers = docker-compose -f $ComposeFile -p $ProjectName ps --format json | ConvertFrom-Json
    
    foreach ($container in $containers) {
        $status = $container.State
        $health = if ($container.Health) { $container.Health } else { "N/A" }
        
        if ($status -eq "running") {
            Write-ColorOutput "✅ $($container.Service): $status ($health)" "Green"
        } else {
            Write-ColorOutput "❌ $($container.Service): $status ($health)" "Red"
        }
    }
    
    # API health checks
    Write-ColorOutput "🌐 API Health Checks:" "Cyan"
    
    # vLLM health
    try {
        $vllmResponse = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 5
        if ($vllmResponse.StatusCode -eq 200) {
            Write-ColorOutput "✅ vLLM API: Healthy" "Green"
        }
    } catch {
        Write-ColorOutput "❌ vLLM API: $($_.Exception.Message)" "Red"
    }
    
    # MCP health
    try {
        $mcpResponse = Invoke-WebRequest -Uri "http://localhost:3001/health" -UseBasicParsing -TimeoutSec 5
        if ($mcpResponse.StatusCode -eq 200) {
            Write-ColorOutput "✅ MCP API: Healthy" "Green"
        }
    } catch {
        Write-ColorOutput "❌ MCP API: $($_.Exception.Message)" "Red"
    }
    
    # Resource usage
    Write-ColorOutput "📊 Resource Usage:" "Cyan"
    try {
        $gpuInfo = nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
        Write-Host "GPU: $gpuInfo"
    } catch {
        Write-ColorOutput "❌ GPU info unavailable" "Red"
    }
    
    Write-Host ""
}

if ($Continuous) {
    Write-ColorOutput "🔄 Starting continuous health monitoring (Ctrl+C to stop)" "Green"
    while ($true) {
        Clear-Host
        Check-ServiceHealth
        Start-Sleep -Seconds $IntervalSeconds
    }
} else {
    Check-ServiceHealth
}
