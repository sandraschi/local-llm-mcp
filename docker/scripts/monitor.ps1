# Performance Monitoring Script for vLLM + MCP (PowerShell)
# Monitors GPU usage, memory, API performance

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("gpu", "docker", "api", "errors", "all", "monitor")]
    [string]$Action
)

$ProjectName = "local-llm-mcp"
$ComposeFile = "docker-compose.vllm-v10.yml"

function Write-ColorOutput($Message, $Color = "White") {
    Write-Host $Message -ForegroundColor $Color
}

function Check-GpuUsage() {
    Write-ColorOutput "🎮 GPU Status:" "Cyan"
    try {
        $gpuInfo = nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
        $gpuInfo
    } catch {
        Write-ColorOutput "❌ nvidia-smi not available" "Red"
    }
    Write-Host ""
}

function Check-DockerStats() {
    Write-ColorOutput "🐳 Docker Container Stats:" "Cyan"
    docker stats --no-stream --format "table {{.Name}}`t{{.CPUPerc}}`t{{.MemUsage}}`t{{.NetIO}}`t{{.BlockIO}}"
    Write-Host ""
}

function Check-ApiPerformance() {
    Write-ColorOutput "🌐 API Performance Check:" "Cyan"
    
    # vLLM API health
    try {
        $vllmResponse = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 5
        if ($vllmResponse.StatusCode -eq 200) {
            Write-ColorOutput "✅ vLLM API: Healthy" "Green"
            
            # Test inference speed
            $startTime = Get-Date
            $testPayload = @{
                model = "llama-3.1-8b"
                prompt = "Hello, how are you?"
                max_tokens = 10
                temperature = 0.1
            } | ConvertTo-Json
            
            Invoke-WebRequest -Uri "http://localhost:8000/v1/completions" -Method POST -Body $testPayload -ContentType "application/json" -UseBasicParsing -TimeoutSec 10 | Out-Null
            $endTime = Get-Date
            $duration = ($endTime - $startTime).TotalSeconds
            Write-ColorOutput "⚡ vLLM Response Time: ${duration}s" "Green"
        }
    } catch {
        Write-ColorOutput "❌ vLLM API: Unhealthy - $($_.Exception.Message)" "Red"
    }
    
    # MCP API health
    try {
        $mcpResponse = Invoke-WebRequest -Uri "http://localhost:3001/health" -UseBasicParsing -TimeoutSec 5
        if ($mcpResponse.StatusCode -eq 200) {
            Write-ColorOutput "✅ MCP API: Healthy" "Green"
        }
    } catch {
        Write-ColorOutput "❌ MCP API: Unhealthy - $($_.Exception.Message)" "Red"
    }
    Write-Host ""
}

function Check-LogsErrors() {
    Write-ColorOutput "🔍 Recent Errors in Logs:" "Cyan"
    $logs = docker-compose -f $ComposeFile -p $ProjectName logs --tail=100
    $errors = $logs | Select-String -Pattern "error|exception|failed" -CaseSensitive:$false | Select-Object -Last 10
    
    if ($errors) {
        $errors | ForEach-Object { Write-Host $_.Line }
    } else {
        Write-ColorOutput "No recent errors found" "Green"
    }
    Write-Host ""
}

function Monitor-Realtime() {
    Write-ColorOutput "📊 Real-time Monitoring (Ctrl+C to stop):" "Cyan"
    while ($true) {
        Clear-Host
        Write-ColorOutput "=== Local LLM MCP Performance Monitor ===" "Yellow"
        Write-Host "Time: $(Get-Date)"
        Write-Host ""
        Check-GpuUsage
        Check-DockerStats
        Check-ApiPerformance
        Start-Sleep -Seconds 5
    }
}

switch ($Action) {
    "gpu" {
        Check-GpuUsage
    }
    "docker" {
        Check-DockerStats
    }
    "api" {
        Check-ApiPerformance
    }
    "errors" {
        Check-LogsErrors
    }
    "all" {
        Check-GpuUsage
        Check-DockerStats
        Check-ApiPerformance
        Check-LogsErrors
    }
    "monitor" {
        Monitor-Realtime
    }
}
