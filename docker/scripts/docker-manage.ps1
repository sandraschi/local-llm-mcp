# Docker Management Script for vLLM v0.10.1.1 + MCP (PowerShell)
# Usage: .\docker-manage.ps1 [start|stop|restart|status|logs|clean]

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("start", "stop", "restart", "status", "logs", "clean", "pull", "rebuild")]
    [string]$Action,
    
    [string]$Service = "",
    [switch]$Follow
)

$ProjectName = "local-llm-mcp"
$ComposeFile = "docker-compose.vllm-v10.yml"

function Write-ColorOutput($Message, $Color = "White") {
    Write-Host $Message -ForegroundColor $Color
}

switch ($Action) {
    "start" {
        Write-ColorOutput "🚀 Starting Local LLM MCP with vLLM v0.10.1.1..." "Green"
        docker-compose -f $ComposeFile -p $ProjectName up -d
        Write-ColorOutput "✅ Services started. Check status with: .\docker-manage.ps1 status" "Green"
    }
    
    "stop" {
        Write-ColorOutput "🛑 Stopping Local LLM MCP services..." "Yellow"
        docker-compose -f $ComposeFile -p $ProjectName down
        Write-ColorOutput "✅ Services stopped." "Green"
    }
    
    "restart" {
        Write-ColorOutput "🔄 Restarting Local LLM MCP services..." "Cyan"
        docker-compose -f $ComposeFile -p $ProjectName restart
        Write-ColorOutput "✅ Services restarted." "Green"
    }
    
    "status" {
        Write-ColorOutput "📊 Service Status:" "Cyan"
        docker-compose -f $ComposeFile -p $ProjectName ps
        Write-Host ""
        Write-ColorOutput "🔍 Health Checks:" "Cyan"
        
        try {
            $vllmHealth = docker-compose -f $ComposeFile -p $ProjectName exec -T vllm-v10 curl -s http://localhost:8000/health 2>$null
            Write-ColorOutput "✅ vLLM health check: $vllmHealth" "Green"
        } catch {
            Write-ColorOutput "❌ vLLM health check failed" "Red"
        }
        
        try {
            $mcpHealth = docker-compose -f $ComposeFile -p $ProjectName exec -T local-llm-mcp curl -s http://localhost:3001/health 2>$null
            Write-ColorOutput "✅ MCP health check: $mcpHealth" "Green"
        } catch {
            Write-ColorOutput "❌ MCP health check failed" "Red"
        }
    }
    
    "logs" {
        if ($Follow) {
            Write-ColorOutput "📝 Following logs (Ctrl+C to stop):" "Cyan"
            docker-compose -f $ComposeFile -p $ProjectName logs -f $Service
        } elseif ($Service) {
            Write-ColorOutput "📝 Logs for service: $Service" "Cyan"
            docker-compose -f $ComposeFile -p $ProjectName logs --tail=50 $Service
        } else {
            Write-ColorOutput "📝 Recent logs (last 50 lines):" "Cyan"
            docker-compose -f $ComposeFile -p $ProjectName logs --tail=50
        }
    }
    
    "clean" {
        Write-ColorOutput "🧹 Cleaning up Docker resources..." "Yellow"
        docker-compose -f $ComposeFile -p $ProjectName down -v
        docker system prune -f
        Write-ColorOutput "✅ Cleanup complete." "Green"
    }
    
    "pull" {
        Write-ColorOutput "📥 Pulling latest base images..." "Cyan"
        docker-compose -f $ComposeFile -p $ProjectName pull
        Write-ColorOutput "✅ Images updated." "Green"
    }
    
    "rebuild" {
        Write-ColorOutput "🔨 Rebuilding images..." "Yellow"
        docker-compose -f $ComposeFile -p $ProjectName build --no-cache
        Write-ColorOutput "✅ Images rebuilt." "Green"
    }
}
