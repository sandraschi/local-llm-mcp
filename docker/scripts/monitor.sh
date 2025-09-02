#!/bin/bash
# Performance Monitoring Script for vLLM + MCP
# Monitors GPU usage, memory, API performance

set -e

PROJECT_NAME="local-llm-mcp"
COMPOSE_FILE="../docker-compose.vllm-v10.yml"

function check_gpu_usage() {
    echo "🎮 GPU Status:"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
    else
        echo "❌ nvidia-smi not available"
    fi
    echo ""
}

function check_docker_stats() {
    echo "🐳 Docker Container Stats:"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"
    echo ""
}

function check_api_performance() {
    echo "🌐 API Performance Check:"
    
    # vLLM API health
    local vllm_health=$(curl -s -w "%{http_code}" -o /dev/null http://localhost:8000/health)
    if [ "$vllm_health" = "200" ]; then
        echo "✅ vLLM API: Healthy"
        
        # Test inference speed
        local start_time=$(date +%s.%N)
        curl -s -X POST http://localhost:8000/v1/completions \
            -H "Content-Type: application/json" \
            -d '{
                "model": "llama-3.1-8b",
                "prompt": "Hello, how are you?",
                "max_tokens": 10,
                "temperature": 0.1
            }' > /dev/null
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc)
        echo "⚡ vLLM Response Time: ${duration}s"
    else
        echo "❌ vLLM API: Unhealthy (HTTP $vllm_health)"
    fi
    
    # MCP API health
    local mcp_health=$(curl -s -w "%{http_code}" -o /dev/null http://localhost:3001/health)
    if [ "$mcp_health" = "200" ]; then
        echo "✅ MCP API: Healthy"
    else
        echo "❌ MCP API: Unhealthy (HTTP $mcp_health)"
    fi
    echo ""
}

function check_logs_errors() {
    echo "🔍 Recent Errors in Logs:"
    docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME logs --tail=100 | grep -i "error\|exception\|failed" | tail -10 || echo "No recent errors found"
    echo ""
}

function monitor_realtime() {
    echo "📊 Real-time Monitoring (Ctrl+C to stop):"
    while true; do
        clear
        echo "=== Local LLM MCP Performance Monitor ==="
        echo "Time: $(date)"
        echo ""
        check_gpu_usage
        check_docker_stats
        check_api_performance
        sleep 5
    done
}

case "$1" in
    "gpu")
        check_gpu_usage
        ;;
    "docker")
        check_docker_stats
        ;;
    "api")
        check_api_performance
        ;;
    "errors")
        check_logs_errors
        ;;
    "all")
        check_gpu_usage
        check_docker_stats
        check_api_performance
        check_logs_errors
        ;;
    "monitor")
        monitor_realtime
        ;;
    *)
        echo "Usage: $0 {gpu|docker|api|errors|all|monitor}"
        echo ""
        echo "Commands:"
        echo "  gpu      - Show GPU usage and temperature"
        echo "  docker   - Show Docker container stats"
        echo "  api      - Test API health and performance"
        echo "  errors   - Show recent errors from logs"
        echo "  all      - Show all status information"
        echo "  monitor  - Real-time monitoring (refreshes every 5s)"
        exit 1
        ;;
esac
