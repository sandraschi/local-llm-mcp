#!/bin/bash
# Docker Management Script for vLLM v0.10.1.1 + MCP
# Usage: ./docker-manage.sh [start|stop|restart|status|logs|clean]

set -e

PROJECT_NAME="local-llm-mcp"
COMPOSE_FILE="docker-compose.vllm-v10.yml"

case "$1" in
    start)
        echo "🚀 Starting Local LLM MCP with vLLM v0.10.1.1..."
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME up -d
        echo "✅ Services started. Check status with: $0 status"
        ;;
    
    stop)
        echo "🛑 Stopping Local LLM MCP services..."
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME down
        echo "✅ Services stopped."
        ;;
    
    restart)
        echo "🔄 Restarting Local LLM MCP services..."
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME restart
        echo "✅ Services restarted."
        ;;
    
    status)
        echo "📊 Service Status:"
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME ps
        echo ""
        echo "🔍 Health Checks:"
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME exec vllm-v10 curl -s http://localhost:8000/health || echo "❌ vLLM health check failed"
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME exec local-llm-mcp curl -s http://localhost:3001/health || echo "❌ MCP health check failed"
        ;;
    
    logs)
        echo "📝 Recent logs (use -f for follow mode):"
        if [ "$2" = "-f" ]; then
            docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME logs -f
        elif [ -n "$2" ]; then
            docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME logs "$2"
        else
            docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME logs --tail=50
        fi
        ;;
    
    clean)
        echo "🧹 Cleaning up Docker resources..."
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME down -v
        docker system prune -f
        echo "✅ Cleanup complete."
        ;;
    
    pull)
        echo "📥 Pulling latest base images..."
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME pull
        echo "✅ Images updated."
        ;;
    
    rebuild)
        echo "🔨 Rebuilding images..."
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME build --no-cache
        echo "✅ Images rebuilt."
        ;;
    
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|clean|pull|rebuild}"
        echo ""
        echo "Commands:"
        echo "  start     - Start all services"
        echo "  stop      - Stop all services"
        echo "  restart   - Restart all services"
        echo "  status    - Show service status and health"
        echo "  logs      - Show recent logs (add -f to follow, or service name)"
        echo "  clean     - Stop services and clean up volumes"
        echo "  pull      - Pull latest base images"
        echo "  rebuild   - Rebuild images from scratch"
        exit 1
        ;;
esac
