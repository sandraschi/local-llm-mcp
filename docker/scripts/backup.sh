#!/bin/bash
# Backup and Restore Script for vLLM Models and Configurations

set -e

BACKUP_DIR="./backups"
MODELS_DIR="./models"
CONFIG_DIR="./docker/config"
COMPOSE_FILE="docker-compose.vllm-v10.yml"

function create_backup() {
    local backup_name="$1"
    if [ -z "$backup_name" ]; then
        backup_name="backup-$(date +%Y%m%d-%H%M%S)"
    fi
    
    local backup_path="$BACKUP_DIR/$backup_name"
    mkdir -p "$backup_path"
    
    echo "💾 Creating backup: $backup_name"
    
    # Backup configurations
    if [ -d "$CONFIG_DIR" ]; then
        cp -r "$CONFIG_DIR" "$backup_path/"
        echo "✅ Configurations backed up"
    fi
    
    # Backup docker-compose
    if [ -f "$COMPOSE_FILE" ]; then
        cp "$COMPOSE_FILE" "$backup_path/"
        echo "✅ Docker compose backed up"
    fi
    
    # Backup environment file
    if [ -f ".env" ]; then
        cp ".env" "$backup_path/"
        echo "✅ Environment file backed up"
    fi
    
    # Create backup manifest
    cat > "$backup_path/backup-manifest.txt" << EOF
Backup created: $(date)
vLLM version: 0.10.1.1
Project: local-llm-mcp
Contents:
- Configuration files
- Docker compose file
- Environment settings
EOF
    
    echo "✅ Backup created at: $backup_path"
}

function list_backups() {
    echo "📋 Available backups:"
    if [ -d "$BACKUP_DIR" ]; then
        ls -la "$BACKUP_DIR"
    else
        echo "No backups found"
    fi
}

function restore_backup() {
    local backup_name="$1"
    if [ -z "$backup_name" ]; then
        echo "❌ Please specify backup name"
        list_backups
        exit 1
    fi
    
    local backup_path="$BACKUP_DIR/$backup_name"
    if [ ! -d "$backup_path" ]; then
        echo "❌ Backup not found: $backup_path"
        exit 1
    fi
    
    echo "🔄 Restoring backup: $backup_name"
    
    # Stop services first
    docker-compose -f $COMPOSE_FILE down 2>/dev/null || true
    
    # Restore configurations
    if [ -d "$backup_path/config" ]; then
        rm -rf "$CONFIG_DIR"
        cp -r "$backup_path/config" "$CONFIG_DIR"
        echo "✅ Configurations restored"
    fi
    
    # Restore docker-compose
    if [ -f "$backup_path/$COMPOSE_FILE" ]; then
        cp "$backup_path/$COMPOSE_FILE" "./"
        echo "✅ Docker compose restored"
    fi
    
    # Restore environment
    if [ -f "$backup_path/.env" ]; then
        cp "$backup_path/.env" "./"
        echo "✅ Environment restored"
    fi
    
    echo "✅ Backup restored. You may need to restart services."
}

case "$1" in
    "create")
        create_backup "$2"
        ;;
    "list")
        list_backups
        ;;
    "restore")
        restore_backup "$2"
        ;;
    *)
        echo "Usage: $0 {create|list|restore} [backup_name]"
        echo ""
        echo "Commands:"
        echo "  create [name]    - Create backup (auto-named if no name given)"
        echo "  list             - List available backups"
        echo "  restore <name>   - Restore from backup"
        exit 1
        ;;
esac
