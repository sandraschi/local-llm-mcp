# Backup and Restore Script for vLLM Models and Configurations (PowerShell)

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("create", "list", "restore")]
    [string]$Action,
    
    [string]$BackupName = ""
)

$BackupDir = ".\backups"
$ModelsDir = ".\models"
$ConfigDir = ".\docker\config"
$ComposeFile = "docker-compose.vllm-v10.yml"

function Write-ColorOutput($Message, $Color = "White") {
    Write-Host $Message -ForegroundColor $Color
}

function Create-Backup($BackupName) {
    if (-not $BackupName) {
        $BackupName = "backup-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    }
    
    $BackupPath = "$BackupDir\$BackupName"
    New-Item -Path $BackupPath -ItemType Directory -Force | Out-Null
    
    Write-ColorOutput "💾 Creating backup: $BackupName" "Cyan"
    
    # Backup configurations
    if (Test-Path $ConfigDir) {
        Copy-Item -Path $ConfigDir -Destination $BackupPath -Recurse -Force
        Write-ColorOutput "✅ Configurations backed up" "Green"
    }
    
    # Backup docker-compose
    if (Test-Path $ComposeFile) {
        Copy-Item -Path $ComposeFile -Destination $BackupPath -Force
        Write-ColorOutput "✅ Docker compose backed up" "Green"
    }
    
    # Backup environment file
    if (Test-Path ".env") {
        Copy-Item -Path ".env" -Destination $BackupPath -Force
        Write-ColorOutput "✅ Environment file backed up" "Green"
    }
    
    # Create backup manifest
    $manifest = @"
Backup created: $(Get-Date)
vLLM version: 0.10.1.1
Project: local-llm-mcp
Contents:
- Configuration files
- Docker compose file
- Environment settings
"@
    $manifest | Out-File -FilePath "$BackupPath\backup-manifest.txt" -Encoding UTF8
    
    Write-ColorOutput "✅ Backup created at: $BackupPath" "Green"
}

function List-Backups() {
    Write-ColorOutput "📋 Available backups:" "Cyan"
    if (Test-Path $BackupDir) {
        Get-ChildItem -Path $BackupDir -Directory | ForEach-Object {
            $manifestPath = "$($_.FullName)\backup-manifest.txt"
            if (Test-Path $manifestPath) {
                $date = (Get-Content $manifestPath | Select-String "Backup created:").ToString().Split(":")[1].Trim()
                Write-Host "  $($_.Name) - $date"
            } else {
                Write-Host "  $($_.Name)"
            }
        }
    } else {
        Write-ColorOutput "No backups found" "Yellow"
    }
}

function Restore-Backup($BackupName) {
    if (-not $BackupName) {
        Write-ColorOutput "❌ Please specify backup name" "Red"
        List-Backups
        return
    }
    
    $BackupPath = "$BackupDir\$BackupName"
    if (-not (Test-Path $BackupPath)) {
        Write-ColorOutput "❌ Backup not found: $BackupPath" "Red"
        return
    }
    
    Write-ColorOutput "🔄 Restoring backup: $BackupName" "Cyan"
    
    # Stop services first
    try {
        docker-compose -f $ComposeFile down 2>$null
    } catch {
        # Services might not be running
    }
    
    # Restore configurations
    if (Test-Path "$BackupPath\config") {
        if (Test-Path $ConfigDir) {
            Remove-Item -Path $ConfigDir -Recurse -Force
        }
        Copy-Item -Path "$BackupPath\config" -Destination $ConfigDir -Recurse -Force
        Write-ColorOutput "✅ Configurations restored" "Green"
    }
    
    # Restore docker-compose
    if (Test-Path "$BackupPath\$ComposeFile") {
        Copy-Item -Path "$BackupPath\$ComposeFile" -Destination ".\" -Force
        Write-ColorOutput "✅ Docker compose restored" "Green"
    }
    
    # Restore environment
    if (Test-Path "$BackupPath\.env") {
        Copy-Item -Path "$BackupPath\.env" -Destination ".\" -Force
        Write-ColorOutput "✅ Environment restored" "Green"
    }
    
    Write-ColorOutput "✅ Backup restored. You may need to restart services." "Green"
}

switch ($Action) {
    "create" {
        Create-Backup $BackupName
    }
    "list" {
        List-Backups
    }
    "restore" {
        Restore-Backup $BackupName
    }
}
