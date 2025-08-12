<#
.SYNOPSIS
    Fixes Markdown formatting issues in files and repositories.
.DESCRIPTION
    This script scans for and fixes common Markdown formatting issues across repositories.
    It uses markdownlint-cli2 with custom rules to ensure consistent formatting.
.PARAMETER Path
    The path to scan for Markdown files (default: current directory).
.PARAMETER Fix
    Automatically fix fixable issues.
.PARAMETER Verbose
    Show detailed output.
#>

param(
    [string]$Path = ".",
    [switch]$Fix,
    [switch]$Verbose
)

# Setup error handling
$ErrorActionPreference = 'Stop'

# Check if markdownlint-cli2 is installed
function Install-MarkdownLint {
    if (-not (Get-Command markdownlint-cli2 -ErrorAction SilentlyContinue)) {
        Write-Host "Installing markdownlint-cli2..." -ForegroundColor Cyan
        npm install -g markdownlint-cli2
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install markdownlint-cli2"
        }
    }
}

# Format output with timestamp and color
function Write-Status {
    param($Message, $Status = "INFO")
    
    $timestamp = Get-Date -Format "HH:mm:ss"
    $statusMap = @{
        "INFO" = "Cyan"
        "SUCCESS" = "Green"
        "WARNING" = "Yellow"
        "ERROR" = "Red"
        "FIXED" = "Green"
    }
    
    $color = $statusMap[$Status]
    if (-not $color) { $color = "White" }
    
    Write-Host "[$timestamp] " -NoNewline
    Write-Host "$Status" -ForegroundColor $color -NoNewline
    Write-Host ": $Message"
}

try {
    # Install markdownlint if needed
    Install-MarkdownLint
    
    # Get absolute path
    $Path = Resolve-Path $Path
    
    # Check if path exists
    if (-not (Test-Path $Path)) {
        throw "Path not found: $Path"
    }
    
    # Find .markdownlint.json in parent directories if not found in current path
    $configPath = $Path
    while ($configPath -ne (Split-Path $configPath -Parent)) {
        if (Test-Path (Join-Path $configPath ".markdownlint.json")) {
            $configPath = Join-Path $configPath ".markdownlint.json"
            break
        }
        $configPath = Split-Path $configPath -Parent
    }
    
    if (-not (Test-Path $configPath)) {
        Write-Status "No .markdownlint.json found, using default rules" -Status WARNING
    }
    
    # Build command arguments
    $markdownArgs = @(
        $Path,
        "--config", $configPath,
        "--no-stdout"
    )
    
    if ($Fix) {
        $markdownArgs += "--fix"
        Write-Status "Fixing Markdown files in: $Path" -Status INFO
    } else {
        Write-Status "Checking Markdown files in: $Path" -Status INFO
    }
    
    if ($Verbose) {
        $markdownArgs += "--debug"
    }
    
    # Run markdownlint
    $output = & markdownlint-cli2 $markdownArgs 2>&1 | Out-String
    
    # Process output
    if ($LASTEXITCODE -eq 0) {
        Write-Status "No Markdown issues found" -Status SUCCESS
    } else {
        # Format error output
        $output -split "`n" | ForEach-Object {
            if ($_ -match '^(.*?):(\d+):(\d+)\s+(\S+)\s+(.*)') {
                $filePath = $matches[1] -replace [regex]::Escape($Path), ""
                $line = $matches[2]
                $column = $matches[3]
                $rule = $matches[4]
                $message = $matches[5]
                
                $displayPath = $filePath.TrimStart('\')
                Write-Host "  ${displayPath}:${line}:${column}" -ForegroundColor Cyan
                Write-Host ("    {0}: {1}" -f $rule, $message) -ForegroundColor White
                
                # Show context if available
                if (Test-Path $matches[1]) {
                    $lines = Get-Content $matches[1]
                    $start = [Math]::Max(1, [int]$line - 2)
                    $end = [Math]::Min($lines.Count, [int]$line + 1)
                    
                    for ($i = $start; $i -le $end; $i++) {
                        $prefix = if ($i -eq [int]$line) { "  >" } else { "   " }
                        Write-Host "$prefix $i | $($lines[$i-1])" -ForegroundColor $(if ($i -eq [int]$line) { "Red" } else { "Gray" })
                    }
                    Write-Host ""
                }
            } elseif ($_.Trim()) {
                Write-Host "  $_" -ForegroundColor Gray
            }
        }
        
        if ($Fix) {
            Write-Status "Some issues were automatically fixed. Please review the changes." -Status FIXED
            Write-Status "Run without -Fix to check for remaining issues." -Status INFO
        } else {
            Write-Status "Markdown issues found. Run with -Fix to automatically fix some issues." -Status WARNING
        }
    }
} catch {
    Write-Status "Error: $_" -Status ERROR
    if ($Verbose) {
        Write-Host $_.ScriptStackTrace -ForegroundColor DarkGray
    }
    exit 1
}
