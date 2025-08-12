<#
.SYNOPSIS
    Lints Markdown files across MCP repositories for consistent formatting.
.DESCRIPTION
    This script scans through all MCP repositories and applies markdownlint rules
    to ensure consistent formatting. It can automatically fix some issues and
    report others that need manual attention.
#>

[CmdletBinding()]
param (
    [Parameter(Position = 0)]
    [string]$ReposPath = "D:\Dev\repos",
    
    [switch]$Fix,
    [switch]$VerboseOutput
)

# Set error action preference
$ErrorActionPreference = 'Stop'

# Function to write colored output
function Write-Status {
    param(
        [string]$Message,
        [string]$Status = "INFO",
        [switch]$NoNewline
    )
    
    $colors = @{
        'SUCCESS' = 'Green'
        'ERROR'   = 'Red'
        'WARNING' = 'Yellow'
        'INFO'    = 'Cyan'
        'DETAIL'  = 'Gray'
    }
    
    $color = $colors[$Status.ToUpper()]
    if (-not $color) { $color = 'White' }
    
    $prefix = "[$(Get-Date -Format 'HH:mm:ss')]"
    $messageToWrite = "$prefix $Message"
    
    if ($NoNewline) {
        Write-Host $messageToWrite -ForegroundColor $color -NoNewline
    } else {
        Write-Host $messageToWrite -ForegroundColor $color
    }
}

# Check if markdownlint is installed
function Test-MarkdownLint {
    try {
        $null = Get-Command markdownlint -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

# Install markdownlint if not present
function Install-MarkdownLint {
    Write-Status -Message "markdownlint not found. Installing..." -Status WARNING
    try {
        npm install -g markdownlint-cli
        Write-Status -Message "markdownlint installed successfully" -Status SUCCESS
        return $true
    } catch {
        Write-Status -Message "Failed to install markdownlint: $_" -Status ERROR
        return $false
    }
}

try {
    # Main execution
    Write-Status -Message "Starting Markdown linting process" -Status INFO
    Write-Status -Message "Scanning for Markdown files in: $ReposPath" -Status DETAIL

    # Check if markdownlint is installed
    if (-not (Test-MarkdownLint)) {
        if (-not (Install-MarkdownLint)) {
            Write-Status -Message "markdownlint is required but could not be installed. Please install it manually." -Status ERROR
            exit 1
        }
    }

    # Get all markdown files in MCP repositories
    $mcpDirs = Get-ChildItem -Path $ReposPath -Directory | Where-Object { $_.Name -match 'mcp' }
    $markdownFiles = $mcpDirs | Get-ChildItem -Recurse -Filter "*.md" -File

    if ($markdownFiles.Count -eq 0) {
        Write-Status -Message "No Markdown files found in MCP repositories." -Status WARNING
        exit 0
    }

    Write-Status -Message "Found $($markdownFiles.Count) Markdown files in $($mcpDirs.Count) MCP repositories" -Status INFO

    # Create configuration if it doesn't exist
    $configPath = Join-Path $ReposPath ".markdownlint.json"
    if (-not (Test-Path $configPath)) {
        $defaultConfig = @{
            "default" = $true
            "MD013" = @{ "line_length" = 120 }
            "MD033" = $false  # Allow inline HTML
            "MD041" = $false  # First line in file should be a top level heading
            "MD047" = $true   # Files should end with a single newline character
        } | ConvertTo-Json -Depth 3
        
        $defaultConfig | Out-File -FilePath $configPath -Encoding utf8
        Write-Status -Message "Created default markdownlint configuration at $configPath" -Status INFO
    }

    # Process each repository
    foreach ($repo in $mcpDirs) {
        $repoName = $repo.Name
        $repoPath = $repo.FullName
        
        Write-Status -Message "`n=== Processing repository: $repoName ===" -Status INFO

        $mdFiles = Get-ChildItem -Path $repoPath -Recurse -Filter "*.md" -File -ErrorAction SilentlyContinue
        if ($null -eq $mdFiles -or $mdFiles.Count -eq 0) {
            Write-Status -Message "  No Markdown files found" -Status DETAIL
            continue
        }

        Write-Status -Message "  Found $($mdFiles.Count) Markdown files" -Status DETAIL

        # Build command arguments
        $markdownLintArgs = @("$repoPath/**/*.md")
        
        if ($Fix) {
            $markdownLintArgs += "--fix"
            Write-Status -Message "  Attempting to fix issues automatically..." -Status DETAIL
        } else {
            Write-Status -Message "  Checking for issues (dry run)..." -Status DETAIL
        }

        # Add config file if it exists
        $configPath = Join-Path $ReposPath ".markdownlint.json"
        if (Test-Path $configPath) {
            $markdownLintArgs += "--config"
            $markdownLintArgs += $configPath
        }

        # Run markdownlint
        try {
            # Capture both stdout and stderr
            $output = & markdownlint @markdownLintArgs 2>&1 | Out-String
            $exitCode = $LASTEXITCODE
            
            # Process the output
            if ($output) {
                # Format the output for better readability
                $formattedOutput = $output.Trim() -split "`n" | ForEach-Object {
                    if ($_ -match ':\d+:\s') {
                        $parts = $_ -split ':\d+:\s', 2
                        $fileInfo = $parts[0] -replace [regex]::Escape($repoPath), "$repoName"
                        $lineInfo = $parts[1]
                        "  $fileInfo`n    $lineInfo"
                    } else {
                        "  $_"
                    }
                }
                
                if ($exitCode -eq 0) {
                    Write-Status -Message "  No Markdown issues found" -Status SUCCESS
                } else {
                    # Display the formatted output
                    $formattedOutput -join "`n" | Write-Host
                    
                    if ($Fix) {
                        Write-Status -Message "  Some issues were automatically fixed. Please review the changes." -Status WARNING
                    } else {
                        Write-Status -Message "  Markdown issues found. Run with -Fix to automatically fix some issues." -Status WARNING
                    }
                }
            } else {
                Write-Status -Message "  No Markdown issues found" -Status SUCCESS
            }
        } catch {
            Write-Status -Message "  Error running markdownlint: $_" -Status ERROR
        }
    }

    Write-Status -Message "`n=== Markdown linting completed ===" -Status SUCCESS
    if (-not $Fix) {
        Write-Status -Message "Note: Run with -Fix to automatically fix fixable issues" -Status INFO
    }
} catch {
    Write-Status -Message "Fatal error: $_" -Status ERROR
    Write-Status -Message $_.ScriptStackTrace -Status DETAIL
    exit 1
}
