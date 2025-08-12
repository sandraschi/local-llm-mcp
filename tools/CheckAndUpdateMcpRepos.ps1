<#
.SYNOPSIS
    Checks all MCP repositories for changes and commits/pushes them.
.DESCRIPTION
    This script scans through all MCP repositories in the specified directory,
    checks for any uncommitted changes, commits them with a timestamp,
    and pushes to the remote repository. It's designed to be safe and reliable,
    with comprehensive error handling and logging.

.PARAMETER ReposPath
    The base path where MCP repositories are located. Defaults to 'D:\Dev\repos'.

.PARAMETER DryRun
    If specified, the script will only show what would be done without making any changes.

.EXAMPLE
    .\CheckAndUpdateMcpRepos.ps1
    .\CheckAndUpdateMcpRepos.ps1 -ReposPath "C:\MyRepos" -DryRun
#>

[CmdletBinding()]
param (
    [Parameter(Position = 0)]
    [ValidateScript({
        if (-not (Test-Path -Path $_ -PathType Container)) {
            throw "Directory '$_' does not exist"
        }
        $true
    })]
    [string]$ReposPath = "D:\Dev\repos",
    
    [switch]$DryRun
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

try {
    Write-Status "Starting MCP repository update process" -Status INFO
    
    # Get all directories that contain '.git' subdirectory
    Write-Status "Scanning for Git repositories in: $ReposPath" -Status DETAIL
    
    $gitRepos = Get-ChildItem -Path $ReposPath -Directory -Recurse -Depth 1 -Force | 
                Where-Object { (Test-Path -Path "$($_.FullName)\.git") }
    
    # Filter for MCP repositories (case-insensitive)
    $mcpRepos = $gitRepos | Where-Object { $_.Name -match 'mcp' }
    
    if ($mcpRepos.Count -eq 0) {
        Write-Status "No MCP repositories found in the specified path." -Status WARNING
        exit 0
    }
    
    Write-Status "Found $($mcpRepos.Count) MCP repositories to process" -Status SUCCESS
    
    foreach ($repo in $mcpRepos) {
        $repoName = $repo.Name
        $repoPath = $repo.FullName
        
        Write-Status "`n=== Processing repository: $repoName ===" -Status INFO
        
        # Save current location
        $originalLocation = Get-Location
        
        try {
            # Change to the repository directory
            Set-Location -Path $repoPath -ErrorAction Stop
            
            try {
                # Get the current branch name
                $branch = git rev-parse --abbrev-ref HEAD
                if (-not $?) { throw "Failed to get current branch" }
                
                # Get remote URL
                $remoteUrl = git config --get remote.origin.url
                Write-Status "  Repository: $repoName" -Status DETAIL
                Write-Status "  Branch: $branch" -Status DETAIL
                Write-Status "  Remote: $remoteUrl" -Status DETAIL
                
                # Check for changes
                $status = git status --porcelain
                
                if ($status) {
                    $changeCount = ($status -split "`n").Count
                    Write-Status "  Found $changeCount uncommitted change(s)" -Status WARNING
                    
                    if (-not $DryRun) {
                        # Stage all changes
                        Write-Status "  Staging changes..." -Status DETAIL -NoNewline
                        git add . 2>&1 | Out-Null
                        if (-not $?) { throw "Failed to stage changes" }
                        Write-Status " Done" -Status SUCCESS -NoNewline
                        Write-Host ""  # New line
                        
                        # Create commit
                        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
                        $commitMessage = "Auto-update: $timestamp"
                        
                        Write-Status "  Creating commit..." -Status DETAIL -NoNewline
                        git commit -m $commitMessage 2>&1 | Out-Null
                        if (-not $?) { throw "Failed to create commit" }
                        Write-Status " Done" -Status SUCCESS -NoNewline
                        Write-Host ""  # New line
                        
                        # Push changes
                        Write-Status "  Pushing to origin/$branch..." -Status DETAIL -NoNewline
                        $pushOutput = git push origin $branch 2>&1 | Out-String
                        if (-not $?) { 
                            Write-Status " Failed" -Status ERROR -NoNewline
                            Write-Host ""  # New line
                            throw "Failed to push changes: $pushOutput"
                        }
                        Write-Status " Done" -Status SUCCESS -NoNewline
                        Write-Host ""  # New line
                        
                        Write-Status "  Successfully updated and pushed changes" -Status SUCCESS
                    } else {
                        Write-Status "  [DRY RUN] Would have committed and pushed $changeCount change(s)" -Status WARNING
                    }
                } else {
                    Write-Status "  No changes to commit" -Status SUCCESS
                }
            }
            catch {
                Write-Status "  Error processing repository: $_" -Status ERROR
                # Continue with next repository
                continue
            }
        }
        finally {
            # Always return to the original directory
            Set-Location -Path $originalLocation
        }
    }
    
    Write-Status "`n=== MCP repository update process completed ===" -Status SUCCESS
}
catch {
    Write-Status "Fatal error: $_" -Status ERROR
    Write-Status $_.ScriptStackTrace -Status DETAIL
    exit 1
}
finally {
    # No need to clean up stack as we're not using it anymore
    
    Write-Status "Script execution completed at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -Status INFO
}
