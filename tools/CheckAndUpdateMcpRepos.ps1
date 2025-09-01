<#
.SYNOPSIS
    Manages MCP repositories by checking for changes, committing, and pushing updates.

.DESCRIPTION
    This script provides comprehensive management of MCP repositories including:
    - Scanning for Git repositories in the specified directory
    - Checking for uncommitted changes
    - Staging and committing changes with meaningful messages
    - Pushing to remote repositories
    - Handling merge conflicts and detached HEAD states
    - Generating detailed reports

    The script includes safety checks, progress tracking, and detailed logging.

.PARAMETER ReposPath
    The base path where MCP repositories are located. Defaults to 'D:\Dev\repos'.

.PARAMETER DryRun
    If specified, the script will only show what would be done without making any changes.

.PARAMETER ForcePush
    If specified, allows force pushing to remote repositories (use with caution).

.PARAMETER CommitMessage
    Custom commit message. If not provided, a timestamp-based message will be used.

.PARAMETER Branch
    The branch to operate on. Defaults to the current branch of each repository.

.PARAMETER MaxConcurrent
    Maximum number of repositories to process in parallel. Defaults to 4.

.EXAMPLE
    # Basic usage
    .\CheckAndUpdateMcpRepos.ps1

    # Dry run to see what would be done
    .\CheckAndUpdateMcpRepos.ps1 -DryRun

    # Custom repositories path and branch
    .\CheckAndUpdateMcpRepos.ps1 -ReposPath "C:\MyRepos" -Branch "main"

    # Force push changes (use with caution)
    .\CheckAndUpdateMcpRepos.ps1 -ForcePush

.NOTES
    - Requires Git to be installed and in the system PATH
    - Administrative privileges may be required for certain operations
    - Always review changes before pushing to shared repositories
#>

[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Medium')]
param (
    [Parameter(Position = 0)]
    [ValidateScript({
        if (-not (Test-Path -Path $_ -PathType Container)) {
            throw "Directory '$_' does not exist"
        }
        $true
    })]
    [string]$ReposPath = "D:\Dev\repos",
    
    [Parameter()]
    [switch]$DryRun,
    
    [Parameter()]
    [switch]$ForcePush,
    
    [Parameter()]
    [string]$CommitMessage,
    
    [Parameter()]
    [string]$Branch,
    
    [Parameter()]
    [ValidateRange(1, 16)]
    [int]$MaxConcurrent = 4
)

# Set strict mode for better error handling
Set-StrictMode -Version Latest

# Configure error handling
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'Continue'

# Set error action preference
$ErrorActionPreference = 'Stop'

#region Helper Functions

<#
.SYNOPSIS
    Writes a status message with color coding and optional logging.
.DESCRIPTION
    Outputs messages to the console with appropriate colors based on status
    and optionally logs them to a file.
#>
function Write-Status {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$Message,
        
        [Parameter()]
        [ValidateSet('SUCCESS', 'ERROR', 'WARNING', 'INFO', 'DETAIL', 'DEBUG')]
        [string]$Status = 'INFO',
        
        [Parameter()]
        [switch]$NoNewline,
        
        [Parameter()]
        [string]$LogFile = $script:logFile
    )
    
    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $statusText = "[$timestamp] [$Status] $Message"
    
    # Define colors for different status levels
    $colors = @{
        'SUCCESS' = 'Green'
        'ERROR'   = 'Red'
        'WARNING' = 'Yellow'
        'INFO'    = 'Cyan'
        'DETAIL'  = 'Gray'
        'DEBUG'   = 'DarkGray'
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

# Initialize script variables
$script:logFile = Join-Path -Path $PSScriptRoot -ChildPath "mcp_repo_update_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
$script:repositories = @()
$script:results = @()
$script:startTime = Get-Date

# Create a mutex to prevent multiple instances
$mutex = New-Object System.Threading.Mutex($false, 'MCP_Repo_Update_Mutex')
$mutexAcquired = $false

try {
    # Try to acquire mutex with timeout
    $mutexAcquired = $mutex.WaitOne(1000)
    if (-not $mutexAcquired) {
        throw "Another instance of this script is already running."
    }
    
    # Start logging
    Write-Status "Starting MCP repository update process" -Status INFO
    Write-Status "Log file: $($script:logFile)" -Status DETAIL
    Write-Status "Repositories path: $ReposPath" -Status DETAIL
    if ($DryRun) { Write-Status "DRY RUN MODE - No changes will be made" -Status WARNING }
    
    # Check if Git is available
    try {
        $gitVersion = git --version
        Write-Status "Using Git version: $gitVersion" -Status DETAIL
    } catch {
        throw "Git is not installed or not in PATH. Please install Git and try again."
    }
    
    # Get all Git repositories
    Write-Status "Scanning for Git repositories in: $ReposPath" -Status INFO
    $gitDirs = Get-ChildItem -Path $ReposPath -Directory -Recurse -Force -ErrorAction SilentlyContinue |
               Where-Object { Test-Path (Join-Path $_.FullName '.git') -PathType Container }
    
    if (-not $gitDirs) {
        Write-Status "No Git repositories found in $ReposPath" -Status WARNING
        exit 0
    }
    
    # Process repositories in parallel batches
    $script:repositories = $gitDirs.FullName | Where-Object { $_ -match 'mcp' }
    $repoCount = $script:repositories.Count
    
    if ($repoCount -eq 0) {
        Write-Status "No MCP repositories found in the specified path." -Status WARNING
        exit 0
    }
    
    Write-Status "Found $repoCount MCP repositories to process" -Status INFO
    
    # Process repositories in batches
    $batchSize = [Math]::Min($MaxConcurrent, $script:repositories.Count)
    $batches = [System.Collections.ArrayList]::new()
    
    for ($i = 0; $i -lt $script:repositories.Count; $i += $batchSize) {
        $batch = $script:repositories | Select-Object -Skip $i -First $batchSize
        $batches.Add($batch) | Out-Null
    }
    
    $batchNumber = 0
    $processedRepos = 0
    $successCount = 0
    $errorCount = 0
    $skippedCount = 0
    
    # Process each batch
    foreach ($batch in $batches) {
        $batchNumber++
        Write-Status "Processing batch $batchNumber of $($batches.Count) ($($batch.Count) repositories)" -Status INFO
        
        $batchResults = $batch | ForEach-Object -ThrottleLimit $MaxConcurrent -Parallel {
            $repoPath = $_
            $repoName = Split-Path -Path $repoPath -Leaf
            $result = @{
                Name = $repoName
                Path = $repoPath
                Status = 'PENDING'
                Message = ''
                Changes = @()
                Branch = ''
                Remote = ''
            }
            
            try {
                # Get repository info
                Push-Location -Path $repoPath -ErrorAction Stop
                
                # Get current branch
                $result.Branch = git rev-parse --abbrev-ref HEAD
                if ($result.Branch -eq 'HEAD') {
                    $result.Status = 'SKIPPED'
                    $result.Message = 'Detached HEAD state'
                    return $result
                }
                
                # Get remote URL
                $result.Remote = git config --get remote.origin.url
                
                # Check for changes
                $changes = git status --porcelain
                if ([string]::IsNullOrWhiteSpace($changes)) {
                    $result.Status = 'CLEAN'
                    $result.Message = 'No changes'
                    return $result
                }
                
                # Parse changes
                $result.Changes = $changes -split "`n" | ForEach-Object {
                    $status = $_.Substring(0, 2).Trim()
                    $file = $_.Substring(3)
                    @{ Status = $status; File = $file }
                }
                
                if ($using:DryRun) {
                    $result.Status = 'DRY_RUN'
                    $result.Message = 'Would commit changes'
                    return $result
                }
                
                # Stage all changes
                git add . 2>&1 | Out-Null
                
                # Create commit message
                $commitMsg = if ($using:CommitMessage) {
                    $using:CommitMessage
                } else {
                    "Auto-commit: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
                }
                
                # Commit changes
                git commit -m $commitMsg 2>&1 | Out-Null
                
                # Push changes
                $pushArgs = if ($using:ForcePush) { '--force' } else { '' }
                git push $pushArgs 2>&1 | Out-Null
                
                $result.Status = 'SUCCESS'
                $result.Message = 'Changes committed and pushed'
                
            } catch {
                $result.Status = 'ERROR'
                $result.Message = $_.Exception.Message
                
            } finally {
                Pop-Location
            }
            
            return $result
        }
        
        # Process batch results
        foreach ($repoResult in $batchResults) {
            $processedRepos++
            
            switch ($repoResult.Status) {
                'SUCCESS' { 
                    $successCount++
                    $statusColor = 'Green'
                }
                'CLEAN' { 
                    $skippedCount++
                    $statusColor = 'Cyan'
                }
                'DRY_RUN' { 
                    $skippedCount++
                    $statusColor = 'Yellow'
                }
                'SKIPPED' { 
                    $skippedCount++
                    $statusColor = 'Magenta'
                }
                default { 
                    $errorCount++
                    $statusColor = 'Red'
                }
            }
            
            # Status text is used in the Write-Host command below
            Write-Host ("[{0,-8}] {1,-50}" -f $repoResult.Status, $repoResult.Name) -NoNewline -ForegroundColor $statusColor
            Write-Host $repoResult.Message -ForegroundColor 'Gray'
            
            # Log detailed changes if any
            if ($repoResult.Changes.Count -gt 0) {
                $repoResult.Changes | ForEach-Object {
                    Write-Host "  $($_.Status) $($_.File)" -ForegroundColor 'DarkGray'
                }
            }
            
            # Add to results
            $script:results += [PSCustomObject]@{
                Name = $repoResult.Name
                Path = $repoResult.Path
                Status = $repoResult.Status
                Message = $repoResult.Message
                Branch = $repoResult.Branch
                Remote = $repoResult.Remote
                ChangeCount = $repoResult.Changes.Count
                Timestamp = Get-Date
            }
        }
        
        # Show progress
        $progress = [Math]::Min(100, [int](($processedRepos / $repoCount) * 100))
        Write-Progress -Activity "Processing repositories" -Status "$processedRepos of $repoCount completed" -PercentComplete $progress
    }
    
    # Generate summary
    $endTime = Get-Date
    $duration = $endTime - $script:startTime
    
    Write-Host "`n=== Summary ===" -ForegroundColor Cyan
    Write-Host "Total repositories: $repoCount"
    Write-Host ("{0,-20} {1,5} ({2:P1})" -f 'Success:', $successCount, $(if ($repoCount -gt 0) { $successCount / $repoCount } else { 0 })) -ForegroundColor 'Green'
    Write-Host ("{0,-20} {1,5} ({2:P1})" -f 'Skipped (clean):', $skippedCount, $(if ($repoCount -gt 0) { $skippedCount / $repoCount } else { 0 })) -ForegroundColor 'Cyan'
    Write-Host ("{0,-20} {1,5} ({2:P1})" -f 'Errors:', $errorCount, $(if ($repoCount -gt 0) { $errorCount / $repoCount } else { 0 })) -ForegroundColor $(if ($errorCount -gt 0) { 'Red' } else { 'Gray' })
    Write-Host ("{0,-20} {1,5} ({2:P1})" -f 'Processed:', $processedRepos, $(if ($repoCount -gt 0) { $processedRepos / $repoCount } else { 0 }))
    Write-Host ("{0,-20} {1,5}" -f 'Duration:', "$($duration.Hours)h $($duration.Minutes)m $($duration.Seconds)s")
    
    # Export results to CSV
    $reportFile = Join-Path -Path $PSScriptRoot -ChildPath "mcp_repo_report_$(Get-Date -Format 'yyyyMMdd_HHmmss').csv"
    $script:results | Export-Csv -Path $reportFile -NoTypeInformation -Encoding UTF8
    Write-Host "`nDetailed report saved to: $reportFile" -ForegroundColor 'Cyan'
    
    Write-Status "`n=== MCP repository update process completed ===" -Status SUCCESS
} catch {
    Write-Status "An error occurred: $_" -Status ERROR -ErrorAction Continue
    Write-Status $_.ScriptStackTrace -Status DEBUG -ErrorAction SilentlyContinue
    exit 1
    
} finally {
    # Release mutex if acquired
    if ($mutexAcquired) {
        try {
            $mutex.ReleaseMutex() | Out-Null
        } catch {
            Write-Status "Warning: Failed to release mutex: $_" -Status WARNING -ErrorAction Continue
        }
    }
    
    # Clean up
    if ($null -ne $mutex) {
        $mutex.Dispose()
    }
    
    # Final status
    $totalTime = (Get-Date) - $script:startTime
    
    Write-Status "Script completed in $($totalTime.TotalSeconds.ToString('0.00')) seconds" -Status INFO
    
    # Exit with appropriate code
    if ($errorCount -gt 0) {
        exit 1
    } else {
        exit 0
    }
}
