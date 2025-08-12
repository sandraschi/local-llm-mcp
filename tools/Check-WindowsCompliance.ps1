<#
.SYNOPSIS
    Checks for Linux/Unix commands in files and suggests PowerShell alternatives.
#>

param (
    [string]$Path = ".",
    [string[]]$FileTypes = @('.ps1', '.psm1', '.bat', '.cmd', '.md')
)

# Command mappings
$commandMappings = @{
    '&&' = ';'  # Command chaining
    '`' = ';'   # Line continuation
    'ls' = 'Get-ChildItem'
    'cat' = 'Get-Content'
    'grep' = 'Select-String'
    'rm ' = 'Remove-Item '
    'mkdir -p' = 'New-Item -ItemType Directory -Force -Path'
    'cp ' = 'Copy-Item '
    'mv ' = 'Move-Item '
    'chmod' = 'icacls or Set-Acl'
    'pwd' = 'Get-Location'
    'echo' = 'Write-Output'
    '| xargs' = '| ForEach-Object'
}

$issuesFound = $false

Get-ChildItem -Path $Path -File -Recurse | Where-Object { 
    $FileTypes -contains $_.Extension 
} | ForEach-Object {
    $file = $_
    try {
        $content = Get-Content -Path $_.FullName -Raw -ErrorAction Stop
        $lineNumber = 0
        
        $content -split "`r`n" | ForEach-Object {
            $lineNumber++
            $line = $_.Trim()
            
            # Skip empty lines and comments
            if ([string]::IsNullOrWhiteSpace($line) -or $line.StartsWith('#')) { 
                return 
            }
            
            # Check for Linux/Unix commands
            foreach ($linuxCmd in $commandMappings.Keys) {
                if ($line -match "\b$([regex]::Escape($linuxCmd))\b") {
                    $script:issuesFound = $true
                    Write-Host "[!] $($file.FullName):$lineNumber" -ForegroundColor Red
                    Write-Host "   Found: $linuxCmd" -ForegroundColor Red
                    Write-Host "   Use: $($commandMappings[$linuxCmd])" -ForegroundColor Green
                    Write-Host "   Line: $line" -ForegroundColor Gray
                    Write-Host ""
                    break
                }
            }
        }
    }
    catch {
        Write-Warning "Error processing $($file.FullName): $_"
    }
}

if (-not $script:issuesFound) {
    Write-Host "[✓] No Linux/Unix commands found." -ForegroundColor Green
    exit 0
}
else {
    Write-Host "[!] Issues found. Please update to use PowerShell commands." -ForegroundColor Red
    exit 1
}
