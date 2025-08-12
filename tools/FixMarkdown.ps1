<#
.SYNOPSIS
    Fixes common Markdown formatting issues in files and directories.
.DESCRIPTION
    This script scans for and fixes common Markdown formatting issues including:
    - Missing blank lines around headings
    - Missing blank lines around lists
    - Missing blank lines around fenced code blocks
    - Line length issues
    - Ordered list numbering issues
.PARAMETER Path
    The path to scan for Markdown files (default: current directory).
.PARAMETER WhatIf
    Show what would happen if the script were to run without making changes.
#>

param(
    [string]$Path = ".",
    [switch]$WhatIf
)

function Test-IsMarkdownFile {
    param([string]$Path)
    return [System.IO.Path]::GetExtension($Path) -in ".md",".markdown"
}

function Add-BlankLines {
    param(
        [string[]]$Content,
        [int]$Index
    )
    
    $before = $Index - 1
    $after = $Index + 1
    
    # Add blank line before if needed
    if ($before -ge 0 -and $Content[$before].Trim() -ne "") {
        $Content = $Content[0..$before] + @("") + $Content[$Index..($Content.Count-1)]
        $Index++  # Adjust index since we added a line
    }
    
    # Add blank line after if needed
    if ($after -lt $Content.Count -and $Content[$after].Trim() -ne "") {
        $Content = $Content[0..$after] + @("") + $Content[($after+1)..($Content.Count-1)]
    }
    
    return $Content
}

# Use an approved PowerShell verb for the function
function Update-MarkdownFormatting {
    param(
        [string]$FilePath,
        [switch]$WhatIf
    )
    
    $content = Get-Content -Path $FilePath -Raw
    $lines = [System.Collections.ArrayList]@($content -split "`r?`n")
    $changesMade = $false
    
    # Process each line
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $line = $lines[$i]
        
        # Fix MD022: Headers need blank lines around them
        if ($line -match '^#+\s+') {
            $lines = Add-BlankLines -Content $lines -Index $i
            $changesMade = $true
        }
        
        # Fix MD032: Lists need blank lines around them
        elseif ($line -match '^[\s]*[-*+]\s+' -or $line -match '^[\s]*\d+\.\s+') {
            $lines = Add-BlankLines -Content $lines -Index $i
            $changesMade = $true
        }
        
        # Fix MD031: Fenced code blocks need blank lines around them
        elseif ($line -match '^```') {
            $lines = Add-BlankLines -Content $lines -Index $i
            $changesMade = $true
        }
        
        # Fix MD013: Line length
        if ($line.Length -gt 120 -and $line -notmatch '^[\s]*[>|]') {
            # Simple line break at last space before 120 chars
            if ($line -match '^(.{0,120})\s(.*)$') {
                # Create a new array with the split line
                $newLines = $lines[0..($i-1)] + @($matches[1], $matches[2]) + $lines[($i+1)..($lines.Count-1)]
                $lines = [System.Collections.ArrayList]@($newLines)
                $changesMade = $true
                $i++ # Skip the newly added line in the next iteration
            }
        }
    }
    
    # Fix MD029: Ordered list numbering
    $inOrderedList = $false
    $listNumber = 1
        
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $line = $lines[$i]
        
        if ($line -match '^(\s*)(\d+)\.\s') {
            $indent = $matches[1]
            $currentNumber = [int]$matches[2]
            
            if ((-not $inOrderedList) -or ($currentNumber -ne $listNumber)) {
                $lines[$i] = $line -replace '^(\s*)\d+(\.\s)', "`$1$listNumber`$2"
                $changesMade = $true
            }
            
            $inOrderedList = $true
            $listNumber++
        } else {
            $inOrderedList = $false
            $listNumber = 1
        }
    }
    
    # Save changes if any were made
    if ($changesMade) {
        $newContent = $lines -join "`n"
        
        if ($WhatIf) {
            Write-Host "[WhatIf] Would update file: $FilePath" -ForegroundColor Cyan
        } else {
            try {
                # Ensure consistent line endings (CRLF for Windows)
                $newContent = $newContent -replace "`r?`n", "`r`n"
                $newContent | Set-Content -Path $FilePath -NoNewline -Force -Encoding UTF8
                Write-Host "Fixed: $FilePath" -ForegroundColor Green
            } catch {
                Write-Host "Error updating $FilePath : $_" -ForegroundColor Red
                return $false
            }
            return $true
        }
    } else {
        Write-Host "No changes needed: $FilePath" -ForegroundColor Gray
        return $false
    }
}

# Main script execution
if (-not (Test-Path $Path)) {
    Write-Error "Path not found: $Path"
    exit 1
}

# Get all Markdown files
$files = if (Test-Path $Path -PathType Container) {
    Get-ChildItem -Path $Path -Recurse -File | Where-Object { Test-IsMarkdownFile $_.FullName }
} else {
    if (Test-IsMarkdownFile $Path) {
        Get-Item $Path
    } else {
        Write-Error "Not a Markdown file: $Path"
        exit 1
    }
}

# Process each file
$filesProcessed = 0
$filesUpdated = 0

foreach ($file in $files) {
    $filesProcessed++
    $result = Update-MarkdownFormatting -FilePath $file.FullName -WhatIf:$WhatIf
    if ($result) { $filesUpdated++ }
}

Write-Host "`nMarkdown formatting complete!" -ForegroundColor Green
Write-Host "Processed $filesProcessed files, updated $filesUpdated files" -ForegroundColor Cyan

# Remove the duplicate completion message
