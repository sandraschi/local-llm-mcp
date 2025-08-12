<#
.SYNOPSIS
    Lists all Git repositories in the specified directory and its subdirectories.
.DESCRIPTION
    This script searches for all .git directories to identify Git repositories
    and lists them with their full paths.
.PARAMETER Path
    The directory to search for Git repositories (default: current directory)
#>

param(
    [string]$Path = "."
)

$repos = @()

# Get all directories containing a .git folder
$gitDirs = Get-ChildItem -Path $Path -Directory -Recurse -Force -ErrorAction SilentlyContinue -Filter ".git" | 
           Where-Object { $_.FullName -notlike '*\node_modules\*' -and $_.FullName -notlike '*\.git\*' }

# Extract the parent directory of each .git folder
$repos = $gitDirs | ForEach-Object { $_.Parent.FullName } | Sort-Object -Unique

# Display the repositories
Write-Host "`nFound $($repos.Count) Git repositories in '$Path':" -ForegroundColor Cyan
$repos | ForEach-Object { Write-Host "- $_" }

# Return the repositories as an array
return $repos
