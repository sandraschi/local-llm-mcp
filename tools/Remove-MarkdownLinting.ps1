<#
.SYNOPSIS
    Removes Markdown linting and pre-commit hooks from all MCP repositories.
.DESCRIPTION
    This script will:
    1. Remove pre-commit hooks related to Markdown linting
    2. Clean up any Markdown linting configuration files
#>

# List of known MCP repositories
$mcpRepos = @(
    "D:\Dev\repos\avatarmcp",
    "D:\Dev\repos\basic-memory",
    "D:\Dev\repos\beyondcomparemcp",
    "D:\Dev\repos\calibremcp",
    "D:\Dev\repos\dockermcp",
    "D:\Dev\repos\eniac-mcp",
    "D:\Dev\repos\eniacmcp",
    "D:\Dev\repos\fastsearch-mcp",
    "D:\Dev\repos\filesystem-mcp",
    "D:\Dev\repos\filesystem-mcp-server",
    "D:\Dev\repos\grandorguemcp",
    "D:\Dev\repos\gtfs-mcp",
    "D:\Dev\repos\handbrakemcp",
    "D:\Dev\repos\hasleo-backup-mcp",
    "D:\Dev\repos\immichmcp",
    "D:\Dev\repos\local-llm-mcp",
    "D:\Dev\repos\mcp-collection",
    "D:\Dev\repos\mcp-filesystem",
    "D:\Dev\repos\miniclaude",
    "D:\Dev\repos\notionmcp",
    "D:\Dev\repos\obsidianmcp",
    "D:\Dev\repos\oscmcp",
    "D:\Dev\repos\plexmcp",
    "D:\Dev\repos\pywinauto-mcp",
    "D:\Dev\repos\qbtmcp",
    "D:\Dev\repos\reaper-mcp",
    "D:\Dev\repos\rustdeskmcp",
    "D:\Dev\repos\sandra-docker-mcp",
    "D:\Dev\repos\sqlitemcp",
    "D:\Dev\repos\system-admin-mcp",
    "D:\Dev\repos\tailscalemcp",
    "D:\Dev\repos\tvtropes-mcp",
    "D:\Dev\repos\vboxmcp",
    "D:\Dev\repos\vroidstudio-mcp",
    "D:\Dev\repos\winrarmcp"
) | Where-Object { Test-Path $_ }

Write-Host "Found $($mcpRepos.Count) MCP repositories to clean up" -ForegroundColor Cyan

foreach ($repo in $mcpRepos) {
    $repoName = Split-Path $repo -Leaf
    Write-Host "`nProcessing repository: $repoName" -ForegroundColor Green
    
    # 1. Remove pre-commit hook if it exists
    $preCommitHook = Join-Path $repo ".git\hooks\pre-commit"
    if (Test-Path $preCommitHook) {
        $hookContent = Get-Content $preCommitHook -Raw -ErrorAction SilentlyContinue
        if ($hookContent -and $hookContent -match "Markdown") {
            Remove-Item $preCommitHook -Force
            Write-Host "  Removed pre-commit hook" -ForegroundColor Yellow
        }
    }
    
    # 2. Remove any Markdown linting configuration files
    $lintingFiles = @(
        ".markdownlint.json",
        ".markdownlint.yaml",
        ".markdownlint.yml",
        ".mdlrc",
        ".mdlrc.json",
        ".mdlrc.yml",
        ".mdlrc.yaml"
    )
    
    foreach ($file in $lintingFiles) {
        $filePath = Join-Path $repo $file
        if (Test-Path $filePath) {
            Remove-Item $filePath -Force -ErrorAction SilentlyContinue
            Write-Host "  Removed $file" -ForegroundColor Yellow
        }
    }
}

Write-Host "`nCleanup complete! All Markdown linting configurations and hooks have been removed." -ForegroundColor Green
