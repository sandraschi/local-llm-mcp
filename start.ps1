Param([switch]$Headless)

# --- SOTA Headless Standard ---
if ($Headless -and ($Host.UI.RawUI.WindowTitle -notmatch 'Hidden')) {
    Start-Process pwsh -ArgumentList '-NoProfile', '-File', $PSCommandPath, '-Headless' -WindowStyle Hidden
    exit
}
$WindowStyle = if ($Headless) { 'Hidden' } else { 'Normal' }
# ------------------------------

$env:FASTMCP_LOG_LEVEL = 'WARNING'
# llm-mcp Start - Standards-Compliant SOTA
Write-Host 'Starting llm-mcp...' -ForegroundColor Cyan

Set-Location $PSScriptRoot
Write-Host 'Starting Standardized Fullstack Hybrid...' -ForegroundColor Green
# Launch backend Hidden by default to prevent console spam
Start-Process pwsh -ArgumentList '-NoProfile', '-Command', 'uv run -m llm_mcp' -WindowStyle Hidden
Set-Location web_sota
npm run dev