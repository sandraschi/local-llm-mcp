Param([switch]$Headless)
$SkipFrontend = $Headless

# --- SOTA Headless Standard ---
if ($Headless -and ($Host.UI.RawUI.WindowTitle -notmatch 'Hidden')) {
    Start-Process pwsh -ArgumentList '-NoProfile', '-File', $PSCommandPath, '-Headless' -WindowStyle Hidden
    exit
}
$WindowStyle = if ($Headless) { 'Hidden' } else { 'Normal' }
# ------------------------------

$env:FASTMCP_LOG_LEVEL = 'WARNING'
Write-Host 'Starting llm-mcp...' -ForegroundColor Cyan

Set-Location $PSScriptRoot

# Clear port zombies before binding
$ports = @(10833)
foreach ($port in $ports) {
    Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "  Clearing zombie on port $port (PID $($_.OwningProcess))" -ForegroundColor Yellow
        Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
    }
}

# Launch backend Hidden by default to prevent console spam
Start-Process pwsh -ArgumentList '-NoProfile', '-Command', 'uv run -m llm_mcp' -WindowStyle Hidden

# Readiness poll (up to 60s)
for ($i = 0; $i -lt 60; $i++) {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:10833/health" -TimeoutSec 2 -UseBasicParsing -ErrorAction SilentlyContinue
        if ($r.StatusCode -eq 200) { Write-Host "  Backend ready" -ForegroundColor Green; break }
    } catch {}
    Start-Sleep 1
}

Set-Location web_sota
if ($SkipFrontend) { return }
npm run dev