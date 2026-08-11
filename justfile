set windows-shell := ["powershell.exe", "-NoProfile", "-Command"]

bootstrap:
    uv sync --extra dev --group dev
    uv run pre-commit install
    Set-Location web_sota; npm ci; if ($LASTEXITCODE -ne 0) { npm install }
    Write-Host "Pre-commit hooks installed." -ForegroundColor Green

default:
    @powershell.exe -NoProfile -ExecutionPolicy Bypass -File ../mcp-central-docs/scripts/just-dashboard.ps1 -Path .

lint:
    Set-Location '{{justfile_directory()}}'
    uv run ruff check .
    Set-Location '{{justfile_directory()}}\web_sota'
    npx @biomejs/biome ci .

fix:
    Set-Location '{{justfile_directory()}}'
    uv run ruff check . --fix --unsafe-fixes
    uv run ruff format .
    Set-Location '{{justfile_directory()}}\web_sota'
    npx @biomejs/biome check --write .

fmt: fix

test:
    Set-Location '{{justfile_directory()}}'
    uv run pytest tests/ -q

serve:
    Set-Location '{{justfile_directory()}}'
    uv run python -m llm_mcp

certify: lint test

mcpb-pack:
    Set-Location '{{justfile_directory()}}'
    Write-Host "Building MCPB bundle..." -ForegroundColor Yellow
    uv run mcpb pack . dist/local-llm-mcp-v1.0.0.mcpb

check-sec:
    Set-Location '{{justfile_directory()}}'
    uv run bandit -r src/

audit-deps:
    Set-Location '{{justfile_directory()}}'
    uv run safety check

# Run CUA-NSIS smoke test (install -> launch -> nav walk -> uninstall)
cua-nsis-test:
    powershell.exe -NoProfile -File "{{justfile_directory()}}\scripts\just\cua-nsis-test.ps1"

# Run CUA webapp test (pre-Tauri: start.ps1 stack + nav walk in browser)
cua-webapp-test:
    powershell.exe -NoProfile -File "{{justfile_directory()}}\scripts\just\cua-webapp-test.ps1"
