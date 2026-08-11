# Troubleshooting

## Server doesn't appear in Claude Desktop
**Cause**: Config JSON is malformed or path is wrong
**Fix**: Validate at jsonlint.com, check for trailing commas. Verify the `args` path points to the repo.

## "command not found: uv"
**Cause**: uv not installed or not in PATH
**Fix**: `winget install astral-sh.uv` then restart terminal

## Port conflict on 8000 / 10833
**Cause**: Another process is using the port
**Fix**: Change `PORT` in `.env` or run `just kill-all` to clear fleet ports

## vLLM fails to load
**Cause**: CUDA not available or incompatible torch version
**Fix**: Check `python -c "import torch; print(torch.cuda.is_available())"`
Install CUDA torch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`

## Ollama connection refused
**Cause**: Ollama not running
**Fix**: Start Ollama: `ollama serve`. Verify at `http://localhost:11434`

## GPU memory exhausted
**Cause**: Model too large for available VRAM
**Fix**: Reduce `gpu_memory_utilization` in config or set `CUDA_VISIBLE_DEVICES=""` for CPU mode

## Tool returns "Failed to register: ..."
**Cause**: Missing optional dependency
**Fix**: Install the relevant dependency (e.g., `pip install unsloth` for Unsloth tools). Unregistered tools are skipped gracefully.

## LM Studio connection fails
**Cause**: LM Studio not running or CORS not enabled
**Fix**: Start LM Studio, enable local API server in Settings. Default URL: `http://localhost:1234`
