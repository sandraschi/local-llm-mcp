"""Engine supervision portmanteau for Local LLM MCP server.

Consolidates process control and model management for the local LLM engines
(Ollama and the native llama.cpp server, e.g. Muse Glimmer 30B) into one tool.

PORTMANTEAU PATTERN RATIONALE:
Starting, stopping, supervising, and model management of local engines would
otherwise be 8+ separate tools. One portmanteau keeps the tool surface small
while giving the agent full control over the local inference stack.

Engine identities:
- "ollama" - Ollama daemon on port 11434 (OpenAI-compat /v1 + native /api)
- "llama"  - llama.cpp llama-server on 11439 with the truncating proxy on 11435
            (Muse Glimmer 30B; single fixed model per server launch)
"""

import asyncio
import json
import os
import socket
import subprocess
import sys
from typing import Any

import httpx
import psutil

from llm_mcp.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from fastmcp import FastMCP

    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False

# --- Per-engine configuration (env-overridable) ---
OLLAMA_PORT = int(os.getenv("LLM_MCP_OLLAMA_PORT", "11434"))
OLLAMA_API = f"http://127.0.0.1:{OLLAMA_PORT}/api"

LLAMA_PORT = int(os.getenv("LLM_MCP_LLAMA_SERVER_PORT", "11439"))
LLAMA_PROXY_PORT = int(os.getenv("LLM_MCP_LLAMA_PROXY_PORT", "11435"))
LLAMA_BASE = f"http://127.0.0.1:{LLAMA_PORT}"
LLAMA_MODEL_DIR = os.getenv("LLM_MCP_LLAMA_MODEL_DIR", r"D:\Dev\models\muse-glimmer")
LLAMA_SERVER_EXE = os.getenv("LLM_MCP_LLAMA_SERVER_EXE", r"D:\Dev\tools\llama.cpp\src\build\bin\llama-server.exe")
LLAMA_PROXY_SCRIPT = os.getenv(
    "LLM_MCP_LLAMA_PROXY_SCRIPT",
    r"C:\Users\sandr\AppData\Local\Temp\opencode\muse-proxy\proxy.py",
)
LLAMA_UV = os.getenv("LLM_MCP_UV", r"C:\Users\sandr\.local\bin\uv.exe")

_READONLY = {"readonly": True}
_MUTATING = {}

_ENGINES = ("ollama", "llama")


# --- helpers ---


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            return True
    except OSError:
        return False


def _find_processes(match_cmdline: list[str]) -> list[dict]:
    """Find processes by cmdline substring via PowerShell CIM.

    psutil's process_iter hangs on this machine (name()/cmdline() block on a
    system process). CIM enumeration is reliable and fast.
    """
    cond = " -or ".join([f"($_.CommandLine -match '{m}')" for m in match_cmdline])
    ps = (
        f"Get-CimInstance Win32_Process | Where-Object {{ $_.CommandLine -and ({cond}) }} | "
        "ForEach-Object { [PSCustomObject]@{pid=$_.ProcessId; name=$_.Name; cmdline=$_.CommandLine} } | "
        "ConvertTo-Json -Compress"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return []
        data = json.loads(out.stdout)
        if isinstance(data, dict):
            data = [data]
        allowed = ("llama-server.exe", "ollama.exe", "python.exe", "uv.exe")
        return [
            {"pid": int(d["pid"]), "name": d.get("name") or "", "cmdline": [d.get("cmdline") or ""]}
            for d in data
            if d.get("pid")
            and "Get-CimInstance Win32_Process" not in (d.get("cmdline") or "")
            and ((d.get("name") or "").lower().startswith("ollama") or (d.get("name") or "").lower() in allowed)
        ]
    except Exception as e:
        logger.warning("process scan failed: %s", e, exc_info=True)
        return []


def _gpu_vram() -> dict:
    """Total VRAM snapshot via nvidia-smi (no heavy imports)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        parts = out.stdout.strip().split(",")
        if len(parts) == 3:
            return {"total_gb": int(parts[0]) / 1024, "used_gb": int(parts[1]) / 1024, "free_gb": int(parts[2]) / 1024}
    except Exception:
        pass
    return {}


def _proc_vram(pids: list[int]) -> dict[int, float]:
    """Per-PID VRAM in GB via nvidia-smi compute-apps."""
    result = {}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in out.stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) == 2 and int(parts[0]) in pids:
                result[int(parts[0])] = int(parts[1]) / 1024
    except Exception:
        pass
    return result


def _kill_pids(pids: list[int]) -> None:
    for pid in pids:
        try:
            p = psutil.Process(pid)
            p.terminate()
            try:
                p.wait(timeout=5)
            except psutil.TimeoutExpired:
                p.kill()
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            logger.warning("Access denied killing PID %d", pid, exc_info=True)


def _spawn_detached(args: list[str], cwd: str | None = None) -> bool:
    """Spawn a detached background process (Windows CREATE_NO_WINDOW)."""
    try:
        kwargs: dict[str, Any] = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        subprocess.Popen(args, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs)
        return True
    except Exception as e:
        logger.error("Failed to spawn %s: %s", args[0], e, exc_info=True)
        return False


async def _http_get(url: str, timeout: float = 3.0) -> dict | None:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url)
            if r.status_code == 200:
                return r.json()
    except Exception:
        return None
    return None


# --- engine state ---


async def _ollama_state() -> dict:
    port_up = _port_open(OLLAMA_PORT)
    procs = _find_processes(["ollama"])
    loaded = []
    if port_up:
        ps = await _http_get(f"{OLLAMA_API}/ps")
        if ps and ps.get("models"):
            loaded = [m.get("name") for m in ps["models"]]
    pids = [p["pid"] for p in procs]
    return {
        "engine": "ollama",
        "running": port_up or bool(procs),
        "port_open": port_up,
        "processes": [{"pid": p["pid"], "name": p["name"]} for p in procs],
        "process_vram_gb": _proc_vram(pids),
        "loaded_models": loaded,
        "url": f"http://127.0.0.1:{OLLAMA_PORT}",
    }


async def _llama_state() -> dict:
    server_up = _port_open(LLAMA_PORT)
    proxy_up = _port_open(LLAMA_PROXY_PORT)
    procs = _find_processes(["llama-server"])
    proxy_procs = _find_processes(["muse-proxy"])
    models = []
    if server_up:
        m = await _http_get(f"{LLAMA_BASE}/v1/models")
        if m and m.get("data"):
            models = [entry.get("id") for entry in m["data"]]
    pids = [p["pid"] for p in procs + proxy_procs]
    return {
        "engine": "llama",
        "running": server_up,
        "server_port_open": server_up,
        "proxy_port_open": proxy_up,
        "processes": [{"pid": p["pid"], "name": p["name"]} for p in procs + proxy_procs],
        "process_vram_gb": _proc_vram(pids),
        "gpu_vram": _gpu_vram(),
        "loaded_models": models,
        "url": f"http://127.0.0.1:{LLAMA_PROXY_PORT}",
        "server_url": LLAMA_BASE,
    }


async def _engine_state(engine: str) -> dict:
    if engine == "ollama":
        return await _ollama_state()
    if engine == "llama":
        return await _llama_state()
    return {"error": f"unknown engine: {engine}", "available": list(_ENGINES)}


# --- actions ---


async def _start_ollama() -> dict:
    if _port_open(OLLAMA_PORT):
        return {"success": True, "message": "Ollama already running", "state": await _ollama_state()}
    # Try the Ollama app first, fall back to `ollama serve`
    spawned = False
    for candidate in (
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
    ):
        if os.path.exists(candidate):
            spawned = _spawn_detached([candidate])
            break
    if not spawned:
        spawned = _spawn_detached(["ollama", "serve"])
    if not spawned:
        return {"success": False, "error": "failed to start Ollama"}
    for _ in range(20):
        await asyncio.sleep(0.5)
        if _port_open(OLLAMA_PORT):
            return {"success": True, "message": "Ollama started", "state": await _ollama_state()}
    return {"success": False, "error": "Ollama did not open its port in time"}


async def _stop_ollama() -> dict:
    procs = _find_processes(["ollama"])
    pids = [p["pid"] for p in procs if p["name"].lower().startswith("ollama")]
    _kill_pids(pids)
    await asyncio.sleep(1.5)
    return {"success": not _port_open(OLLAMA_PORT), "message": "Ollama stopped", "killed_pids": pids}


async def _start_llama() -> dict:
    if _port_open(LLAMA_PROXY_PORT):
        return {"success": True, "message": "llama server already running", "state": await _llama_state()}
    gguf = os.path.join(LLAMA_MODEL_DIR, "meta", "muse-glimmer-30B-kquant-17gb.gguf")
    mmproj = os.path.join(LLAMA_MODEL_DIR, "meta", "mmproj-kquant.gguf")
    drafter = os.path.join(LLAMA_MODEL_DIR, "meta", "dflash-kquant.gguf")
    template = os.path.join(LLAMA_MODEL_DIR, "muse-template.jinja")
    if not os.path.exists(gguf):
        return {"success": False, "error": f"model file missing: {gguf}"}
    server_args = [
        LLAMA_SERVER_EXE,
        "-m",
        gguf,
        "--mmproj",
        mmproj,
        "--model-draft",
        drafter,
        "--port",
        str(LLAMA_PORT),
        "--host",
        "127.0.0.1",
        "-ngl",
        "99",
        "--ctx-size",
        "131072",
        "--parallel",
        "1",
        "--chat-template-file",
        template,
        "--reasoning",
        "on",
        "--reasoning-format",
        "deepseek",
        "--reasoning-budget",
        "1024",
    ]
    proxy_args = ["run", "--with", "fastapi", "--with", "uvicorn", "--with", "httpx", LLAMA_PROXY_SCRIPT]
    ok = _spawn_detached(server_args)
    ok = _spawn_detached([LLAMA_UV, *proxy_args]) and ok
    if not ok:
        return {"success": False, "error": "failed to spawn llama server"}
    for _ in range(120):
        await asyncio.sleep(1)
        if _port_open(LLAMA_PROXY_PORT):
            return {"success": True, "message": "llama server started", "state": await _llama_state()}
    return {"success": False, "error": "llama server did not become healthy in time"}


async def _stop_llama() -> dict:
    pids = [p["pid"] for p in _find_processes(["llama-server"])]
    pids += [p["pid"] for p in _find_processes(["muse-proxy"])]
    _kill_pids(pids)
    await asyncio.sleep(2)
    return {
        "success": not _port_open(LLAMA_PROXY_PORT),
        "message": "llama server stopped",
        "killed_pids": pids,
    }


async def _list_models(engine: str) -> dict:
    if engine in ("ollama", "all"):
        tags = await _http_get(f"{OLLAMA_API}/tags")
        if tags and tags.get("models"):
            return {"engine": "ollama", "models": [m.get("name") for m in tags["models"]]}
        if engine == "ollama":
            return {"engine": "ollama", "models": [], "note": "Ollama not reachable"}
    if engine in ("llama", "all"):
        state = await _llama_state()
        return {"engine": "llama", "models": state["loaded_models"]}
    return {"error": f"unknown engine: {engine}", "available": list(_ENGINES)}


async def _load_model(engine: str, model: str) -> dict:
    if engine == "ollama":
        try:
            async with httpx.AsyncClient(timeout=600) as client:
                r = await client.post(f"{OLLAMA_API}/generate", json={"model": model, "keep_alive": "30m"})
            if r.status_code in (200, 201):
                return {"success": True, "message": f"loaded {model} in Ollama (keep_alive 30m)"}
            return {"success": False, "error": f"ollama load failed: HTTP {r.status_code}: {r.text[:200]}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    if engine == "llama":
        if not model or model == "muse-glimmer-30b":
            return await _start_llama()
        return {
            "success": False,
            "error": "llama server serves a single fixed model; use engine=llama without model or reconfigure",
        }
    return {"success": False, "error": f"unknown engine: {engine}"}


async def _unload_model(engine: str, model: str) -> dict:
    if engine == "ollama":
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(f"{OLLAMA_API}/generate", json={"model": model, "keep_alive": 0})
            if r.status_code in (200, 201):
                return {"success": True, "message": f"unloaded {model} from Ollama"}
            return {"success": False, "error": f"ollama unload failed: HTTP {r.status_code}: {r.text[:200]}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    if engine == "llama":
        return await _stop_llama()
    return {"success": False, "error": f"unknown engine: {engine}"}


# --- portmanteau entry ---


async def llm_engine(
    operation: str,
    engine: str = "all",
    model: str | None = None,
) -> dict[str, Any]:
    """Supervise local LLM engines (Ollama + native llama.cpp server) and manage their models.

    PORTMANTEAU PATTERN: Consolidates 8 engine operations into one tool.

    SUPPORTED OPERATIONS:
    - status: Live state of engines (process, ports, health, loaded models)
    - start: Start an engine (engine=ollama|llama)
    - stop: Stop an engine (engine=ollama|llama)
    - list_models: List available/loaded models (engine=ollama|llama|all)
    - load_model: Load a model into memory (engine=ollama requires model; engine=llama starts the server)
    - unload_model: Unload a model (engine=ollama requires model; engine=llama stops the server)

    Args:
        operation: Operation to perform (see SUPPORTED OPERATIONS above)
        engine: Target engine: ollama | llama | all (status/list only)
        model: Model name (required for ollama load/unload)

    ## Return Format
    {"success": bool, "message": str, "state": {...}} - operation-specific

    ## Examples
    llm_engine(operation="status")
    llm_engine(operation="start", engine="llama")
    llm_engine(operation="list_models", engine="all")
    llm_engine(operation="load_model", engine="ollama", model="qwen3.6:27b")
    """
    try:
        if operation == "status":
            results = {}
            for eng in list(_ENGINES):
                if engine in ("all", eng):
                    results[eng] = await _engine_state(eng)
            return {"success": True, "message": "engine status", "engines": results}

        if operation == "start":
            if engine not in _ENGINES:
                return {"success": False, "error": f"engine must be one of {list(_ENGINES)}"}
            return await {"ollama": _start_ollama, "llama": _start_llama}[engine]()

        if operation == "stop":
            if engine not in _ENGINES:
                return {"success": False, "error": f"engine must be one of {list(_ENGINES)}"}
            return await {"ollama": _stop_ollama, "llama": _stop_llama}[engine]()

        if operation == "list_models":
            return await _list_models(engine)

        if operation == "load_model":
            if engine not in _ENGINES:
                return {"success": False, "error": f"engine must be one of {list(_ENGINES)}"}
            return await _load_model(engine, model or "")

        if operation == "unload_model":
            if engine not in _ENGINES:
                return {"success": False, "error": f"engine must be one of {list(_ENGINES)}"}
            return await _unload_model(engine, model or "")

        return {
            "success": False,
            "error": f"Unknown operation: {operation}",
            "available_operations": ["status", "start", "stop", "list_models", "load_model", "unload_model"],
        }
    except Exception as e:
        logger.error(f"Error in llm_engine operation {operation}: {e}", exc_info=True)
        return {"success": False, "error": f"Operation failed: {e!s}", "operation": operation}


def register_llm_engine_tools(mcp):
    """Register the engine supervision portmanteau tool with the MCP server."""
    if not FASTMCP_AVAILABLE:
        logger.error("Cannot register engine tools - FastMCP not available")
        return mcp

    @mcp.tool(annotations=_READONLY)
    async def llm_engine_status_tool() -> dict[str, Any]:
        """Status of local LLM engines (Ollama + llama.cpp server): processes, ports, loaded models."""
        return await llm_engine(operation="status", engine="all")

    @mcp.tool(annotations=_MUTATING)
    async def llm_engine_control_tool(
        operation: str,
        engine: str = "all",
        model: str | None = None,
    ) -> dict[str, Any]:
        """Control local LLM engines and models.

        Operations:
        - start: start an engine (engine=ollama|llama)
        - stop: stop an engine (engine=ollama|llama)
        - list_models: list models (engine=ollama|llama|all)
        - load_model: load a model (engine=ollama requires model; engine=llama starts the server)
        - unload_model: unload a model (engine=ollama requires model; engine=llama stops the server)
        """
        return await llm_engine(operation=operation, engine=engine, model=model)

    logger.info("Registered engine supervision portmanteau tool")
    return mcp
