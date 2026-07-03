"""Unified provider health service with liveness checks, caching, and circuit breaking.

Provides a single source of truth for Ollama and LM Studio reachability,
used by every code path that touches local providers. Implements:

- Fast liveness probes (3s connect timeout, 10s read timeout)
- Result caching with 30-second TTL
- Circuit breaker: 3 consecutive failures -> mark unavailable for 60 seconds
- LM Studio Docker port conflict detection (validates response shape)
- Per-provider structured health reports
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp

OLLAMA_BASE = "http://localhost:11434"
LMSTUDIO_BASE = "http://localhost:1234"
HEALTH_CACHE_TTL = 30.0
CIRCUIT_BREAKER_THRESHOLD = 3
CIRCUIT_BREAKER_COOLDOWN = 60.0
HEALTH_CONNECT_TIMEOUT = 3.0
HEALTH_READ_TIMEOUT = 10.0

_provider_health_cache: dict[str, _CachedHealth] = {}
_circuit_state: dict[str, _CircuitBreaker] = {}


@dataclass
class ProviderHealth:
    """Structured health report for one provider."""

    provider: str
    reachable: bool
    base_url: str
    latency_ms: float | None = None
    model_count: int | None = None
    error: str | None = None
    error_type: str | None = None
    suggestion: str | None = None
    checked_at: float = field(default_factory=time.monotonic)


@dataclass
class _CachedHealth:
    health: ProviderHealth
    timestamp: float


@dataclass
class _CircuitBreaker:
    failures: int = 0
    opened_at: float = 0.0

    @property
    def is_open(self) -> bool:
        if self.failures < CIRCUIT_BREAKER_THRESHOLD:
            return False
        return (time.monotonic() - self.opened_at) < CIRCUIT_BREAKER_COOLDOWN

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = 0.0

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= CIRCUIT_BREAKER_THRESHOLD:
            self.opened_at = time.monotonic()


def _get_circuit(provider: str) -> _CircuitBreaker:
    if provider not in _circuit_state:
        _circuit_state[provider] = _CircuitBreaker()
    return _circuit_state[provider]


def _cached_health(provider: str) -> ProviderHealth | None:
    entry = _provider_health_cache.get(provider)
    if entry is None:
        return None
    if (time.monotonic() - entry.timestamp) > HEALTH_CACHE_TTL:
        return None
    return entry.health


def _cache_health(health: ProviderHealth) -> None:
    _provider_health_cache[health.provider] = _CachedHealth(
        health=health, timestamp=time.monotonic()
    )


def _force_refresh(provider: str) -> None:
    _provider_health_cache.pop(provider, None)


async def check_ollama_health(force: bool = False) -> ProviderHealth:
    """Probe Ollama liveness via GET /api/tags with fast timeout."""
    if not force:
        cached = _cached_health("ollama")
        if cached is not None:
            return cached

    circuit = _get_circuit("ollama")
    if circuit.is_open:
        return ProviderHealth(
            provider="ollama",
            reachable=False,
            base_url=OLLAMA_BASE,
            error="Circuit breaker open — too many consecutive failures",
            error_type="circuit_open",
            suggestion="Wait for the cooldown period or restart the Ollama daemon",
        )

    start = time.monotonic()
    try:
        timeout = aiohttp.ClientTimeout(
            total=HEALTH_READ_TIMEOUT, connect=HEALTH_CONNECT_TIMEOUT
        )
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{OLLAMA_BASE}/api/tags") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    model_count = len(data.get("models", []))
                    latency = (time.monotonic() - start) * 1000
                    circuit.record_success()
                    health = ProviderHealth(
                        provider="ollama",
                        reachable=True,
                        base_url=OLLAMA_BASE,
                        latency_ms=round(latency, 1),
                        model_count=model_count,
                    )
                    _cache_health(health)
                    return health

                circuit.record_failure()
                body = await resp.text()
                health = ProviderHealth(
                    provider="ollama",
                    reachable=False,
                    base_url=OLLAMA_BASE,
                    error=f"HTTP {resp.status}: {body[:200]}",
                    error_type="http_error",
                    suggestion="Verify the Ollama daemon is running (`ollama serve`)",
                )
                _cache_health(health)
                return health

    except aiohttp.ClientConnectorError as e:
        circuit.record_failure()
        health = ProviderHealth(
            provider="ollama",
            reachable=False,
            base_url=OLLAMA_BASE,
            error=str(e),
            error_type="connection_refused" if "refused" in str(e).lower() else "connection_error",
            suggestion="Ollama is not running. Start it with `ollama serve` or install from https://ollama.com",
        )
        _cache_health(health)
        return health

    except TimeoutError:
        circuit.record_failure()
        health = ProviderHealth(
            provider="ollama",
            reachable=False,
            base_url=OLLAMA_BASE,
            error="Connection timed out — Ollama daemon may be hung",
            error_type="timeout",
            suggestion="The Ollama daemon process exists but is not responding. Restart it.",
        )
        _cache_health(health)
        return health

    except Exception as e:
        circuit.record_failure()
        health = ProviderHealth(
            provider="ollama",
            reachable=False,
            base_url=OLLAMA_BASE,
            error=str(e),
            error_type="unknown",
            suggestion="Check if Ollama is installed and running",
        )
        _cache_health(health)
        return health


async def check_lmstudio_health(force: bool = False) -> ProviderHealth:
    """Probe LM Studio liveness with Docker port-conflict detection."""
    if not force:
        cached = _cached_health("lmstudio")
        if cached is not None:
            return cached

    circuit = _get_circuit("lmstudio")
    if circuit.is_open:
        return ProviderHealth(
            provider="lmstudio",
            reachable=False,
            base_url=LMSTUDIO_BASE,
            error="Circuit breaker open — too many consecutive failures",
            error_type="circuit_open",
            suggestion="Wait for the cooldown period or check if Docker is occupying port 1234",
        )

    start = time.monotonic()
    try:
        timeout = aiohttp.ClientTimeout(
            total=HEALTH_READ_TIMEOUT, connect=HEALTH_CONNECT_TIMEOUT
        )
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{LMSTUDIO_BASE}/v1/models") as resp:
                if resp.status != 200:
                    circuit.record_failure()
                    body = await resp.text()
                    health = ProviderHealth(
                        provider="lmstudio",
                        reachable=False,
                        base_url=LMSTUDIO_BASE,
                        error=f"HTTP {resp.status}: {body[:200]}",
                        error_type="http_error",
                        suggestion="Verify LM Studio is running and has a model loaded",
                    )
                    _cache_health(health)
                    return health

                content_type = resp.headers.get("Content-Type", "")
                raw_body = await resp.text()

                # Docker port conflict detection: Docker API returns different JSON shape
                if "text/html" in content_type:
                    circuit.record_failure()
                    health = ProviderHealth(
                        provider="lmstudio",
                        reachable=False,
                        base_url=LMSTUDIO_BASE,
                        error="Port 1234 responded with HTML — likely Docker Desktop, not LM Studio",
                        error_type="docker_conflict",
                        suggestion=(
                            "Docker Desktop is occupying port 1234. "
                            "Stop Docker or change LM Studio's port in Settings."
                        ),
                    )
                    _cache_health(health)
                    return health

                try:
                    data = __import__("json").loads(raw_body)
                except Exception:
                    circuit.record_failure()
                    health = ProviderHealth(
                        provider="lmstudio",
                        reachable=False,
                        base_url=LMSTUDIO_BASE,
                        error=(
                            "Port 1234 responded with non-JSON content "
                            "— likely Docker or another service"
                        ),
                        error_type="docker_conflict",
                        suggestion=(
                            "Docker Desktop or another service is occupying port 1234. "
                            "Stop Docker or change LM Studio's port in Settings."
                        ),
                    )
                    _cache_health(health)
                    return health

                # Validate LM Studio response shape
                if not isinstance(data, dict) or "data" not in data:
                    circuit.record_failure()
                    health = ProviderHealth(
                        provider="lmstudio",
                        reachable=False,
                        base_url=LMSTUDIO_BASE,
                        error=(
                            f"Response at port 1234 is JSON but does not match "
                            f"LM Studio API shape: {raw_body[:200]}"
                        ),
                        error_type="wrong_service",
                        suggestion=(
                            "Something other than LM Studio is listening on port 1234 "
                            "(Docker, another service). Stop conflicting services."
                        ),
                    )
                    _cache_health(health)
                    return health

                model_count = len(data.get("data", []))
                latency = (time.monotonic() - start) * 1000
                circuit.record_success()
                health = ProviderHealth(
                    provider="lmstudio",
                    reachable=True,
                    base_url=LMSTUDIO_BASE,
                    latency_ms=round(latency, 1),
                    model_count=model_count,
                )
                _cache_health(health)
                return health

    except aiohttp.ClientConnectorError as e:
        circuit.record_failure()
        health = ProviderHealth(
            provider="lmstudio",
            reachable=False,
            base_url=LMSTUDIO_BASE,
            error=str(e),
            error_type="connection_refused" if "refused" in str(e).lower() else "connection_error",
            suggestion="LM Studio is not running. Start it from the LM Studio application.",
        )
        _cache_health(health)
        return health

    except TimeoutError:
        circuit.record_failure()
        health = ProviderHealth(
            provider="lmstudio",
            reachable=False,
            base_url=LMSTUDIO_BASE,
            error="Connection timed out — the process at port 1234 may be hung",
            error_type="timeout",
            suggestion="A process occupies port 1234 but is not responding. Check if Docker is running.",
        )
        _cache_health(health)
        return health

    except Exception as e:
        circuit.record_failure()
        health = ProviderHealth(
            provider="lmstudio",
            reachable=False,
            base_url=LMSTUDIO_BASE,
            error=str(e),
            error_type="unknown",
            suggestion="Check if LM Studio is running",
        )
        _cache_health(health)
        return health


async def check_all_providers(force: bool = False) -> dict[str, ProviderHealth]:
    """Run health checks for Ollama and LM Studio in parallel."""
    results = await asyncio.gather(
        check_ollama_health(force=force),
        check_lmstudio_health(force=force),
    )
    return {
        "ollama": results[0],
        "lmstudio": results[1],
    }


def get_cached_provider_health(provider: str) -> ProviderHealth | None:
    """Get cached health without making a request. Returns None if no cache."""
    return _cached_health(provider)


def invalidate_provider_health(provider: str | None = None) -> None:
    """Clear health cache for a specific provider or all providers."""
    if provider:
        _provider_health_cache.pop(provider, None)
        _circuit_state.pop(provider, None)
    else:
        _provider_health_cache.clear()
        _circuit_state.clear()


def provider_health_to_dict(health: ProviderHealth) -> dict[str, Any]:
    """Serialize a ProviderHealth to a JSON-safe dict."""
    return {
        "provider": health.provider,
        "reachable": health.reachable,
        "base_url": health.base_url,
        "latency_ms": health.latency_ms,
        "model_count": health.model_count,
        "error": health.error,
        "error_type": health.error_type,
        "suggestion": health.suggestion,
    }
