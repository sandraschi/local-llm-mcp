"""REST endpoints for optional RAG-as-a-service."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from ....services.rag_service import _HAS_DEPS, get_rag

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/rag/status")
async def rag_status() -> dict:
    """Check if RAG service is available and get stats."""
    if not _HAS_DEPS:
        return {"available": False, "error": "lancedb/pyarrow/sentence-transformers not installed"}
    try:
        svc = get_rag()
        return await svc.stats()
    except Exception as e:
        return {"available": False, "error": str(e)}


@router.post("/rag/ingest")
async def rag_ingest(body: dict[str, Any]) -> dict:
    """Ingest a text document into the vector store.

    Body:
        text (str): Document text to index.
        source (str, optional): URL, file path, or identifier.
        metadata (dict, optional): Arbitrary key/value pairs.

    Returns:
        {"success": true, "source": "...", "total_docs": N}
    """
    if not _HAS_DEPS:
        raise HTTPException(501, "RAG not available - install lancedb pyarrow sentence-transformers")
    text = body.get("text", "").strip()
    if not text:
        raise HTTPException(400, "text is required")
    svc = get_rag()
    return await svc.ingest(text, body.get("source", ""), body.get("metadata"))


@router.get("/rag/search")
async def rag_search(query: str = "", limit: int = 5) -> list[dict]:
    """Search the vector store by semantic similarity.

    Args:
        query: Natural-language search string.
        limit: Max results (default 5, max 50).

    Returns:
        List of {text, source, metadata, score} sorted by relevance.
    """
    if not _HAS_DEPS:
        raise HTTPException(501, "RAG not available - install lancedb pyarrow sentence-transformers")
    if not query.strip():
        raise HTTPException(400, "query is required")
    limit = min(max(limit, 1), 50)
    svc = get_rag()
    return await svc.search(query, limit)


@router.delete("/rag/clear")
async def rag_clear() -> dict:
    """Drop all documents from the vector store."""
    if not _HAS_DEPS:
        raise HTTPException(501, "RAG not available")
    svc = get_rag()
    return await svc.clear()
