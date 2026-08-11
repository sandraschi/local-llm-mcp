"""Optional RAG-as-a-service with LanceDB.

Lets lightweight MCP repos outsource vector search to this server
instead of shipping their own sentence-transformers + LanceDB stack.

Usage:
    POST /api/v1/rag/ingest  {"text": "...", "source": "url-or-ref", "metadata": {}}
    GET  /api/v1/rag/search  ?query=...&limit=5
    DELETE /api/v1/rag/clear
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_rag_service: RAGService | None = None

try:
    import lancedb
    import pyarrow as pa
    from sentence_transformers import SentenceTransformer

    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False
    lancedb = None
    pa = None
    SentenceTransformer = None


class RAGService:
    """Persistent LanceDB vector store with sentence-transformers embeddings."""

    def __init__(self, db_path: str | None = None, model_name: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path or str(Path(os.getcwd()) / "data" / "rag" / "lancedb")
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._table = None
        self._initialized = False

    async def initialize(self):
        if self._initialized:
            return
        if not _HAS_DEPS:
            raise RuntimeError("RAG dependencies not installed. Run: uv add lancedb pyarrow sentence-transformers")

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._db = lancedb.connect(self.db_path)
        logger.info("RAG service loading model %s ...", self.model_name)
        t0 = time.time()
        self._model = SentenceTransformer(self.model_name)
        logger.info("RAG model loaded in %.1fs", time.time() - t0)

        try:
            self._table = self._db.open_table("docs")
        except Exception:
            schema = pa.schema(
                [
                    pa.field("vector", pa.list_(pa.float32(), 384)),
                    pa.field("text", pa.string()),
                    pa.field("source", pa.string()),
                    pa.field("metadata", pa.string()),
                    pa.field("created_at", pa.float64()),
                ]
            )
            self._table = self._db.create_table("docs", schema=schema)
        self._initialized = True

    def _embed(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()  # type: ignore

    async def ingest(self, text: str, source: str = "", metadata: dict | None = None) -> dict:
        if not self._initialized:
            await self.initialize()
        vec = self._embed(text)
        self._table.add(
            [
                {
                    "vector": vec,
                    "text": text,
                    "source": source or "unknown",
                    "metadata": json.dumps(metadata or {}),
                    "created_at": time.time(),
                }
            ]
        )
        count = self._table.count_rows()
        return {"success": True, "source": source, "total_docs": count}

    async def search(self, query: str, limit: int = 5) -> list[dict]:
        if not self._initialized:
            await self.initialize()
        vec = self._embed(query)
        try:
            results = self._table.search(vec).limit(limit).to_list()
        except Exception as e:
            logger.warning("RAG search failed: %s", e)
            return []
        return [
            {
                "text": r["text"],
                "source": r.get("source", ""),
                "metadata": json.loads(r.get("metadata", "{}")),
                "score": round(float(r.get("_distance", 0)), 4),
            }
            for r in results
        ]

    async def clear(self) -> dict:
        if self._initialized:
            self._db.drop_table("docs")
            self._table = None
            self._initialized = False
        return {"success": True, "cleared": True}

    async def stats(self) -> dict:
        if not self._initialized:
            await self.initialize()
        count = self._table.count_rows() if self._table else 0
        return {
            "available": _HAS_DEPS,
            "model": self.model_name,
            "total_docs": count,
            "db_path": self.db_path,
        }


import json


def get_rag() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
