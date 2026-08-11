# RAG as a Service

local-llm-mcp can serve as a **shared vector search backend** for lightweight fleet repos that don't want to ship their own embedding + LanceDB stack.

## Prerequisites

```bash
uv sync --extra rag
```

This installs `lancedb` and `pyarrow`. The embedding model (`all-MiniLM-L6-v2`) downloads on first use (~90 MB).

## Endpoints

### Check availability

```
GET /api/v1/rag/status
```

Returns `{"available": true, "model": "all-MiniLM-L6-v2", "total_docs": N}` or a clear error if deps are missing.

### Ingest documents

```bash
curl -X POST http://localhost:10833/api/v1/rag/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "Your document text here...", "source": "https://example.com/doc", "metadata": {"author": "user"}}'
```

Returns `{"success": true, "source": "...", "total_docs": N}`.

### Search

```bash
curl "http://localhost:10833/api/v1/rag/search?query=what+is+attention&limit=5"
```

Returns ranked list of `{text, source, metadata, score}`.

### Clear index

```bash
curl -X DELETE http://localhost:10833/api/v1/rag/clear
```

## Usage from another MCP server

### Python (httpx)

```python
import httpx

BASE = "http://localhost:10833"

def ingest(text: str, source: str = ""):
    httpx.post(f"{BASE}/api/v1/rag/ingest", json={"text": text, "source": source})

def search(query: str, limit: int = 5):
    r = httpx.get(f"{BASE}/api/v1/rag/search", params={"query": query, "limit": limit})
    return r.json()
```

### Bash / curl

```bash
# Ingest a paper abstract
curl -s -X POST http://localhost:10833/api/v1/rag/ingest \
  -H "Content-Type: application/json" \
  -d "$(cat paper-abstract.txt | jq -R '{text: ., source: "paper-123"}')"

# Semantic search
curl -s "http://localhost:10833/api/v1/rag/search?query=transformer+attention&limit=3" | jq
```

## How it works

- Embedding: `all-MiniLM-L6-v2` (384-dim, ~90 MB, runs on CPU)
- Vector store: LanceDB at `data/rag/lancedb/` (persistent)
- Text column + source + metadata JSON for downstream filtering

## Optional dependency

The RAG service is **opt-in**. Without `lancedb` + `pyarrow`, all endpoints return HTTP 501 with a clear install hint. No existing functionality is affected.
