---
name: rag-expert
description: Local vector search via LanceDB + sentence-transformers, shared as a service for the whole fleet
---

# RAG Expert — local-llm-mcp

This server hosts an optional **RAG-as-a-service** endpoint that lightweight fleet MCP repos
can use instead of shipping their own vector stack. The service uses `all-MiniLM-L6-v2`
embeddings stored in LanceDB.

## When to use

| Situation | Use shared RAG? |
|-----------|----------------|
| Repo has < 500 lines of Python | Yes — skip lanceDB + sentence-transformers dep |
| Repo needs cross-corpus search | Yes — one index, one query |
| Repo has private/sensitive data | No — keep local RAG |
| Repo has custom chunking/metadata | No — keep local RAG |
| Repo already has LanceDB | No — don't change |

## API

| Method | Path | Body / Params | Returns |
|--------|------|---------------|---------|
| `POST` | `/api/v1/rag/ingest` | `{"text", "source"?, "metadata"?}` | `{"success", "total_docs"}` |
| `GET` | `/api/v1/rag/search` | `?query=&limit=` | `[{text, source, metadata, score}]` |
| `DELETE` | `/api/v1/rag/clear` | — | `{"success", "cleared": true}` |
| `GET` | `/api/v1/rag/status` | — | `{"available", "model", "total_docs"}` |

## Examples

```python
import httpx
rag = "http://localhost:10833/api/v1/rag"

# Ingest
httpx.post(f"{rag}/ingest", json={"text": "Attention is all you need...", "source": "arxiv:1706.03762"})

# Search
r = httpx.get(f"{rag}/search", params={"query": "transformer paper", "limit": 3})
for hit in r.json():
    print(f"  [{hit['score']}] {hit['source']}: {hit['text'][:80]}...")
```

## Recipes

### Ingest from a URL (fetch + index)

```python
import httpx

def ingest_url(url: str):
    resp = httpx.get(url, timeout=30)
    resp.raise_for_status()
    text = resp.text[:50000]  # cap at 50k chars
    httpx.post("http://localhost:10833/api/v1/rag/ingest", json={
        "text": text,
        "source": url,
    })
```

### Search and return matched sources

```python
def search_sources(query: str) -> list[str]:
    r = httpx.get("http://localhost:10833/api/v1/rag/search", params={"query": query, "limit": 10})
    return list(set(h["source"] for h in r.json() if h.get("source")))
```
