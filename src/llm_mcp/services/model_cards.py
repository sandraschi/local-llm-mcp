"""Model-card catalog served from the mcd info depot.

Reads Markdown cards from ``mcp-central-docs/models/*.md`` (the fleet info
depot) and exposes them as list/detail dicts. Same pattern as the
``/api/v1/skills`` endpoints in ``server.py`` — files on disk are the source
of truth, no database, no hardcoded lists.

Resolution order for the depot directory:
1. ``MCD_MODELS_DIR`` env var (explicit override, used in tests).
2. Sibling checkout: ``<repos>/mcp-central-docs/models`` relative to this file.
"""

import os
import re
from pathlib import Path

from llm_mcp.utils.logging import get_logger

logger = get_logger(__name__)

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_FIELD_RE = re.compile(r"^(title|status|last_updated)\s*:\s*(.+?)\s*$", re.MULTILINE)
_HEADING_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_STATUS_LINE_RE = re.compile(r"^\*\*Status:\*\*\s*(.+?)\s*$", re.MULTILINE)
_VERDICT_RE = re.compile(r"\*\*Verdict\s*:?\s*(.+?)\*\*", re.IGNORECASE)


def get_models_dir() -> Path | None:
    """Resolve the mcd models directory, or None if not found."""
    override = os.environ.get("MCD_MODELS_DIR")
    if override:
        candidate = Path(override)
        if candidate.is_dir():
            return candidate
        logger.warning("MCD_MODELS_DIR=%s is not a directory", override)
        return None
    # .../local-llm-mcp/src/llm_mcp/services/model_cards.py -> parents[4] = repos root
    repos_root = Path(__file__).resolve().parents[4]
    candidate = repos_root / "mcp-central-docs" / "models"
    if candidate.is_dir():
        return candidate
    logger.warning("mcd models dir not found at %s", candidate)
    return None


def _parse_card(path: Path) -> dict:
    """Parse one card file into a summary dict (no body)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    meta: dict[str, str] = (
        dict(_FIELD_RE.findall(_FRONTMATTER_RE.search(text).group(1))) if _FRONTMATTER_RE.search(text) else {}
    )
    heading = _HEADING_RE.search(text)
    status_line = _STATUS_LINE_RE.search(text)
    verdict = _VERDICT_RE.search(text)
    return {
        "name": path.stem,
        "title": meta.get("title") or (heading.group(1) if heading else path.stem),
        "status": meta.get("status", "unknown"),
        "last_updated": meta.get("last_updated", ""),
        "status_line": status_line.group(1) if status_line else "",
        "verdict": verdict.group(1).strip() if verdict else "",
        "words": len(text.split()),
    }


def list_model_cards() -> dict:
    """List all model cards in the depot (summaries, no bodies)."""
    models_dir = get_models_dir()
    if models_dir is None:
        return {"cards": [], "total": 0, "error": "mcd models dir not found"}
    cards = [_parse_card(p) for p in sorted(models_dir.glob("*.md"))]
    return {"cards": cards, "total": len(cards)}


def get_model_card(name: str) -> dict:
    """Return the full Markdown body of one card by file stem."""
    models_dir = get_models_dir()
    if models_dir is None:
        return {"error": "mcd models dir not found", "content": ""}
    safe = re.sub(r"[^A-Za-z0-9_\-]", "", name)
    card_file = models_dir / f"{safe}.md"
    if not card_file.is_file():
        return {"error": f"model card '{name}' not found", "content": ""}
    text = card_file.read_text(encoding="utf-8", errors="replace")
    summary = _parse_card(card_file)
    return {"name": safe, **summary, "content": text}
