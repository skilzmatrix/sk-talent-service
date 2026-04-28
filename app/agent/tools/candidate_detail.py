"""Tool for loading a full candidate profile from Supabase."""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from langchain_core.tools import tool

from app.services import persistence_service

logger = logging.getLogger(__name__)


def _normalize_candidate_id(raw: str) -> str:
    s = (raw or "").strip().strip('`"\'')
    lower = s.lower()
    if lower.startswith("uuid:"):
        s = s.split(":", 1)[1].strip()
    return s


def _json_safe(obj: Any) -> Any:
    """Ensure tool output is JSON-serializable for the LLM / tracing."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return str(obj)


@tool
def get_candidate_details(candidate_id: str) -> dict[str, Any]:
    """
    Fetch the complete profile and resume details for a specific candidate by their ID.
    Use when the user wants more detail about one candidate from search results.

    Args:
        candidate_id: The candidate UUID string returned by search_candidates (id field).
    """
    cid = _normalize_candidate_id(candidate_id)
    if not cid:
        return {"error": "candidate_id is required."}

    try:
        uuid.UUID(cid)
    except ValueError:
        return {
            "error": (
                f"Invalid candidate_id {candidate_id!r}. "
                "Pass the exact UUID from search_candidates (id field), not a name."
            )
        }

    try:
        details = persistence_service.get_candidate_by_id(cid)
    except Exception as e:
        logger.exception("get_candidate_details: Supabase read failed")
        return {"error": f"Database error: {e!s}"}

    if not details:
        return {"error": f"No candidate found for id {cid!r}."}

    return _json_safe(details)
