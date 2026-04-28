"""Tool for searching candidates (Pinecone + Supabase enrichment)."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool

from app.services import persistence_service, pinecone_service

logger = logging.getLogger(__name__)


def _summarize_for_llm(row: dict[str, Any], relevance: float | None) -> dict[str, Any]:
    """Trim payload so tool results fit comfortably in context."""
    summary = row.get("summary")
    if isinstance(summary, str) and len(summary) > 1200:
        summary = summary[:1199] + "…"
    skills = row.get("skills") or []
    if isinstance(skills, list) and len(skills) > 50:
        skills = skills[:50]
    return {
        "id": str(row.get("id", "")),
        "full_name": row.get("full_name"),
        "job_role": row.get("job_role"),
        "summary": summary,
        "skills": skills,
        "location": row.get("location"),
        "relevance_score": relevance,
    }


@tool
def search_candidates(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Search the vector database for candidates matching a specific query or skill set.
    Use this tool when the user asks to find candidates, e.g., 'Find me a Python developer'.

    Args:
        query: The semantic search query, e.g. "Senior Python developer with AWS".
        top_k: Number of candidates to return (default 5, max 25).
    """
    q = (query or "").strip()
    if not q:
        return [{"error": "Search query cannot be empty."}]

    k = max(1, min(int(top_k), 25))

    try:
        rows, _weights = pinecone_service.search_candidates(query_text=q, top_k=k)
    except Exception as e:
        logger.exception("search_candidates: Pinecone search failed")
        return [{"error": f"Vector search failed: {e!s}"}]

    enriched: list[dict[str, Any]] = []
    for r in rows:
        # pinecone_operations finalized rows use candidate_id, not id
        cid = r.get("candidate_id")
        if not cid:
            continue
        cid_str = str(cid).strip()
        rel = r.get("composite_score")
        if rel is None:
            rel = r.get("best_score")

        try:
            details = persistence_service.get_candidate_by_id(cid_str)
        except Exception as e:
            logger.warning("get_candidate_by_id failed for %s: %s", cid_str, e)
            details = None

        if details:
            enriched.append(_summarize_for_llm(details, rel))
        else:
            enriched.append(
                {
                    "id": cid_str,
                    "full_name": r.get("full_name"),
                    "job_role": r.get("job_role"),
                    "summary": None,
                    "skills": [],
                    "location": None,
                    "relevance_score": rel,
                    "note": "Vector hit found but no matching candidate row in Supabase (re-index or sync DB).",
                }
            )

    if not enriched:
        return [
            {
                "message": "No candidates matched this query. Try different keywords, or confirm Pinecone is populated and Supabase has candidates.",
            }
        ]
    return enriched
