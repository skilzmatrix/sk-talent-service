"""Service wrapper for Pinecone vector operations."""

from __future__ import annotations

import logging
from typing import Any

from app import pinecone_operations

logger = logging.getLogger(__name__)


def vectorize_candidate(candidate_id: str, record: dict[str, Any]) -> list[str]:
    """Returns list of section names that were embedded."""
    try:
        return pinecone_operations.upsert_candidate_vectors(candidate_id, record)
    except Exception:
        logger.exception("Failed to vectorize candidate %s", candidate_id)
        return []


def search_candidates(query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Search for top matching candidates given a job description."""
    try:
        # Fetch enough section-level hits to fairly rank candidates (composite score needs breadth).
        section_fetch = max(40, top_k * 10)
        results = pinecone_operations.query_candidates(query_text, top_k=section_fetch)
        return results[:top_k]
    except Exception:
        logger.exception("Failed to search candidates")
        raise


def delete_candidate_vectors(candidate_id: str) -> None:
    try:
        pinecone_operations.delete_candidate_vectors(candidate_id)
    except Exception:
        logger.exception("Failed to delete vectors for candidate %s", candidate_id)
