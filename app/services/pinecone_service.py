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


def search_candidates(
    query_text: str,
    top_k: int = 5,
    keyword_weight: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Search for top candidates; returns (results, ranking_weights) with semantic/keyword shares."""
    try:
        section_fetch = max(40, top_k * 10)
        results, weights = pinecone_operations.query_candidates(
            query_text,
            top_k=section_fetch,
            keyword_weight=keyword_weight,
        )
        return results[:top_k], weights
    except Exception:
        logger.exception("Failed to search candidates")
        raise


def delete_candidate_vectors(candidate_id: str) -> None:
    try:
        pinecone_operations.delete_candidate_vectors(candidate_id)
    except Exception:
        logger.exception("Failed to delete vectors for candidate %s", candidate_id)
