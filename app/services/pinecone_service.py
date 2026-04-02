"""Service wrapper for Pinecone vector operations."""

from __future__ import annotations

import logging
from typing import Any

from app import pinecone_operations

logger = logging.getLogger(__name__)


def vectorize_candidate(candidate_id: str, record: dict[str, Any]) -> None:
    try:
        pinecone_operations.upsert_candidate_vectors(candidate_id, record)
    except Exception:
        logger.exception("Failed to vectorize candidate %s", candidate_id)
