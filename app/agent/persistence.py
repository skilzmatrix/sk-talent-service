"""Supabase persistence for chat histories."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import List

from langchain_core.messages import BaseMessage, messages_from_dict, message_to_dict

logger = logging.getLogger(__name__)

# Module-level client cache — avoids creating a new TCP connection on every
# load/save call while remaining safe for multi-worker deployments (each worker
# process holds its own singleton).
_supabase_client = None


def _get_client():
    global _supabase_client
    if _supabase_client is None:
        from app.supabase_operations import _client

        _supabase_client = _client()
    return _supabase_client


def load_chat_history(conversation_id: str) -> List[BaseMessage]:
    """Load chat history from Supabase. Returns an empty list on any error."""
    try:
        client = _get_client()
        response = (
            client.table("chat_histories")
            .select("messages")
            .eq("conversation_id", conversation_id)
            .limit(1)
            .execute()
        )

        if not response.data:
            return []

        messages_raw = response.data[0].get("messages", [])
        if isinstance(messages_raw, str):
            messages_raw = json.loads(messages_raw)

        if not isinstance(messages_raw, list):
            return []

        return messages_from_dict(messages_raw)

    except Exception:
        logger.exception("Failed to load chat history for conversation %s", conversation_id)
        return []


def save_chat_history(conversation_id: str, messages: List[BaseMessage]) -> None:
    """Upsert the full message list for a conversation. Logs errors instead of raising."""
    try:
        client = _get_client()
        messages_dict = [message_to_dict(m) for m in messages]
        client.table("chat_histories").upsert(
            {
                "conversation_id": conversation_id,
                "messages": messages_dict,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        ).execute()
    except Exception:
        logger.exception("Failed to save chat history for conversation %s", conversation_id)
