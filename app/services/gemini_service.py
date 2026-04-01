"""Service wrapper for Gemini operations."""

from __future__ import annotations

from typing import Any

from google import genai

from app.gemini_operations import invoke


def run_operation(
    client: genai.Client, operation: str, payload: dict[str, Any]
) -> tuple[str, str | Any]:
    return invoke(client, operation, payload)
