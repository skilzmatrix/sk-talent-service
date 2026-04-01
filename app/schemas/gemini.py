"""Schemas for Gemini proxy requests and responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GeminiRequest(BaseModel):
    operation: str
    payload: dict[str, Any] = Field(default_factory=dict)


class GeminiResponse(BaseModel):
    error: str | None = None
    kind: str | None = None
    value: Any = None
