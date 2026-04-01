"""Shared dependencies for route handlers."""

from __future__ import annotations

from fastapi import HTTPException
from google import genai

from app.core.config import GEMINI_API_KEY


def get_gemini_client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Server missing GEMINI_API_KEY. Set it in repo root .env.local or backend/.env.",
        )
    return genai.Client(api_key=GEMINI_API_KEY)
