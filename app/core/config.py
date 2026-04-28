"""Application configuration and environment loading."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

API_TITLE = "Skillz Talent AI API"
API_VERSION = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(REPO_ROOT / ".env")
load_dotenv(REPO_ROOT / ".env.local")
load_dotenv(BACKEND_ROOT / ".env")

GEMINI_API_KEY = (os.environ.get("GEMINI_API_KEY") or os.environ.get("API_KEY") or "").strip()
DEFAULT_CORS_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"

# https://ai.google.dev/gemini-api/docs/models
# General text/JSON via google.genai:
GEMINI_MODEL = (os.environ.get("GEMINI_MODEL") or "gemini-3.1-pro-preview").strip()
# LangGraph ReAct + custom tools: use the customtools-optimized model when available
GEMINI_MODEL_AGENT = (os.environ.get("GEMINI_MODEL_AGENT") or "gemini-3.1-pro-preview-customtools").strip()


def get_cors_origins() -> list[str]:
    raw = os.environ.get("CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]
