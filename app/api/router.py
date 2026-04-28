"""Top-level API router."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes.gemini import router as gemini_router
from app.api.routes.health import router as health_router
from app.api.routes.records import router as records_router
from app.api.routes.chat import router as chat_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(gemini_router)
api_router.include_router(records_router)
api_router.include_router(chat_router)
