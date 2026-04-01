"""Gemini proxy routes."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from app.core.dependencies import get_gemini_client
from app.schemas.gemini import GeminiRequest, GeminiResponse
from app.services.gemini_service import run_operation

router = APIRouter()


@router.post("/api/gemini", response_model=GeminiResponse)
async def gemini_endpoint(body: GeminiRequest) -> GeminiResponse:
    client = get_gemini_client()
    try:
        kind, value = await asyncio.to_thread(
            run_operation, client, body.operation, body.payload
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        return GeminiResponse(error=str(exc) or "Gemini request failed")
    return GeminiResponse(kind=kind, value=value)
