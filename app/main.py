"""FastAPI application entrypoint."""

from __future__ import annotations

import warnings

# LangChain's Google integration may emit FutureWarning on import; keep dev logs readable.
warnings.filterwarnings("ignore", category=FutureWarning, module="langchain_google_genai.chat_models")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import API_TITLE, API_VERSION, get_cors_origins


def create_app() -> FastAPI:
    app = FastAPI(title=API_TITLE, version=API_VERSION)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router)
    return app


app = create_app()
