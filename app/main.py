"""FastAPI app — Gemini proxy for the Angular frontend."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from pydantic import BaseModel, Field

from app.gemini_operations import invoke
from app import supabase_operations

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(REPO_ROOT / ".env")
load_dotenv(REPO_ROOT / ".env.local")
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GEMINI_KEY = (os.environ.get("GEMINI_API_KEY") or os.environ.get("API_KEY") or "").strip()

app = FastAPI(title="Skillz Talent AI API", version="1.0.0")

_cors = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
_origins = [o.strip() for o in _cors.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _client() -> genai.Client:
    if not GEMINI_KEY:
        raise HTTPException(
            status_code=503,
            detail="Server missing GEMINI_API_KEY. Set it in repo root .env.local or backend/.env.",
        )
    return genai.Client(api_key=GEMINI_KEY)


class GeminiRequest(BaseModel):
    operation: str
    payload: dict = Field(default_factory=dict)


class GeminiResponse(BaseModel):
    error: str | None = None
    kind: str | None = None
    value: Any = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/gemini", response_model=GeminiResponse)
async def gemini_endpoint(body: GeminiRequest) -> GeminiResponse:
    client = _client()
    try:
        kind, value = await asyncio.to_thread(invoke, client, body.operation, body.payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        return GeminiResponse(error=str(e) or "Gemini request failed")
    return GeminiResponse(kind=kind, value=value)


class ResumeRecord(BaseModel):
    file_name: str
    summary: str
    skills: list[Any] = []
    experiences: list[Any] = []
    projects: list[Any] = []
    certifications: list[Any] = []


@app.post("/api/resumes", status_code=201)
async def save_resume(body: ResumeRecord) -> dict[str, Any]:
    try:
        result = await asyncio.to_thread(supabase_operations.save_resume, body.model_dump())
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/resumes")
async def get_resumes() -> list[dict[str, Any]]:
    try:
        return await asyncio.to_thread(supabase_operations.get_resumes)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


class JobDescriptionRecord(BaseModel):
    job_title: str
    responsibilities: str
    content: str


@app.post("/api/job-descriptions", status_code=201)
async def save_job_description(body: JobDescriptionRecord) -> dict[str, Any]:
    try:
        result = await asyncio.to_thread(
            supabase_operations.save_job_description, body.model_dump()
        )
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/job-descriptions")
async def get_job_descriptions() -> list[dict[str, Any]]:
    try:
        return await asyncio.to_thread(supabase_operations.get_job_descriptions)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


class CandidateRecord(BaseModel):
    full_name: str
    job_role: str
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin_profile: str = ""
    domain_industry: str = ""
    work_authorization: str = ""
    preferred_location: str = ""
    open_to_relocation: str = ""
    expected_salary: str = ""
    employment_type: str = ""
    summary: str
    skills: list[str] = []


@app.post("/api/candidates", status_code=201)
async def save_candidate(body: CandidateRecord) -> dict[str, Any]:
    try:
        result = await asyncio.to_thread(
            supabase_operations.save_candidate, body.model_dump()
        )
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/candidates")
async def get_candidates() -> list[dict[str, Any]]:
    try:
        return await asyncio.to_thread(supabase_operations.get_candidates)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
