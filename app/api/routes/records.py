"""Persistence routes for resumes, job descriptions, and candidates."""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any, Callable

import uuid

from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from google import genai

from pydantic import BaseModel, Field

from app.core.config import GEMINI_API_KEY
from app.schemas.records import (
    CandidateProfileUpdate,
    CandidateRecord,
    JobDescriptionRecord,
    PaginatedRecordsResponse,
    ResumeRecord,
)
from app.services import persistence_service, pinecone_service, talent_search_service


class TalentSearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    # Fraction of final score from JD keyword overlap (0–1). Semantic uses the remainder.
    # If omitted, uses env TALENT_SEARCH_KEYWORD_WEIGHT or server default.
    keyword_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    # When True, Gemini reranks the vector top‑K using full DB profiles and adds reasoning.
    use_llm_rerank: bool = Field(default=True)
    work_authorization: str | None = None
    experience: str | None = None
    location: str | None = None
    city: str | None = None
    state: str | None = None
    linkedin_profile: str | None = None
    domain_industry: str | None = None
    preferred_location: str | None = None
    open_to_relocation: str | None = None
    expected_salary: str | None = None
    employment_type: str | None = None
    skills: list[str] | None = None

router = APIRouter()

ALLOWED_RESUME_EXTENSIONS = {"pdf", "docx"}


def _normalize_candidate_filter(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _normalize_candidate_skills(skills: list[str] | None) -> list[str]:
    if not skills:
        return []
    normalized_skills: list[str] = []
    for item in skills:
        for skill in item.split(","):
            normalized = skill.strip()
            if normalized:
                normalized_skills.append(normalized)
    return normalized_skills


def _build_candidate_filters(
    work_authorization: str | None = None,
    experience: str | None = None,
    location: str | None = None,
    city: str | None = None,
    state: str | None = None,
    linkedin_profile: str | None = None,
    domain_industry: str | None = None,
    preferred_location: str | None = None,
    open_to_relocation: str | None = None,
    expected_salary: str | None = None,
    employment_type: str | None = None,
    skills: list[str] | None = None,
) -> dict[str, Any]:
    filters = {
        "work_authorization": _normalize_candidate_filter(work_authorization),
        "experience": _normalize_candidate_filter(experience),
        "location": _normalize_candidate_filter(location),
        "city": _normalize_candidate_filter(city),
        "state": _normalize_candidate_filter(state),
        "linkedin_profile": _normalize_candidate_filter(linkedin_profile),
        "domain_industry": _normalize_candidate_filter(domain_industry),
        "preferred_location": _normalize_candidate_filter(preferred_location),
        "open_to_relocation": _normalize_candidate_filter(open_to_relocation),
        "expected_salary": _normalize_candidate_filter(expected_salary),
        "employment_type": _normalize_candidate_filter(employment_type),
        "skills": _normalize_candidate_skills(skills),
    }
    return {
        key: value
        for key, value in filters.items()
        if value not in (None, [], "")
    }


def _build_resume_upload_name(filename: str | None) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if filename and "." in filename else ""
    if ext not in ALLOWED_RESUME_EXTENSIONS:
        allowed = ", ".join(sorted(f".{item}" for item in ALLOWED_RESUME_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported resume file type. Allowed types: {allowed}.",
        )
    return f"{uuid.uuid4().hex}.{ext}"


def _raise_storage_error(exc: Exception) -> None:
    if isinstance(exc, RuntimeError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail=str(exc)) from exc


async def _run_storage_call(fn: Callable[..., Any], *args: Any) -> Any:
    try:
        return await asyncio.to_thread(fn, *args)
    except Exception as exc:
        _raise_storage_error(exc)


@router.post("/api/resumes/upload-file", status_code=201)
async def upload_resume_file(file: UploadFile = File(...)) -> dict[str, str]:
    try:
        file_bytes = await file.read()
        unique_name = _build_resume_upload_name(file.filename)
        url = await asyncio.to_thread(
            persistence_service.upload_candidate_resume,
            unique_name,
            file_bytes,
            file.content_type or "application/octet-stream",
        )
        return {"url": url}
    except HTTPException:
        raise
    except Exception as exc:
        _raise_storage_error(exc)
        return {"url": ""}


@router.post("/api/resumes", status_code=201)
async def save_resume(body: ResumeRecord) -> dict[str, Any]:
    return await _run_storage_call(persistence_service.save_resume, body.model_dump())


@router.get("/api/resumes")
async def get_resumes(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> PaginatedRecordsResponse:
    data = await _run_storage_call(
        persistence_service.get_resumes_paginated,
        page,
        page_size,
    )
    return PaginatedRecordsResponse.model_validate(data)


@router.post("/api/job-descriptions", status_code=201)
async def save_job_description(body: JobDescriptionRecord) -> dict[str, Any]:
    return await _run_storage_call(
        persistence_service.save_job_description, body.model_dump()
    )


@router.get("/api/job-descriptions")
async def get_job_descriptions() -> list[dict[str, Any]]:
    return await _run_storage_call(persistence_service.get_job_descriptions)


@router.post("/api/candidates/upload-resume", status_code=201)
async def upload_candidate_resume(file: UploadFile = File(...)) -> dict[str, str]:
    try:
        file_bytes = await file.read()
        unique_name = _build_resume_upload_name(file.filename)
        url = await asyncio.to_thread(
            persistence_service.upload_candidate_resume,
            unique_name,
            file_bytes,
            file.content_type or "application/octet-stream",
        )
        return {"url": url}
    except HTTPException:
        raise
    except Exception as exc:
        _raise_storage_error(exc)
        return {"url": ""}


@router.get("/api/storage/signed-url")
async def get_signed_url(path: str = Query(...)) -> dict[str, str]:
    try:
        signed_url = await asyncio.to_thread(
            persistence_service.get_signed_resume_url, path
        )
        if not signed_url:
            raise HTTPException(status_code=404, detail="File not found")
        return {"url": signed_url}
    except HTTPException:
        raise
    except Exception as exc:
        _raise_storage_error(exc)
        return {"url": ""}


@router.post("/api/candidates", status_code=201)
async def save_candidate(body: CandidateRecord) -> dict[str, Any]:
    result = await _run_storage_call(persistence_service.save_candidate, body.model_dump())
    candidate_id = result.get("id")
    embedded_sections: list[str] = []
    if candidate_id:
        embedded_sections = await asyncio.to_thread(
            pinecone_service.vectorize_candidate, candidate_id, result
        )
    result["embedded_sections"] = embedded_sections
    return result


@router.put("/api/candidates/{candidate_id}")
async def update_candidate(candidate_id: str, body: CandidateProfileUpdate) -> dict[str, Any]:
    result = await _run_storage_call(
        persistence_service.update_candidate, candidate_id, body.model_dump()
    )
    embedded_sections = await asyncio.to_thread(
        pinecone_service.vectorize_candidate, candidate_id, result
    )
    result["embedded_sections"] = embedded_sections
    return result


@router.get("/api/candidates")
async def get_candidates(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    q: str | None = Query(default=None),
    work_authorization: str | None = Query(default=None),
    experience: str | None = Query(default=None),
    location: str | None = Query(default=None),
    city: str | None = Query(default=None),
    state: str | None = Query(default=None),
    linkedin_profile: str | None = Query(default=None),
    domain_industry: str | None = Query(default=None),
    preferred_location: str | None = Query(default=None),
    open_to_relocation: str | None = Query(default=None),
    expected_salary: str | None = Query(default=None),
    employment_type: str | None = Query(default=None),
    skills: list[str] | None = Query(default=None),
) -> PaginatedRecordsResponse:
    filters = _build_candidate_filters(
        work_authorization=work_authorization,
        experience=experience,
        location=location,
        city=city,
        state=state,
        linkedin_profile=linkedin_profile,
        domain_industry=domain_industry,
        preferred_location=preferred_location,
        open_to_relocation=open_to_relocation,
        expected_salary=expected_salary,
        employment_type=employment_type,
        skills=skills,
    )
    data = await _run_storage_call(
        persistence_service.get_candidates_paginated,
        page,
        page_size,
        q,
        filters,
    )
    return PaginatedRecordsResponse.model_validate(data)


@router.delete("/api/candidates/{candidate_id}", status_code=200)
async def delete_candidate(
    candidate_id: str,
    resume_path: str | None = Query(default=None),
) -> dict[str, str]:
    await _run_storage_call(persistence_service.delete_candidate, candidate_id, resume_path)
    asyncio.create_task(
        asyncio.to_thread(pinecone_service.delete_candidate_vectors, candidate_id)
    )
    return {"status": "deleted"}


@router.post("/api/talent-search")
async def talent_search(body: TalentSearchRequest) -> dict[str, Any]:
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query text is required.")
    candidate_filters = _build_candidate_filters(
        work_authorization=body.work_authorization,
        experience=body.experience,
        location=body.location,
        city=body.city,
        state=body.state,
        linkedin_profile=body.linkedin_profile,
        domain_industry=body.domain_industry,
        preferred_location=body.preferred_location,
        open_to_relocation=body.open_to_relocation,
        expected_salary=body.expected_salary,
        employment_type=body.employment_type,
        skills=body.skills,
    )
    # Skills are applied in relational search; Pinecone metadata filters support scalar fields only.
    vector_filters = {
        key: value
        for key, value in candidate_filters.items()
        if key != "skills"
    }
    gemini_client: genai.Client | None = None
    if GEMINI_API_KEY:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    try:
        results, ranking_weights, meta = await asyncio.to_thread(
            partial(
                talent_search_service.run_talent_search,
                body.query.strip(),
                body.top_k,
                body.keyword_weight,
                gemini_client,
                vector_filters,
                use_llm_rerank=body.use_llm_rerank,
            )
        )
        out: dict[str, Any] = {
            "results": results,
            "ranking_weights": ranking_weights,
            "llm_rerank": meta.get("llm_rerank"),
        }
        if meta.get("llm_error"):
            out["llm_error"] = meta["llm_error"]
        if meta.get("llm_hint"):
            out["llm_hint"] = meta["llm_hint"]
        return out
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
