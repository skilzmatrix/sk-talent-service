"""Persistence routes for resumes, job descriptions, and candidates."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from fastapi import APIRouter, HTTPException

from app.schemas.records import CandidateRecord, JobDescriptionRecord, ResumeRecord
from app.services import persistence_service

router = APIRouter()


def _raise_storage_error(exc: Exception) -> None:
    if isinstance(exc, RuntimeError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail=str(exc)) from exc


async def _run_storage_call(fn: Callable[..., Any], *args: Any) -> Any:
    try:
        return await asyncio.to_thread(fn, *args)
    except Exception as exc:
        _raise_storage_error(exc)


@router.post("/api/resumes", status_code=201)
async def save_resume(body: ResumeRecord) -> dict[str, Any]:
    return await _run_storage_call(persistence_service.save_resume, body.model_dump())


@router.get("/api/resumes")
async def get_resumes() -> list[dict[str, Any]]:
    return await _run_storage_call(persistence_service.get_resumes)


@router.post("/api/job-descriptions", status_code=201)
async def save_job_description(body: JobDescriptionRecord) -> dict[str, Any]:
    return await _run_storage_call(
        persistence_service.save_job_description, body.model_dump()
    )


@router.get("/api/job-descriptions")
async def get_job_descriptions() -> list[dict[str, Any]]:
    return await _run_storage_call(persistence_service.get_job_descriptions)


@router.post("/api/candidates", status_code=201)
async def save_candidate(body: CandidateRecord) -> dict[str, Any]:
    return await _run_storage_call(persistence_service.save_candidate, body.model_dump())


@router.get("/api/candidates")
async def get_candidates() -> list[dict[str, Any]]:
    return await _run_storage_call(persistence_service.get_candidates)
