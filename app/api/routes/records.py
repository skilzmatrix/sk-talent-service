"""Persistence routes for resumes, job descriptions, and candidates."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import uuid

from fastapi import APIRouter, HTTPException, Query, UploadFile, File

from app.schemas.records import CandidateProfileUpdate, CandidateRecord, JobDescriptionRecord, ResumeRecord
from app.services import persistence_service, pinecone_service

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


@router.post("/api/resumes/upload-file", status_code=201)
async def upload_resume_file(file: UploadFile = File(...)) -> dict[str, str]:
    try:
        file_bytes = await file.read()
        ext = file.filename.rsplit(".", 1)[-1] if file.filename else "pdf"
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        url = await asyncio.to_thread(
            persistence_service.upload_candidate_resume,
            unique_name,
            file_bytes,
            file.content_type or "application/octet-stream",
        )
        return {"url": url}
    except Exception as exc:
        _raise_storage_error(exc)
        return {"url": ""}


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


@router.post("/api/candidates/upload-resume", status_code=201)
async def upload_candidate_resume(file: UploadFile = File(...)) -> dict[str, str]:
    try:
        file_bytes = await file.read()
        ext = file.filename.rsplit(".", 1)[-1] if file.filename else "pdf"
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        url = await asyncio.to_thread(
            persistence_service.upload_candidate_resume,
            unique_name,
            file_bytes,
            file.content_type or "application/octet-stream",
        )
        return {"url": url}
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
    if candidate_id:
        asyncio.create_task(
            asyncio.to_thread(pinecone_service.vectorize_candidate, candidate_id, result)
        )
    return result


@router.put("/api/candidates/{candidate_id}")
async def update_candidate(candidate_id: str, body: CandidateProfileUpdate) -> dict[str, Any]:
    return await _run_storage_call(
        persistence_service.update_candidate, candidate_id, body.model_dump()
    )


@router.get("/api/candidates")
async def get_candidates() -> list[dict[str, Any]]:
    return await _run_storage_call(persistence_service.get_candidates)


@router.delete("/api/candidates/{candidate_id}", status_code=200)
async def delete_candidate(
    candidate_id: str,
    resume_path: str | None = Query(default=None),
) -> dict[str, str]:
    await _run_storage_call(persistence_service.delete_candidate, candidate_id, resume_path)
    return {"status": "deleted"}
