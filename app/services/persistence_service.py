"""Service wrapper for Supabase persistence operations."""

from __future__ import annotations

from typing import Any

from app import supabase_operations


def save_resume(record: dict[str, Any]) -> dict[str, Any]:
    return supabase_operations.save_resume(record)


def get_resumes() -> list[dict[str, Any]]:
    return supabase_operations.get_resumes()


def save_job_description(record: dict[str, Any]) -> dict[str, Any]:
    return supabase_operations.save_job_description(record)


def get_job_descriptions() -> list[dict[str, Any]]:
    return supabase_operations.get_job_descriptions()


def upload_candidate_resume(file_name: str, file_bytes: bytes, content_type: str) -> str:
    return supabase_operations.upload_candidate_resume(file_name, file_bytes, content_type)


def get_signed_resume_url(path: str) -> str:
    return supabase_operations.get_signed_resume_url(path)


def save_candidate(record: dict[str, Any]) -> dict[str, Any]:
    return supabase_operations.save_candidate(record)


def update_candidate(candidate_id: str, record: dict[str, Any]) -> dict[str, Any]:
    return supabase_operations.update_candidate(candidate_id, record)


def get_candidates() -> list[dict[str, Any]]:
    return supabase_operations.get_candidates()


def get_candidate_by_id(candidate_id: str) -> dict[str, Any] | None:
    return supabase_operations.get_candidate_by_id(candidate_id)


def delete_candidate(candidate_id: str, resume_path: str | None = None) -> None:
    supabase_operations.delete_candidate(candidate_id, resume_path)
