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


def save_candidate(record: dict[str, Any]) -> dict[str, Any]:
    return supabase_operations.save_candidate(record)


def get_candidates() -> list[dict[str, Any]]:
    return supabase_operations.get_candidates()
