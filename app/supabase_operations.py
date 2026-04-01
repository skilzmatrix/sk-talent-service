"""Supabase operations for resume storage."""

from __future__ import annotations

import os
from typing import Any

from supabase import create_client, Client


def _client() -> Client:
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not url or not key:
        raise RuntimeError(
            "Supabase not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in backend/.env"
        )
    return create_client(url, key)


def save_resume(record: dict[str, Any]) -> dict[str, Any]:
    client = _client()
    response = (
        client.table("resumes")
        .insert({
            "file_name": record["file_name"],
            "summary": record["summary"],
            "skills": record.get("skills", []),
            "experiences": record.get("experiences", []),
            "projects": record.get("projects", []),
            "certifications": record.get("certifications", []),
        })
        .execute()
    )
    if not response.data:
        raise RuntimeError("Failed to save resume to Supabase.")
    return response.data[0]


def get_resumes() -> list[dict[str, Any]]:
    client = _client()
    response = (
        client.table("resumes")
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )
    return response.data or []


def save_job_description(record: dict[str, Any]) -> dict[str, Any]:
    client = _client()
    response = (
        client.table("job_descriptions")
        .insert({
            "job_title": record["job_title"],
            "responsibilities": record["responsibilities"],
            "content": record["content"],
        })
        .execute()
    )
    if not response.data:
        raise RuntimeError("Failed to save job description to Supabase.")
    return response.data[0]


def get_job_descriptions() -> list[dict[str, Any]]:
    client = _client()
    response = (
        client.table("job_descriptions")
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )
    return response.data or []


def save_candidate(record: dict[str, Any]) -> dict[str, Any]:
    client = _client()
    response = (
        client.table("candidates")
        .insert({
            "full_name": record.get("full_name", ""),
            "job_role": record.get("job_role", ""),
            "email": record.get("email", ""),
            "phone": record.get("phone", ""),
            "location": record.get("location", ""),
            "linkedin_profile": record.get("linkedin_profile", ""),
            "domain_industry": record.get("domain_industry", ""),
            "work_authorization": record.get("work_authorization", ""),
            "preferred_location": record.get("preferred_location", ""),
            "open_to_relocation": record.get("open_to_relocation", ""),
            "expected_salary": record.get("expected_salary", ""),
            "employment_type": record.get("employment_type", ""),
            "summary": record.get("summary", ""),
            "skills": record.get("skills", []),
        })
        .execute()
    )
    if not response.data:
        raise RuntimeError("Failed to save candidate to Supabase.")
    return response.data[0]


def get_candidates() -> list[dict[str, Any]]:
    client = _client()
    response = (
        client.table("candidates")
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )
    return response.data or []
