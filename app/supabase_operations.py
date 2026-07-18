"""Supabase operations for resume storage."""

from __future__ import annotations

import os
import re
from typing import Any
from urllib.parse import unquote, urlparse

from supabase import create_client, Client


CANDIDATE_GLOBAL_SEARCH_FIELDS = (
    "full_name",
    "job_role",
    "summary",
    "email",
    "location",
    "city",
    "state",
    "domain_industry",
    "work_authorization",
    "experience",
    "preferred_location",
    "expected_salary",
    "employment_type",
    "linkedin_profile",
    "open_to_relocation",
)

CANDIDATE_TEXT_FILTER_FIELDS = (
    "work_authorization",
    "location",
    "city",
    "state",
    "linkedin_profile",
    "domain_industry",
    "preferred_location",
    "open_to_relocation",
    "expected_salary",
    "employment_type",
)


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
            "experience": record.get("experience", ""),
            "skills": record.get("skills", []),
            "experiences": record.get("experiences", []),
            "projects": record.get("projects", []),
            "certifications": record.get("certifications", []),
            "resume_url": record.get("resume_url", ""),
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


def get_resumes_paginated(page: int, page_size: int) -> dict[str, Any]:
    client = _client()
    start = (page - 1) * page_size
    end = start + page_size - 1
    response = (
        client.table("resumes")
        .select("*", count="exact")
        .order("created_at", desc=True)
        .range(start, end)
        .execute()
    )
    total_items = int(response.count or 0)
    total_pages = (total_items + page_size - 1) // page_size if total_items else 0
    return {
        "items": response.data or [],
        "page": page,
        "page_size": page_size,
        "total_items": total_items,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_previous": page > 1,
    }


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


def upload_candidate_resume(file_name: str, file_bytes: bytes, content_type: str) -> str:
    client = _client()
    path = f"resumes/{file_name}"
    client.storage.from_("candidate_resumes").upload(
        path,
        file_bytes,
        {"content-type": content_type},
    )
    return path


def _normalize_resume_storage_path(path: str | None) -> str:
    """Convert stored resume references to a bucket-relative storage key."""
    raw = str(path or "").strip()
    if not raw:
        return ""

    parsed = urlparse(raw)
    if parsed.scheme and parsed.netloc:
        raw = parsed.path or ""

    raw = unquote(raw).split("?", 1)[0].split("#", 1)[0]

    marker = "/candidate_resumes/"
    if marker in raw:
        raw = raw.split(marker, 1)[1]

    raw = raw.lstrip("/")
    bucket_prefix = "candidate_resumes/"
    if raw.startswith(bucket_prefix):
        raw = raw[len(bucket_prefix):]

    return raw


def _remove_storage_object(client: Client, path: str) -> None:
    normalized = _normalize_resume_storage_path(path)
    if not normalized:
        return
    client.storage.from_("candidate_resumes").remove([normalized])


def upload_chat_attachment(
    file_name: str,
    file_bytes: bytes,
    content_type: str,
    size_bytes: int,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    client = _client()
    folder = f"chat_attachments/{conversation_id}" if conversation_id else "chat_attachments/unassigned"
    storage_key = f"{folder}/{file_name}"
    client.storage.from_("candidate_resumes").upload(
        storage_key,
        file_bytes,
        {"content-type": content_type},
    )

    try:
        response = (
            client.table("chat_attachments")
            .insert(
                {
                    "conversation_id": conversation_id,
                    "file_name": file_name,
                    "mime_type": content_type,
                    "size_bytes": size_bytes,
                    "storage_key": storage_key,
                    "bucket": "candidate_resumes",
                }
            )
            .execute()
        )
    except Exception:
        try:
            _remove_storage_object(client, storage_key)
        except Exception:
            pass
        raise

    if not response.data:
        try:
            _remove_storage_object(client, storage_key)
        except Exception:
            pass
        raise RuntimeError("Failed to persist chat attachment metadata.")

    return response.data[0]


def list_chat_attachments(conversation_id: str) -> list[dict[str, Any]]:
    client = _client()
    response = (
        client.table("chat_attachments")
        .select("id, conversation_id, file_name, mime_type, size_bytes, storage_key, bucket, created_at")
        .eq("conversation_id", conversation_id)
        .order("created_at", desc=True)
        .execute()
    )
    return response.data or []


def delete_chat_attachment(attachment_id: str) -> dict[str, Any] | None:
    client = _client()
    lookup = (
        client.table("chat_attachments")
        .select("id, conversation_id, file_name, mime_type, size_bytes, storage_key, bucket, created_at")
        .eq("id", attachment_id)
        .limit(1)
        .execute()
    )
    if not lookup.data:
        return None

    record = lookup.data[0]
    storage_key = str(record.get("storage_key") or "")

    if storage_key:
        try:
            _remove_storage_object(client, storage_key)
        except Exception:
            pass

    client.table("chat_attachments").delete().eq("id", attachment_id).execute()
    return record


def delete_chat_attachments_for_conversation(conversation_id: str) -> None:
    client = _client()
    rows = list_chat_attachments(conversation_id)
    for row in rows:
        storage_key = str(row.get("storage_key") or "")
        if not storage_key:
            continue
        try:
            _remove_storage_object(client, storage_key)
        except Exception:
            pass

    client.table("chat_attachments").delete().eq("conversation_id", conversation_id).execute()


def delete_chat_history(conversation_id: str) -> None:
    client = _client()
    client.table("chat_histories").delete().eq("conversation_id", conversation_id).execute()


def delete_conversation_with_attachments(conversation_id: str) -> None:
    delete_chat_attachments_for_conversation(conversation_id)
    delete_chat_history(conversation_id)


def get_signed_resume_url(path: str, expires_in: int = 3600) -> str:
    path = _normalize_resume_storage_path(path)
    if not path:
        return ""
    client = _client()
    result = client.storage.from_("candidate_resumes").create_signed_url(path, expires_in)
    return result.get("signedURL", "")


def _split_location_parts(location: str) -> tuple[str, str]:
    raw = location.strip()
    if not raw:
        return "", ""

    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return raw, ""


def _normalize_location_fields(record: dict[str, Any]) -> tuple[str, str, str]:
    location = str(record.get("location", "") or "").strip()
    city = str(record.get("city", "") or "").strip()
    state = str(record.get("state", "") or "").strip()

    if location and (not city or not state):
        inferred_city, inferred_state = _split_location_parts(location)
        if not city:
            city = inferred_city
        if not state:
            state = inferred_state

    if not location and city and state:
        location = f"{city}, {state}"
    elif not location and city:
        location = city

    return location, city, state


def save_candidate(record: dict[str, Any]) -> dict[str, Any]:
    client = _client()
    
    # Check for duplicates and remove them (keep only the latest)
    email = record.get("email", "").strip()
    phone = record.get("phone", "").strip()
    
    if email or phone:
        if email:
            email_lower = email.lower()
            response = (
                client.table("candidates")
                .select("id, resume_url")
                .ilike("email", email_lower)
                .execute()
            )
            for dup in response.data or []:
                dup_id = dup.get("id")
                dup_resume = dup.get("resume_url")
                delete_candidate(dup_id, dup_resume)
        
        if phone:
            response = (
                client.table("candidates")
                .select("id, resume_url")
                .eq("phone", phone)
                .execute()
            )
            for dup in response.data or []:
                dup_id = dup.get("id")
                dup_resume = dup.get("resume_url")
                delete_candidate(dup_id, dup_resume)
    
    location, city, state = _normalize_location_fields(record)
    response = (
        client.table("candidates")
        .insert({
            "full_name": record.get("full_name", ""),
            "job_role": record.get("job_role", ""),
            "email": record.get("email", ""),
            "phone": record.get("phone", ""),
            "location": location,
            "city": city,
            "state": state,
            "linkedin_profile": record.get("linkedin_profile", ""),
            "domain_industry": record.get("domain_industry", ""),
            "work_authorization": record.get("work_authorization", ""),
            "experience": record.get("experience", ""),
            "preferred_location": record.get("preferred_location", ""),
            "open_to_relocation": record.get("open_to_relocation", ""),
            "expected_salary": record.get("expected_salary", ""),
            "employment_type": record.get("employment_type", ""),
            "summary": record.get("summary", ""),
            "skills": record.get("skills", []),
            "experiences": record.get("experiences", []),
            "projects": record.get("projects", []),
            "certifications": record.get("certifications", []),
            "resume_url": record.get("resume_url", ""),
        })
        .execute()
    )
    if not response.data:
        raise RuntimeError("Failed to save candidate to Supabase.")
    return response.data[0]


def update_candidate(candidate_id: str, record: dict[str, Any]) -> dict[str, Any]:
    client = _client()
    current = get_candidate_by_id(candidate_id)
    if not current:
        raise RuntimeError(f"Candidate {candidate_id} not found.")

    merged_record = {**current, **record}
    location, city, state = _normalize_location_fields(merged_record)
    response = (
        client.table("candidates")
        .update({
            "full_name": merged_record.get("full_name", ""),
            "job_role": merged_record.get("job_role", ""),
            "email": merged_record.get("email", ""),
            "phone": merged_record.get("phone", ""),
            "location": location,
            "city": city,
            "state": state,
            "linkedin_profile": merged_record.get("linkedin_profile", ""),
            "domain_industry": merged_record.get("domain_industry", ""),
            "work_authorization": merged_record.get("work_authorization", ""),
            "experience": merged_record.get("experience", ""),
            "preferred_location": merged_record.get("preferred_location", ""),
            "open_to_relocation": merged_record.get("open_to_relocation", ""),
            "expected_salary": merged_record.get("expected_salary", ""),
            "employment_type": merged_record.get("employment_type", ""),
            "summary": merged_record.get("summary", ""),
            "skills": merged_record.get("skills", []),
            "experiences": merged_record.get("experiences", []),
            "projects": merged_record.get("projects", []),
            "certifications": merged_record.get("certifications", []),
            "resume_url": merged_record.get("resume_url", ""),
        })
        .eq("id", candidate_id)
        .execute()
    )
    if not response.data:
        raise RuntimeError("Failed to update candidate.")
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


def _sanitize_filter_text(value: str) -> str:
    return value.replace("%", "").replace(",", " ").strip()


def _normalize_linkedin_filter(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"^https?://", "", normalized)
    normalized = re.sub(r"^www\.", "", normalized)
    return normalized.rstrip("/")


def _build_ilike_pattern(value: str) -> str:
    return f"*{value}*"


def _normalize_skill_filters(filters: dict[str, Any] | None) -> list[str]:
    if not filters:
        return []
    raw_skills = filters.get("skills")
    if not isinstance(raw_skills, list):
        return []
    return [skill.strip().lower() for skill in raw_skills if skill and skill.strip()]


def _candidate_matches_skills(candidate: dict[str, Any], skills: list[str]) -> bool:
    if not skills:
        return True
    candidate_skills = candidate.get("skills")
    if not isinstance(candidate_skills, list):
        return False

    normalized_candidate_skills = [
        str(skill).strip().lower() for skill in candidate_skills if str(skill).strip()
    ]
    if not normalized_candidate_skills:
        return False

    for search_skill in skills:
        if any(search_skill in candidate_skill for candidate_skill in normalized_candidate_skills):
            return True
    return False


def _parse_experience_years(value: Any) -> float | None:
    raw = str(value or "").strip().lower()
    if not raw:
        return None
    cleaned = (
        raw.replace("years", "")
        .replace("year", "")
        .replace("yrs", "")
        .replace("yr", "")
        .replace("+", "")
        .replace(",", "")
        .strip()
    )
    try:
        years = float(cleaned)
    except ValueError:
        return None
    if years < 0 or years > 80:
        return None
    return years


def _normalize_experience_min_filter(filters: dict[str, Any] | None) -> float | None:
    if not filters:
        return None
    raw_value = filters.get("experience")
    if not isinstance(raw_value, str):
        return None
    return _parse_experience_years(raw_value)


def _candidate_matches_experience_min(candidate: dict[str, Any], min_years: float | None) -> bool:
    if min_years is None:
        return True
    years = _parse_experience_years(candidate.get("experience"))
    if years is None:
        return False
    return years >= min_years


def _apply_candidate_filters(db_query: Any, filters: dict[str, Any] | None) -> Any:
    if not filters:
        return db_query

    for field in CANDIDATE_TEXT_FILTER_FIELDS:
        raw_value = filters.get(field)
        if not isinstance(raw_value, str):
            continue
        safe_value = _sanitize_filter_text(raw_value)
        if field == "linkedin_profile" and safe_value:
            safe_value = _normalize_linkedin_filter(safe_value)
        if safe_value:
            db_query = db_query.ilike(field, _build_ilike_pattern(safe_value))

    return db_query


def get_candidates_paginated(
    page: int,
    page_size: int,
    query: str | None = None,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    client = _client()
    start = (page - 1) * page_size
    end = start + page_size - 1
    skill_filters = _normalize_skill_filters(filters)
    experience_min = _normalize_experience_min_filter(filters)
    db_query = client.table("candidates").select("*", count="exact")
    term = (query or "").strip()
    if term:
        safe_term = _sanitize_filter_text(term)
        if safe_term:
            pattern = _build_ilike_pattern(safe_term)
            db_query = db_query.or_(
                ",".join(
                    [f"{field}.ilike.{pattern}" for field in CANDIDATE_GLOBAL_SEARCH_FIELDS]
                )
            )

    db_query = _apply_candidate_filters(db_query, filters)

    needs_python_filter = bool(skill_filters) or experience_min is not None
    if needs_python_filter:
        response = db_query.order("created_at", desc=True).execute()
        filtered_items = [
            item
            for item in (response.data or [])
            if _candidate_matches_skills(item, skill_filters)
            and _candidate_matches_experience_min(item, experience_min)
        ]
        total_items = len(filtered_items)
        total_pages = (total_items + page_size - 1) // page_size if total_items else 0
        return {
            "items": filtered_items[start : start + page_size],
            "page": page,
            "page_size": page_size,
            "total_items": total_items,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_previous": page > 1,
        }

    response = db_query.order("created_at", desc=True).range(start, end).execute()
    total_items = int(response.count or 0)
    total_pages = (total_items + page_size - 1) // page_size if total_items else 0
    return {
        "items": response.data or [],
        "page": page,
        "page_size": page_size,
        "total_items": total_items,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_previous": page > 1,
    }


def get_candidate_by_id(candidate_id: str) -> dict[str, Any] | None:
    client = _client()
    response = (
        client.table("candidates")
        .select("*")
        .eq("id", candidate_id)
        .limit(1)
        .execute()
    )
    if response.data:
        return response.data[0]
    return None


def delete_candidate(candidate_id: str, resume_path: str | None = None) -> None:
    client = _client()
    if resume_path:
        resume_path = _normalize_resume_storage_path(resume_path)
        try:
            if resume_path:
                client.storage.from_("candidate_resumes").remove([resume_path])
        except Exception:
            pass
    client.table("candidates").delete().eq("id", candidate_id).execute()
