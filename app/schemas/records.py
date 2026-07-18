"""Schemas for persisted records."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ResumeRecord(BaseModel):
    file_name: str
    summary: str
    experience: str = ""
    skills: list[Any] = Field(default_factory=list)
    experiences: list[Any] = Field(default_factory=list)
    projects: list[Any] = Field(default_factory=list)
    certifications: list[Any] = Field(default_factory=list)
    resume_url: str = ""


class JobDescriptionRecord(BaseModel):
    job_title: str
    responsibilities: str
    content: str


class CandidateRecord(BaseModel):
    full_name: str
    job_role: str
    email: str = ""
    phone: str = ""
    location: str = ""
    city: str = ""
    state: str = ""
    linkedin_profile: str = ""
    domain_industry: str = ""
    work_authorization: str = ""
    experience: str = ""
    preferred_location: str = ""
    open_to_relocation: str = ""
    expected_salary: str = ""
    employment_type: str = ""
    summary: str
    skills: list[str] = Field(default_factory=list)
    experiences: list[Any] = Field(default_factory=list)
    projects: list[Any] = Field(default_factory=list)
    certifications: list[Any] = Field(default_factory=list)
    resume_url: str = ""


class CandidateProfileUpdate(BaseModel):
    full_name: str | None = None
    job_role: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    city: str | None = None
    state: str | None = None
    linkedin_profile: str | None = None
    domain_industry: str | None = None
    work_authorization: str | None = None
    experience: str | None = None
    preferred_location: str | None = None
    open_to_relocation: str | None = None
    expected_salary: str | None = None
    employment_type: str | None = None
    summary: str | None = None
    skills: list[str] | None = None

    @field_validator("skills", mode="before")
    @classmethod
    def _normalize_skills(cls, value: Any) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, list):
            normalized: list[str] = []
            for item in value:
                if item is None:
                    continue
                for skill in str(item).split(","):
                    stripped = skill.strip()
                    if stripped:
                        normalized.append(stripped)
            return normalized
        if isinstance(value, str):
            return [skill.strip() for skill in value.split(",") if skill.strip()]
        return value


class PaginatedRecordsResponse(BaseModel):
    items: list[dict[str, Any]] = Field(default_factory=list)
    page: int
    page_size: int
    total_items: int
    total_pages: int
    has_next: bool
    has_previous: bool
