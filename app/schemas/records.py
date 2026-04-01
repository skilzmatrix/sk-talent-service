"""Schemas for persisted records."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ResumeRecord(BaseModel):
    file_name: str
    summary: str
    skills: list[Any] = Field(default_factory=list)
    experiences: list[Any] = Field(default_factory=list)
    projects: list[Any] = Field(default_factory=list)
    certifications: list[Any] = Field(default_factory=list)


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
    linkedin_profile: str = ""
    domain_industry: str = ""
    work_authorization: str = ""
    preferred_location: str = ""
    open_to_relocation: str = ""
    expected_salary: str = ""
    employment_type: str = ""
    summary: str
    skills: list[str] = Field(default_factory=list)
