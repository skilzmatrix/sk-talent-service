"""Gemini prompts and handlers (mirrors frontend gemini.service.ts)."""

from __future__ import annotations

import json
from typing import Any, Literal

from google import genai
from google.genai import types

MODEL = "gemini-2.5-flash"

SchemaCandidateProfile: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "full_name": {"type": "STRING", "description": "Candidate's full name."},
        "job_role": {"type": "STRING", "description": "Current or most recent job title / target role."},
        "email": {"type": "STRING", "description": "Email address. Empty string if absent."},
        "phone": {"type": "STRING", "description": "Phone number. Empty string if absent."},
        "location": {"type": "STRING", "description": "City, State or City, Country. Empty string if absent."},
        "linkedin_profile": {"type": "STRING", "description": "Full LinkedIn URL. Empty string if absent."},
        "domain_industry": {"type": "STRING", "description": "Primary domain or industry (e.g. 'Data Engineering, IT Consulting')."},
        "work_authorization": {
            "type": "STRING",
            "description": "Work authorisation status. One of: Initial OPT, STEM OPT, H1-B, Green Card, US Citizen. Empty string if not found.",
        },
        "preferred_location": {
            "type": "STRING",
            "description": "Preferred work location (e.g. Remote, New York). Empty string if not found.",
        },
        "open_to_relocation": {
            "type": "STRING",
            "description": "Yes or No. Empty string if not mentioned.",
        },
        "expected_salary": {
            "type": "STRING",
            "description": "Expected salary range. Empty string if not found.",
        },
        "employment_type": {
            "type": "STRING",
            "description": "Employment type: Full-time, Part-time, Contract, or Freelance. Empty string if not found.",
        },
        "summary": {"type": "STRING", "description": "2-4 sentence professional summary."},
        "skills": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "Key technical and soft skills.",
        },
        "experiences": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "title": {"type": "STRING"},
                    "company": {"type": "STRING"},
                    "duration": {
                        "type": "STRING",
                        "description": 'e.g., "Jan 2020 - Present"',
                    },
                    "responsibilities": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                },
                "required": ["title", "company", "duration", "responsibilities"],
            },
            "description": "Work experiences with title, company, duration, and responsibilities.",
        },
        "projects": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "description": {"type": "STRING"},
                    "technologies": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "url": {"type": "STRING"},
                },
                "required": ["name"],
            },
            "description": "Personal or professional projects.",
        },
        "certifications": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "issuer": {"type": "STRING"},
                    "date": {"type": "STRING"},
                },
                "required": ["name"],
            },
            "description": "Certifications, licenses, or credentials.",
        },
    },
    "required": ["full_name", "job_role", "summary", "skills", "experiences"],
}

SchemaResumeAnalysis: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "summary": {
            "type": "STRING",
            "description": "A 2-3 sentence summary of the candidate's profile.",
        },
        "skills": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "A list of key technical and soft skills.",
        },
        "experiences": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "title": {"type": "STRING"},
                    "company": {"type": "STRING"},
                    "duration": {
                        "type": "STRING",
                        "description": 'e.g., "Jan 2020 - Present"',
                    },
                    "responsibilities": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                },
                "required": ["title", "company", "duration", "responsibilities"],
            },
        },
        "projects": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "description": {"type": "STRING"},
                    "technologies": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "url": {"type": "STRING"},
                },
                "required": ["name"],
            },
            "description": "Personal or professional projects.",
        },
        "certifications": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "issuer": {"type": "STRING"},
                    "date": {"type": "STRING"},
                },
                "required": ["name"],
            },
            "description": "Certifications, licenses, or credentials.",
        },
        "fitScore": {
            "type": "INTEGER",
            "description": "Only populated when a job description is provided. A score from 0–100 representing how well the candidate matches the job description.",
        },
        "fitReasoning": {
            "type": "STRING",
            "description": "Only populated when a job description is provided. 2-3 sentences explaining the fit score.",
        },
        "matchedSkills": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "Only populated when a job description is provided. Skills the candidate has that are explicitly required or desired by the JD.",
        },
        "missingSkills": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "Only populated when a job description is provided. Skills required or desired by the JD that the candidate does not appear to have.",
        },
    },
    "required": ["summary", "skills", "experiences"],
}

def _generate_text(client: genai.Client, prompt: str) -> str:
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return (response.text or "").strip()


def _generate_json(
    client: genai.Client, prompt: str, schema: dict[str, Any]
) -> Any:
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    raw = (response.text or "").strip()
    return json.loads(raw)


def invoke(
    client: genai.Client, operation: str, payload: dict[str, Any]
) -> tuple[Literal["text", "json"], str | Any]:
    if operation == "generateJobDescription":
        title = payload.get("title", "")
        responsibilities = payload.get("responsibilities", "")
        prompt = f"""Generate a professional and detailed job description for the role of "{title}".
      Key responsibilities include: {responsibilities}.
      The description should include:
      - A brief company introduction (use a generic placeholder).
      - A role summary.
      - A list of key responsibilities.
      - A list of required qualifications and skills.
      - A list of preferred qualifications.
      - A concluding statement about equal opportunity.
      Format the output in Markdown."""
        return "text", _generate_text(client, prompt)

    if operation == "createCandidateProfile":
        resume_text = payload.get("resumeText", "")
        prompt = f"""You are an expert HR data extractor. Parse the following resume and extract the complete candidate profile into the required JSON schema.

Extract ALL of the following:
- Contact and personal information (name, email, phone, location, LinkedIn, etc.)
- Professional summary (2-4 sentences)
- All technical and soft skills
- All work experiences with title, company, duration, and key responsibilities for each role
- All personal or professional projects with name, description, technologies, and URL if available
- All certifications, licenses, or credentials with name, issuer, and date if available

For any text field that is not present in the resume, use an empty string "".
For open_to_relocation, choose from: Yes, No, or "" if not mentioned.
For employment_type, choose from: Full-time, Part-time, Contract, Freelance, or "" if not mentioned.
For work_authorization, choose from: Initial OPT, STEM OPT, H1-B, Green Card, US Citizen, or "" if not mentioned.
For experiences, projects, or certifications not found, use an empty array [].

Resume:
---
{resume_text}
---"""
        return "json", _generate_json(client, prompt, SchemaCandidateProfile)

    if operation == "analyzeResume":
        resume_text = payload.get("resumeText", "")
        job_description = payload.get("jobDescription", "").strip()
        if job_description:
            prompt = f"""You are an expert technical recruiter. Analyze the following resume against the provided job description.

      Extract all key information in JSON format and also populate these JD-match fields:
      - fitScore: an integer 0–100 representing overall candidate fit for the role
      - fitReasoning: 2-3 sentences explaining why you gave that score
      - matchedSkills: skills the candidate has that the JD explicitly requires or values
      - missingSkills: skills the JD requires or values that the candidate does not appear to have

      Job Description:
      ---
      {job_description}
      ---

      Resume:
      ---
      {resume_text}
      ---"""
        else:
            prompt = f"""Analyze the following resume text and extract the key information in JSON format.
      Resume:
      ---
      {resume_text}
      ---"""
        return "json", _generate_json(client, prompt, SchemaResumeAnalysis)

    if operation == "compareCandidates":
        job_description = payload.get("jobDescription", "")
        resume_a = payload.get("resumeAText", "")
        resume_b = payload.get("resumeBText", "")
        prompt = f"""Act as an expert technical recruiter. You are tasked with comparing two candidates for a role based on the provided job description and their resumes.

      **Job Description:**
      ---
      {job_description}
      ---

      **Candidate A Resume:**
      ---
      {resume_a}
      ---

      **Candidate B Resume:**
      ---
      {resume_b}
      ---

      Please provide a detailed comparison in Markdown format. Your analysis should include:
      1.  **Overall Summary:** A brief, high-level summary of each candidate's suitability for the role.
      2.  **Side-by-Side Comparison:** A table comparing the candidates on key aspects from the job description (e.g., required skills, years of experience, specific technologies). Rate each candidate on each aspect (e.g., "Strong Match", "Partial Match", "Not Mentioned").
      3.  **Strengths & Weaknesses:** A bulleted list of strengths and potential weaknesses for each candidate.
      4.  **Final Recommendation:** Your final recommendation on which candidate seems to be a better fit and why."""
        return "text", _generate_text(client, prompt)

    if operation == "optimizeJobAd":
        job_ad_text = payload.get("jobAdText", "")
        prompt = f"""Act as a Diversity, Equity, and Inclusion (DEI) consultant and a senior copywriter specializing in recruitment marketing. Analyze the following job advertisement.

      **Original Job Ad:**
      ---
      {job_ad_text}
      ---

      Provide a detailed analysis and an improved version in Markdown format. Your response should have two main sections:
      1.  **Analysis and Recommendations:**
          - **Inclusivity & Bias:** Identify any words or phrases that might be non-inclusive, gender-coded, or could discourage certain groups from applying. Explain why and suggest alternatives.
          - **Clarity & Engagement:** Assess the ad's readability, tone, and ability to attract top talent. Suggest improvements to make it more compelling and clear.
          - **Structure:** Recommend changes to the structure for better flow (e.g., leading with benefits, clarifying the role's impact).
      2.  **Optimized Job Ad:**
          - Provide the full, rewritten job advertisement incorporating all your recommendations."""
        return "text", _generate_text(client, prompt)

    raise ValueError(f"Unknown operation: {operation}")
