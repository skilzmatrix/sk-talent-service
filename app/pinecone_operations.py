"""Pinecone vector operations for candidate resume sections.

Embeds each resume section via Google AI REST API (gemini-embedding-001)
and upserts the resulting dense vectors directly to Pinecone.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests
from pinecone import Pinecone

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{EMBEDDING_MODEL}:embedContent"
)
EMBEDDING_DIMS = 768  # matches the Pinecone index dimension
SECTIONS = ("summary", "skills", "experiences", "projects", "certifications")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _embed_text(text: str, api_key: str) -> list[float]:
    """Embed a single string using the Google AI REST API."""
    resp = requests.post(
        EMBEDDING_API_URL,
        params={"key": api_key},
        json={
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": EMBEDDING_DIMS,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]["values"]


def _get_pinecone_index(api_key: str, index_name: str, index_host: str):
    """Return a Pinecone Index object."""
    pc = Pinecone(api_key=api_key)
    if index_host:
        return pc.Index(name=index_name, host=index_host)
    return pc.Index(name=index_name)


def _build_section_text(section: str, record: dict[str, Any]) -> str | None:
    """Convert a candidate record section into embeddable text. Returns None if empty."""
    if section == "summary":
        val = record.get("summary", "").strip()
        return f"SUMMARY: {val}" if val else None

    if section == "skills":
        skills = record.get("skills", [])
        if not skills:
            return None
        return f"SKILLS: {', '.join(skills)}"

    if section == "experiences":
        exps = record.get("experiences", [])
        if not exps:
            return None
        parts: list[str] = []
        for e in exps:
            title = e.get("title", "")
            company = e.get("company", "")
            duration = e.get("duration", "")
            responsibilities = e.get("responsibilities", [])
            header = f"{title} at {company} ({duration})"
            body = "; ".join(responsibilities) if responsibilities else ""
            parts.append(f"{header} — {body}" if body else header)
        return "WORK EXPERIENCE: " + " | ".join(parts)

    if section == "projects":
        projects = record.get("projects", [])
        if not projects:
            return None
        parts = []
        for p in projects:
            name = p.get("name", "")
            desc = p.get("description", "")
            techs = p.get("technologies", [])
            line = name
            if desc:
                line += f": {desc}"
            if techs:
                line += f" — {', '.join(techs)}"
            parts.append(line)
        return "PROJECTS: " + " | ".join(parts)

    if section == "certifications":
        certs = record.get("certifications", [])
        if not certs:
            return None
        parts = []
        for c in certs:
            name = c.get("name", "")
            issuer = c.get("issuer", "")
            date = c.get("date", "")
            line = name
            if issuer:
                line += f" — {issuer}"
            if date:
                line += f" ({date})"
            parts.append(line)
        return "CERTIFICATIONS: " + " | ".join(parts)

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def upsert_candidate_vectors(candidate_id: str, record: dict[str, Any]) -> list[str]:
    """Embed each resume section and upsert vectors to Pinecone.

    Returns the list of section names that were successfully embedded.
    """
    gemini_key = (
        os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("API_KEY", "").strip()
    )
    pinecone_key = os.environ.get("PINECONE_API_KEY", "").strip()
    index_name = os.environ.get("PINECONE_INDEX_NAME", "").strip()
    index_host = os.environ.get("PINECONE_INDEX_HOST", "").strip()

    if not gemini_key:
        raise RuntimeError("GEMINI_API_KEY not configured.")
    if not pinecone_key or not index_name:
        raise RuntimeError(
            "Pinecone not configured. Set PINECONE_API_KEY and PINECONE_INDEX_NAME in backend/.env"
        )

    metadata_base = {
        "candidate_id": candidate_id,
        "full_name": record.get("full_name", ""),
        "job_role": record.get("job_role", ""),
    }

    vectors: list[dict[str, Any]] = []
    embedded_sections: list[str] = []
    for section in SECTIONS:
        text = _build_section_text(section, record)
        if not text:
            continue
        try:
            values = _embed_text(text, gemini_key)
        except Exception:
            logger.exception("Failed to embed section '%s' for candidate %s", section, candidate_id)
            continue
        vectors.append(
            {
                "id": f"{candidate_id}_{section}",
                "values": values,
                "metadata": {**metadata_base, "section": section, "text": text},
            }
        )
        embedded_sections.append(section)

    if not vectors:
        logger.warning("No embeddable sections for candidate %s", candidate_id)
        return []

    index = _get_pinecone_index(pinecone_key, index_name, index_host)
    index.upsert(vectors=vectors)
    logger.info("Upserted %d vectors for candidate %s", len(vectors), candidate_id)
    return embedded_sections


def query_candidates(query_text: str, top_k: int = 25) -> list[dict[str, Any]]:
    """Embed the query text and search Pinecone for matching candidate sections.

    Returns raw Pinecone matches (id, score, metadata) grouped and ranked by candidate.
    We fetch top_k individual section matches, then aggregate by candidate_id
    and return the top candidates sorted by best match score.
    """
    gemini_key = (
        os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("API_KEY", "").strip()
    )
    pinecone_key = os.environ.get("PINECONE_API_KEY", "").strip()
    index_name = os.environ.get("PINECONE_INDEX_NAME", "").strip()
    index_host = os.environ.get("PINECONE_INDEX_HOST", "").strip()

    if not gemini_key:
        raise RuntimeError("GEMINI_API_KEY not configured.")
    if not pinecone_key or not index_name:
        raise RuntimeError("Pinecone not configured.")

    query_vector = _embed_text(query_text, gemini_key)
    index = _get_pinecone_index(pinecone_key, index_name, index_host)
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    candidates: dict[str, dict[str, Any]] = {}
    for match in results.get("matches", []):
        meta = match.get("metadata", {})
        cid = meta.get("candidate_id", "")
        if not cid:
            continue
        score = match.get("score", 0)
        section = meta.get("section", "")
        section_text = meta.get("text", "")

        if cid not in candidates:
            candidates[cid] = {
                "candidate_id": cid,
                "full_name": meta.get("full_name", ""),
                "job_role": meta.get("job_role", ""),
                "best_score": score,
                "matched_sections": [],
            }

        entry = candidates[cid]
        if score > entry["best_score"]:
            entry["best_score"] = score
        entry["matched_sections"].append({
            "section": section,
            "score": round(score, 4),
            "text": section_text,
        })

    # Deduplicate sections per candidate (keep best score per section)
    finalized: list[dict[str, Any]] = []
    num_section_types = len(SECTIONS)
    for entry in candidates.values():
        by_section: dict[str, dict[str, Any]] = {}
        for m in entry["matched_sections"]:
            sec = m["section"]
            if sec not in by_section or m["score"] > by_section[sec]["score"]:
                by_section[sec] = m
        sections = sorted(by_section.values(), key=lambda x: x["score"], reverse=True)
        scores = [m["score"] for m in sections]
        best = max(scores) if scores else 0.0
        avg = sum(scores) / len(scores) if scores else 0.0
        breadth = min(len(sections) / num_section_types, 1.0)
        # Composite: peak similarity + consistency across sections + coverage bonus
        composite = 0.52 * best + 0.33 * avg + 0.15 * breadth
        finalized.append({
            "candidate_id": entry["candidate_id"],
            "full_name": entry["full_name"],
            "job_role": entry["job_role"],
            "best_score": round(best, 4),
            "avg_score": round(avg, 4),
            "composite_score": round(composite, 4),
            "sections_matched": len(sections),
            "matched_sections": sections,
        })

    finalized.sort(key=lambda c: c["composite_score"], reverse=True)
    return finalized


def delete_candidate_vectors(candidate_id: str) -> None:
    """Delete all section vectors for a candidate from Pinecone."""
    pinecone_key = os.environ.get("PINECONE_API_KEY", "").strip()
    index_name = os.environ.get("PINECONE_INDEX_NAME", "").strip()
    index_host = os.environ.get("PINECONE_INDEX_HOST", "").strip()

    if not pinecone_key or not index_name:
        logger.warning("Pinecone not configured — skipping vector deletion for %s", candidate_id)
        return

    vector_ids = [f"{candidate_id}_{section}" for section in SECTIONS]
    index = _get_pinecone_index(pinecone_key, index_name, index_host)
    index.delete(ids=vector_ids)
    logger.info("Deleted %d vectors for candidate %s", len(vector_ids), candidate_id)
