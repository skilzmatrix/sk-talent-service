"""Pinecone vector operations for candidate resume sections.

Embeds each resume section via Google AI REST API (gemini-embedding-001)
and upserts the resulting dense vectors directly to Pinecone.
"""

from __future__ import annotations

import logging
import os
import re
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

# Relative importance for role-fit (experience + skills weighted highest)
SECTION_WEIGHTS: dict[str, float] = {
    "experiences": 0.30,
    "skills": 0.28,
    "summary": 0.18,
    "projects": 0.16,
    "certifications": 0.08,
}

# Hybrid: dense semantic similarity + lexical JD term coverage
_SEMANTIC_BLEND = (0.44, 0.38, 0.18)  # best, weighted_avg, breadth
# Default share of **keyword (lexical) overlap** in final score; semantic = 1 - this.
# Override with env `TALENT_SEARCH_KEYWORD_WEIGHT` or per-request `keyword_weight`.
_DEFAULT_KEYWORD_WEIGHT = 0.44

_STOPWORDS = frozenset(
    """
    a an the and or but if in on at to for of is are was were be been being
    as by with from that this these those it its we you they he she them their
    our your my me us will can could should would may might must shall do does did
    have has had not no yes all any some both each few more most other such than
    then so very just also only into about over after before between through
    during under again further once here there when where why how what which who
    whom about into onto upon out off down up off am being been
    """.split()
)
_MAX_JD_TOKENS = 120


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens; drops very short tokens and common stopwords."""
    words = re.findall(r"[a-z0-9][a-z0-9+#.\-]*", text.lower())
    out: set[str] = set()
    for w in words:
        w = w.strip(".-+")
        if len(w) < 2 or w in _STOPWORDS:
            continue
        out.add(w)
    return out


def _jd_tokens_for_overlap(jd: str) -> set[str]:
    """Cap JD vocabulary so long postings don't crush lexical scores."""
    toks = _tokenize(jd)
    if len(toks) <= _MAX_JD_TOKENS:
        return toks
    # Prefer longer tokens (often technologies: kubernetes, typescript)
    ranked = sorted(toks, key=len, reverse=True)
    return set(ranked[:_MAX_JD_TOKENS])


def _lexical_coverage(jd: str, resume_corpus: str) -> float:
    """Share of informative JD terms that appear in resume text [0, 1]."""
    jd_toks = _jd_tokens_for_overlap(jd)
    if not jd_toks:
        return 0.0
    resume_toks = _tokenize(resume_corpus)
    if not resume_toks:
        return 0.0
    hits = len(jd_toks & resume_toks)
    return hits / len(jd_toks)


def _weighted_section_avg(sections: list[dict[str, Any]]) -> float:
    """Mean cosine score weighted by SECTION_WEIGHTS for matched sections only."""
    num = 0.0
    den = 0.0
    for m in sections:
        sec = m.get("section", "")
        w = SECTION_WEIGHTS.get(sec, 0.12)
        num += w * float(m["score"])
        den += w
    return num / den if den > 0 else 0.0


def _semantic_composite(best: float, weighted_avg: float, breadth: float) -> float:
    a, b, c = _SEMANTIC_BLEND
    return a * best + b * weighted_avg + c * breadth


def resolve_keyword_weight(explicit: float | None) -> tuple[float, float]:
    """Return (semantic_weight, keyword_weight) for the final hybrid; both in [0, 1], sum to 1."""
    if explicit is not None:
        kw = max(0.0, min(1.0, float(explicit)))
    else:
        raw = os.environ.get("TALENT_SEARCH_KEYWORD_WEIGHT", "").strip()
        if raw:
            try:
                kw = max(0.0, min(1.0, float(raw)))
            except ValueError:
                kw = _DEFAULT_KEYWORD_WEIGHT
        else:
            kw = _DEFAULT_KEYWORD_WEIGHT
    sem = 1.0 - kw
    return sem, kw


def _hybrid_score(semantic: float, lexical: float, w_sem: float, w_kw: float) -> float:
    return w_sem * semantic + w_kw * lexical


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


def query_candidates(
    query_text: str,
    top_k: int = 25,
    *,
    keyword_weight: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Embed the JD, query Pinecone, rank by hybrid score (semantic + keyword overlap).

    ``keyword_weight`` is the fraction of the final score from lexical overlap (0–1).
    If omitted, uses ``TALENT_SEARCH_KEYWORD_WEIGHT`` or the module default.

    Returns ``(results, {"semantic": w_sem, "keyword": w_kw})`` for API transparency.
    """
    w_sem, w_kw = resolve_keyword_weight(keyword_weight)
    ranking_weights = {"semantic": round(w_sem, 4), "keyword": round(w_kw, 4)}

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
    for match in results.get("matches", []) or []:
        meta = match.get("metadata") or {} if hasattr(match, "get") else {}
        if not isinstance(meta, dict):
            meta = {}
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
        wavg = _weighted_section_avg(sections)
        semantic = _semantic_composite(best, wavg, breadth)
        corpus = " ".join(str(m.get("text", "")) for m in sections)
        lexical = _lexical_coverage(query_text, corpus)
        hybrid = _hybrid_score(semantic, lexical, w_sem, w_kw)
        finalized.append({
            "candidate_id": entry["candidate_id"],
            "full_name": entry["full_name"],
            "job_role": entry["job_role"],
            "best_score": round(best, 4),
            "avg_score": round(avg, 4),
            "composite_score": round(hybrid, 4),
            "semantic_composite": round(semantic, 4),
            "lexical_score": round(lexical, 4),
            "sections_matched": len(sections),
            "matched_sections": sections,
        })

    finalized.sort(key=lambda c: c["composite_score"], reverse=True)
    return finalized, ranking_weights


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
