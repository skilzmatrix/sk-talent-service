"""Talent search: vector retrieval + optional Gemini rerank with reasoning."""

from __future__ import annotations

import logging
from typing import Any

from google import genai

from app.services import gemini_service, persistence_service, pinecone_service

logger = logging.getLogger(__name__)

_WORK_AUTH_VALUES = {
    "initial opt",
    "stem opt",
    "h1-b",
    "h1b",
    "green card",
    "us citizen",
}

_WORK_AUTH_CANONICAL = {
    "h1b": "H1-B",
    "h1-b": "H1-B",
    "initial opt": "Initial OPT",
    "stem opt": "STEM OPT",
    "green card": "Green Card",
    "us citizen": "US Citizen",
}


def _truncate(s: str, max_len: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _compact_profile(row: dict[str, Any]) -> dict[str, Any]:
    """Shrink DB profile for LLM context limits."""
    experiences: list[dict[str, Any]] = []
    for e in (row.get("experiences") or [])[:6]:
        responsibilities = (e.get("responsibilities") or [])[:5]
        experiences.append(
            {
                "title": e.get("title", ""),
                "company": e.get("company", ""),
                "duration": e.get("duration", ""),
                "responsibilities": responsibilities,
            }
        )
    projects: list[dict[str, Any]] = []
    for p in (row.get("projects") or [])[:5]:
        projects.append(
            {
                "name": p.get("name", ""),
                "description": _truncate(str(p.get("description", "")), 500),
                "technologies": (p.get("technologies") or [])[:15],
            }
        )
    return {
        "full_name": row.get("full_name", ""),
        "job_role": row.get("job_role", ""),
        "summary": _truncate(str(row.get("summary", "")), 1400),
        "skills": (row.get("skills") or [])[:45],
        "domain_industry": row.get("domain_industry", ""),
        "work_authorization": row.get("work_authorization", ""),
        "experience": row.get("experience", ""),
        "location": row.get("location", ""),
        "experiences": experiences,
        "projects": projects,
        "certifications": (row.get("certifications") or [])[:10],
    }


_RESULT_METADATA_FIELDS = (
    "experience",
    "work_authorization",
    "city",
    "state",
    "location",
    "employment_type",
    "open_to_relocation",
    "domain_industry",
    "preferred_location",
    "email",
)


def _enrich_results_with_metadata(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach available candidate metadata from Supabase onto search hits."""
    enriched: list[dict[str, Any]] = []
    for result in results:
        row = dict(result)
        cid = row.get("candidate_id")
        if not cid:
            enriched.append(row)
            continue
        try:
            profile = persistence_service.get_candidate_by_id(str(cid))
        except Exception:
            logger.exception("Failed to load metadata for candidate %s", cid)
            profile = None
        if not profile:
            enriched.append(row)
            continue

        # Prefer live DB identity fields when present
        if profile.get("full_name"):
            row["full_name"] = profile.get("full_name")
        if profile.get("job_role"):
            row["job_role"] = profile.get("job_role")

        metadata: dict[str, Any] = {}
        for field in _RESULT_METADATA_FIELDS:
            value = profile.get(field)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                metadata[field] = text
        skills = profile.get("skills") or []
        if isinstance(skills, list):
            cleaned = [str(s).strip() for s in skills if str(s).strip()]
            if cleaned:
                metadata["skills"] = cleaned[:12]

        row["metadata"] = metadata
        # Flat convenience fields for clients
        for field, value in metadata.items():
            if field != "skills":
                row[field] = value
        if "skills" in metadata:
            row["skills"] = metadata["skills"]
        enriched.append(row)
    return enriched


def _merge_llm_rankings(
    hybrid_results: list[dict[str, Any]], rankings: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {r["candidate_id"]: dict(r) for r in hybrid_results}
    valid = [x for x in rankings if x.get("candidate_id") in by_id]
    valid.sort(key=lambda x: int(x.get("rank", 999)))
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for g in valid:
        cid = str(g["candidate_id"])
        if cid in seen:
            continue
        seen.add(cid)
        row = dict(by_id[cid])
        row["llm_rank"] = int(g.get("rank", 0))
        row["llm_fit_score"] = int(g.get("fit_score", 0))
        row["llm_reasoning"] = str(g.get("reasoning", "")).strip()
        row["llm_strengths"] = list(g.get("key_strengths") or [])
        row["llm_gaps"] = list(g.get("key_gaps") or [])
        merged.append(row)
    for r in hybrid_results:
        cid = r["candidate_id"]
        if cid not in seen:
            merged.append(dict(r))
    return merged


def _canonicalize_work_authorization(value: str) -> str:
    key = value.strip().lower().replace("_", " ")
    if key in _WORK_AUTH_CANONICAL:
        return _WORK_AUTH_CANONICAL[key]
    for raw, canonical in _WORK_AUTH_CANONICAL.items():
        if raw in key or key in raw:
            return canonical
    # Preserve known casing variants already stored in DB
    for allowed in ("Initial OPT", "STEM OPT", "H1-B", "Green Card", "US Citizen"):
        if value.strip().lower() == allowed.lower():
            return allowed
    return value.strip()


def _normalize_extracted_filters(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}

    out: dict[str, Any] = {}
    for field in (
        "city",
        "state",
        "location",
        "employment_type",
        "open_to_relocation",
        "domain_industry",
        "preferred_location",
    ):
        value = str(raw.get(field) or "").strip()
        if value:
            if field == "state":
                value = value.upper()[:2] if len(value) <= 3 else value
            out[field] = value

    work_auth = str(raw.get("work_authorization") or "").strip()
    if work_auth:
        canonical = _canonicalize_work_authorization(work_auth)
        if canonical.lower().replace("_", " ") in _WORK_AUTH_VALUES or canonical in {
            "Initial OPT",
            "STEM OPT",
            "H1-B",
            "Green Card",
            "US Citizen",
        }:
            out["work_authorization"] = canonical

    experience_min = raw.get("experience_min")
    try:
        years = float(experience_min)
    except (TypeError, ValueError):
        years = 0.0
    if years > 0:
        out["experience_min"] = years

    return out


def _coerce_explicit_filters(metadata_filters: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata_filters:
        return {}
    out = dict(metadata_filters)
    # Convert string experience ("5") into numeric min for Pinecone $gte
    if "experience_min" not in out and out.get("experience"):
        try:
            years = float(str(out.get("experience")).strip().replace("+", ""))
        except ValueError:
            years = 0.0
        if years > 0:
            out["experience_min"] = years
    out.pop("experience", None)
    out.pop("skills", None)
    return {k: v for k, v in out.items() if v not in (None, "", [])}


def extract_filters_from_query(
    gemini_client: genai.Client | None,
    query: str,
) -> dict[str, Any]:
    """Use Gemini to pull hard boolean/metadata filters from natural-language query."""
    if gemini_client is None or not query.strip():
        return {}
    try:
        _, parsed = gemini_service.run_operation(
            gemini_client,
            "extractTalentFilters",
            {"query": query},
        )
    except Exception:
        logger.exception("Failed to extract talent filters from query")
        return {}
    if not isinstance(parsed, dict):
        return {}
    return _normalize_extracted_filters(parsed)


def _merge_filters(
    auto_filters: dict[str, Any],
    explicit_filters: dict[str, Any],
) -> dict[str, Any]:
    """Explicit request filters override auto-extracted ones."""
    merged = dict(auto_filters)
    merged.update(explicit_filters)
    return merged


def run_talent_search(
    query: str,
    top_k: int,
    keyword_weight: float | None,
    gemini_client: genai.Client | None,
    metadata_filters: dict[str, Any] | None = None,
    *,
    use_llm_rerank: bool = True,
    auto_extract_filters: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, Any]]:
    """Vector/hybrid retrieval, then optional Gemini rerank using full DB profiles.

    Returns ``(results, ranking_weights, meta)``.
    """
    explicit = _coerce_explicit_filters(metadata_filters)
    auto: dict[str, Any] = {}
    if auto_extract_filters:
        auto = extract_filters_from_query(gemini_client, query)
    applied_filters = _merge_filters(auto, explicit)

    results, weights = pinecone_service.search_candidates(
        query,
        top_k,
        keyword_weight,
        metadata_filters=applied_filters or None,
    )
    meta: dict[str, Any] = {
        "llm_rerank": "skipped",
        "applied_filters": applied_filters,
        "auto_filters": auto,
        "filters_relaxed": False,
    }

    # If hard filters wiped the candidate pool, retry once without them.
    if not results and applied_filters:
        logger.info(
            "Talent search returned 0 hits with filters %s; retrying without metadata filters",
            applied_filters,
        )
        results, weights = pinecone_service.search_candidates(
            query,
            top_k,
            keyword_weight,
            metadata_filters=None,
        )
        meta["filters_relaxed"] = True
        meta["filters_relaxed_reason"] = (
            "No vector matches for inferred filters; showed unfiltered semantic results."
        )

    results = [{**r, "retrieval_rank": idx + 1} for idx, r in enumerate(results)]
    results = _enrich_results_with_metadata(results)

    if not results or not use_llm_rerank or gemini_client is None:
        if not use_llm_rerank:
            meta["llm_rerank"] = "disabled"
        elif gemini_client is None:
            meta["llm_rerank"] = "skipped_no_gemini"
            meta["llm_hint"] = "Configure GEMINI_API_KEY on the server to enable Gemini reranking."
        return results, weights, meta

    candidates_for_llm: list[dict[str, Any]] = []
    for r in results:
        cid = r.get("candidate_id")
        if not cid:
            continue
        # Reuse already-enriched metadata when possible; still load full profile for rerank.
        row = persistence_service.get_candidate_by_id(str(cid))
        if not row:
            candidates_for_llm.append(
                {"candidate_id": str(cid), "profile": {"note": "Profile not found in database."}}
            )
        else:
            candidates_for_llm.append(
                {"candidate_id": str(cid), "profile": _compact_profile(row)}
            )

    try:
        _, parsed = gemini_service.run_operation(
            gemini_client,
            "rerankTalentSearch",
            {"jobDescription": query, "candidates": candidates_for_llm},
        )
    except Exception as exc:
        logger.exception("Gemini talent rerank failed")
        meta["llm_rerank"] = "error"
        meta["llm_error"] = str(exc)
        return results, weights, meta

    if not isinstance(parsed, dict):
        meta["llm_rerank"] = "error"
        meta["llm_error"] = "Invalid Gemini response"
        return results, weights, meta

    rankings = parsed.get("rankings") or []
    if not isinstance(rankings, list):
        meta["llm_rerank"] = "error"
        meta["llm_error"] = "Missing rankings array"
        return results, weights, meta

    merged = _merge_llm_rankings(results, rankings)
    meta["llm_rerank"] = "ok"
    return merged, weights, meta
