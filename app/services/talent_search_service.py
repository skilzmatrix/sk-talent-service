"""Talent search: vector retrieval + optional Gemini rerank with reasoning."""

from __future__ import annotations

import logging
from typing import Any

from google import genai

from app.services import gemini_service, persistence_service, pinecone_service

logger = logging.getLogger(__name__)


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
        "location": row.get("location", ""),
        "experiences": experiences,
        "projects": projects,
        "certifications": (row.get("certifications") or [])[:10],
    }


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


def run_talent_search(
    query: str,
    top_k: int,
    keyword_weight: float | None,
    gemini_client: genai.Client | None,
    *,
    use_llm_rerank: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, Any]]:
    """Vector/hybrid retrieval, then optional Gemini rerank using full DB profiles.

    Returns ``(results, ranking_weights, meta)``.
    """
    results, weights = pinecone_service.search_candidates(query, top_k, keyword_weight)
    meta: dict[str, Any] = {"llm_rerank": "skipped"}

    results = [{**r, "retrieval_rank": idx + 1} for idx, r in enumerate(results)]

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
