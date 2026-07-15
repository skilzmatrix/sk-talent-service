#!/usr/bin/env python3
"""Backfill candidates.experience (years) with Gemini, one candidate at a time.

Uses each candidate's stored summary + experiences from Supabase, asks Gemini
for total professional years, then updates only the experience column.

Usage (from backend/):
  python backfill_experience.py              # dry-run (no DB writes)
  python backfill_experience.py --confirm    # write updates
  python backfill_experience.py --confirm --force   # recompute even if set
  python backfill_experience.py --confirm --limit 5
  python backfill_experience.py --confirm --delay 1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SCHEMA_EXPERIENCE_YEARS: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "experience": {
            "type": "STRING",
            "description": (
                "Total years of professional work experience as a number string "
                '(e.g. "3", "7.5", "12"). Empty string if unknown.'
            ),
        },
    },
    "required": ["experience"],
}


def _has_experience(value: Any) -> bool:
    return bool(str(value or "").strip())


def _build_profile_text(candidate: dict[str, Any]) -> str:
    """Build resume-like text from stored candidate fields for Gemini."""
    lines: list[str] = [
        f"Name: {candidate.get('full_name') or ''}",
        f"Role: {candidate.get('job_role') or ''}",
        f"Summary: {candidate.get('summary') or ''}",
        "",
        "Work Experience:",
    ]

    experiences = candidate.get("experiences") or []
    if not isinstance(experiences, list) or not experiences:
        lines.append("(no structured work experience on record)")
    else:
        for exp in experiences:
            if not isinstance(exp, dict):
                continue
            title = str(exp.get("title") or "").strip() or "Unknown title"
            company = str(exp.get("company") or "").strip() or "Unknown company"
            duration = str(exp.get("duration") or "").strip() or "Unknown duration"
            lines.append(f"- {title} at {company} ({duration})")
            responsibilities = exp.get("responsibilities") or []
            if isinstance(responsibilities, list):
                for item in responsibilities[:8]:
                    text = str(item or "").strip()
                    if text:
                        lines.append(f"  • {text}")

    return "\n".join(lines).strip()


def _normalize_experience_years(raw: Any) -> str:
    value = str(raw or "").strip()
    if not value:
        return ""
    # Accept "7.5 years" → "7.5"
    lowered = value.lower().replace("years", "").replace("year", "").replace("yrs", "").replace("yr", "")
    cleaned = lowered.replace(",", "").strip()
    try:
        number = float(cleaned)
    except ValueError:
        return value
    if number < 0 or number > 80:
        return ""
    if number == int(number):
        return str(int(number))
    return f"{number:.1f}".rstrip("0").rstrip(".") if "." in f"{number:.1f}" else f"{number:.1f}"


def _estimate_experience_with_gemini(profile_text: str) -> str:
    from google import genai
    from google.genai import types

    from app.core.config import GEMINI_API_KEY, GEMINI_MODEL

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY (or API_KEY) is not set.")

    prompt = f"""You are an expert HR data extractor. Estimate total professional years of experience
from the candidate profile below.

Rules:
- Compute total professional years from work history dates/durations.
- Do not double-count overlapping roles.
- Round to one decimal when needed (e.g. "3.5").
- Prefer whole numbers when close (e.g. "7" not "7.0").
- Use an empty string "" if experience cannot be determined.
- Return JSON matching the schema with only the "experience" field.

Candidate profile:
---
{profile_text}
---"""

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SCHEMA_EXPERIENCE_YEARS,
        ),
    )
    raw = (response.text or "").strip()
    parsed = json.loads(raw)
    return _normalize_experience_years(parsed.get("experience"))


def _update_experience(candidate_id: str, experience: str) -> None:
    from app.supabase_operations import _client

    response = (
        _client()
        .table("candidates")
        .update({"experience": experience})
        .eq("id", candidate_id)
        .execute()
    )
    if not response.data:
        raise RuntimeError(f"No row updated for candidate {candidate_id}")


def backfill_experience(
    *,
    confirm: bool = False,
    force: bool = False,
    limit: int | None = None,
    delay: float = 0.5,
) -> None:
    from app import supabase_operations

    logger.info("Fetching candidates from Supabase...")
    try:
        candidates = supabase_operations.get_candidates()
    except Exception as exc:
        logger.error("Failed to fetch candidates: %s", exc)
        sys.exit(1)

    if not candidates:
        logger.warning("No candidates found.")
        return

    # Oldest first so new uploads keep priority if interrupted mid-run
    candidates = sorted(candidates, key=lambda c: str(c.get("created_at") or ""))

    pending: list[dict[str, Any]] = []
    skipped_existing = 0
    skipped_no_data = 0

    for candidate in candidates:
        if not candidate.get("id"):
            continue
        if not force and _has_experience(candidate.get("experience")):
            skipped_existing += 1
            continue
        profile_text = _build_profile_text(candidate)
        experiences = candidate.get("experiences") or []
        summary = str(candidate.get("summary") or "").strip()
        if (not isinstance(experiences, list) or not experiences) and not summary:
            skipped_no_data += 1
            logger.warning(
                "Skipping %s (%s): no summary or experiences to parse",
                candidate.get("id"),
                candidate.get("full_name") or "Unknown",
            )
            continue
        pending.append(candidate)

    if limit is not None:
        pending = pending[: max(0, limit)]

    mode = "WRITE" if confirm else "DRY-RUN"
    logger.info(
        "Mode=%s | to_process=%s | skip_existing=%s | skip_no_data=%s | delay=%ss",
        mode,
        len(pending),
        skipped_existing,
        skipped_no_data,
        delay,
    )

    if not pending:
        logger.info("Nothing to backfill.")
        return

    updated = 0
    failed = 0
    empty = 0
    start = time.time()

    for index, candidate in enumerate(pending, start=1):
        candidate_id = str(candidate["id"])
        name = candidate.get("full_name") or "Unknown"
        profile_text = _build_profile_text(candidate)

        try:
            years = _estimate_experience_with_gemini(profile_text)
        except Exception as exc:
            failed += 1
            logger.error("[%s/%s] Gemini failed for %s (%s): %s", index, len(pending), name, candidate_id, exc)
            if delay > 0:
                time.sleep(delay)
            continue

        if not years:
            empty += 1
            logger.warning(
                "[%s/%s] %s (%s): Gemini returned empty experience",
                index,
                len(pending),
                name,
                candidate_id,
            )
        else:
            logger.info(
                "[%s/%s] %s (%s): experience=%s%s",
                index,
                len(pending),
                name,
                candidate_id,
                years,
                "" if confirm else " [dry-run]",
            )
            if confirm:
                try:
                    _update_experience(candidate_id, years)
                    updated += 1
                except Exception as exc:
                    failed += 1
                    logger.error("Failed to update %s: %s", candidate_id, exc)

        if delay > 0 and index < len(pending):
            time.sleep(delay)

    elapsed = time.time() - start
    logger.info(
        "\nBackfill complete (%s)\n"
        "  Processed: %s\n"
        "  Updated:   %s\n"
        "  Empty:     %s\n"
        "  Failed:    %s\n"
        "  Skipped existing: %s\n"
        "  Skipped no data:  %s\n"
        "  Time: %.1fs",
        mode,
        len(pending),
        updated,
        empty,
        failed,
        skipped_existing,
        skipped_no_data,
        elapsed,
    )
    if not confirm:
        logger.info("Dry-run only. Re-run with --confirm to write experience values.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill candidate experience years via Gemini.")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Persist updates to Supabase (default is dry-run).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even when experience is already set.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N candidates.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to wait between Gemini calls (default: 0.5).",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        logger.error("--limit must be >= 1")
        sys.exit(1)
    if args.delay < 0:
        logger.error("--delay must be >= 0")
        sys.exit(1)

    backfill_experience(
        confirm=args.confirm,
        force=args.force,
        limit=args.limit,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
