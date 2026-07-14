#!/usr/bin/env python3
"""Remove duplicate candidates, keeping the earliest (by created_at)."""

from __future__ import annotations

import logging
import sys
from collections import defaultdict
from typing import Any

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def cleanup_duplicates(dry_run: bool = True) -> None:
    """Remove duplicate candidates, keeping the first one."""
    from app import supabase_operations, pinecone_operations
    
    logger.info("Fetching all candidates...")
    candidates = supabase_operations.get_candidates()
    
    if not candidates:
        logger.warning("No candidates found.")
        return
    
    # Group by email (case-insensitive)
    email_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    phone_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    
    for cand in candidates:
        email = (cand.get("email") or "").strip().lower()
        phone = (cand.get("phone") or "").strip()
        
        if email:
            email_groups[email].append(cand)
        if phone:
            phone_groups[phone].append(cand)
    
    # Collect duplicates to delete (keeping LATEST, deleting the rest)
    to_delete_ids: set[str] = set()
    
    for email, group in email_groups.items():
        if len(group) > 1:
            # Sort by created_at descending, keep last (most recent)
            sorted_group = sorted(group, key=lambda c: c.get("created_at", ""), reverse=True)
            for cand in sorted_group[1:]:
                to_delete_ids.add(cand["id"])
    
    for phone, group in phone_groups.items():
        if len(group) > 1:
            sorted_group = sorted(group, key=lambda c: c.get("created_at", ""), reverse=True)
            for cand in sorted_group[1:]:
                to_delete_ids.add(cand["id"])
    
    if not to_delete_ids:
        logger.info("✅ No duplicates found!")
        return
    
    logger.info(f"Found {len(to_delete_ids)} duplicate records to delete.")
    
    if dry_run:
        logger.info("\n📋 DRY RUN - Would delete (keeping MOST RECENT resume):")
        for cand_id in sorted(to_delete_ids):
            cand = next((c for c in candidates if c["id"] == cand_id), {})
            logger.info(
                f"  - {cand['id']}: {cand.get('full_name', 'N/A')} "
                f"({cand.get('email', 'N/A')}, {cand.get('phone', 'N/A')})"
            )
        logger.info("\nRun with --confirm to actually delete.")
        return
    
    # Actually delete
    logger.info("🗑️  Deleting duplicates...")
    deleted = 0
    failed = 0
    
    for cand_id in to_delete_ids:
        cand = next((c for c in candidates if c["id"] == cand_id), {})
        try:
            # Delete from Supabase
            supabase_operations.delete_candidate(cand_id, cand.get("resume_url"))
            deleted += 1
            logger.info(f"✅ Deleted {cand['full_name']} ({cand_id})")
        except Exception as e:
            failed += 1
            logger.error(f"❌ Failed to delete {cand_id}: {e}")
    
    logger.info(f"\n🎯 Cleanup complete!")
    logger.info(f"  Deleted: {deleted}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Remaining candidates: {len(candidates) - deleted}")


if __name__ == "__main__":
    dry_run = "--confirm" not in sys.argv
    if not dry_run:
        logger.warning("⚠️  Running in CONFIRM mode - will actually delete!")
    cleanup_duplicates(dry_run=dry_run)
