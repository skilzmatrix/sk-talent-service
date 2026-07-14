#!/usr/bin/env python3
"""Find duplicate candidates by email and phone."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def find_duplicates() -> None:
    """Find and report duplicate candidates."""
    from app import supabase_operations
    
    logger.info("Fetching all candidates...")
    candidates = supabase_operations.get_candidates()
    
    if not candidates:
        logger.warning("No candidates found.")
        return
    
    logger.info(f"Found {len(candidates)} candidates. Analyzing for duplicates...")
    
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
    
    # Find duplicates
    email_dups = {k: v for k, v in email_groups.items() if len(v) > 1}
    phone_dups = {k: v for k, v in phone_groups.items() if len(v) > 1}
    
    total_dup_count = 0
    
    if email_dups:
        logger.info(f"\n📧 Email Duplicates ({len(email_dups)} groups):")
        for email, group in sorted(email_dups.items()):
            logger.info(f"  Email: {email} ({len(group)} candidates)")
            for cand in group:
                logger.info(
                    f"    - ID: {cand['id']}, Name: {cand.get('full_name', 'N/A')}, Phone: {cand.get('phone', 'N/A')}"
                )
            total_dup_count += len(group) - 1
    
    if phone_dups:
        logger.info(f"\n📱 Phone Duplicates ({len(phone_dups)} groups):")
        for phone, group in sorted(phone_dups.items()):
            logger.info(f"  Phone: {phone} ({len(group)} candidates)")
            for cand in group:
                logger.info(
                    f"    - ID: {cand['id']}, Name: {cand.get('full_name', 'N/A')}, Email: {cand.get('email', 'N/A')}"
                )
            total_dup_count += len(group) - 1
    
    if not email_dups and not phone_dups:
        logger.info("✅ No duplicates found!")
        return
    
    logger.info(f"\n⚠️  Total duplicate records to clean up: {total_dup_count}")


if __name__ == "__main__":
    find_duplicates()
