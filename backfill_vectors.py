#!/usr/bin/env python3
"""Backfill Pinecone vectors for all existing candidates."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def backfill_vectors(batch_size: int = 10, max_workers: int = 5) -> None:
    """Backfill all candidates' vectors in Pinecone.
    
    Args:
        batch_size: Number of candidates to process before logging progress
        max_workers: Number of concurrent embedding/upsert operations
    """
    from app import supabase_operations, pinecone_operations
    
    logger.info("Starting vector backfill...")
    logger.info(f"Fetching all candidates from Supabase (batch processing with {max_workers} workers)...")
    
    try:
        candidates = supabase_operations.get_candidates()
    except Exception as e:
        logger.error(f"Failed to fetch candidates: {e}")
        sys.exit(1)
    
    if not candidates:
        logger.warning("No candidates found in database.")
        return
    
    logger.info(f"Found {len(candidates)} candidates to process.")
    
    total = len(candidates)
    processed = 0
    failed = 0
    skipped = 0
    start_time = time.time()
    
    # Use ThreadPoolExecutor for concurrent vectorization
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        
        for candidate in candidates:
            candidate_id = candidate.get("id")
            if not candidate_id:
                logger.warning("Skipping candidate with no ID")
                skipped += 1
                continue
            
            # Submit the vectorization task
            future = executor.submit(
                pinecone_operations.upsert_candidate_vectors,
                candidate_id,
                candidate
            )
            futures[future] = candidate_id
        
        # Process completed futures as they finish
        for future in as_completed(futures):
            candidate_id = futures[future]
            try:
                embedded_sections = future.result()
                processed += 1
                
                if embedded_sections:
                    logger.info(
                        f"[{processed}/{total}] Vectorized candidate {candidate_id}: "
                        f"{', '.join(embedded_sections)}"
                    )
                else:
                    logger.warning(
                        f"[{processed}/{total}] No embeddable sections for candidate {candidate_id}"
                    )
                    failed += 1
                
                # Log progress every batch
                if processed % batch_size == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Progress: {processed}/{total} ({100*processed/total:.1f}%) | "
                        f"Rate: {rate:.1f}/sec | Remaining: ~{(total-processed)/rate:.0f}s"
                    )
                    
            except Exception as e:
                failed += 1
                logger.error(f"Failed to vectorize candidate {candidate_id}: {e}")
    
    elapsed = time.time() - start_time
    logger.info(
        f"\n✅ Backfill complete!\n"
        f"  Processed: {processed}/{total}\n"
        f"  Failed: {failed}\n"
        f"  Skipped: {skipped}\n"
        f"  Time: {elapsed:.1f}s\n"
        f"  Rate: {processed/elapsed:.1f} candidates/sec"
    )


if __name__ == "__main__":
    # Parse CLI args
    max_workers = 5
    batch_size = 10
    
    if len(sys.argv) > 1:
        try:
            max_workers = int(sys.argv[1])
        except ValueError:
            logger.error("First arg must be max_workers (int)")
            sys.exit(1)
    
    if len(sys.argv) > 2:
        try:
            batch_size = int(sys.argv[2])
        except ValueError:
            logger.error("Second arg must be batch_size (int)")
            sys.exit(1)
    
    backfill_vectors(batch_size=batch_size, max_workers=max_workers)
