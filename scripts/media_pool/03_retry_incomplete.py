"""
03_retry_incomplete.py — clears incomplete concepts from the checkpoint so that
01_run_pool_agent.py --resume will re-process them.

A concept is flagged for retry if ANY of these are true:
  - It is in the checkpoint but has NO entry in the pool
  - Its pool entry has no videos  (--missing-videos)
  - Its pool entry has no citations  (--missing-citations)
  - Its pool entry has no wikipedia  (--missing-wikipedia)

By default all four conditions are checked. Use flags to narrow the scope.

Usage:
    python scripts/media_pool/03_retry_incomplete.py --domain neuro
    python scripts/media_pool/03_retry_incomplete.py --domain neuro --missing-videos
    python scripts/media_pool/03_retry_incomplete.py --domain neuro --dry-run
"""

import argparse
import sys
import os

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from schema import load_pool, load_checkpoint, save_checkpoint

POOL_PATH_TPL = "data/media/kg_{domain}_media_pool.json"
CKPT_PATH_TPL = "data/media/checkpoint_{domain}.json"


def main():
    parser = argparse.ArgumentParser(description="Clear incomplete concepts from checkpoint for retry")
    parser.add_argument("--domain", required=True, help="Domain name (e.g. neuro, udl)")
    parser.add_argument("--missing-videos", action="store_true", help="Retry concepts with no videos")
    parser.add_argument("--missing-citations", action="store_true", help="Retry concepts with no citations")
    parser.add_argument("--missing-wikipedia", action="store_true", help="Retry concepts with no wikipedia")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be cleared without changing files")
    args = parser.parse_args()

    # If no specific flags given, check all fields
    check_all = not (args.missing_videos or args.missing_citations or args.missing_wikipedia)
    check_videos = check_all or args.missing_videos
    check_citations = check_all or args.missing_citations
    check_wikipedia = check_all or args.missing_wikipedia

    pool_path = POOL_PATH_TPL.format(domain=args.domain)
    ckpt_path = CKPT_PATH_TPL.format(domain=args.domain)

    pool = load_pool(pool_path)
    checkpoint = load_checkpoint(ckpt_path)

    if not checkpoint:
        print(f"No checkpoint found at {ckpt_path} — nothing to do.")
        return

    print(f"Pool entries    : {len(pool)}")
    print(f"Checkpointed    : {len(checkpoint)}")
    print()

    to_retry = []
    reasons = {}

    for concept in list(checkpoint.keys()):
        entry = pool.get(concept)
        r = []

        if entry is None:
            r.append("not in pool")
        else:
            if check_videos and not entry.get("videos"):
                r.append("no videos")
            if check_citations and not entry.get("citations"):
                r.append("no citations")
            if check_wikipedia and not entry.get("wikipedia"):
                r.append("no wikipedia")

        if r:
            to_retry.append(concept)
            reasons[concept] = ", ".join(r)

    if not to_retry:
        print("All checkpointed concepts look complete. Nothing to retry.")
        return

    # Summary by reason category
    buckets = {}
    for r in reasons.values():
        buckets[r] = buckets.get(r, 0) + 1
    print(f"Concepts flagged for retry: {len(to_retry)}")
    for reason, count in sorted(buckets.items(), key=lambda x: -x[1]):
        print(f"  {count:>4}  {reason}")
    print()

    if args.dry_run:
        print("-- DRY RUN: no files modified --")
        print("First 20 concepts that would be retried:")
        for c in to_retry[:20]:
            print(f"  {c}  [{reasons[c]}]")
        return

    # Remove flagged concepts from checkpoint
    for concept in to_retry:
        del checkpoint[concept]

    save_checkpoint(ckpt_path, checkpoint)
    print(f"Checkpoint updated: {len(to_retry)} concepts cleared for retry.")
    print(f"Remaining done    : {len(checkpoint)}")
    print()
    print("Now run:")
    print(f"  python scripts/media_pool/01_run_pool_agent.py --domain {args.domain} --workers 4 --resume")


if __name__ == "__main__":
    main()
