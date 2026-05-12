"""
Pool verifier — re-verify all URLs in an existing media pool with HEAD requests.

Useful to:
  - Run after 01_run_pool_agent.py to double-check every URL
  - Re-run periodically (monthly) to remove stale links
  - Generate a report of pool health

Usage:
    python scripts/media_pool/02_verify_pool.py --domain neuro
    python scripts/media_pool/02_verify_pool.py --domain udl --fix   # remove broken entries
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import date
from typing import Dict, Any

import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
from schema import load_pool, save_pool

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEDIA_DIR = os.path.join(REPO_ROOT, "data", "media")
USER_AGENT = "GraphAIxLearning/1.0 (https://github.com/FEM-modena/graphrag-aixlearning)"


def head_url(url: str, timeout: int = 8) -> bool:
    """Return True if the URL resolves (HTTP < 400)."""
    try:
        resp = requests.head(
            url,
            timeout=timeout,
            allow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )
        return resp.status_code < 400
    except Exception:
        return False


def verify_youtube(entry: Dict) -> bool:
    """Verify a YouTube video is still accessible via oEmbed (no API key needed)."""
    video_id = entry.get("video_id", "")
    if not video_id:
        return False
    try:
        resp = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"},
            timeout=8,
            headers={"User-Agent": USER_AGENT},
        )
        return resp.status_code == 200
    except Exception:
        return False


def verify_doi(entry: Dict) -> bool:
    """Verify DOI still resolves."""
    doi_url = entry.get("doi_url", "")
    return head_url(doi_url) if doi_url else False


def verify_wikipedia(entry: Dict) -> bool:
    """Verify Wikipedia URL still returns a valid article."""
    url = entry.get("url", "")
    return head_url(url) if url else False


def main():
    parser = argparse.ArgumentParser(description="Verify all URLs in an existing media pool")
    parser.add_argument("--domain", default="neuro", choices=["neuro", "udl"])
    parser.add_argument("--fix", action="store_true", help="Remove broken entries from the pool")
    parser.add_argument("--delay", type=float, default=0.3, help="Seconds between HTTP requests")
    args = parser.parse_args()

    pool_path = os.path.join(MEDIA_DIR, f"kg_{args.domain}_media_pool.json")

    if not os.path.exists(pool_path):
        logger.error(f"Pool file not found: {pool_path}")
        logger.error("Run 01_run_pool_agent.py first.")
        sys.exit(1)

    entries = load_pool(pool_path)
    today = date.today().isoformat()

    total_videos = ok_videos = broken_videos = 0
    total_citations = ok_citations = broken_citations = 0
    total_wiki = ok_wiki = broken_wiki = 0

    for concept_name, entry in entries.items():
        # Videos
        kept_videos = []
        for v in entry.get("videos", []):
            total_videos += 1
            ok = verify_youtube(v)
            time.sleep(args.delay)
            if ok:
                v["verified_date"] = today
                ok_videos += 1
                kept_videos.append(v)
            else:
                broken_videos += 1
                logger.warning(f"  [broken video] {concept_name}: {v.get('title')} ({v.get('video_id')})")
        if args.fix:
            entry["videos"] = kept_videos

        # Citations
        kept_cites = []
        for c in entry.get("citations", []):
            total_citations += 1
            ok = verify_doi(c)
            time.sleep(args.delay)
            if ok:
                c["verified_date"] = today
                ok_citations += 1
                kept_cites.append(c)
            else:
                broken_citations += 1
                logger.warning(f"  [broken doi] {concept_name}: {c.get('title')} ({c.get('doi')})")
        if args.fix:
            entry["citations"] = kept_cites

        # Wikipedia
        wiki = entry.get("wikipedia")
        if wiki:
            total_wiki += 1
            ok = verify_wikipedia(wiki)
            time.sleep(args.delay)
            if ok:
                wiki["verified_date"] = today
                ok_wiki += 1
            else:
                broken_wiki += 1
                logger.warning(f"  [broken wiki] {concept_name}: {wiki.get('url')}")
                if args.fix:
                    entry["wikipedia"] = None

    # Summary
    print("\n" + "=" * 55)
    print(f"  Verification report — {args.domain} pool")
    print("=" * 55)
    print(f"  Videos:    {ok_videos}/{total_videos} OK  ({broken_videos} broken)")
    print(f"  Citations: {ok_citations}/{total_citations} OK  ({broken_citations} broken)")
    print(f"  Wikipedia: {ok_wiki}/{total_wiki} OK  ({broken_wiki} broken)")
    total_ok = ok_videos + ok_citations + ok_wiki
    total_all = total_videos + total_citations + total_wiki
    print(f"  Total:     {total_ok}/{total_all} OK  ({total_all - total_ok} broken)")

    if args.fix and (broken_videos + broken_citations + broken_wiki) > 0:
        # Read pool metadata to preserve it
        with open(pool_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        model = meta.get("generated_by", "unknown")
        save_pool(pool_path, args.domain, model, entries)
        print(f"\n  Fixed pool written to: {pool_path}")
    elif not args.fix and (broken_videos + broken_citations + broken_wiki) > 0:
        print("\n  Run with --fix to remove broken entries from the pool.")


if __name__ == "__main__":
    main()
