"""
Media Pool Enricher — fetch full YouTube metadata (view_count, like_count, duration),
score each video, apply quality filters, and re-sort each concept's video list.

Run this AFTER 01_run_pool_agent.py to upgrade an existing pool with engagement signals.
The pool JSON is updated in-place (atomic write via temp file).

Ideas implemented:
  #1 — engagement ranking (view_count + like_ratio composite score)
  #3 — trusted-channel whitelist (+0.2 bonus to quality_score)
  #5 — duration_seconds stored for later retriever-side bucketing

Usage:
    python scripts/media_pool/04_enrich_pool.py --domain neuro
    python scripts/media_pool/04_enrich_pool.py --domain udl --min-views 1000
    python scripts/media_pool/04_enrich_pool.py --domain neuro --dry-run
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import date

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

# Sidecar file: video IDs dropped by enrichment filters, keyed by video_id.
# Loaded by 01_run_pool_agent.py to prevent re-collecting known-bad videos.
_REJECTED_FILE_TMPL = "kg_{domain}_rejected_ids.json"


def _load_rejected_ids(domain: str) -> dict:
    path = os.path.join(MEDIA_DIR, _REJECTED_FILE_TMPL.format(domain=domain))
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_rejected_ids(domain: str, rejected: dict, dry_run: bool = False) -> None:
    path = os.path.join(MEDIA_DIR, _REJECTED_FILE_TMPL.format(domain=domain))
    if dry_run:
        logger.info("[dry-run] Would write %d rejected IDs to %s", len(rejected), path)
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rejected, f, ensure_ascii=False, indent=2)
    logger.info("Rejected IDs sidecar updated: %d total → %s", len(rejected), path)


def _filter_reason(entry: dict, min_views: int) -> str:
    vc = entry.get("view_count")
    ds = entry.get("duration_seconds")
    if vc is not None and vc < min_views:
        return "low_views"
    if ds is not None and ds < 60:
        return "too_short"
    if ds is not None and ds > 3600:
        return "too_long"
    return "unknown"

# =============================================================================
# Trusted channel whitelist (Idea 3)
# Keys are domain names matching --domain arg; values are lowercase substrings.
# A channel matches if ANY substring appears in the channel name (case-insensitive).
# =============================================================================

TRUSTED_CHANNELS: dict[str, set[str]] = {
    "neuro": {
        "ted", "ted-ed", "ted education", "kurzgesagt",
        "scishow", "scishow psych", "crash course", "3blue1brown",
        "mit opencourseware", "hhmi biointeractive", "noba project",
        "sentis", "asap science", "the brain", "neuroscientifically",
        "psychology today", "stanford", "harvard",
    },
    "udl": {
        "cast", "understood", "khan academy", "edutopia",
        "iris center", "do-it uw", "national center on disability",
        "pacer center", "vanderbilt", "kennedy center",
        "special education", "inclusion", "accessibility",
    },
}

# Default: union of all domains — used when the domain is unknown
_ALL_TRUSTED = {ch for chs in TRUSTED_CHANNELS.values() for ch in chs}


def _is_trusted(channel: str, domain: str) -> bool:
    lookup = TRUSTED_CHANNELS.get(domain, _ALL_TRUSTED)
    c_lower = (channel or "").lower()
    return any(t in c_lower for t in lookup)


# =============================================================================
# yt-dlp metadata fetch (Idea 1)
# =============================================================================

try:
    import yt_dlp
    _YT_DLP_AVAILABLE = True
except ImportError:
    _YT_DLP_AVAILABLE = False


def _fetch_yt_metadata(video_id: str) -> dict:
    """
    Fetch full video metadata via yt-dlp (no download).
    Returns empty dict on any failure.
    """
    if not _YT_DLP_AVAILABLE:
        return {}
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "ignoreerrors": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False) or {}
        return info
    except Exception as exc:
        logger.debug("[enrich] yt-dlp failed for %s: %s", video_id, exc)
        return {}


# =============================================================================
# Quality score (Idea 1 + 3)
# =============================================================================

def _compute_quality_score(
    view_count: int | None,
    like_count: int | None,
    duration_s: int | None,
    trusted: bool,
) -> float:
    """
    Composite quality score in [0.0, 1.0].

    view_score  = log10(view_count) / 7   (log scale; 7 ≈ log10(10M))
    like_ratio  = like_count / view_count  (default 0.04 when unavailable)
    score       = 0.7 * view_score + 0.3 * like_ratio
    trusted     → +0.20 bonus, capped at 1.0
    """
    views = max(view_count or 0, 1)
    view_score = min(math.log10(views) / 7.0, 1.0)

    if like_count is not None and view_count and view_count > 0:
        like_ratio = min(like_count / view_count, 1.0)
    else:
        like_ratio = 0.04  # YouTube average engagement rate

    score = 0.7 * view_score + 0.3 * like_ratio
    if trusted:
        score = min(score + 0.20, 1.0)

    return round(score, 4)


# =============================================================================
# Filters (Idea 1)
# =============================================================================

def _passes_filters(
    entry: dict,
    min_views: int,
    min_duration_s: int = 60,
    max_duration_s: int = 3600,
) -> bool:
    """Return False if the video should be dropped from the pool."""
    view_count = entry.get("view_count")
    duration_s = entry.get("duration_seconds")

    if view_count is not None and view_count < min_views:
        return False
    if duration_s is not None:
        if duration_s < min_duration_s or duration_s > max_duration_s:
            return False
    return True


# =============================================================================
# Main enrichment loop
# =============================================================================

def enrich_pool(
    domain: str,
    min_views: int = 500,
    dry_run: bool = False,
    yt_delay: float = 1.0,
) -> None:
    pool_path = os.path.join(MEDIA_DIR, f"kg_{domain}_media_pool.json")
    if not os.path.exists(pool_path):
        logger.error("Pool not found: %s — run 01_run_pool_agent.py first.", pool_path)
        sys.exit(1)

    entries = load_pool(pool_path)
    total_videos = enriched = filtered = 0
    concepts_updated = 0
    rejected_this_run: dict = {}

    for concept_name, entry in entries.items():
        videos = entry.get("videos", [])
        if not videos:
            continue

        enriched_videos = []
        for v in videos:
            total_videos += 1
            video_id = v.get("video_id", "")
            if not video_id:
                enriched_videos.append(v)
                continue

            # Skip if already enriched (has view_count) unless forced
            if v.get("view_count") is not None:
                # Re-score in case trusted list changed
                v["trusted_channel"] = _is_trusted(v.get("channel", ""), domain)
                v["quality_score"] = _compute_quality_score(
                    v.get("view_count"), v.get("like_count"),
                    v.get("duration_seconds"), v["trusted_channel"]
                )
                enriched_videos.append(v)
                enriched += 1
                continue

            logger.info("[enrich] %s → %s (%s)", concept_name, v.get("title", "?")[:50], video_id)
            meta = _fetch_yt_metadata(video_id)
            time.sleep(yt_delay)

            view_count = meta.get("view_count")
            like_count = meta.get("like_count")
            duration_s = meta.get("duration")  # yt-dlp returns seconds as int

            v["view_count"] = view_count
            v["like_count"] = like_count
            v["duration_seconds"] = duration_s
            v["trusted_channel"] = _is_trusted(v.get("channel", ""), domain)
            v["quality_score"] = _compute_quality_score(
                view_count, like_count, duration_s, v["trusted_channel"]
            )
            enriched += 1

            if _passes_filters(v, min_views=min_views):
                enriched_videos.append(v)
            else:
                filtered += 1
                logger.info(
                    "  [drop] views=%s dur=%ss score=%.3f",
                    view_count, duration_s, v.get("quality_score", 0),
                )
                if video_id:
                    rejected_this_run[video_id] = {
                        "title": v.get("title", "")[:80],
                        "channel": v.get("channel", ""),
                        "reason": _filter_reason(v, min_views),
                        "view_count": view_count,
                        "duration_seconds": duration_s,
                        "rejected_date": date.today().isoformat(),
                    }

        # Sort by quality_score descending (trusted + high-view first)
        enriched_videos.sort(key=lambda x: x.get("quality_score") or 0, reverse=True)
        entry["videos"] = enriched_videos

        if enriched_videos != videos:
            concepts_updated += 1

    # Merge rejected IDs into the persistent sidecar
    existing_rejected = _load_rejected_ids(domain)
    existing_rejected.update(rejected_this_run)

    # Summary
    print("\n" + "=" * 55)
    print(f"  Enrichment report — {domain} pool")
    print("=" * 55)
    print(f"  Videos processed:       {total_videos}")
    print(f"  Enriched:               {enriched}")
    print(f"  Dropped (filter):       {filtered}")
    print(f"  Concepts updated:       {concepts_updated}")
    print(f"  Rejected IDs (total):   {len(existing_rejected)}")
    if not _YT_DLP_AVAILABLE:
        print("\n  WARNING: yt-dlp not installed — no metadata fetched.")
        print("  Install with: pip install yt-dlp")

    if dry_run:
        print("\n  [dry-run] No changes written.")
        return

    # Read metadata header to preserve it
    with open(pool_path, "r", encoding="utf-8") as f:
        meta_header = json.load(f)
    model = meta_header.get("generated_by", "unknown")
    save_pool(pool_path, domain, model, entries)
    print(f"\n  Updated pool written to: {pool_path}")

    # Write rejected IDs sidecar AFTER the pool (so partial runs don't blacklist)
    _save_rejected_ids(domain, existing_rejected)


def main():
    parser = argparse.ArgumentParser(
        description="Enrich an existing media pool with YouTube engagement metrics."
    )
    parser.add_argument("--domain", default="neuro", choices=["neuro", "udl"],
                        help="Domain pool to enrich (default: neuro)")
    parser.add_argument("--min-views", type=int, default=500,
                        help="Drop videos with fewer views than this (default: 500)")
    parser.add_argument("--yt-delay", type=float, default=1.0,
                        help="Seconds to wait between yt-dlp calls (default: 1.0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute scores but do not write changes to disk")
    args = parser.parse_args()

    enrich_pool(
        domain=args.domain,
        min_views=args.min_views,
        dry_run=args.dry_run,
        yt_delay=args.yt_delay,
    )


if __name__ == "__main__":
    main()
