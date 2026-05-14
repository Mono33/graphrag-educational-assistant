"""
YouTube tool — searches YouTube via yt-dlp (no API key, no quota).
Verifies embeddability via YouTube's official oEmbed endpoint before accepting a video.
"""

import logging
import time
from datetime import date
from typing import Any, Dict

import threading

import requests

try:
    import yt_dlp

    _YT_DLP_AVAILABLE = True
except ImportError:
    _YT_DLP_AVAILABLE = False

logger = logging.getLogger(__name__)

_LAST_CALL = 0.0
_MIN_INTERVAL = 1.5  # polite: ~1 search every 1.5s across all threads
_LOCK = threading.Lock()


def _wait():
    global _LAST_CALL
    with _LOCK:
        elapsed = time.time() - _LAST_CALL
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        _LAST_CALL = time.time()


def _is_embeddable(video_id: str) -> bool:
    """
    Check via YouTube oEmbed — official endpoint, no API key, no quota.
    Returns True if the video is publicly embeddable.
    """
    try:
        resp = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"},
            timeout=8,
        )
        return resp.status_code == 200
    except Exception:
        return False


def search_videos(query: str, max_results: int = 3) -> Dict[str, Any]:
    """
    Search YouTube for educational videos using yt-dlp (no API key required).
    Verifies embeddability via oEmbed before accepting each video.

    Returns:
        {"results": [...], "count": int} — only embeddable, non-live videos
        {"error": str, "results": [], "count": 0} on failure
    """
    if not _YT_DLP_AVAILABLE:
        return {
            "error": "yt-dlp not installed — run: pip install yt-dlp",
            "results": [],
            "count": 0,
        }

    today = date.today().isoformat()
    _wait()

    fetch_count = max_results * 3  # fetch extra, filter down to embeddable
    search_query = f"ytsearch{fetch_count}:{query}"

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,  # metadata only, no download
        "ignoreerrors": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_query, download=False)
        entries = info.get("entries", []) if info else []
    except Exception as e:
        logger.warning(f"[youtube_tool] yt-dlp search failed for '{query}': {e}")
        return {"error": str(e), "results": [], "count": 0}

    if not entries:
        return {"results": [], "count": 0}

    results = []
    for entry in entries:
        if len(results) >= max_results:
            break
        if not entry:
            continue

        video_id = entry.get("id")
        if not video_id:
            continue

        # Skip live and upcoming broadcasts
        live_status = entry.get("live_status", "")
        if live_status in ("is_live", "is_upcoming"):
            continue

        # Verify embeddability via oEmbed (authoritative, quota-free)
        if not _is_embeddable(video_id):
            continue

        # In extract_flat mode, license is rarely populated; youtube_embed is the safe default.
        # youtube_embed = permitted by YouTube TOS Section 6C for public embeds.
        license_ = entry.get("license") or ""
        rights_status = "youtube_cc" if "creative commons" in license_.lower() else "youtube_embed"

        results.append(
            {
                "title": entry.get("title", ""),
                "video_id": video_id,
                "url": f"https://youtu.be/{video_id}",
                "embed_url": f"https://www.youtube.com/embed/{video_id}",
                "channel": entry.get("channel") or entry.get("uploader", ""),
                "rights_status": rights_status,
                "verified_date": today,
                "language": "en",
                "duration_hint": str(entry.get("duration", "")),
            }
        )

    return {"results": results, "count": len(results)}


# Tool definition for OpenAI function-calling format
TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_youtube",
        "description": (
            "Search YouTube for embeddable educational videos. "
            "No API key needed — no quota limits. "
            "Returns only videos verified as publicly embeddable via oEmbed. "
            "Include context from the Knowledge Graph to make queries specific. "
            "Example good query: 'metacognition classroom strategies ADHD students' "
            "instead of just 'metacognition'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "YouTube search query. Be specific and educational. "
                        "Include concept name + educational context + target audience when relevant."
                    ),
                }
            },
            "required": ["query"],
        },
    },
}
