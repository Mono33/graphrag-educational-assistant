"""
Wikipedia tool — verifies that a topic has a real, non-disambiguation Wikipedia article.
Returns the canonical URL and language.
"""

import logging
import threading
import time
from datetime import date
from typing import Any

import requests

logger = logging.getLogger(__name__)

_USER_AGENT = "GraphAIxLearning/1.0 (https://github.com/FEM-modena/graphrag-aixlearning)"
_LAST_CALL = 0.0
_MIN_INTERVAL = 0.5
_LOCK = threading.Lock()


def _wait():
    global _LAST_CALL
    with _LOCK:
        elapsed = time.time() - _LAST_CALL
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        _LAST_CALL = time.time()


def get_article(topic: str, language: str = "en") -> dict[str, Any]:
    """
    Fetch a Wikipedia article summary for a topic.

    Returns:
        {"found": True, "title": ..., "url": ..., "rights_status": "cc_by_sa", ...}
        {"found": False, "reason": ...} if not found or disambiguation
    """
    _wait()
    today = date.today().isoformat()
    lang = language if language in ("en", "it", "de", "fr", "es") else "en"
    encoded_topic = requests.utils.quote(topic.replace(" ", "_"))
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{encoded_topic}"

    try:
        resp = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=10)
        if resp.status_code == 404:
            return {"found": False, "reason": "Article not found"}
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"[wikipedia_tool] Request failed for '{topic}' ({lang}): {e}")
        return {"found": False, "reason": str(e)}

    page_type = data.get("type", "")
    if page_type == "disambiguation":
        return {"found": False, "reason": "Disambiguation page — be more specific"}

    canonical_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
    if not canonical_url:
        canonical_url = f"https://{lang}.wikipedia.org/wiki/{data.get('title', topic).replace(' ', '_')}"

    return {
        "found": True,
        "title": data.get("title", topic),
        "url": canonical_url,
        "extract": data.get("extract", "")[:300],
        "rights_status": "cc_by_sa",
        "verified_date": today,
        "language": lang,
    }


# Tool definition for OpenAI function-calling format
TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_wikipedia",
        "description": (
            "Check if a Wikipedia article exists for a topic. "
            "Returns the canonical URL if the article is real (not a disambiguation page). "
            "Try the main concept name first; if not found, try a more specific or common form."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Wikipedia article title or concept name. Use title case. Example: 'Metacognition'",
                },
                "language": {
                    "type": "string",
                    "description": "Language code: 'en' for English (default), 'it' for Italian.",
                    "default": "en",
                },
            },
            "required": ["topic"],
        },
    },
}
