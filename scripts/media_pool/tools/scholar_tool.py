"""
Academic paper tool — searches OpenAlex for open-access papers.
No API key required. Polite pool: 10 req/s, 100 000 req/day.
DOI resolution verified via HEAD request before accepting an entry.
"""

import logging
import threading
import time
from datetime import date
from typing import Any

import requests

logger = logging.getLogger(__name__)

_BASE = "https://api.openalex.org"
_USER_AGENT = "GraphAIxLearning/1.0 (mailto:info@fem.digital)"
_LAST_CALL = 0.0
_MIN_INTERVAL = 0.5  # 2 req/s — well within the 10 req/s polite pool
_LOCK = threading.Lock()


def _wait():
    global _LAST_CALL
    with _LOCK:
        elapsed = time.time() - _LAST_CALL
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        _LAST_CALL = time.time()


def _verify_doi(doi: str) -> bool:
    """HEAD request to doi.org — returns True if DOI resolves."""
    try:
        resp = requests.head(
            f"https://doi.org/{doi}",
            timeout=8,
            allow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        )
        return resp.status_code < 400
    except Exception:
        return False


def search_papers(query: str, max_results: int = 3) -> dict[str, Any]:
    """
    Search OpenAlex for open-access papers with verified DOIs.

    Returns:
        {"results": [...], "count": int} — only OA papers with verified DOIs
        {"error": str, "results": [], "count": 0} on failure
    """
    _wait()
    today = date.today().isoformat()

    try:
        resp = requests.get(
            f"{_BASE}/works",
            params={
                "search": query,
                "filter": "is_oa:true",
                "per-page": max_results * 3,  # fetch more, filter down to verified DOIs
                "select": "title,authorships,publication_year,doi,primary_location,open_access",
            },
            headers={"User-Agent": _USER_AGENT},
            timeout=15,
        )
        if resp.status_code == 429:
            logger.warning("[scholar_tool] OpenAlex rate limit (429) — unexpected at this request rate")
            return {"error": "Rate limited by OpenAlex", "results": [], "count": 0}
        resp.raise_for_status()
        works = resp.json().get("results", [])
    except Exception as e:
        logger.warning(f"[scholar_tool] OpenAlex search failed for '{query}': {e}")
        return {"error": str(e), "results": [], "count": 0}

    results = []
    for work in works:
        if len(results) >= max_results:
            break

        # OpenAlex returns doi as full URL: "https://doi.org/10.xxx/..."
        doi_raw = work.get("doi") or ""
        doi = doi_raw.replace("https://doi.org/", "").strip()
        if not doi:
            continue

        if not _verify_doi(doi):
            continue

        authors_raw = work.get("authorships", [])
        authors = [
            a["author"]["display_name"]
            for a in authors_raw[:5]
            if a.get("author") and a["author"].get("display_name")
        ]
        if len(authors_raw) > 5:
            authors.append("et al.")

        oa_pdf = (work.get("open_access") or {}).get("oa_url")

        source = ((work.get("primary_location") or {}).get("source")) or {}
        journal = source.get("display_name")

        results.append(
            {
                "title": work.get("title", ""),
                "authors": authors,
                "year": work.get("publication_year"),
                "doi": doi,
                "doi_url": f"https://doi.org/{doi}",
                "open_access_pdf": oa_pdf,
                "journal": journal,
                "rights_status": "open_access_paper",
                "verified_date": today,
            }
        )

    return {"results": results, "count": len(results)}


# Tool definition for OpenAI function-calling format
TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_semantic_scholar",
        "description": (
            "Search for open-access academic papers (powered by OpenAlex). "
            "Only returns papers that are open-access AND have a verified DOI. "
            "Use academic-style queries with key concepts and research context. "
            "Example: 'metacognition self-regulated learning interventions students review'"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Academic search query. Include the concept + research context. "
                        "Example: 'working memory cognitive load classroom interventions'"
                    ),
                }
            },
            "required": ["query"],
        },
    },
}
