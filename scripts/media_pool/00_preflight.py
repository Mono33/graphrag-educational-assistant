"""
Preflight check — verify all connections before running the pool agent.
Run this first to confirm your environment is ready.

Usage:
    python scripts/media_pool/00_preflight.py
"""

import io
import os
import sys

import requests
from dotenv import load_dotenv

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError with emoji)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Allow imports from scripts/media_pool/tools/
sys.path.insert(0, os.path.dirname(__file__))

load_dotenv()

REQUIRED_CHECKS = []   # Neo4j + LM Studio — must all pass
OPTIONAL_WARNS = []    # API keys — warn only, don't block


def check(name: str, ok: bool, detail: str = "", required: bool = True):
    if ok:
        status = "✅ OK "
    elif required:
        status = "❌ FAIL"
    else:
        status = "⚠️  WARN"
    msg = f"  {status}  {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    if required:
        REQUIRED_CHECKS.append(ok)
    else:
        OPTIONAL_WARNS.append(ok)


def check_neo4j():
    print("\n[Neo4j]")
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    check("NEO4J_URI set", bool(uri), uri or "missing")
    check("NEO4J_PASSWORD set", bool(password), "****" if password else "missing")

    if uri and password:
        try:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(uri, auth=(user, password))
            with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
                result = session.run("RETURN 1 AS n")
                row = result.single()
                check("Neo4j connection", row and row["n"] == 1)
            driver.close()
        except Exception as e:
            check("Neo4j connection", False, str(e))


def check_lmstudio():
    print("\n[LM Studio]")
    base_url = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
    model = os.getenv("LMSTUDIO_MODEL", "google/gemma-4-26b-a4b")

    check("LMSTUDIO_BASE_URL", True, base_url)
    check("LMSTUDIO_MODEL", True, model)

    try:
        resp = requests.get(f"{base_url}/models", timeout=5)
        resp.raise_for_status()
        models = [m["id"] for m in resp.json().get("data", [])]
        model_available = any(model in m or m in model for m in models)
        check("LM Studio reachable", True, f"{len(models)} models loaded")
        check(f"Model '{model}' available", model_available, str(models) if not model_available else "")
    except Exception as e:
        check("LM Studio reachable", False, str(e))


def check_youtube():
    print("\n[YouTube Data API]  (optional)")
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        check("YOUTUBE_API_KEY", False, "not set — add to .env to enable YouTube search", required=False)
        return
    check("YOUTUBE_API_KEY", True, api_key[:8] + "...", required=False)
    try:
        resp = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={"part": "snippet", "q": "test", "maxResults": 1, "key": api_key},
            timeout=10,
        )
        resp.raise_for_status()
        check("YouTube API v3 call", True, required=False)
    except Exception as e:
        check("YouTube API v3 call", False, str(e), required=False)


def check_semantic_scholar():
    print("\n[Semantic Scholar]  (optional)")
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        check("SEMANTIC_SCHOLAR_API_KEY", True, api_key[:8] + "... (1000 req/5min)", required=False)
    else:
        check("SEMANTIC_SCHOLAR_API_KEY", True, "not set — free tier (100 req/5min)", required=False)
        # Skip the live test call to avoid burning free-tier quota during preflight
        print("  ℹ️        Skipping live API test (no key — preserving rate limit quota)")
        return
    try:
        headers = {"x-api-key": api_key}
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": "metacognition", "fields": "title", "limit": 1},
            headers=headers,
            timeout=10,
        )
        if resp.status_code == 429:
            check("Semantic Scholar API call", False, "rate limited — wait and retry", required=False)
        else:
            resp.raise_for_status()
            check("Semantic Scholar API call", True, required=False)
    except Exception as e:
        check("Semantic Scholar API call", False, str(e), required=False)


def check_wikipedia():
    print("\n[Wikipedia]")
    try:
        resp = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/Metacognition",
            headers={"User-Agent": "GraphAIxLearning/1.0"},
            timeout=10,
        )
        resp.raise_for_status()
        check("Wikipedia API call", True)
    except Exception as e:
        check("Wikipedia API call", False, str(e))


def check_output_dirs():
    print("\n[Output directories]")
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    media_dir = os.path.join(repo_root, "data", "media")
    os.makedirs(media_dir, exist_ok=True)
    check("data/media/ exists", os.path.isdir(media_dir), media_dir)


if __name__ == "__main__":
    print("=" * 55)
    print("  GraphAIxLearning — Media Pool Agent Preflight Check")
    print("=" * 55)

    check_neo4j()
    check_lmstudio()
    check_youtube()
    check_semantic_scholar()
    check_wikipedia()
    check_output_dirs()

    req_passed = sum(REQUIRED_CHECKS)
    req_total = len(REQUIRED_CHECKS)
    opt_passed = sum(OPTIONAL_WARNS)
    opt_total = len(OPTIONAL_WARNS)

    print(f"\n{'=' * 55}")
    print(f"  Required: {req_passed}/{req_total} passed   Optional: {opt_passed}/{opt_total} configured")

    if req_passed == req_total:
        if opt_passed == opt_total:
            print("  All systems ready. Run 01_run_pool_agent.py")
        else:
            print("  Ready to run (some optional APIs not configured — see ⚠️  above)")
            print("  YouTube: add YOUTUBE_API_KEY to .env for video search")
            print("  Scholar: add SEMANTIC_SCHOLAR_API_KEY for higher rate limits")
    else:
        print("  ❌ Fix required checks before running the agent")
        sys.exit(1)
