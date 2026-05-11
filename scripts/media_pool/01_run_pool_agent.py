"""
Media Pool Agent — LM Studio tool-calling agent that builds a verified media pool.

The local LLM (Gemma 4 via LM Studio) drives the entire process:
  1. Queries the live Neo4j KG to understand each concept's relationships
  2. Generates context-aware search queries based on what it finds
  3. Calls YouTube, Semantic Scholar, and Wikipedia via tool use
  4. Saves only verified, rights-tagged entries to the pool

The pool is written to data/media/kg_{domain}_media_pool.json.
A checkpoint is saved after each concept so the run is fully resumable.

Usage:
    python scripts/media_pool/01_run_pool_agent.py --domain neuro
    python scripts/media_pool/01_run_pool_agent.py --domain neuro --workers 4   # concurrent
    python scripts/media_pool/01_run_pool_agent.py --domain neuro --resume      # continue from checkpoint
    python scripts/media_pool/01_run_pool_agent.py --domain neuro --concept "Metacognition"
"""

import argparse
import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Allow tool imports from the same scripts/media_pool/ subtree
sys.path.insert(0, os.path.dirname(__file__))

from schema import load_pool, save_pool, load_checkpoint, save_checkpoint
from tools import neo4j_tool, youtube_tool, scholar_tool, wikipedia_tool

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "google/gemma-4-26b-a4b")
MAX_ITERATIONS = int(os.getenv("POOL_AGENT_MAX_ITER", "25"))  # tool calls per concept

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEDIA_DIR = os.path.join(REPO_ROOT, "data", "media")

# =============================================================================
# System prompt
# =============================================================================

SYSTEM_PROMPT = """You are an expert educational media curator building a verified media pool for an AI-powered learning platform focused on neuroscience, education, and Universal Design for Learning (UDL).

Your task: for each educational concept, find and save high-quality, rights-clear media resources.

## Workflow for each concept (follow this order):

1. **QUERY THE KNOWLEDGE GRAPH** — Call `query_neo4j` to understand the concept's relationships and educational context. Look at its neighbors, relationship types (SUGGESTS, FACILITATES, SUPPORTS, ENHANCES, MITIGATED_BY, etc.), and labels. This context makes your searches more targeted.

2. **SEARCH FOR VIDEOS** — Call `search_youtube` 2-3 times with specific, contextual queries that incorporate what you learned from the KG. Don't just search the bare concept name — include the educational context (e.g., "metacognition strategies ADHD students classroom" instead of just "metacognition").

3. **SEARCH FOR PAPERS** — Call `search_semantic_scholar` 1-2 times with academic-style queries. Only open-access papers with verified DOIs are returned.

4. **SEARCH WIKIPEDIA** — Call `search_wikipedia` once with the concept name. Try English first; if the concept is very Italian-specific, try Italian ("it").

5. **SAVE VERIFIED RESULTS** — For each good result returned by the search tools, call `save_to_pool`. Only save what the search tools actually returned — never invent or modify video IDs, DOIs, or URLs.

## Important rules:
- NEVER invent video IDs, DOIs, or URLs. Only use data returned by the search tools.
- If a search returns no results, try a different query variation.
- Target: 2-3 videos, 1-2 citations, 1 Wikipedia article per concept.
- Prioritize relevance: a highly relevant result beats 3 mediocre ones.
- When done with a concept (all saves complete), say "DONE" in your final message.
"""

# =============================================================================
# Tool definitions
# =============================================================================

SAVE_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "save_to_pool",
        "description": (
            "Save a verified media entry to the pool. "
            "ONLY call this with data returned by search_youtube, search_semantic_scholar, or search_wikipedia. "
            "Never fabricate video_id, doi, or url values."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entry_type": {
                    "type": "string",
                    "enum": ["video", "citation", "wikipedia"],
                    "description": "Type of media entry to save.",
                },
                "concept_name": {
                    "type": "string",
                    "description": "The concept this media is for (exactly as provided in the task).",
                },
                "data": {
                    "type": "object",
                    "description": (
                        "Entry data from the search tool result. "
                        "For video: include title, video_id, url, embed_url, channel, rights_status, verified_date, language. "
                        "For citation: include title, authors, doi, doi_url, rights_status, verified_date (and year, journal, open_access_pdf if available). "
                        "For wikipedia: include title, url, rights_status, verified_date, language."
                    ),
                },
                "graph_context": {
                    "type": "string",
                    "description": "Optional: note the KG relationship that makes this resource relevant (e.g., 'ADHD -[SUGGESTS]-> Metacognition').",
                },
            },
            "required": ["entry_type", "concept_name", "data"],
        },
    },
}

ALL_TOOLS = [
    neo4j_tool.TOOL_DEFINITION,
    youtube_tool.TOOL_DEFINITION,
    scholar_tool.TOOL_DEFINITION,
    wikipedia_tool.TOOL_DEFINITION,
    SAVE_TOOL_DEFINITION,
]

# =============================================================================
# Tool dispatcher
# =============================================================================

def dispatch_tool(tool_name: str, tool_args: Dict[str, Any], concept_name: str, pool_entries: Dict) -> str:
    """Execute a tool call and return the result as a JSON string."""
    try:
        if tool_name == "query_neo4j":
            result = neo4j_tool.run_query(tool_args["cypher"])

        elif tool_name == "search_youtube":
            result = youtube_tool.search_videos(tool_args["query"])

        elif tool_name == "search_semantic_scholar":
            result = scholar_tool.search_papers(tool_args["query"])

        elif tool_name == "search_wikipedia":
            result = wikipedia_tool.get_article(
                tool_args["topic"], tool_args.get("language", "en")
            )

        elif tool_name == "save_to_pool":
            result = _handle_save(tool_args, pool_entries)

        else:
            result = {"error": f"Unknown tool: {tool_name}"}

    except Exception as e:
        logger.error(f"[dispatch] Tool '{tool_name}' raised: {e}")
        result = {"error": str(e)}

    return json.dumps(result, ensure_ascii=False)


def _handle_save(args: Dict[str, Any], pool_entries: Dict) -> Dict:
    """Add a verified entry to the in-memory pool dict."""
    entry_type = args.get("entry_type")
    concept = args.get("concept_name", "")
    data = args.get("data", {})
    graph_ctx = args.get("graph_context")

    if not concept or not data:
        return {"saved": False, "reason": "Missing concept_name or data"}

    today = date.today().isoformat()

    # Validate required fields per type
    if entry_type == "video":
        if not data.get("title") or not data.get("video_id"):
            return {"saved": False, "reason": "Missing required video fields: title, video_id"}
        vid_id = data["video_id"]
        # YouTube video IDs are always exactly 11 alphanumeric/dash/underscore characters.
        # Reject anything outside that range — it's a hallucinated or corrupted ID.
        if not (len(vid_id) == 11 and all(c.isalnum() or c in "-_" for c in vid_id)):
            return {"saved": False, "reason": f"Invalid video_id '{vid_id}' — must be exactly 11 chars (YouTube format)"}
        # Always derive url/embed_url from video_id to prevent LLM typos
        data["url"] = f"https://youtu.be/{vid_id}"
        data["embed_url"] = f"https://www.youtube.com/embed/{vid_id}"
        data.setdefault("verified_date", today)
        data.setdefault("rights_status", "youtube_embed")
        if graph_ctx:
            data["graph_context"] = graph_ctx
        pool_entries.setdefault(concept, {"videos": [], "citations": [], "wikipedia": None})
        pool_entries[concept]["videos"].append(data)

    elif entry_type == "citation":
        if not data.get("title") or not data.get("doi"):
            return {"saved": False, "reason": "Missing required citation fields: title, doi"}
        doi = data["doi"].strip()
        # DOIs must start with "10." followed by a registrant code and suffix.
        # Reject anything that doesn't match the basic pattern.
        if not doi.startswith("10.") or "/" not in doi or len(doi) < 8:
            return {"saved": False, "reason": f"Invalid DOI '{doi}' — must start with '10.' and contain '/'"}
        data["doi"] = doi
        # Always derive doi_url from doi
        data["doi_url"] = f"https://doi.org/{doi}"
        data.setdefault("verified_date", today)
        data.setdefault("rights_status", "open_access_paper")
        if graph_ctx:
            data["graph_context"] = graph_ctx
        pool_entries.setdefault(concept, {"videos": [], "citations": [], "wikipedia": None})
        pool_entries[concept]["citations"].append(data)

    elif entry_type == "wikipedia":
        if not data.get("title") or not data.get("url"):
            return {"saved": False, "reason": "Missing required wikipedia fields: title, url"}
        data.setdefault("verified_date", today)
        data.setdefault("rights_status", "cc_by_sa")
        pool_entries.setdefault(concept, {"videos": [], "citations": [], "wikipedia": None})
        pool_entries[concept]["wikipedia"] = data

    else:
        return {"saved": False, "reason": f"Unknown entry_type: {entry_type}"}

    logger.info(f"  [saved] {entry_type} for '{concept}': {data.get('title', data.get('url', ''))}")
    return {"saved": True, "entry_type": entry_type, "concept": concept}


# =============================================================================
# Agent loop (one concept at a time)
# =============================================================================

def run_agent_for_concept(
    client: OpenAI,
    concept_name: str,
    domain: str,
    pool_entries: Dict,
    max_iterations: int = MAX_ITERATIONS,
) -> int:
    """
    Run the tool-calling agent loop for a single concept.
    Returns the number of entries saved for this concept.
    """
    saved_before = sum(
        len(pool_entries.get(concept_name, {}).get("videos", []))
        + len(pool_entries.get(concept_name, {}).get("citations", []))
        + (1 if pool_entries.get(concept_name, {}).get("wikipedia") else 0)
        for _ in [0]
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Research educational media for the concept: **{concept_name}**\n\n"
                f"Domain: {domain}\n\n"
                "Start by querying the knowledge graph to understand this concept's "
                "relationships and context. Then search for verified educational media "
                "and save the best results. When finished, say DONE."
            ),
        },
    ]

    for iteration in range(max_iterations):
        try:
            response = client.chat.completions.create(
                model=LMSTUDIO_MODEL,
                messages=messages,
                tools=ALL_TOOLS,
                tool_choice="auto",
                temperature=0.3,
                max_tokens=2048,
            )
        except Exception as e:
            logger.error(f"  [agent] LLM call failed (iteration {iteration}): {e}")
            break

        choice = response.choices[0]
        msg = choice.message

        # Append assistant message (may contain tool_calls)
        messages.append(msg.model_dump(exclude_none=True))

        if choice.finish_reason == "stop":
            logger.debug(f"  [agent] Finished for '{concept_name}' after {iteration + 1} iterations")
            break

        if not (msg.tool_calls):
            # No tool calls and not stopped — force stop
            break

        # Dispatch all tool calls in this turn
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            logger.debug(f"  [tool] {tool_name}({list(tool_args.keys())})")
            result_str = dispatch_tool(tool_name, tool_args, concept_name, pool_entries)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                }
            )
    else:
        logger.warning(f"  [agent] Hit max iterations ({max_iterations}) for '{concept_name}'")

    saved_after = sum(
        len(pool_entries.get(concept_name, {}).get("videos", []))
        + len(pool_entries.get(concept_name, {}).get("citations", []))
        + (1 if pool_entries.get(concept_name, {}).get("wikipedia") else 0)
        for _ in [0]
    )
    return saved_after - saved_before


# =============================================================================
# Concurrent worker
# =============================================================================

_WRITE_LOCK = threading.Lock()


def _worker(
    concept_name: str,
    domain: str,
    max_iterations: int,
    pool_path: str,
    checkpoint_path: str,
    checkpoint: Dict,
) -> tuple:
    """
    Process a single concept in its own thread.
    Each worker has its own local pool dict and OpenAI client.
    Results are merged into the shared pool file under a write lock.
    """
    local_entries: Dict = {}
    client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key="lm-studio")

    try:
        n_saved = run_agent_for_concept(client, concept_name, domain, local_entries, max_iterations)
    except Exception as e:
        logger.error(f"  [worker] Agent failed for '{concept_name}': {e}")
        n_saved = 0

    # Atomic: merge local results into shared pool + update checkpoint
    with _WRITE_LOCK:
        current_pool = load_pool(pool_path)
        if local_entries:
            current_pool.update(local_entries)
        save_pool(pool_path, domain, LMSTUDIO_MODEL, current_pool)
        checkpoint[concept_name] = "done"
        save_checkpoint(checkpoint_path, checkpoint)

    return concept_name, n_saved


# =============================================================================
# Concept list fetcher
# =============================================================================

def fetch_concept_list(domain: str) -> List[str]:
    """
    Fetch all concept names from Neo4j for the given domain.
    Nodes have a 'domain' property with values 'neuro' or 'udl'.
    """
    cypher = (
        "MATCH (n) WHERE n.domain = $domain AND n.name IS NOT NULL "
        "RETURN n.name AS name ORDER BY n.name"
    )
    # neo4j_tool.run_query doesn't support parameters — inline the value safely
    cypher = f"MATCH (n) WHERE n.domain = '{domain}' AND n.name IS NOT NULL RETURN n.name AS name ORDER BY n.name"

    result = neo4j_tool.run_query(cypher, limit_rows=2000)
    if "error" in result:
        raise RuntimeError(f"Failed to fetch concept list: {result['error']}")

    names = [row["name"] for row in result["rows"] if row.get("name")]
    logger.info(f"Fetched {len(names)} concepts for domain '{domain}'")
    return names


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run the media pool LLM agent")
    parser.add_argument("--domain", default="neuro", choices=["neuro", "udl"], help="KG domain to process")
    parser.add_argument("--limit", type=int, default=0, help="Max concepts to process (0 = all)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent workers (default 1)")
    parser.add_argument("--max-iter", type=int, default=MAX_ITERATIONS, help="Max tool call iterations per concept")
    parser.add_argument("--concept", type=str, default=None, help="Process a single concept name (for testing)")
    args = parser.parse_args()

    os.makedirs(MEDIA_DIR, exist_ok=True)
    pool_path = os.path.join(MEDIA_DIR, f"kg_{args.domain}_media_pool.json")
    checkpoint_path = os.path.join(MEDIA_DIR, f"checkpoint_{args.domain}.json")

    # Load existing state
    checkpoint = load_checkpoint(checkpoint_path) if args.resume else {}
    if args.resume:
        existing_pool = load_pool(pool_path)
        logger.info(f"Resuming: {len(existing_pool)} concepts in pool, {len(checkpoint)} checkpointed")

    logger.info(f"LM Studio: {LMSTUDIO_BASE_URL} | model: {LMSTUDIO_MODEL} | workers: {args.workers}")

    # Get concept list
    if args.concept:
        concepts = [args.concept]
    else:
        concepts = fetch_concept_list(args.domain)

    if args.limit > 0:
        concepts = concepts[: args.limit]

    pending = [c for c in concepts if c not in checkpoint]
    logger.info(f"Concepts to process: {len(pending)} (skipping {len(concepts) - len(pending)} done)")

    if not pending:
        logger.info("All concepts already processed. Run with --resume to confirm pool state.")
        return

    total_saved = 0
    completed = 0

    if args.workers == 1:
        # Single-worker path (original sequential loop)
        client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key="lm-studio")
        pool_entries = load_pool(pool_path) if args.resume else {}

        for i, concept_name in enumerate(pending, 1):
            logger.info(f"[{i}/{len(pending)}] Processing: {concept_name}")
            try:
                n_saved = run_agent_for_concept(client, concept_name, args.domain, pool_entries, args.max_iter)
                total_saved += n_saved
                logger.info(f"  → {n_saved} entries saved")
            except Exception as e:
                logger.error(f"  → Error: {e}")

            checkpoint[concept_name] = "done"
            save_checkpoint(checkpoint_path, checkpoint)
            save_pool(pool_path, args.domain, LMSTUDIO_MODEL, pool_entries)

    else:
        # Multi-worker path — each worker handles one concept at a time
        logger.info(f"Starting {args.workers} concurrent workers")

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _worker, c, args.domain, args.max_iter, pool_path, checkpoint_path, checkpoint
                ): c
                for c in pending
            }

            for future in as_completed(futures):
                concept = futures[future]
                try:
                    name, n_saved = future.result()
                    completed += 1
                    total_saved += n_saved
                    logger.info(f"  [{completed}/{len(pending)}] {name}: {n_saved} entries saved")
                except Exception as e:
                    logger.error(f"  [error] {concept}: {e}")

    neo4j_tool.close()

    logger.info("=" * 55)
    logger.info(f"Done. Total entries saved: {total_saved}")
    logger.info(f"Pool written to: {pool_path}")
    logger.info("Run 02_verify_pool.py to re-verify all URLs.")


if __name__ == "__main__":
    main()
