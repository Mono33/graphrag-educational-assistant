"""
Media MCP tools — Phase 3 of CORE 5 #20.

Five tools that expose Aix's existing media subsystem to MCP clients:

* ``media.lookup_curated``  — Phase 0 curated mapping (file-backed, offline,
                              no API key, no quota cost). Wraps
                              ``MediaLookup.find_media`` and returns videos,
                              images, external resources, citations, and OER
                              textbooks already vetted by domain experts.

* ``media.search_youtube``  — Phase 4 live search via YouTube Data API
                              (when ``YOUTUBE_API_KEY`` is set) or a fallback
                              search-URL generator otherwise. Wraps
                              ``ExternalMediaAPI.search_youtube``.

* ``media.search_academic`` — Phase 4 live search via Semantic Scholar
                              (rate-limited, with retry-w/-backoff). Wraps
                              ``ExternalMediaAPI.search_semantic_scholar``.

* ``media.search_oer``      — Phase 4 live search across DOAB,
                              OpenTextbookLibrary, and BC Campus. Wraps
                              ``ExternalMediaAPI.search_oer_textbooks``.

* ``media.generate_diagram``— Phase 5 Mermaid.js diagram generation.
                              Uses gpt-4o-mini for the Mermaid code and
                              the free mermaid.ink renderer for SVG/PNG
                              URLs. Wraps ``MermaidGenerator.generate``.

Design notes
------------
* All tools are read-only or generate-only (no KG mutation, no DB writes).
* Backing components are lazily instantiated and cached at module scope so
  cold start doesn't pay full cost on every tool call. The same pattern as
  Phase 1's ``kg_tools._GRAPHRAG_TOOLS`` cache.
* External-API tools degrade gracefully: missing API keys → fallback paths
  (already implemented by the underlying ``ExternalMediaAPI``); transport
  errors → structured error response (we never crash the tool call).
* Pydantic models on the response side preserve the public contract that
  the curated mapping already exposes (``VideoResource``, ``Citation`` etc.)
  while giving each client a JSON-schema-rich payload.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Literal, Optional

from fastmcp import FastMCP
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


DomainLiteral = Literal["neuro", "udl"]


# ---------------------------------------------------------------------------
# Public Pydantic schemas (the JSON shape MCP clients see).
# We mirror the underlying dataclasses so a client could roundtrip if needed.
# ---------------------------------------------------------------------------


class MediaCuratedSummary(BaseModel):
    """Summary of curated media for a single concept."""

    concept_name: str
    found: bool = Field(
        description="Whether curated media was found for this concept name "
        "(possibly via fuzzy match)."
    )
    counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Per-kind item counts (videos, images, resources, "
        "citations, open_textbooks).",
    )
    items: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Per-kind list of items (each item is a dict-form of "
        "the underlying dataclass).",
    )


class MediaCuratedResult(BaseModel):
    """Aggregated result for ``media.lookup_curated``."""

    domain: DomainLiteral
    requested: int
    matched: int
    by_concept: List[MediaCuratedSummary]


class MediaErrorResult(BaseModel):
    """Structured error returned by live-API tools when something goes wrong.

    Live-API tools always return *something* — never raise — so MCP clients
    don't have to special-case transport errors.
    """

    error: str
    details: Optional[str] = None


# ---------------------------------------------------------------------------
# Lazy backing-component singletons.
# ---------------------------------------------------------------------------
_MEDIA_LOOKUPS: Dict[str, Any] = {}
_EXTERNAL_API: Dict[str, Any] = {}  # single-key dict ('default') for symmetry
_MERMAID_GENERATOR: Dict[str, Any] = {}


def _get_media_lookup(domain: str):
    """Cached ``MediaLookup`` per domain (lazy init)."""
    if domain not in _MEDIA_LOOKUPS:
        from aix.agent.media.media_lookup import MediaLookup

        _MEDIA_LOOKUPS[domain] = MediaLookup(domain=domain)
    return _MEDIA_LOOKUPS[domain]


def _get_external_api():
    """Cached ``ExternalMediaAPI`` (singleton, env-driven keys)."""
    if "default" not in _EXTERNAL_API:
        from aix.agent.media.external_apis import ExternalMediaAPI

        _EXTERNAL_API["default"] = ExternalMediaAPI(
            youtube_api_key=os.environ.get("YOUTUBE_API_KEY"),
            semantic_scholar_api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"),
        )
    return _EXTERNAL_API["default"]


def _get_mermaid_generator():
    """Cached ``MermaidGenerator`` (singleton, OpenRouter-compatible)."""
    if "default" not in _MERMAID_GENERATOR:
        from aix.agent.media.mermaid_generator import MermaidGenerator

        _MERMAID_GENERATOR["default"] = MermaidGenerator()
    return _MERMAID_GENERATOR["default"]


def _to_dict(item: Any) -> Dict[str, Any]:
    """Best-effort conversion of a dataclass / Pydantic model / dict to dict."""
    if is_dataclass(item):
        return asdict(item)
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if isinstance(item, dict):
        return item
    return {"value": str(item)}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(mcp: FastMCP) -> None:
    """Register all five media.* tools onto the shared FastMCP instance."""

    # ---- media.lookup_curated --------------------------------------------
    @mcp.tool(
        name="media.lookup_curated",
        description=(
            "Look up Aix's curated media (videos, images, external resources, "
            "academic citations, open-textbook references) for one or more "
            "concept names. Reads from the file-backed media mapping — "
            "offline, no API key required, no quota cost. Use this BEFORE "
            "the live-search tools when you only need expert-vetted content."
        ),
        tags={"media", "curated", "offline"},
    )
    def media_lookup_curated(
        concepts: List[str],
        domain: DomainLiteral = "neuro",
    ) -> MediaCuratedResult:
        """Return curated media bundles for a list of concept names.

        Args:
            concepts: One or more concept names (case-insensitive, fuzzy-matched
                against the curated mapping). Italian or English.
            domain: 'neuro' or 'udl'. Defaults to 'neuro'.
        """
        if not isinstance(concepts, list) or not concepts:
            raise ValueError("`concepts` must be a non-empty list of strings")

        lookup = _get_media_lookup(domain)
        summaries: List[MediaCuratedSummary] = []
        matched = 0

        for name in concepts:
            content = lookup.find_media_for_concept(name)
            if content is None or not content.has_content():
                summaries.append(
                    MediaCuratedSummary(
                        concept_name=name,
                        found=False,
                        counts={
                            "videos": 0,
                            "images": 0,
                            "resources": 0,
                            "citations": 0,
                            "open_textbooks": 0,
                        },
                        items={},
                    )
                )
                continue

            matched += 1
            summaries.append(
                MediaCuratedSummary(
                    concept_name=name,
                    found=True,
                    counts={
                        "videos": len(content.videos),
                        "images": len(content.images),
                        "resources": len(content.resources),
                        "citations": len(content.citations),
                        "open_textbooks": len(content.open_textbooks),
                    },
                    items={
                        "videos": [_to_dict(v) for v in content.videos],
                        "images": [_to_dict(i) for i in content.images],
                        "resources": [_to_dict(r) for r in content.resources],
                        "citations": [_to_dict(c) for c in content.citations],
                        "open_textbooks": [
                            _to_dict(t) for t in content.open_textbooks
                        ],
                    },
                )
            )

        return MediaCuratedResult(
            domain=domain,  # type: ignore[arg-type]
            requested=len(concepts),
            matched=matched,
            by_concept=summaries,
        )

    # ---- media.search_youtube --------------------------------------------
    @mcp.tool(
        name="media.search_youtube",
        description=(
            "Search YouTube for educational videos in real time. Uses the "
            "YouTube Data API when YOUTUBE_API_KEY is set in the environment; "
            "otherwise falls back to a deterministic search-URL generator so "
            "the call still succeeds. Always returns at least one item or "
            "an empty list — never raises."
        ),
        tags={"media", "youtube", "live"},
    )
    async def media_search_youtube(
        query: str,
        max_results: int = 5,
        language: str = "en",
    ) -> Dict[str, Any]:
        """Search YouTube for educational videos.

        Args:
            query: Search query (Italian or English).
            max_results: Cap on returned videos (1..20).
            language: Two-letter ISO language code (e.g. 'en', 'it').
        """
        if not query or not query.strip():
            raise ValueError("`query` must be a non-empty string")
        if max_results < 1 or max_results > 20:
            raise ValueError("`max_results` must be between 1 and 20")

        api = _get_external_api()
        try:
            videos = await api.search_youtube(
                query=query.strip(),
                max_results=max_results,
                language=language or "en",
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("[media.search_youtube] failed: %s", exc)
            return {
                "query": query,
                "count": 0,
                "videos": [],
                "error": "live YouTube search failed",
                "details": str(exc),
            }

        return {
            "query": query,
            "count": len(videos),
            "language": language,
            "api_key_present": bool(os.environ.get("YOUTUBE_API_KEY")),
            "videos": [_to_dict(v) for v in videos],
        }

    # ---- media.search_academic -------------------------------------------
    @mcp.tool(
        name="media.search_academic",
        description=(
            "Search Semantic Scholar for academic papers in real time. "
            "Rate-limited (100 calls / 5 min on the free tier; raise the cap "
            "by setting SEMANTIC_SCHOLAR_API_KEY). Supports a year cutoff and "
            "an open-access filter. Always returns a list (possibly empty) or "
            "an inlined error — never raises."
        ),
        tags={"media", "academic", "live", "semantic-scholar"},
    )
    async def media_search_academic(
        query: str,
        max_results: int = 5,
        year_from: Optional[int] = None,
        open_access_only: bool = False,
    ) -> Dict[str, Any]:
        """Search Semantic Scholar for academic papers.

        Args:
            query: Search query (English works best on Semantic Scholar).
            max_results: Cap on returned papers (1..25).
            year_from: Optional minimum publication year (e.g. 2018).
            open_access_only: When True, filter to open-access papers.
        """
        if not query or not query.strip():
            raise ValueError("`query` must be a non-empty string")
        if max_results < 1 or max_results > 25:
            raise ValueError("`max_results` must be between 1 and 25")
        if year_from is not None and (year_from < 1900 or year_from > 2100):
            raise ValueError("`year_from` must be a plausible 4-digit year")

        api = _get_external_api()
        try:
            papers = await api.search_semantic_scholar(
                query=query.strip(),
                max_results=max_results,
                year_from=year_from,
                open_access_only=open_access_only,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("[media.search_academic] failed: %s", exc)
            return {
                "query": query,
                "count": 0,
                "papers": [],
                "error": "live academic search failed",
                "details": str(exc),
            }

        return {
            "query": query,
            "count": len(papers),
            "year_from": year_from,
            "open_access_only": open_access_only,
            "api_key_present": bool(
                os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
            ),
            "papers": [_to_dict(p) for p in papers],
        }

    # ---- media.search_oer -------------------------------------------------
    @mcp.tool(
        name="media.search_oer",
        description=(
            "Search Open Educational Resources (OER) textbooks across DOAB, "
            "Open Textbook Library, and BC Campus OpenEd. All results are "
            "domain-expert approved and copyright-safe (CC BY 4.0 by default). "
            "No API key required."
        ),
        tags={"media", "oer", "textbook", "live"},
    )
    async def media_search_oer(
        query: str,
        max_results: int = 5,
        language: str = "en",
    ) -> Dict[str, Any]:
        """Search Open Educational Resource catalogues.

        Args:
            query: Topic to search.
            max_results: Cap on total textbooks returned (1..15).
            language: Language preference, two-letter code (e.g. 'en', 'it').
        """
        if not query or not query.strip():
            raise ValueError("`query` must be a non-empty string")
        if max_results < 1 or max_results > 15:
            raise ValueError("`max_results` must be between 1 and 15")

        api = _get_external_api()
        try:
            textbooks = await api.search_oer_textbooks(
                query=query.strip(),
                max_results=max_results,
                language=language or "en",
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("[media.search_oer] failed: %s", exc)
            return {
                "query": query,
                "count": 0,
                "textbooks": [],
                "error": "live OER search failed",
                "details": str(exc),
            }

        return {
            "query": query,
            "count": len(textbooks),
            "language": language,
            "textbooks": [_to_dict(t) for t in textbooks],
        }

    # ---- media.generate_diagram ------------------------------------------
    @mcp.tool(
        name="media.generate_diagram",
        description=(
            "Generate an educational Mermaid.js diagram for a concept. "
            "Uses gpt-4o-mini to produce the Mermaid code, then renders it "
            "via the FREE mermaid.ink service (returns both SVG and PNG "
            "URLs the client can embed directly). Diagram types: mindmap, "
            "flowchart, sequence, timeline, hierarchy, comparison, process."
        ),
        tags={"media", "diagram", "mermaid", "generative"},
    )
    async def media_generate_diagram(
        concept: str,
        diagram_type: Literal[
            "mindmap",
            "flowchart",
            "sequence",
            "timeline",
            "hierarchy",
            "comparison",
            "process",
        ] = "mindmap",
        related_concepts: Optional[List[str]] = None,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """Generate a Mermaid.js diagram for an educational concept.

        Args:
            concept: The central concept (Italian or English).
            diagram_type: One of the supported Mermaid diagram types.
            related_concepts: Optional list of related concepts to include
                as branches/nodes; used to enrich the diagram structure.
            validate: When True, the generator validates the rendered SVG
                actually loads from mermaid.ink before returning success.
        """
        if not concept or not concept.strip():
            raise ValueError("`concept` must be a non-empty string")

        gen = _get_mermaid_generator()
        try:
            result = await gen.generate(
                concept=concept.strip(),
                diagram_type=diagram_type,
                related_concepts=related_concepts or None,
                validate=validate,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("[media.generate_diagram] failed: %s", exc)
            return {
                "success": False,
                "concept": concept,
                "diagram_type": diagram_type,
                "error": "diagram generation failed",
                "details": str(exc),
            }

        # MermaidResult exposes ``to_dict``; fall back to dataclass coercion.
        if hasattr(result, "to_dict"):
            return result.to_dict()
        return _to_dict(result)

    # Touch _ to silence "unused" linters — FastMCP decorators register them.
    _ = (
        media_lookup_curated,
        media_search_youtube,
        media_search_academic,
        media_search_oer,
        media_generate_diagram,
    )
