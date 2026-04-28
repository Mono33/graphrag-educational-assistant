"""
Phase 2 MCP resources — KG metadata, methodology catalogues, media coverage.

Why resources (not tools)?
--------------------------
MCP distinguishes *tools* (interactive operations the LLM chooses to invoke
mid-turn) from *resources* (browseable read-only data the client can list and
attach to context). The four artifacts here are pure metadata:

* ``kg://schema``                — node label categories + similarity threshold
                                   + methodology-category groupings, both domains.
* ``kg://concepts/{domain}``     — every concept name with curated media coverage
                                   in the requested domain.
* ``methodology://list``         — methodology categories per domain.
* ``media://stats``              — curated-media coverage stats per domain
                                   (videos, images, resources, citations,
                                    open textbooks).

Resources are returned as Python ``dict`` / ``list`` and FastMCP auto-encodes
them to JSON via the ``application/json`` MIME type. This is the same
serialisation the tools layer uses, so MCP clients get a consistent shape.

Performance
-----------
``MediaLookup`` is the heaviest dependency (it loads ~94k-line JSON mapping
into a dict on first use) — we cache one instance per domain and reuse it for
every read. Domain config introspection is cheap (~ms). No Neo4j calls happen
in any of the resources, so reads stay sub-second even on a cold process.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastmcp import FastMCP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy backing-component singletons (mirrors aix.mcp.tools.kg_tools pattern).
# ---------------------------------------------------------------------------
_MEDIA_LOOKUPS: Dict[str, Any] = {}

_VALID_DOMAINS = ("neuro", "udl")


def _get_media_lookup(domain: str):
    """Return a cached ``MediaLookup`` for the given domain (lazy init)."""
    if domain not in _MEDIA_LOOKUPS:
        from aix.agent.media.media_lookup import MediaLookup

        _MEDIA_LOOKUPS[domain] = MediaLookup(domain=domain)
    return _MEDIA_LOOKUPS[domain]


def _domain_schema_dict(domain: str) -> Dict[str, Any]:
    """Build a serializable schema snapshot for a single domain.

    Mirrors the shape returned by the ``kg.get_schema`` tool so clients get
    the same view whether they read the resource or call the tool.
    """
    from aix.domains import get_domain_config

    config = get_domain_config(domain)
    if config is None:
        return {"domain": domain, "error": f"unknown domain: {domain}"}

    def _safe(callable_, default):
        try:
            return callable_()
        except Exception:
            return default

    return {
        "domain": domain,
        "display_name": getattr(config, "display_name", domain),
        "description": _safe(config.get_description, None),
        "label_categories": _safe(config.get_label_category_map, {}),
        "similarity_threshold": _safe(config.get_similarity_threshold, None),
        "methodology_categories": _safe(config.get_methodology_categories, {}),
    }


def _media_stats_for_domain(domain: str) -> Dict[str, Any]:
    """Compute curated-media coverage stats for a single domain.

    Iterates the in-memory media mapping (already loaded by ``MediaLookup``)
    and counts how many concepts have at least one item per media kind.
    Cheap (single pass over an already-resident dict).
    """
    media = _get_media_lookup(domain)
    if not getattr(media, "loaded", False):
        return {
            "domain": domain,
            "loaded": False,
            "total_concepts": 0,
            "concepts_with_media": 0,
            "by_kind": {
                "videos": 0,
                "images": 0,
                "resources": 0,
                "citations": 0,
                "open_textbooks": 0,
            },
        }

    by_kind = {
        "videos": 0,
        "images": 0,
        "resources": 0,
        "citations": 0,
        "open_textbooks": 0,
    }
    concepts_with_media = 0

    for concept in media.media_by_concept.values():
        has_any = False
        for kind in by_kind:
            items = concept.get(kind) or []
            if items:
                by_kind[kind] += 1
                has_any = True
        if has_any:
            concepts_with_media += 1

    return {
        "domain": domain,
        "loaded": True,
        "total_concepts": len(media.media_by_concept),
        "concepts_with_media": concepts_with_media,
        "by_kind": by_kind,
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(mcp: FastMCP) -> None:
    """Register the four Phase 2 KG/methodology/media resources."""

    # ---- kg://schema ------------------------------------------------------
    @mcp.resource(
        uri="kg://schema",
        name="kg.schema",
        description=(
            "Full schema snapshot for both Knowledge Graph domains "
            "('neuro' and 'udl'): node label categories, default similarity "
            "threshold used by Node2Vec retrieval, and methodology category "
            "groupings. Use this resource as the LLM's grounding before "
            "calling kg.search or kg.get_context. Equivalent to calling the "
            "kg.get_schema tool once per domain, but available as a single "
            "discoverable read."
        ),
        mime_type="application/json",
        tags={"kg", "schema", "metadata"},
    )
    def kg_schema() -> Dict[str, Any]:
        return {
            "domains": [_domain_schema_dict(d) for d in _VALID_DOMAINS],
        }

    # ---- kg://concepts/{domain} ------------------------------------------
    @mcp.resource(
        uri="kg://concepts/{domain}",
        name="kg.concepts",
        description=(
            "All concept names with curated media coverage in the requested "
            "domain ('neuro' or 'udl'). Useful for autocomplete, validating "
            "user input, or surfacing what's available before crafting a "
            "kg.search query. Cheap (file-backed, no Neo4j round-trip)."
        ),
        mime_type="application/json",
        tags={"kg", "concepts", "discovery"},
    )
    def kg_concepts(domain: str) -> Dict[str, Any]:
        domain_norm = (domain or "").strip().lower()
        if domain_norm not in _VALID_DOMAINS:
            return {
                "domain": domain,
                "error": f"unknown domain '{domain}'. Valid: {list(_VALID_DOMAINS)}",
                "concepts": [],
                "count": 0,
            }
        media = _get_media_lookup(domain_norm)
        all_concepts: List[str] = list(media.get_all_concepts() or [])
        return {
            "domain": domain_norm,
            "count": len(all_concepts),
            "concepts": all_concepts,
        }

    # ---- methodology://list ----------------------------------------------
    @mcp.resource(
        uri="methodology://list",
        name="methodology.list",
        description=(
            "Pedagogical methodology category groupings per domain. The "
            "'neuro' domain exposes neuroscience-grounded methodologies "
            "(executive function, motivation, etc.); the 'udl' domain "
            "exposes Universal Design for Learning principles. Useful when "
            "a client wants to filter or scope a lesson plan to a specific "
            "methodology family before invoking the agent."
        ),
        mime_type="application/json",
        tags={"methodology", "metadata", "education"},
    )
    def methodology_list() -> Dict[str, Any]:
        from aix.domains import get_domain_config

        out: Dict[str, Any] = {}
        for domain in _VALID_DOMAINS:
            config = get_domain_config(domain)
            if config is None:
                out[domain] = {"error": "domain unavailable"}
                continue
            try:
                cats = config.get_methodology_categories() or {}
            except Exception as exc:
                cats = {"error": str(exc)}
            out[domain] = {
                "display_name": getattr(config, "display_name", domain),
                "categories": cats,
            }
        return {"domains": out}

    # ---- media://stats ----------------------------------------------------
    @mcp.resource(
        uri="media://stats",
        name="media.stats",
        description=(
            "Curated-media coverage statistics per domain — total concepts, "
            "concepts with at least one media item, and per-kind counts "
            "(videos, images, external resources, academic citations, open "
            "textbooks). Lets a client decide whether to bother with media "
            "lookups for a given domain before invoking media.* tools."
        ),
        mime_type="application/json",
        tags={"media", "stats", "metadata"},
    )
    def media_stats() -> Dict[str, Any]:
        return {
            "domains": [_media_stats_for_domain(d) for d in _VALID_DOMAINS],
        }

    _ = (kg_schema, kg_concepts, methodology_list, media_stats)
