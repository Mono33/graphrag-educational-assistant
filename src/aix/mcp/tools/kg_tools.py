"""
KG (Knowledge Graph) MCP tools — Phase 1 of CORE 5 #20.

Wraps four read-only entry points to the educational Neo4j KG:

* ``kg.search``         — free-text natural-language query → ranked nodes,
                          relationships, methodology recommendations.
* ``kg.get_context``    — same shape ``POST /api/v1/context`` returns
                          (educational ranking + media counts), so any MCP
                          client can replace a /context HTTP call with a
                          single tool invocation.
* ``kg.list_concepts``  — fast catalogue of concept names available in a
                          domain — useful as a "discovery" pre-step before
                          a search.
* ``kg.get_schema``     — node labels + categories + relationship intent,
                          for clients that want to craft a precise query
                          rather than fuzz over the search.

All four tools are read-only and idempotent. They lazily instantiate their
backing components (``GraphRAGTool``, ``MediaLookup``, domain configs) on
first call and cache them in module-level singletons so subsequent calls
inside the same MCP session avoid the ~13s schema cache warm-up cost.

Phase 5 (HTTP transport) will reuse the same singletons because the FastMCP
ASGI mount lives in the same uvicorn process as the FastAPI app — see
``aix.mcp.server`` for the architecture.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from typing import Any, Dict, List, Literal, Optional

from fastmcp import FastMCP
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public output schemas
# ---------------------------------------------------------------------------
# We expose Pydantic models (not raw dicts) so the FastMCP-generated tool
# schema sent to MCP clients carries proper field-level descriptions and
# constraints. This is the same UX MCP Inspector / Claude Desktop / Cursor
# render in their tool pickers.
#
# IMPORTANT: these mirror existing public contracts where possible:
#   * ``KgSearchResult`` is a JSON-friendly view of ``GraphRAGResult``
#     (the dataclass returned by ``GraphRAGTool.search()``).
#   * ``KgSchemaResult`` is bespoke to the schema introspection use case.
# ---------------------------------------------------------------------------


DomainLiteral = Literal["neuro", "udl"]


class KgSearchNode(BaseModel):
    """A node returned by a GraphRAG search."""

    name: Optional[str] = Field(default=None, description="Node display name.")
    label: Optional[str] = Field(
        default=None, description="Neo4j label (e.g. 'Methodology', 'Concept')."
    )
    description: Optional[str] = Field(
        default=None, description="Natural-language description if present in the KG."
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Other node properties as returned by Neo4j.",
    )


class KgSearchRelationship(BaseModel):
    source: str
    type: str
    target: str


class KgSearchResult(BaseModel):
    """JSON-friendly view of ``GraphRAGResult`` for MCP consumers."""

    domain: DomainLiteral
    query: str
    nodes: List[KgSearchNode]
    relationships: List[KgSearchRelationship]
    recommendations: List[Dict[str, Any]] = Field(
        description="Ranked methodology recommendations (educational context)."
    )
    confidence: str = Field(
        description="Educational-context confidence assessment "
        "('LOW' / 'MEDIUM' / 'HIGH')."
    )
    cypher_query: Optional[str] = Field(
        default=None,
        description="Cypher query the Text2Cypher layer generated. "
        "Useful for transparency / debugging.",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KgConceptList(BaseModel):
    """Lightweight list of concepts with media coverage."""

    domain: DomainLiteral
    count: int
    concepts: List[str] = Field(description="Concept names available in the domain.")


class KgSchemaResult(BaseModel):
    """Schema introspection result for a domain."""

    domain: DomainLiteral
    display_name: str
    description: Optional[str] = None
    label_categories: Dict[str, str] = Field(
        description="Map of Neo4j node label → human-readable category."
    )
    similarity_threshold: Optional[float] = Field(
        default=None,
        description="Default cosine similarity threshold used by Node2Vec retrieval.",
    )
    methodology_categories: Dict[str, Any] = Field(
        default_factory=dict,
        description="Domain-specific methodology category groupings.",
    )


# ---------------------------------------------------------------------------
# Lazy backing-component singletons
# ---------------------------------------------------------------------------
# Cold start of the GraphRAGTool is the single biggest latency hit (Neo4j
# driver init + schema cache warm-up; ~13s on the dev box, see terminals
# 23.txt). We cache one instance per domain so the second call in the same
# process is sub-second.
_GRAPHRAG_TOOLS: Dict[str, Any] = {}
_MEDIA_LOOKUPS: Dict[str, Any] = {}


def _get_graphrag_tool(domain: str):
    """Return a cached ``GraphRAGTool`` for the given domain (lazy init)."""
    if domain not in _GRAPHRAG_TOOLS:
        from aix.agent.tools.graphrag_tool import GraphRAGTool

        _GRAPHRAG_TOOLS[domain] = GraphRAGTool(domain=domain)
    return _GRAPHRAG_TOOLS[domain]


def _get_media_lookup(domain: str):
    if domain not in _MEDIA_LOOKUPS:
        from aix.agent.media.media_lookup import MediaLookup

        _MEDIA_LOOKUPS[domain] = MediaLookup(domain=domain)
    return _MEDIA_LOOKUPS[domain]


def _node_dict_to_model(raw: Dict[str, Any]) -> KgSearchNode:
    """Best-effort coercion of a raw Neo4j-derived dict into ``KgSearchNode``."""
    if not isinstance(raw, dict):
        return KgSearchNode(
            name=str(raw),
            label=None,
            description=None,
            properties={},
        )
    name = raw.get("name") or raw.get("title") or raw.get("id")
    label = raw.get("label") or raw.get("type") or raw.get("category")
    description = raw.get("description") or raw.get("summary")
    skip = {"name", "title", "id", "label", "type", "category", "description", "summary"}
    properties = {k: v for k, v in raw.items() if k not in skip}
    return KgSearchNode(
        name=name,
        label=label,
        description=description,
        properties=properties,
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(mcp: FastMCP) -> None:
    """Register all four kg.* tools onto the shared FastMCP instance."""

    # ---- kg.search --------------------------------------------------------
    @mcp.tool(
        name="kg.search",
        description=(
            "Search the educational Knowledge Graph for teaching strategies, "
            "methodologies, and neuroscience concepts. Accepts natural-language "
            "Italian or English. Returns matched nodes, their relationships, "
            "and ranked methodology recommendations.\n\n"
            "Use this tool when the teacher's question is open-ended and you "
            "want the GraphRAG pipeline to: (1) translate to Cypher, "
            "(2) retrieve from Neo4j, (3) build educational context. For a "
            "drop-in replacement of POST /api/v1/context, use kg.get_context."
        ),
        tags={"kg", "search", "graphrag"},
    )
    async def kg_search(
        query: str,
        domain: DomainLiteral = "neuro",
    ) -> KgSearchResult:
        """Free-text Knowledge Graph search.

        Args:
            query: The teacher's question in Italian or English (e.g.
                "Quali strategie per studenti con ADHD?").
            domain: Knowledge domain — 'neuro' (neuroscience methodologies)
                or 'udl' (Universal Design for Learning).
        """
        tool = _get_graphrag_tool(domain)
        result = await tool.search(query)

        return KgSearchResult(
            domain=domain,  # type: ignore[arg-type]
            query=query,
            nodes=[_node_dict_to_model(n) for n in result.nodes],
            relationships=[
                KgSearchRelationship(**r) if isinstance(r, dict) else KgSearchRelationship(
                    source=str(r[0]), type=str(r[1]), target=str(r[2])
                )
                for r in result.relationships
            ],
            recommendations=list(result.recommendations or []),
            confidence=str(result.confidence),
            cypher_query=result.cypher_query,
            metadata=dict(result.metadata or {}),
        )

    # ---- kg.get_context ---------------------------------------------------
    # We delegate to the same underlying GraphRAGTool the /api/v1/context
    # route uses, then enrich the response with media counts mirroring the
    # /context Pydantic shape. Any MCP client can therefore call kg.get_context
    # and get the same payload they would get from POST /api/v1/context.
    @mcp.tool(
        name="kg.get_context",
        description=(
            "Return the same educational-context payload that "
            "POST /api/v1/context produces: ranked methodologies, retrieved "
            "nodes, and a media-coverage summary. Use this tool when the "
            "client wants the canonical 'context bundle' shape rather than "
            "the raw search output of kg.search."
        ),
        tags={"kg", "context", "graphrag", "education"},
    )
    async def kg_get_context(
        query: str,
        domain: DomainLiteral = "neuro",
        max_methodologies: int = 10,
    ) -> Dict[str, Any]:
        """Get the educational-context bundle for a query.

        Args:
            query: Teacher question in Italian or English.
            domain: 'neuro' or 'udl'.
            max_methodologies: Cap on ranked recommendations (1..50).
        """
        if max_methodologies < 1 or max_methodologies > 50:
            raise ValueError("max_methodologies must be between 1 and 50")

        tool = _get_graphrag_tool(domain)
        result = await tool.search(query)

        # Mirror the public ContextResponse shape closely enough that an MCP
        # client can swap kg.get_context for POST /api/v1/context with no
        # downstream changes. We stay conservative on field names.
        recommendations = list(result.recommendations or [])[:max_methodologies]

        media_counts: Dict[str, int] = {"videos": 0, "images": 0, "resources": 0}
        try:
            media = _get_media_lookup(domain)
            concept_names = [
                (n.get("name") or n.get("title") or "")
                for n in result.nodes
                if isinstance(n, dict)
            ]
            concept_names = [c for c in concept_names if c]
            if concept_names:
                combined = media.get_combined_media(concept_names)
                media_counts = {
                    "videos": len(getattr(combined, "videos", []) or []),
                    "images": len(getattr(combined, "images", []) or []),
                    "resources": len(getattr(combined, "resources", []) or []),
                }
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[kg.get_context] media lookup failed: %s", exc)

        return {
            "domain": domain,
            "query": query,
            "confidence": str(result.confidence),
            "cypher_query": result.cypher_query,
            "nodes_count": len(result.nodes),
            "relationships_count": len(result.relationships),
            "methodologies": recommendations,
            "media_counts": media_counts,
            "metadata": dict(result.metadata or {}),
        }

    # ---- kg.list_concepts -------------------------------------------------
    @mcp.tool(
        name="kg.list_concepts",
        description=(
            "List all concept names that have curated media coverage in the "
            "given domain. Cheap (file-backed read, no Neo4j round-trip). "
            "Use this BEFORE kg.search when the client wants to discover "
            "what's available before crafting a query."
        ),
        tags={"kg", "discovery", "metadata"},
    )
    def kg_list_concepts(
        domain: DomainLiteral = "neuro",
        limit: int = 200,
    ) -> KgConceptList:
        """List concept names available in the domain.

        Args:
            domain: 'neuro' or 'udl'.
            limit: Maximum number of concept names to return (1..1000).
        """
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        media = _get_media_lookup(domain)
        all_concepts = media.get_all_concepts() or []
        truncated = list(all_concepts)[:limit]
        return KgConceptList(
            domain=domain,  # type: ignore[arg-type]
            count=len(truncated),
            concepts=truncated,
        )

    # ---- kg.get_schema ----------------------------------------------------
    @mcp.tool(
        name="kg.get_schema",
        description=(
            "Return the schema of a domain — node label categories, "
            "similarity threshold, methodology categories. Useful when an "
            "MCP client wants to craft a precise query rather than fuzz "
            "over kg.search."
        ),
        tags={"kg", "schema", "metadata"},
    )
    def kg_get_schema(
        domain: DomainLiteral = "neuro",
    ) -> KgSchemaResult:
        """Inspect the schema of a domain.

        Args:
            domain: 'neuro' or 'udl'.
        """
        from aix.domains import get_domain_config

        config = get_domain_config(domain)
        if config is None:
            raise ValueError(f"Unknown domain: {domain}")

        try:
            label_map = config.get_label_category_map() or {}
        except Exception:
            label_map = {}

        try:
            similarity_threshold = float(config.get_similarity_threshold())
        except Exception:
            similarity_threshold = None

        try:
            methodology_categories = config.get_methodology_categories() or {}
        except Exception:
            methodology_categories = {}

        try:
            description = config.get_description()
        except Exception:
            description = None

        try:
            display_name = getattr(config, "display_name", domain)
        except Exception:
            display_name = domain

        return KgSchemaResult(
            domain=domain,  # type: ignore[arg-type]
            display_name=str(display_name),
            description=description,
            label_categories=label_map,
            similarity_threshold=similarity_threshold,
            methodology_categories=methodology_categories,
        )

    # Touch _ to satisfy linters about "unused" inner functions — they ARE
    # used: FastMCP's decorators register them on the server.
    _ = (kg_search, kg_get_context, kg_list_concepts, kg_get_schema)
