"""
MCP resources for the Aix GraphRAG server — Phase 2 of CORE 5 #20.

Resources are *read-only data exposures* that MCP clients can list and read
without invoking any side-effecting tool. They are the "table of contents"
of the server: clients use them for discovery, autocomplete, and grounding
the LLM in our domain shape *before* a tool call.

Naming conventions
------------------
* URI scheme follows the resource's *concern*, not transport:
    - ``kg://...``         — Knowledge Graph metadata (schema, concepts)
    - ``methodology://...`` — pedagogical methodology catalogues
    - ``media://...``       — curated media coverage statistics
* Static resources have no path parameters: ``kg://schema``.
* Template resources interpolate parameters: ``kg://concepts/{domain}``.

Registration is centralised behind ``register(mcp)`` so ``aix.mcp.server``
has exactly one call site per phase, matching the existing ``kg_tools``
pattern.
"""

from __future__ import annotations

import logging

from fastmcp import FastMCP

logger = logging.getLogger(__name__)


def register(mcp: FastMCP) -> None:
    """Register every Phase 2 resource onto the shared FastMCP instance."""
    from aix.mcp.resources import kg_resources

    kg_resources.register(mcp)
    logger.info("[MCP] Registered Phase 2 resources (kg://, methodology://, media://)")


__all__ = ["register"]
