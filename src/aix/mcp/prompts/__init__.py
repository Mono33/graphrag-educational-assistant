"""
MCP prompts for the Aix GraphRAG server — Phase 2 of CORE 5 #20.

Prompts are *parameterised, server-defined LLM templates* that any MCP
client can render and submit. They let teachers (or any consumer of the
server, including external apps via Streamable HTTP in Phase 5) invoke
standardised educational workflows from any client UI without each client
having to know how to format an Aix-specific query or lesson-plan brief.

Conventions
-----------
* Names are lowercase-kebab and namespaced by intent rather than transport
  (``educational-query`` rather than ``aix.educational-query``) — matches
  the reference Anthropic / FastMCP examples.
* Each prompt accepts only typed primitives (``str``, ``int``, ``Optional``).
  This keeps the JSON-schema FastMCP generates clean and renderable in
  Claude Desktop / Cursor IDE prompt pickers.
* Prompts return ``list[{"role": "user" | "system", "content": str}]``,
  the standard MCP message envelope.
"""

from __future__ import annotations

import logging

from fastmcp import FastMCP

logger = logging.getLogger(__name__)


def register(mcp: FastMCP) -> None:
    """Register every Phase 2 prompt onto the shared FastMCP instance."""
    from aix.mcp.prompts import educational_prompts

    educational_prompts.register(mcp)
    logger.info("[MCP] Registered Phase 2 prompts (educational-query, lesson-plan-request)")


__all__ = ["register"]
