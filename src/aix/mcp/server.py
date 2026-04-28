"""
Single ``FastMCP`` server instance shared by all transports.

This module is the *one* place where tools, resources, and prompts get
registered. Both entry points (``stdio_main.py`` for local clients and
``http_app.py`` mounted under FastAPI at ``/mcp/`` for remote clients)
import the same ``mcp`` instance from here, so every primitive is
available on every transport with zero duplication.

Phase 1 (LANDED): 4 KG tools (kg.search, kg.get_context, kg.list_concepts,
                  kg.get_schema). Stdio entry only.
Phase 2 (LANDED): 4 kg:// resources + 2 prompt templates.
Phase 3 (LANDED): 5 media.* tools (lookup_curated, search_youtube,
                  search_academic, search_oer, generate_diagram).
Phase 4 (LANDED): 1 agent.run_lesson_plan tool (wraps stream_agent_events
                  from #7 with MCP progress notifications).
Phase 5 (LANDED): Streamable HTTP transport mounted at /mcp/ in
                  ``aix.api.main``, validated by FastMCP's ``JWTVerifier``
                  configured with the same HS256 secret + audience used by
                  fastapi-users' bearer backend (see ``aix.mcp.http_app``).

Design notes:
- Tool registration is delegated to ``register(mcp)`` functions inside each
  ``aix.mcp.tools.<x>_tools`` module. ``build_mcp_server()`` is the single
  composition root — calling it twice is a no-op (idempotent guard) so it
  is safe to import the module from multiple entry points.
- We bias toward *graceful degradation*: if a tool group fails to register
  (e.g. Neo4j unreachable during cold-start of the stdio entry on a dev
  laptop without the DB up), we log a warning and keep going so the rest
  of the surface still works. This mirrors the ``try/except`` discipline
  used in ``aix.api.main``.
"""

from __future__ import annotations

import logging
import os
from typing import Final

from fastmcp import FastMCP

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server identity
# ---------------------------------------------------------------------------
# The MCP spec uses ``name`` for client-side display in tool pickers / chooser
# UIs and ``version`` for compatibility checks. ``instructions`` is shown to
# the LLM as a top-level system message when the server is first negotiated —
# treat it as the "README the LLM reads".
_SERVER_NAME: Final[str] = "aix-graphrag"
_SERVER_VERSION: Final[str] = os.environ.get("AIX_MCP_VERSION", "0.1.0")
_SERVER_INSTRUCTIONS: Final[str] = (
    "Aix GraphRAG MCP server — exposes the educational Knowledge Graph "
    "(neuro + UDL domains) and lesson-planning agent as MCP tools.\n\n"
    "Use ``kg.search`` for free-text queries (Italian or English), "
    "``kg.get_context`` for the same shape ``POST /api/v1/context`` returns "
    "(includes educational ranking + media), ``kg.list_concepts`` to discover "
    "what's in a domain before a search, and ``kg.get_schema`` to inspect "
    "node labels and relationship types when crafting a precise query.\n\n"
    "Domains: 'neuro' (neuroscience-grounded methodologies) and 'udl' "
    "(Universal Design for Learning). All tools default to 'neuro' if "
    "domain is unspecified."
)

# ---------------------------------------------------------------------------
# Global FastMCP instance
# ---------------------------------------------------------------------------
mcp: FastMCP = FastMCP(
    name=_SERVER_NAME,
    version=_SERVER_VERSION,
    instructions=_SERVER_INSTRUCTIONS,
)


# ---------------------------------------------------------------------------
# Composition root — idempotent registration of all primitives
# ---------------------------------------------------------------------------
_REGISTERED: bool = False


def build_mcp_server() -> FastMCP:
    """Register every tool / resource / prompt onto the shared ``mcp`` instance.

    Idempotent: calling this twice is a no-op. This lets both the stdio entry
    point and the (future) HTTP mount call it on startup without worrying
    about duplicate registrations.

    Returns the same ``mcp`` instance that lives at module scope, for
    callers that prefer an explicit handle.
    """
    global _REGISTERED
    if _REGISTERED:
        return mcp

    # Phase 1 — KG tools.
    try:
        from aix.mcp.tools import kg_tools

        kg_tools.register(mcp)
        logger.info("[MCP] Registered kg.* tools (Phase 1)")
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("[MCP] Failed to register kg.* tools: %s", exc)

    # Phase 2 — Resources + prompts.
    try:
        from aix.mcp import resources as mcp_resources

        mcp_resources.register(mcp)
        logger.info("[MCP] Registered Phase 2 resources")
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("[MCP] Failed to register Phase 2 resources: %s", exc)

    try:
        from aix.mcp import prompts as mcp_prompts

        mcp_prompts.register(mcp)
        logger.info("[MCP] Registered Phase 2 prompts")
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("[MCP] Failed to register Phase 2 prompts: %s", exc)

    # Phase 3 — Media tools.
    try:
        from aix.mcp.tools import media_tools

        media_tools.register(mcp)
        logger.info("[MCP] Registered media.* tools (Phase 3)")
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("[MCP] Failed to register media.* tools: %s", exc)

    # Phase 4 — Agent tool.
    try:
        from aix.mcp.tools import agent_tools

        agent_tools.register(mcp)
        logger.info("[MCP] Registered agent.* tools (Phase 4)")
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("[MCP] Failed to register agent.* tools: %s", exc)

    _REGISTERED = True
    logger.info(
        "[MCP] Server '%s' v%s ready (Phases 1+2+3+4+5 — KG, resources, prompts, media, agent; stdio + Streamable HTTP)",
        _SERVER_NAME,
        _SERVER_VERSION,
    )
    return mcp
