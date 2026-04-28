"""
CORE 5 #20 — MCP (Model Context Protocol) tool servers.

This package exposes Aix's GraphRAG, media, and agent tools as MCP servers
so external AI clients (Claude Desktop, Cursor IDE, Lovable, ChatGPT desktop,
partner LangGraph agents, etc.) can call them through the standard JSON-RPC
protocol — without depending on our specific FastAPI surface or Python imports.

Architecture (Option C — hybrid stdio + Streamable HTTP):
    src/aix/mcp/
    ├── __init__.py            (this file — public re-exports)
    ├── server.py              single ``FastMCP`` instance, registers all primitives
    ├── tools/
    │   ├── kg_tools.py        Phase 1 — 4 kg.* tools (KG search / context / concepts / schema)
    │   ├── media_tools.py     Phase 3 — 5 media.* tools (lookup_curated,
    │   │                                                  search_youtube, search_academic,
    │   │                                                  search_oer, generate_diagram)
    │   └── agent_tools.py     Phase 4 — 1 agent.run_lesson_plan tool (full pipeline,
    │                                     same shape as POST /api/v1/agent/run)
    ├── resources/
    │   ├── __init__.py        Phase 2 — register dispatcher
    │   └── kg_resources.py    Phase 2 — 4 resources (kg://schema, kg://concepts/{domain},
    │                                                 methodology://list, media://stats)
    ├── prompts/
    │   ├── __init__.py        Phase 2 — register dispatcher
    │   └── educational_prompts.py  Phase 2 — 2 prompts (educational-query,
    │                                                    lesson-plan-request)
    ├── stdio_main.py          ``python -m aix.mcp.stdio_main`` for local MCP clients
    └── http_app.py            Phase 5 — Streamable HTTP ASGI app mounted at /mcp/
                                          inside aix/api/main.py, behind the same
                                          JWT Bearer secret as /api/v1/agent/*

The same ``mcp`` instance backs both transports — there is exactly one source
of truth for tool definitions, resources, and prompts. Adding a new tool means
adding one ``@mcp.tool`` decorator; both stdio and HTTP automatically expose it.

See: ``docs/integrations/MCP_Setup.md`` for client setup (stdio + HTTP).
See: ``docs/product/ClickUp_Agentic_GraphRAG_Update.md`` → Subtask 20.
"""

from aix.mcp.server import mcp, build_mcp_server
from aix.mcp.http_app import build_mcp_http_app, MCP_MOUNT_PATH

__all__ = ["mcp", "build_mcp_server", "build_mcp_http_app", "MCP_MOUNT_PATH"]
