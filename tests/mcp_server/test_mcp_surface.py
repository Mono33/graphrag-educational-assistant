"""
Lock the FastMCP server's *surface* — every tool, resource, and prompt name
shipped in Phases 1-5 must remain discoverable by clients.

External MCP clients (Claude Desktop, Cursor IDE, Lovable, partner LangGraph
agents) bind to primitive names by string. Renaming or removing one is a
silent breaking change for every client out there. These tests catch that.
"""

from __future__ import annotations

import pytest

# The exact inventory shipped by Phases 1-5. Update this set deliberately if
# (and only if) you intend to introduce a breaking change — and bump the
# server version alongside it. See ``aix.mcp.server._SERVER_VERSION``.
EXPECTED_TOOLS = {
    # Phase 1 — KG
    "kg.search",
    "kg.get_context",
    "kg.list_concepts",
    "kg.get_schema",
    # Phase 3 — Media
    "media.lookup_curated",
    "media.search_youtube",
    "media.search_academic",
    "media.search_oer",
    "media.generate_diagram",
    # Phase 4 — Agent
    "agent.run_lesson_plan",
}

EXPECTED_RESOURCES_STATIC = {
    "kg://schema",
    "methodology://list",
    "media://stats",
}

# Templated resources are listed via ``list_resource_templates`` rather than
# ``list_resources`` — see CHANGELOG of the smoke script for the rationale.
EXPECTED_RESOURCE_TEMPLATES = {
    "kg://concepts/{domain}",
}

EXPECTED_PROMPTS = {
    "educational-query",
    "lesson-plan-request",
}


@pytest.mark.asyncio
async def test_tool_inventory(mcp_client):
    """All 10 tools shipped in Phases 1-5 are discoverable."""
    tools = await mcp_client.list_tools()
    names = {t.name for t in tools}
    missing = EXPECTED_TOOLS - names
    extra = names - EXPECTED_TOOLS
    assert not missing, f"Missing expected tools: {sorted(missing)}"
    assert not extra, (
        "Unexpected tools registered (update EXPECTED_TOOLS deliberately if "
        f"this is intentional): {sorted(extra)}"
    )
    assert len(names) == 10, f"Tool count drifted: got {len(names)}, expected 10"


@pytest.mark.asyncio
async def test_resource_inventory(mcp_client):
    """All 3 static resources are discoverable."""
    resources = await mcp_client.list_resources()
    uris = {str(r.uri) for r in resources}
    missing = EXPECTED_RESOURCES_STATIC - uris
    assert not missing, f"Missing expected static resources: {sorted(missing)}"


@pytest.mark.asyncio
async def test_resource_template_inventory(mcp_client):
    """The templated ``kg://concepts/{domain}`` resource is discoverable."""
    templates = await mcp_client.list_resource_templates()
    template_uris = {str(t.uriTemplate) for t in templates}
    missing = EXPECTED_RESOURCE_TEMPLATES - template_uris
    assert not missing, (
        f"Missing expected templated resources: {sorted(missing)}"
    )


@pytest.mark.asyncio
async def test_prompt_inventory(mcp_client):
    """Both Phase 2 prompts are discoverable."""
    prompts = await mcp_client.list_prompts()
    names = {p.name for p in prompts}
    missing = EXPECTED_PROMPTS - names
    assert not missing, f"Missing expected prompts: {sorted(missing)}"


def test_server_identity(mcp_server):
    """The server advertises the canonical name + non-empty instructions.

    These are the strings MCP clients show in their tool-picker UI. Drift
    here would surprise users who already configured their client to look
    for ``aix-graphrag``.
    """
    assert mcp_server.name == "aix-graphrag"
    instructions = getattr(mcp_server, "instructions", None) or ""
    assert "GraphRAG" in instructions or "Knowledge Graph" in instructions, (
        "Server instructions should mention GraphRAG / Knowledge Graph — "
        "this is what the LLM sees as the system prompt."
    )
