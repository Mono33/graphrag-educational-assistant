"""
Lock the input/output schema of the cheap (no-Neo4j) KG tools.

We deliberately exercise the two tools that read from local config files
rather than hitting Neo4j, so this test runs sub-second and never fails
because of an external service:

    * ``kg.list_concepts`` — reads ``MediaLookup`` (file-backed JSON catalog).
    * ``kg.get_schema``    — reads ``aix.domains`` (Python module config).

These two tools are also the most commonly used by external clients as
"discovery" pre-steps before a richer ``kg.search`` call, so locking
their shape here protects the most-used surface.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _structured(call_result):
    """Pull the structured payload out of a FastMCP CallToolResult.

    ``Client.call_tool`` returns an object whose ``.structured_content``
    holds the typed Pydantic dict (or ``.data`` in older fastmcp builds).
    We accept either to keep the test resilient across minor releases.
    """
    payload = getattr(call_result, "structured_content", None)
    if payload is None:
        payload = getattr(call_result, "data", None)
    assert payload is not None, (
        "FastMCP returned a tool result without structured_content / data — "
        "schema serialisation is broken."
    )
    return payload


# ---------------------------------------------------------------------------
# kg.list_concepts
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_kg_list_concepts_neuro(mcp_client):
    """``kg.list_concepts`` returns the canonical ``KgConceptList`` shape."""
    result = await mcp_client.call_tool(
        "kg.list_concepts",
        {"domain": "neuro", "limit": 5},
    )
    payload = _structured(result)

    assert payload["domain"] == "neuro"
    assert "count" in payload
    assert "concepts" in payload
    assert isinstance(payload["concepts"], list)
    assert payload["count"] == len(payload["concepts"])
    assert payload["count"] <= 5
    # The neuro KG is non-empty, so we expect at least one concept.
    assert payload["count"] >= 1, (
        "Neuro domain should have at least one curated concept — got 0. "
        "Check that data/aix/domains/neuro/media_concepts.json exists."
    )


@pytest.mark.asyncio
async def test_kg_list_concepts_udl(mcp_client):
    """Same shape for the UDL domain (smoke that both domains stay wired)."""
    result = await mcp_client.call_tool(
        "kg.list_concepts",
        {"domain": "udl", "limit": 3},
    )
    payload = _structured(result)
    assert payload["domain"] == "udl"
    assert isinstance(payload["concepts"], list)
    assert payload["count"] <= 3


@pytest.mark.asyncio
async def test_kg_list_concepts_validates_limit(mcp_client):
    """``limit`` outside 1..1000 must surface as a tool error.

    FastMCP wraps ``ValueError`` raised inside a tool body into an
    ``isError=true`` response. We accept that as the failure signal.
    """
    from fastmcp.exceptions import ToolError

    with pytest.raises((ToolError, Exception)) as exc_info:
        await mcp_client.call_tool(
            "kg.list_concepts",
            {"domain": "neuro", "limit": 9999},
        )
    msg = str(exc_info.value).lower()
    assert "limit" in msg or "1000" in msg or "validation" in msg, (
        f"Unexpected tool error message: {exc_info.value!r}"
    )


# ---------------------------------------------------------------------------
# kg.get_schema
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_kg_get_schema_neuro(mcp_client):
    """``kg.get_schema`` returns the canonical ``KgSchemaResult`` shape."""
    result = await mcp_client.call_tool("kg.get_schema", {"domain": "neuro"})
    payload = _structured(result)

    assert payload["domain"] == "neuro"
    assert "display_name" in payload
    assert isinstance(payload["display_name"], str)
    assert isinstance(payload["label_categories"], dict)
    assert isinstance(payload["methodology_categories"], dict)
    # similarity_threshold can be None if the domain doesn't expose one;
    # we just assert it's either None or a positive float.
    sim = payload.get("similarity_threshold")
    assert sim is None or (isinstance(sim, (int, float)) and sim > 0), (
        f"similarity_threshold must be None or a positive number, got {sim!r}"
    )


@pytest.mark.asyncio
async def test_kg_get_schema_udl(mcp_client):
    """Same shape for the UDL domain."""
    result = await mcp_client.call_tool("kg.get_schema", {"domain": "udl"})
    payload = _structured(result)
    assert payload["domain"] == "udl"
    assert "display_name" in payload
