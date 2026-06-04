"""
Lock the response shape of ``agent.run_lesson_plan`` (Phase 4 tool).

The MCP wrapper and the public HTTP endpoint ``POST /api/v1/agent/run``
share the SAME response model (``AgentRunResponse``). If a future refactor
diverges them, partner integrations that talk over MCP will break in ways
that look like bugs in the partner's code rather than in ours.

This test:
    1. Monkey-patches ``stream_agent_events`` to emit a canned 6-event
       sequence (planner → retriever → writer_pending → writer → critic →
       done) — the same approach ``tests/api/test_agent_routes.py`` uses.
    2. Calls ``agent.run_lesson_plan`` via the in-memory MCP client.
    3. Asserts the structured response contains every field the public
       API contract promises, with the right types.

The test does NOT exercise OpenRouter, Neo4j, or any LLM — those are
out of scope for a contract test (they're tested by the manual smoke
runs in Phase 4 ``--phase4-verify``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Canned event stream — mirrors the shape ``test_agent_routes.test_agent_run_happy_path``
# uses, so MCP and HTTP wrappers exercise the same fixture.
# ---------------------------------------------------------------------------
_CANNED_EVENTS: list[dict[str, Any]] = [
    {
        "kind": "planner",
        "payload": {
            "intent": "lesson_creation",
            "intent_label": "Creazione lezione",
            "scope": "in_scope",
            "scope_label": "Nel Knowledge Graph",
            "key_concepts": ["attenzione", "memoria di lavoro"],
            "search_queries": ["strategie attenzione adolescenti"],
        },
    },
    {
        "kind": "retriever",
        "payload": {
            "nodes_count": 7,
            "relationships_count": 12,
            "recommendations_count": 3,
            "media_counts": {"videos": 2, "articles": 1, "oer": 4},
            "media": {"videos": [], "resources": []},
            "top_concepts": ["Attenzione sostenuta"],
            "retrieval_confidence": "HIGH",
        },
    },
    {"kind": "writer_pending", "payload": {"revision": 1}},
    {
        "kind": "writer",
        "payload": {"revision": 1},
        "lesson_plan_md": "# Bozza",
    },
    {"kind": "critic", "payload": {"approved": True, "score": 4.5}},
    {
        "kind": "done",
        "lesson_plan_md": "# Lezione finale\n\nContenuto strutturato della lezione.",
        "meta": {
            "duration_seconds": 12.3,
            "approved": True,
            "revision_count": 0,
            "scores": {
                "average_score": 4.5,
                "scientific_accuracy": 4.5,
                "pedagogical_quality": 4.5,
                "neuroscience_alignment": 4.5,
                "udl_compliance": 4.5,
                "completeness": 4.5,
            },
            "nodes_count": 7,
            "recommendations_count": 3,
            "media_counts": {"videos": 2, "articles": 1, "oer": 4},
            "search_queries_count": 1,
        },
    },
]


def _patch_stream_agent_events(monkeypatch) -> None:
    """Replace the real ``stream_agent_events`` with one that emits the canned events.

    We patch *both* the source module and the MCP tool module — Python's
    import binding means patching the source alone wouldn't catch references
    already imported at function-call time inside the MCP tool body.
    """
    from aix.webui.agent.service import StreamEvent

    async def _fake_stream(**_kwargs) -> AsyncIterator[StreamEvent]:
        for e in _CANNED_EVENTS:
            yield StreamEvent(
                kind=e["kind"],
                payload=e.get("payload", {}),
                lesson_plan_md=e.get("lesson_plan_md"),
                error=e.get("error"),
                meta=e.get("meta", {}),
            )

    import aix.mcp.tools.agent_tools as agent_tools_mod
    import aix.webui.agent.service as service_mod

    monkeypatch.setattr(service_mod, "stream_agent_events", _fake_stream)
    # The MCP tool imports lazily inside the function body (line 219 of
    # agent_tools.py), so the stream symbol is re-resolved on each call —
    # patching service_mod is enough. We still patch the module attribute
    # explicitly for safety in case someone refactors to a top-level import.
    if hasattr(agent_tools_mod, "stream_agent_events"):
        monkeypatch.setattr(agent_tools_mod, "stream_agent_events", _fake_stream)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_agent_run_lesson_plan_happy_path(mcp_client, monkeypatch):
    """A full canned stream produces a complete ``AgentRunResponse``."""
    _patch_stream_agent_events(monkeypatch)

    result = await mcp_client.call_tool(
        "agent.run_lesson_plan",
        {
            "query": "Crea una lezione sull'attenzione per la classe terza media",
            "domain": "neuro",
            "language": "it",
            "max_revisions": 0,
        },
    )

    payload = getattr(result, "structured_content", None) or getattr(
        result, "data", None
    )
    assert payload is not None, "agent.run_lesson_plan returned no structured payload"

    # Top-level shape — same contract as POST /api/v1/agent/run.
    assert set(payload.keys()) >= {"lesson_plan_md", "meta", "planner", "retriever"}, (
        f"Top-level keys drifted: {sorted(payload.keys())}"
    )

    # Lesson body propagated from the ``done`` event.
    assert payload["lesson_plan_md"].startswith("# Lezione finale")

    # Meta block (matches AgentRunMeta).
    meta = payload["meta"]
    assert meta["approved"] is True
    assert meta["duration_seconds"] == pytest.approx(12.3)
    assert meta["revision_count"] == 0
    assert meta["nodes_count"] == 7
    assert meta["recommendations_count"] == 3
    assert meta["media_counts"] == {"videos": 2, "articles": 1, "oer": 4}
    assert meta["search_queries_count"] == 1

    # Planner explainability.
    planner = payload["planner"]
    assert planner["intent"] == "lesson_creation"
    assert planner["scope"] == "in_scope"
    assert planner["key_concepts"] == ["attenzione", "memoria di lavoro"]

    # Retriever explainability.
    retriever = payload["retriever"]
    assert retriever["nodes_count"] == 7
    assert retriever["retrieval_confidence"] == "HIGH"
    assert retriever["media_counts"] == {"videos": 2, "articles": 1, "oer": 4}


@pytest.mark.asyncio
async def test_agent_run_lesson_plan_validates_query(mcp_client):
    """``query`` shorter than 3 chars must raise — no LLM call attempted."""
    from fastmcp.exceptions import ToolError

    with pytest.raises((ToolError, Exception)) as exc_info:
        await mcp_client.call_tool(
            "agent.run_lesson_plan",
            {"query": "x", "domain": "neuro"},
        )
    msg = str(exc_info.value).lower()
    assert "query" in msg or "3" in msg or "validation" in msg, (
        f"Unexpected tool error: {exc_info.value!r}"
    )


@pytest.mark.asyncio
async def test_agent_run_lesson_plan_propagates_runtime_error(mcp_client, monkeypatch):
    """An ``error`` StreamEvent → tool failure (FastMCP isError=true)."""
    from fastmcp.exceptions import ToolError

    from aix.webui.agent.service import StreamEvent

    async def _failing_stream(**_kwargs):
        yield StreamEvent(kind="error", error="Knowledge Graph unreachable")

    import aix.webui.agent.service as service_mod

    monkeypatch.setattr(service_mod, "stream_agent_events", _failing_stream)

    with pytest.raises((ToolError, Exception)) as exc_info:
        await mcp_client.call_tool(
            "agent.run_lesson_plan",
            {"query": "Crea una lezione su qualcosa", "domain": "neuro"},
        )
    assert "Knowledge Graph unreachable" in str(exc_info.value), (
        f"Expected the agent error message to bubble up, got: {exc_info.value!r}"
    )
