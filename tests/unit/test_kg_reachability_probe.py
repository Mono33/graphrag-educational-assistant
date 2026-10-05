"""The Knowledge Graph probe: asked only when it matters, and believed (§6.3).

The teacher is told one of two very different things when the search comes back
empty — «this question is outside the Knowledge Graph» or «the Knowledge Graph
did not answer» — and the only way to tell them apart is to ask the database.
Asking costs a round trip, so these tests pin *when* it is asked as firmly as
what is done with the answer.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.unit


def _result(nodes: list) -> SimpleNamespace:
    return SimpleNamespace(
        nodes=nodes,
        relationships=[],
        recommendations=[],
        search_results=[],
        confidence="MEDIUM",
        curated_media={},
        external_resources={},
        is_hybrid=False,
    )


async def _retrieve(nodes: list, *, reachable: bool) -> dict:
    from tests.unit.agent_doubles import _plan

    from aix.agent.graph import nodes as graph_nodes

    probe = AsyncMock(return_value=reachable)
    retriever = SimpleNamespace(retrieve=AsyncMock(return_value=_result(nodes)))

    with (
        patch.object(graph_nodes, "get_retriever", return_value=retriever),
        patch.object(graph_nodes, "_knowledge_graph_is_reachable", probe),
    ):
        state = {
            "plan": _plan().__dict__,
            "teacher_query": "una lezione sulla guerra fredda",
            "educational_profile": {},
            "domain": "neuro",
        }
        out = await graph_nodes.retrieve_node(state)

    return {"out": out, "probe_calls": probe.await_count}


async def test_nothing_found_and_the_database_is_silent():
    got = await _retrieve([], reachable=False)

    assert got["out"]["kg_unreachable"] is True
    assert got["probe_calls"] == 1


async def test_nothing_found_because_the_question_is_outside_the_graph():
    """The database answers; it simply has nothing on the Cold War. Not an outage."""
    got = await _retrieve([], reachable=True)

    assert got["out"]["kg_unreachable"] is False
    assert got["probe_calls"] == 1


async def test_a_search_that_found_concepts_never_pays_for_the_probe():
    got = await _retrieve([{"title": "Scaffolding"}], reachable=True)

    assert got["out"]["kg_unreachable"] is False
    assert got["probe_calls"] == 0, "the probe is the empty-result path only"
