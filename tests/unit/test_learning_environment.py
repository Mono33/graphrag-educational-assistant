"""The graph's classroom nodes, read when the class needs them (UDL expert, 2026-10-16)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from aix.agent.agents.retriever_agent import RetrieverAgent

pytestmark = pytest.mark.unit

ROWS = {
    "Colour": [
        {
            "name": "Neutral tones",
            "labels": ["Colour"],
            "description": "Neutral colours reduce overstimulation, aiding focus.",
        }
    ],
    "Checkpoint": [
        {
            "name": "address distractions",
            "labels": ["Checkpoint"],
            "description": "Minimize sensory distractions by structuring the environment.",
            "guideline": "recruit interest",
        }
    ],
}


class _Tool:
    def __init__(self, fail: bool = False):
        self.calls: list[dict] = []
        self.fail = fail

    async def read(self, cypher, **params):
        if self.fail:
            raise RuntimeError("database down")
        self.calls.append(params)
        return ROWS["Checkpoint"] if "Checkpoint" in cypher else ROWS["Colour"]


async def test_the_environment_nodes_and_checkpoints_come_back_as_nodes():
    agent = RetrieverAgent(domain="udl", enable_media_lookup=False, enable_external_apis=False)
    agent._tool = tool = _Tool()

    nodes = await agent.fetch_learning_environment()

    assert [n["name"] for n in nodes] == ["Neutral tones", "address distractions"]
    assert nodes[1]["guideline"] == "recruit interest"
    assert tool.calls[0]["domain"] == "udl"
    assert "Smells" not in tool.calls[0]["labels"]


async def test_sensory_needs_add_smells_and_textures():
    agent = RetrieverAgent(domain="udl", enable_media_lookup=False, enable_external_apis=False)
    agent._tool = tool = _Tool()

    await agent.fetch_learning_environment(sensory=True)

    assert {"Smells", "Textures"} <= set(tool.calls[0]["labels"])


async def test_a_failed_read_is_no_environment_not_a_failed_lesson():
    agent = RetrieverAgent(domain="udl", enable_media_lookup=False, enable_external_apis=False)
    agent._tool = _Tool(fail=True)

    assert await agent.fetch_learning_environment() == []


def _result(nodes):
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


async def _retrieve(profile, *, domain="udl", enabled="true", monkeypatch=None):
    from tests.unit.agent_doubles import _plan

    from aix.agent.graph import nodes as graph_nodes

    monkeypatch.setenv("AIX_KG_ENVIRONMENT_ENABLED", enabled)
    fetch = AsyncMock(return_value=[{"name": "Neutral tones", "labels": ["Colour"]}])
    retriever = SimpleNamespace(
        retrieve=AsyncMock(
            return_value=_result([{"name": "Group till  15", "labels": ["Context"]}])
        ),
        fetch_learning_environment=fetch,
    )
    with patch.object(graph_nodes, "get_retriever", return_value=retriever):
        out = await graph_nodes.retrieve_node(
            {
                "plan": _plan().__dict__,
                "teacher_query": "il ciclo dell'acqua",
                "educational_profile": profile,
                "domain": domain,
            }
        )
    return out, fetch


ADHD = {"group": {"disabilities": ["ADHD"]}}


async def test_a_class_with_adhd_gets_the_room(monkeypatch):
    out, fetch = await _retrieve(ADHD, monkeypatch=monkeypatch)

    assert fetch.await_count == 1
    assert [n["name"] for n in out["retrieved_nodes"]] == ["Neutral tones", "Group till  15"]


async def test_not_without_the_need_nor_outside_udl_nor_when_switched_off(monkeypatch):
    _, fetch = await _retrieve({"group": {"disabilities": ["DLDS"]}}, monkeypatch=monkeypatch)
    assert fetch.await_count == 0
    _, fetch = await _retrieve(ADHD, domain="neuro", monkeypatch=monkeypatch)
    assert fetch.await_count == 0
    _, fetch = await _retrieve(ADHD, enabled="false", monkeypatch=monkeypatch)
    assert fetch.await_count == 0
