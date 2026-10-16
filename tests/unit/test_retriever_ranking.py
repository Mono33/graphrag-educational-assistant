"""The retriever's merged result is ranked once: every reader takes its head."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aix.agent.agents.planner_agent import RetrievalPlan
from aix.agent.agents.retriever_agent import RetrieverAgent
from aix.agent.tools.graphrag_tool import GraphRAGResult

pytestmark = pytest.mark.unit

#: The first search was about the class: it used to head the merged list.
BY_QUERY = {
    "classe": GraphRAGResult(
        nodes=[
            {"name": "Group till  15", "labels": ["Context"], "rank_score": 1.6},
            {"name": "Cohesive", "labels": ["Context"], "rank_score": 1.6},
        ],
        relationships=[],
        recommendations=[
            {"name": "Group till  15", "category": "Contesto Classe", "relevance_score": 0.5}
        ],
        confidence="MEDIUM",
    ),
    "adhd": GraphRAGResult(
        nodes=[
            {"name": "Difficulty sustaining focus", "labels": ["Adhd"], "rank_score": 1.6},
            {"name": "Focus Tasks", "labels": ["InstructionalTechnique"], "rank_score": 1.7},
        ],
        relationships=[],
        recommendations=[{"name": "Focus Tasks", "relevance_score": 0.9}],
        confidence="HIGH",
    ),
}


async def test_strategies_head_the_merged_result():
    agent = RetrieverAgent(domain="udl", enable_media_lookup=False, enable_external_apis=False)

    async def _search(query):
        return BY_QUERY[query]

    agent._tool = SimpleNamespace(search=_search)
    plan = RetrievalPlan(
        query_intent="lesson_creation", key_concepts=[], search_queries=["classe", "adhd"]
    )

    result = await agent.retrieve(plan)

    assert [n["name"] for n in result.nodes] == [
        "Focus Tasks",
        "Difficulty sustaining focus",
        "Group till  15",
        "Cohesive",
    ]
    assert [r["name"] for r in result.recommendations] == ["Focus Tasks", "Group till  15"]
