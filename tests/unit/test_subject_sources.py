"""The lesson's subject sources are about the subject (staging, 2026-10-16).

The Writer of the water-cycle lesson was handed «Sam Neill» and «Poltergeist di
Enfield» as its Wikipedia summaries, and the profile's need code «DLDS» was
searched in the graph as if it were a word.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from aix.agent.agents.retriever_agent import RetrieverAgent
from aix.agent.media.external_apis import ExternalMediaAPI, _shares_a_word

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("query", "title", "about"),
    [
        ("water cycle", "Sam Neill", False),
        ("evaporation", "Poltergeist di Enfield", False),
        ("condensation", "Condensato di Bose-Einstein", False),
        ("ciclo dell'acqua", "Ciclo dell'acqua", True),
        ("water cycle", "Water cycle", True),
        ("Guerra fredda", "Guerra fredda", True),
    ],
)
def test_a_search_hit_must_be_about_the_query(query, title, about):
    assert _shares_a_word(query, title) is about


async def test_an_unrelated_search_hit_is_no_summary():
    api = ExternalMediaAPI()
    response = SimpleNamespace(
        status=200, json=AsyncMock(return_value={"query": {"search": [{"title": "Sam Neill"}]}})
    )

    class _Get:
        async def __aenter__(self):
            return response

        async def __aexit__(self, *exc):
            return False

    session = SimpleNamespace(get=lambda *a, **k: _Get())
    with (
        patch.object(api, "_get_session", AsyncMock(return_value=session)),
        patch.object(api, "get_wikipedia_summary", AsyncMock()) as summary,
    ):
        assert await api._wikipedia_search("water cycle", "it") is None
    summary.assert_not_awaited()


async def test_the_english_wikipedia_when_the_italian_has_no_page():
    agent = RetrieverAgent(domain="udl", enable_media_lookup=False)
    found = SimpleNamespace(
        title="Water cycle",
        summary="The water cycle…",
        url="https://en.wikipedia.org/wiki/Water_cycle",
        thumbnail_url=None,
    )

    async def _summary(concept, language="en"):
        return None if language == "it" else found

    api = SimpleNamespace(
        get_wikipedia_summary=AsyncMock(side_effect=_summary),
        search_openalex=AsyncMock(return_value=[]),
        search_oer_textbooks=AsyncMock(return_value=[]),
    )
    agent._external_api = api

    resources = await agent._fetch_external_resources(["water cycle"], [])

    assert [w["title"] for w in resources["wikipedia"]] == ["Water cycle"]
    assert [c.kwargs.get("language") for c in api.get_wikipedia_summary.await_args_list] == [
        "it",
        "en",
    ]


async def test_the_profile_needs_are_searched_by_their_label_not_their_code():
    from tests.unit.agent_doubles import _plan

    from aix.agent.graph import nodes as graph_nodes

    retriever = SimpleNamespace(
        retrieve=AsyncMock(
            return_value=SimpleNamespace(
                nodes=[{"name": "x"}],
                relationships=[],
                recommendations=[],
                search_results=[],
                confidence="MEDIUM",
                curated_media={},
                external_resources={},
                is_hybrid=False,
            )
        )
    )
    with patch.object(graph_nodes, "get_retriever", return_value=retriever):
        await graph_nodes.retrieve_node(
            {
                "plan": _plan().__dict__,
                "teacher_query": "q",
                "educational_profile": {"group": {"disabilities": ["DLDS"]}},
                "domain": "neuro",
            }
        )

    queries = retriever.retrieve.await_args.args[0].search_queries
    assert "Difficoltà linguistiche (straniero/a)" in queries
    assert "DLDS" not in queries
