"""The private retriever inside the graph (Fase 2 · PR 6, §8.1 / §8.9).

Two retrievers now run in the same superstep, which is the situation the
reducers in ``state.py`` were written for: both branches return
``current_step``, and without a reducer LangGraph raises
``InvalidUpdateError: can receive only one value per step``. §8.9 says to prove
that by running them, not by reading the code, so the topology test here builds
the real graph.

The rest is about what must keep working when the private half does not: an API
or CLI run registers no search function, and a search that fails must cost the
lesson its handout, never its lesson.
"""

from __future__ import annotations

import pytest

from aix.agent.graph import material_source
from aix.agent.graph.nodes import retrieve_private_node
from aix.agent.graph.state import merge_material_items

pytestmark = pytest.mark.unit

SESSION = "lesson-1"

PASSAGE = {
    "source_kind": "teacher_material",
    "source_id": "v1",
    "text": "Yalta divise l'Europa in due sfere.",
    "score": 0.91,
    "citation": "[[material:v1#page=1]]",
    "instruction_trust": "data_only",
}


@pytest.fixture(autouse=True)
def _clean_registry():
    yield
    material_source.deregister(SESSION)


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------


async def test_the_node_returns_the_passages_it_was_given():
    async def _search(query):
        assert "guerra fredda" in query, "the subject concepts sharpen the query"
        return [PASSAGE]

    material_source.register(SESSION, _search)

    updates = await retrieve_private_node(
        {
            "session_id": SESSION,
            "teacher_query": "amplia la lezione",
            "subject_concepts": ["guerra fredda", "Yalta"],
        }
    )

    assert updates["material_context_items"] == [PASSAGE]
    assert updates["current_step"] == "retrieve_complete"


async def test_without_a_registered_search_nothing_is_retrieved():
    """An API or CLI run has no teacher session, and keeps its old topology."""
    updates = await retrieve_private_node({"session_id": "no-such-session"})

    assert "material_context_items" not in updates
    assert updates["current_step"] == "retrieve_complete"


async def test_a_failed_search_costs_the_handout_not_the_lesson():
    async def _explode(_query):
        raise RuntimeError("the embedding provider is down")

    material_source.register(SESSION, _explode)

    updates = await retrieve_private_node({"session_id": SESSION, "teacher_query": "x"})

    assert "error" not in updates, "the Knowledge Graph alone still writes a lesson"
    assert updates["current_step"] == "retrieve_complete"


async def test_the_search_is_removed_with_its_session():
    material_source.register(SESSION, lambda _q: [])
    material_source.deregister(SESSION)

    assert material_source.get(SESSION) is None


# ---------------------------------------------------------------------------
# The reducer (§8.9)
# ---------------------------------------------------------------------------


def test_both_branches_survive_the_merge():
    kg_turn = [{"citation": "[[material:a]]", "score": 0.4}]
    private_turn = [{"citation": "[[material:b]]", "score": 0.9}]

    assert len(merge_material_items(kg_turn, private_turn)) == 2


def test_the_order_comes_from_the_score_not_from_which_branch_landed_first():
    low = [{"citation": "[[material:a]]", "score": 0.2}]
    high = [{"citation": "[[material:b]]", "score": 0.95}]

    assert merge_material_items(low, high) == merge_material_items(high, low)
    assert merge_material_items(low, high)[0]["citation"] == "[[material:b]]"


def test_the_same_passage_twice_gets_one_share_of_the_budget():
    once = [{"citation": "[[material:a#page=3]]", "score": 0.8}]

    assert len(merge_material_items(once, list(once))) == 1


def test_nothing_to_merge_keeps_what_there_was():
    existing = [{"citation": "[[material:a]]", "score": 0.5}]

    assert merge_material_items(existing, None) == existing
    assert merge_material_items(None, None) == []


# ---------------------------------------------------------------------------
# The topology, built for real
# ---------------------------------------------------------------------------


async def _topology(monkeypatch, *, private: str):
    monkeypatch.setenv("AIX_PRIVATE_RAG_ENABLED", private)
    from aix.agent.graph.lesson_planner_graph import build_lesson_planner_graph_async

    graph = (await build_lesson_planner_graph_async()).get_graph()
    return (
        {node for node in graph.nodes},
        {(edge.source, edge.target) for edge in graph.edges},
    )


async def test_the_two_retrievers_run_beside_each_other(monkeypatch):
    nodes, edges = await _topology(monkeypatch, private="true")

    assert "retrieve_private" in nodes
    assert ("plan", "retrieve") in edges
    assert ("plan", "retrieve_private") in edges
    assert ("retrieve_private", "write") in edges, "write joins both branches"


async def test_with_the_flag_off_the_topology_is_the_one_it_always_was(monkeypatch):
    nodes, edges = await _topology(monkeypatch, private="false")

    assert "retrieve_private" not in nodes
    assert not any(target == "retrieve_private" for _source, target in edges)


# ---------------------------------------------------------------------------
# What to search for (measured on the first real lesson, 2026-09-16)
# ---------------------------------------------------------------------------


def test_the_search_is_for_the_subject_not_the_request():
    """The request's words describe the task and matched the English PDF's boilerplate."""
    from aix.agent.graph.nodes import _private_search_query

    query = _private_search_query(
        {
            "teacher_query": "crea una lezione basandoti sui documenti allegati",
            "educational_profile": {"specific_topic": "guerra fredda"},
            "subject_concepts": ["Conferenza di Yalta", "cortina di ferro", "guerra fredda"],
        }
    )

    assert query == "guerra fredda · Conferenza di Yalta · cortina di ferro"
    assert "documenti allegati" not in query


def test_without_a_subject_the_request_is_the_query():
    from aix.agent.graph.nodes import _private_search_query

    assert _private_search_query({"teacher_query": "riassumi il documento"}) == (
        "riassumi il documento"
    )
