"""Conditional retrieval: reuse the corpus when a turn adds nothing new.

Most follow-ups do not need a second Knowledge Graph search — "riscrivila più
corta" and "adatta per ADHD" work from the evidence already gathered. Only a
turn that introduces a concept the corpus does not cover has to go back to
Neo4j.

The decision is the planner's rather than a manual toggle, because the planner
already extracts the turn's ``key_concepts``; comparing those against what the
thread holds costs nothing. A toggle would ask the teacher to predict whether
their own question introduces a new concept.

Gated by ``AIX_CONDITIONAL_RETRIEVAL_ENABLED``. With the flag off the
``plan → retrieve`` edge stays unconditional, which the first test pins.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from tests.unit.agent_doubles import _critique, _plan, _retrieval_result

from aix.agent.graph import nodes as graph_nodes
from aix.agent.graph.lesson_planner_graph import (
    build_lesson_planner_graph,
    get_compiled_graph,
    reset_compiled_graph,
)
from aix.agent.graph.nodes import (
    _corpus_covers_this_turn,
    reuse_retrieval_node,
    should_retrieve,
)
from aix.agent.graph.state import create_initial_state

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("AIX_CORRECTIVE_RAG_ENABLED", raising=False)
    reset_compiled_graph()
    yield
    reset_compiled_graph()


@pytest.fixture
def _on(monkeypatch):
    monkeypatch.setenv("AIX_CONDITIONAL_RETRIEVAL_ENABLED", "true")


@pytest.fixture
def _off(monkeypatch):
    monkeypatch.delenv("AIX_CONDITIONAL_RETRIEVAL_ENABLED", raising=False)


def _followup(**over):
    """A state shaped like a second turn: history present, corpus in place."""
    state = dict(create_initial_state(query="riscrivila più corta"))
    state.update(
        conversation_history=[{"role": "user", "content": "prima domanda"}],
        key_concepts=["memoria di lavoro"],
        retrieved_nodes=[{"name": "Memoria di Lavoro"}],
    )
    state.update(over)
    return state


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


def test_flag_off_keeps_the_unconditional_edge(_off):
    compiled = build_lesson_planner_graph()
    edges = {(e.source, e.target, bool(e.conditional)) for e in compiled.get_graph().edges}

    assert ("plan", "retrieve", False) in edges, "the pre-T6 edge, unconditional"
    assert "reuse" not in compiled.get_graph().nodes


def test_flag_on_forks_the_planner(_on):
    compiled = build_lesson_planner_graph()
    graph = compiled.get_graph()
    edges = {(e.source, e.target) for e in graph.edges}

    assert "reuse" in graph.nodes
    assert ("plan", "retrieve") in edges
    assert ("plan", "reuse") in edges
    assert ("reuse", "write") in edges
    assert {e for e in graph.edges if e.source == "plan" and not e.conditional} == set()


# ---------------------------------------------------------------------------
# The routing decision
# ---------------------------------------------------------------------------


def test_flag_off_always_retrieves(_off):
    assert should_retrieve(_followup()) == "retrieve"


def test_upstream_error_keeps_todays_path(_on):
    """Every remaining node no-ops to the end; the route must not change that."""
    assert should_retrieve(_followup(error="boom")) == "retrieve"


@pytest.mark.parametrize(
    "override, expected",
    [(True, "retrieve"), (False, "reuse")],
    ids=["forced on", "forced off"],
)
def test_teacher_override_wins_in_both_directions(override, expected, _on):
    # Set up so the planner's own verdict would be the opposite of the override.
    state = _followup(force_retrieval=override)
    if override is True:
        assert _corpus_covers_this_turn(state), "planner would have said reuse"
    assert should_retrieve(state) == expected


def test_first_turn_always_retrieves(_on):
    assert should_retrieve(_followup(conversation_history=[])) == "retrieve"


def test_empty_corpus_retrieves(_on):
    """A follow-up on a thread whose earlier turn failed has nothing to reuse."""
    assert should_retrieve(_followup(retrieved_nodes=[])) == "retrieve"


def test_reuses_when_the_turn_adds_no_new_concept(_on):
    assert should_retrieve(_followup()) == "reuse"


def test_retrieves_when_the_turn_introduces_a_new_concept(_on):
    state = _followup(key_concepts=["memoria di lavoro", "carico cognitivo"])
    assert should_retrieve(state) == "retrieve"


# ---------------------------------------------------------------------------
# Concept coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "concepts, corpus, covered, why",
    [
        (["memoria di lavoro"], ["Memoria di Lavoro"], True, "case is not significant"),
        (["memoria"], ["Memoria di Lavoro"], True, "concept inside a node name"),
        (["memoria di lavoro"], ["Memoria"], True, "node name inside the concept"),
        (["carico cognitivo"], ["Memoria di Lavoro"], False, "genuinely new"),
        (["memoria", "carico"], ["Memoria di Lavoro"], False, "one uncovered is enough"),
        ([], ["Memoria"], True, "the planner named nothing to look for"),
        (["memoria"], [], False, "nothing retrieved yet"),
        (["  ", None], ["Memoria"], True, "blank concepts are ignored"),
    ],
)
def test_corpus_coverage(concepts, corpus, covered, why):
    state = {
        "key_concepts": concepts,
        "retrieved_nodes": [{"name": n} for n in corpus],
    }
    assert _corpus_covers_this_turn(state) is covered, why


def test_coverage_also_reads_the_recommendations():
    state = {
        "key_concepts": ["chunking"],
        "retrieved_nodes": [],
        "recommendations": [{"name": "Chunking"}],
    }
    assert _corpus_covers_this_turn(state) is True


async def test_reuse_node_marks_the_turn():
    result = await reuse_retrieval_node(_followup())
    assert result["retrieval_reused"] is True
    assert result["current_step"] == "retrieve_complete"


# ---------------------------------------------------------------------------
# End to end — the acceptance criterion
# ---------------------------------------------------------------------------


async def _two_turns(monkeypatch, *, second_turn_concepts):
    """Run two turns on one thread; return (final state of turn 2, retriever mock)."""
    monkeypatch.setattr(
        "aix.agent.graph.checkpointer.get_checkpointer",
        AsyncMock(return_value=InMemorySaver()),
    )

    first_plan = _plan()
    second_plan = _plan()
    second_plan.key_concepts = second_turn_concepts

    planner = AsyncMock(plan=AsyncMock(side_effect=[first_plan, second_plan]))
    retriever = AsyncMock(retrieve=AsyncMock(return_value=_retrieval_result()))
    writer = AsyncMock(write=AsyncMock(return_value="# Lezione"))
    critic = AsyncMock(critique=AsyncMock(side_effect=[_critique(True), _critique(True)]))

    graph = await get_compiled_graph()
    config = {"configurable": {"thread_id": "t6"}}

    with (
        patch.object(graph_nodes, "get_planner", return_value=planner),
        patch.object(graph_nodes, "get_retriever", return_value=retriever),
        patch.object(graph_nodes, "get_writer", return_value=writer),
        patch.object(graph_nodes, "get_critic", return_value=critic),
    ):
        await graph.ainvoke(create_initial_state(query="Crea una lezione"), config=config)
        second = await graph.ainvoke(
            create_initial_state(
                query="riscrivila più corta",
                conversation_history=[{"role": "user", "content": "Crea una lezione"}],
            ),
            config=config,
        )
    return second, retriever


async def test_followup_without_new_concepts_skips_neo4j(monkeypatch, _on):
    second, retriever = await _two_turns(monkeypatch, second_turn_concepts=["memoria di lavoro"])

    assert retriever.retrieve.await_count == 1, "only the first turn searched the graph"
    assert second["retrieval_reused"] is True
    assert second["retrieved_nodes"] == [{"name": "Memoria di lavoro", "labels": ["Concept"]}]
    assert second["final_lesson_plan"] == "# Lezione"


async def test_followup_with_a_new_concept_searches_again(monkeypatch, _on):
    second, retriever = await _two_turns(
        monkeypatch, second_turn_concepts=["memoria di lavoro", "carico cognitivo"]
    )

    assert retriever.retrieve.await_count == 2, "the new concept forced a second search"
    assert not second.get("retrieval_reused")


async def test_flag_off_searches_on_every_turn(monkeypatch, _off):
    _second, retriever = await _two_turns(monkeypatch, second_turn_concepts=["memoria di lavoro"])

    assert retriever.retrieve.await_count == 2, "unchanged behaviour when the flag is off"
