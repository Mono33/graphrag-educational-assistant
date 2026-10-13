"""Conditional retrieval: reuse the corpus when a turn adds nothing new.

Most follow-ups do not need a second Knowledge Graph search — "riscrivila più
corta" and "adatta per ADHD" work from the evidence already gathered. Only a
turn that introduces a concept the corpus does not cover has to go back to
Neo4j.

The decision is the planner's rather than a manual toggle: a toggle would ask
the teacher to predict whether their own question introduces a new concept.

It was first written as a comparison between the turn's ``key_concepts`` and
the corpus the thread holds, which cost nothing because the planner already
extracts them. On real threads that never once chose reuse. The planner was
called with the bare query and no thread context, so asked to "traduci la
lezione in inglese" it read the turn as a fresh request and returned the
concepts of a lesson nobody had asked for — generic pedagogy, in English,
against subject nodes in Italian. So the planner is now given the thread and
the corpus, and asked the question directly; the concept comparison stays as
the fallback for a turn it did not answer.

Gated by ``AIX_CONDITIONAL_RETRIEVAL_ENABLED``. With the flag off the
``plan → retrieve`` edge stays unconditional, which the first test pins.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from tests.unit.agent_doubles import _critique, _plan, _retrieval_result

from aix.agent.agents.planner_agent import _as_bool, _format_corpus, _format_history
from aix.agent.graph import nodes as graph_nodes
from aix.agent.graph.lesson_planner_graph import (
    build_lesson_planner_graph,
    get_compiled_graph,
    reset_compiled_graph,
)
from aix.agent.graph.nodes import (
    _corpus_covers_this_turn,
    plan_node,
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
    monkeypatch.setenv("AIX_CONDITIONAL_RETRIEVAL_ENABLED", "false")


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


# ---------------------------------------------------------------------------
# The planner's verdict — the criterion that actually decides
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "verdict, expected",
    [(True, "retrieve"), (False, "reuse")],
    ids=["needs new facts", "works on what is there"],
)
def test_planner_verdict_decides(verdict, expected, _on):
    assert should_retrieve(_followup(requires_new_retrieval=verdict)) == expected


def test_verdict_beats_the_concept_fallback(_on):
    """The concepts say reuse; the planner, which saw the thread, says search."""
    state = _followup(requires_new_retrieval=True)
    assert _corpus_covers_this_turn(state), "the fallback would have said reuse"
    assert should_retrieve(state) == "retrieve"


def test_the_teacher_still_outranks_the_planner(_on):
    state = _followup(requires_new_retrieval=True, force_retrieval=False)
    assert should_retrieve(state) == "reuse"


def test_no_verdict_falls_back_to_comparing_concepts(_on):
    """A first turn, an old checkpoint, or a planner that ignored the keys."""
    assert should_retrieve(_followup(requires_new_retrieval=None)) == "reuse"
    assert (
        should_retrieve(_followup(requires_new_retrieval=None, key_concepts=["carico cognitivo"]))
        == "retrieve"
    )


# ---------------------------------------------------------------------------
# Asking the question: plan_node only does it when there is a thread to ask about
# ---------------------------------------------------------------------------


async def _plan_node_kwargs(state):
    plan = _plan()
    plan.requires_new_retrieval = False
    plan.retrieval_reason = "translation of the existing lesson"
    planner = AsyncMock(plan=AsyncMock(return_value=plan))
    with patch.object(graph_nodes, "get_planner", return_value=planner):
        updates = await plan_node(state)
    return planner.plan.await_args.kwargs, updates


async def test_plan_node_hands_the_planner_the_thread(_on):
    state = _followup(retrieved_nodes=[{"name": "Memoria di Lavoro"}])
    kwargs, updates = await _plan_node_kwargs(state)

    assert kwargs["conversation_history"] == state["conversation_history"]
    assert kwargs["existing_corpus"] == ["memoria di lavoro"]
    assert updates["requires_new_retrieval"] is False
    assert updates["plan"]["retrieval_reason"] == "translation of the existing lesson"


async def test_first_turn_prompt_is_unchanged(_on):
    kwargs, _ = await _plan_node_kwargs(_followup(conversation_history=[]))
    assert "conversation_history" not in kwargs
    assert "existing_corpus" not in kwargs


async def test_plan_node_hands_the_planner_the_attachments(_off):
    """Whatever the retrieval flag: the Planner plans on the teacher's document."""
    state = _followup(conversation_history=[])
    state["teacher_provided_context"] = "### guerra-fredda.pdf\nDue mondi contrapposti."

    kwargs, _ = await _plan_node_kwargs(state)

    assert kwargs["teacher_materials"] == "### guerra-fredda.pdf\nDue mondi contrapposti."


# ---------------------------------------------------------------------------
# What the planner is shown, and how its answer is read back
# ---------------------------------------------------------------------------


def test_history_is_trimmed_to_the_recent_tail():
    history = [{"role": "user", "content": f"turno {i}"} for i in range(20)]
    formatted = _format_history(history)

    assert formatted.count("\n") == 5, "six turns, five separators"
    assert "turno 19" in formatted
    assert "turno 13" not in formatted


def test_long_turns_are_cut_and_flattened():
    formatted = _format_history([{"role": "user", "content": "a\nb" + "x" * 1000}])
    assert "\n" not in formatted.replace("- user: ", "")
    assert formatted.endswith("...")


def test_empty_history_says_so():
    assert _format_history([]) == "(none)"
    assert _format_history(None) == "(none)"
    assert _format_history([{"role": "user"}]) == "(none)", "a turn with no content is not a turn"


def test_corpus_is_capped_with_a_count():
    formatted = _format_corpus([f"concetto {i}" for i in range(50)])
    assert "concetto 39" in formatted
    assert "concetto 40" not in formatted
    assert formatted.endswith(", and 10 more")


def test_empty_corpus_says_so():
    assert _format_corpus([]) == "(nothing retrieved yet)"
    assert _format_corpus(["  ", None]) == "(nothing retrieved yet)"


@pytest.mark.parametrize(
    "raw, expected",
    [
        (True, True),
        (False, False),
        ("true", True),
        ("FALSE", False),
        ("yes", True),
        ("1", True),
        ("0", False),
        (None, None),
        ("maybe", None),
        (1, None),
    ],
)
def test_verdict_is_read_without_trusting_the_json_typing(raw, expected):
    """Anything unrecognised reads as None, which is the fallback signal."""
    assert _as_bool(raw) is expected
