"""AgentState partition, reducers, and cross-turn persistence.

Three things are locked in here.

**The partition.** ``AgentInput`` and ``PERSISTED_ACROSS_TURNS`` must together
cover ``AgentState`` exactly. Getting this wrong is silent and expensive in both
directions: a key wrongly left out of the input schema survives into the next
turn (a stale ``revision_count`` arrives at the ceiling and disables the
revision loop), and a key wrongly put in it is nulled on every follow-up (which
is what used to happen to the whole retrieval corpus).

**The reducers.** ``current_step`` and ``error`` are written by every node, so
they are the two keys that fail first when two nodes share a superstep. Both
must be order-independent — the runtime does not apply concurrent updates in
declaration order.

**Cross-turn persistence.** The behaviour the other two exist to produce: a
second turn on the same ``thread_id`` still sees the first turn's retrieval.
"""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from aix.agent.graph.state import (
    PERSISTED_ACROSS_TURNS,
    AgentInput,
    AgentState,
    create_initial_state,
    merge_current_step,
    merge_error,
)

pytestmark = pytest.mark.unit


def _keys(td: Any) -> set[str]:
    return set(td.__annotations__)


# ---------------------------------------------------------------------------
# 1. The partition is exact
# ---------------------------------------------------------------------------


def test_input_and_persisted_cover_the_state_exactly():
    missing = _keys(AgentState) - _keys(AgentInput) - PERSISTED_ACROSS_TURNS
    assert not missing, (
        f"unclassified AgentState keys: {sorted(missing)}. Every key must either "
        "be in AgentInput (reset each turn) or PERSISTED_ACROSS_TURNS (survives)."
    )


def test_partition_halves_do_not_overlap():
    overlap = _keys(AgentInput) & PERSISTED_ACROSS_TURNS
    assert not overlap, f"keys claiming both halves: {sorted(overlap)}"


def test_partition_invents_nothing():
    unknown = (_keys(AgentInput) | PERSISTED_ACROSS_TURNS) - _keys(AgentState)
    assert not unknown, f"keys that are not on AgentState: {sorted(unknown)}"


@pytest.mark.parametrize(
    "key, why",
    [
        ("revision_count", "would arrive at the ceiling and disable the revision loop"),
        ("error", "a stale error short-circuits every node before it runs"),
        ("approved", "a previous approval would end the new turn immediately"),
        ("lesson_plan_draft", "write_node would revise a stale draft instead of writing"),
    ],
)
def test_load_bearing_keys_are_reset_every_turn(key, why):
    assert key in _keys(AgentInput), f"{key} must reset per turn: {why}"


def test_create_initial_state_does_not_null_the_persisted_keys():
    """The reset used to happen here, by spelling out every output as None."""
    nulled = PERSISTED_ACROSS_TURNS & set(create_initial_state(query="q"))
    assert not nulled, f"persisted keys reset by a new turn: {sorted(nulled)}"


# ---------------------------------------------------------------------------
# 2. Reducers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "current, update, expected",
    [
        ("start", "plan_complete", "plan_complete"),
        # The revision loop walks back to the writer; the marker does not.
        ("critique_complete", "write_complete", "critique_complete"),
        # Failure outranks every stage, whichever side it arrives on.
        ("critique_complete", "error", "error"),
        ("error", "plan_complete", "error"),
        # A new run resets the marker even though "start" ranks lowest.
        ("critique_complete", "start", "start"),
        (None, "plan_complete", "plan_complete"),
        ("plan_complete", None, "plan_complete"),
    ],
)
def test_merge_current_step(current, update, expected):
    assert merge_current_step(current, update) == expected


@pytest.mark.parametrize(
    "a, b",
    [
        ("plan_complete", "retrieve_complete"),
        ("write_complete", "error"),
        ("retrieve_complete", "retrieve_complete"),
        ("plan_complete", "unknown_step"),
    ],
)
def test_merge_current_step_is_order_independent(a, b):
    assert merge_current_step(a, b) == merge_current_step(b, a)


@pytest.mark.parametrize(
    "current, update, expected",
    [
        (None, "boom", "boom"),
        ("boom", "boom", "boom"),
        (None, None, None),
        ("stale", None, None),  # the per-run reset from the input schema
        ("a", "b", "a | b"),
        ("b", "a", "a | b"),  # sorted, so arrival order cannot change it
    ],
)
def test_merge_error(current, update, expected):
    assert merge_error(current, update) == expected


def test_merge_error_keeps_both_failures_order_independently():
    assert merge_error("kg down", "llm timeout") == merge_error("llm timeout", "kg down")
    assert "kg down" in merge_error("kg down", "llm timeout")
    assert "llm timeout" in merge_error("kg down", "llm timeout")


# ---------------------------------------------------------------------------
# 3. The reducers make a parallel superstep possible
# ---------------------------------------------------------------------------


async def test_two_nodes_in_one_superstep_do_not_raise_invalid_update():
    """Without reducers this raises InvalidUpdateError on the shared keys.

    This is the shape the PrivateMaterialRetriever will introduce: two
    retrievers fanning out from the planner and converging on a composer.
    """

    async def fan(_state: AgentState) -> dict:
        return {"current_step": "plan_complete"}

    async def branch_kg(_state: AgentState) -> dict:
        return {"current_step": "retrieve_complete", "retrieved_nodes": [{"src": "kg"}]}

    async def branch_private(_state: AgentState) -> dict:
        return {"current_step": "retrieve_complete", "error": "private index cold"}

    async def join(_state: AgentState) -> dict:
        return {"current_step": "write_complete"}

    g = StateGraph(AgentState, input_schema=AgentInput)
    for name, fn in (
        ("fan", fan),
        ("kg", branch_kg),
        ("private", branch_private),
        ("join", join),
    ):
        g.add_node(name, fn)
    g.set_entry_point("fan")
    g.add_edge("fan", "kg")
    g.add_edge("fan", "private")
    g.add_edge("kg", "join")
    g.add_edge("private", "join")
    g.add_edge("join", END)

    final = await g.compile().ainvoke(create_initial_state(query="q"))

    assert final["retrieved_nodes"] == [{"src": "kg"}]
    assert final["error"] == "private index cold"
    assert final["current_step"] == "write_complete"


# ---------------------------------------------------------------------------
# 4. Cross-turn persistence — the point of the whole exercise
# ---------------------------------------------------------------------------


async def test_retrieval_survives_the_next_turn_while_per_run_keys_reset():
    """Second turn, same thread: the corpus is still there, the counters are not."""

    async def retrieve(state: AgentState) -> dict:
        # Only searches when it has nothing — the T6 behaviour in miniature.
        if state.get("retrieved_nodes"):
            return {"current_step": "retrieve_complete"}
        return {
            "retrieved_nodes": [{"name": "memoria di lavoro"}],
            "retrieval_confidence": "HIGH",
            "current_step": "retrieve_complete",
        }

    async def critique(state: AgentState) -> dict:
        return {
            "revision_count": state.get("revision_count", 0) + 1,
            "approved": False,
            "current_step": "critique_complete",
        }

    g = StateGraph(AgentState, input_schema=AgentInput)
    g.add_node("retrieve", retrieve)
    g.add_node("critique", critique)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "critique")
    g.add_edge("critique", END)
    app = g.compile(checkpointer=InMemorySaver())

    cfg = {"configurable": {"thread_id": "lesson-1"}}

    first = await app.ainvoke(create_initial_state(query="prima domanda"), config=cfg)
    assert first["retrieved_nodes"] == [{"name": "memoria di lavoro"}]
    assert first["revision_count"] == 1

    second = await app.ainvoke(create_initial_state(query="riscrivi più corto"), config=cfg)

    assert second["retrieved_nodes"] == [{"name": "memoria di lavoro"}], (
        "the retrieval corpus must survive so a follow-up can reuse it"
    )
    assert second["retrieval_confidence"] == "HIGH"
    assert second["revision_count"] == 1, "per-run counters must restart, not accumulate"
    assert second["teacher_query"] == "riscrivi più corto"


async def test_caller_cannot_smuggle_a_persisted_key_through_the_input():
    """Extra keys are dropped, so the input cannot forge the retrieval corpus."""

    async def noop(_state: AgentState) -> dict:
        return {"current_step": "retrieve_complete"}

    g = StateGraph(AgentState, input_schema=AgentInput)
    g.add_node("noop", noop)
    g.set_entry_point("noop")
    g.add_edge("noop", END)

    state = dict(create_initial_state(query="q"))
    state["retrieved_nodes"] = [{"forged": True}]

    final = await g.compile().ainvoke(state)

    assert not final.get("retrieved_nodes")
