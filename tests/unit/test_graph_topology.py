"""Graph topology and branch-routing coverage.

Before this file nothing compiled the graph, and the function that carries
the revision decision — :func:`should_continue_to_revision` — had no tests
at all. Every edge in the pipeline was therefore unverified.

What is locked in here:

1. **Default topology** (``AIX_CORRECTIVE_RAG_ENABLED`` unset/false) — the exact
   node set and edge set, so a stray ``add_node`` / ``add_edge`` shows up as a
   failing test rather than as behaviour drift.
2. **Branch function** — table-driven, including the precedence of ``error``
   over every other signal and the ``max_revisions`` ceiling. Corrective RAG,
   kept off by the pilot, is no longer covered (2026-10-13).
4. **One end-to-end run** with all four agents mocked: no network, no Neo4j,
   no LLM. Proves the nodes actually hand state to each other in the compiled
   graph, which none of the per-node unit tests can show.

The agent mocks follow the pattern already used in
``test_plan_node_profile_wins_over_history.py``: a real ``RetrievalPlan``
dataclass, and ``AsyncMock`` on the module-level ``get_*`` accessors.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from tests.unit.agent_doubles import _critique, _plan, _retrieval_result

from aix.agent.graph import nodes as graph_nodes
from aix.agent.graph.lesson_planner_graph import (
    build_lesson_planner_graph,
    get_compiled_graph,
    reset_compiled_graph,
)
from aix.agent.graph.nodes import should_continue_to_revision
from aix.agent.graph.state import create_initial_state

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_graph_singleton():
    """Keep the process-wide compiled graph from leaking between tests.

    The topology is read from the environment at build time, so a graph cached
    under one value of ``AIX_CORRECTIVE_RAG_ENABLED`` must not survive into a
    test that sets the other.
    """
    reset_compiled_graph()
    yield
    reset_compiled_graph()


def _edges(compiled) -> set[tuple[str, str]]:
    """The compiled graph's edges as ``{(source, target)}``."""
    return {(e.source, e.target) for e in compiled.get_graph().edges}


def _conditional_edges(compiled) -> set[tuple[str, str]]:
    return {(e.source, e.target) for e in compiled.get_graph().edges if e.conditional}


# ---------------------------------------------------------------------------
# 1. Default topology — plan → retrieve → write → critique → [write | END]
# ---------------------------------------------------------------------------


def test_default_topology_has_exactly_the_five_agent_nodes(monkeypatch):
    monkeypatch.delenv("AIX_CORRECTIVE_RAG_ENABLED", raising=False)
    compiled = build_lesson_planner_graph()

    assert set(compiled.get_graph().nodes) == {
        "__start__",
        "plan",
        "retrieve",
        "write",
        # Q1 · 1.3 — always present, unlike the flagged nodes: the node no-ops
        # when the validator is off, so the graph has one shape, not two.
        "validate",
        "critique",
        "__end__",
    }


def test_default_topology_wires_the_linear_path_and_both_loops(monkeypatch):
    monkeypatch.delenv("AIX_CORRECTIVE_RAG_ENABLED", raising=False)
    compiled = build_lesson_planner_graph()

    assert _edges(compiled) == {
        ("__start__", "plan"),
        ("plan", "retrieve"),
        ("retrieve", "write"),  # direct edge — the grader is not in this topology
        ("write", "validate"),
        ("validate", "write"),  # the feasibility rewrite
        ("validate", "critique"),
        ("critique", "write"),  # the revision loop
        ("critique", "__end__"),
    }
    # Two decisions: is the lesson feasible, and is it good enough.
    assert _conditional_edges(compiled) == {
        ("validate", "write"),
        ("validate", "critique"),
        ("critique", "write"),
        ("critique", "__end__"),
    }


# ---------------------------------------------------------------------------
# 2b. The process-wide compiled-graph singleton
# ---------------------------------------------------------------------------


async def test_compiled_graph_is_built_once_per_process(monkeypatch):
    """Two runs must share one compiled graph.

    Regression guard for the original defect: the Pipeline cached on
    ``self._graph``, but callers built a fresh Orchestrator (and Pipeline) per
    run, so the cache was always cold and ``compile()`` ran for every lesson.
    """
    monkeypatch.setattr(
        "aix.agent.graph.checkpointer.get_checkpointer", AsyncMock(return_value=None)
    )

    first = await get_compiled_graph()
    second = await get_compiled_graph()

    assert first is second


async def test_concurrent_first_callers_share_one_graph(monkeypatch):
    """The init lock must make simultaneous cold starts converge on one graph."""
    import asyncio

    monkeypatch.setattr(
        "aix.agent.graph.checkpointer.get_checkpointer", AsyncMock(return_value=None)
    )

    graphs = await asyncio.gather(*(get_compiled_graph() for _ in range(8)))

    assert len({id(g) for g in graphs}) == 1


async def test_graph_recompiles_when_the_checkpointer_is_replaced(monkeypatch):
    """A cached graph must not outlive the saver it closed over.

    ``close_checkpointer()`` tears the saver down and a later call builds a new
    one. Since the compiled graph holds the old saver, reuse would write
    checkpoints through a dead connection — so the cache is keyed on saver
    identity rather than merely on "has a graph been built".
    """
    from langgraph.checkpoint.memory import InMemorySaver

    saver_a, saver_b = InMemorySaver(), InMemorySaver()

    monkeypatch.setattr(
        "aix.agent.graph.checkpointer.get_checkpointer", AsyncMock(return_value=saver_a)
    )
    first = await get_compiled_graph()
    assert await get_compiled_graph() is first, "same saver → same graph"

    monkeypatch.setattr(
        "aix.agent.graph.checkpointer.get_checkpointer", AsyncMock(return_value=saver_b)
    )
    second = await get_compiled_graph()

    assert second is not first, "new saver → rebuilt graph"
    assert await get_compiled_graph() is second, "and the new one is now cached"


async def test_pipeline_and_orchestrator_expose_the_shared_graph(monkeypatch):
    """The public accessors that replaced the two `noqa: SLF001` reach-throughs."""
    from aix.agent.orchestrator import AgentOrchestrator

    monkeypatch.setattr(
        "aix.agent.graph.checkpointer.get_checkpointer", AsyncMock(return_value=None)
    )

    # Two independently-built orchestrators, as the service layer creates them.
    first = await AgentOrchestrator(domain="neuro").get_pipeline().get_graph()
    second = await AgentOrchestrator(domain="udl").get_pipeline().get_graph()

    assert first is second, "domain travels in state, so it must not fork the graph"
    assert first is await get_compiled_graph()


# ---------------------------------------------------------------------------
# 3a. should_continue_to_revision
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "state, expected, why",
    [
        ({"error": "boom"}, "error", "an upstream failure short-circuits"),
        (
            {"error": "boom", "approved": True},
            "error",
            "error outranks approval — otherwise a failed run could 'finish'",
        ),
        ({"approved": True}, "finish", "the critic approved"),
        (
            {"approved": False, "revision_count": 2, "max_revisions": 2},
            "revise",
            "the second of two permitted revisions (the count includes this rejection)",
        ),
        (
            {"approved": False, "revision_count": 3, "max_revisions": 2},
            "finish",
            "ceiling reached — ship the draft rather than loop",
        ),
        (
            {"approved": False, "revision_count": 1, "max_revisions": 0},
            "finish",
            "a zero budget disables the revision loop entirely",
        ),
        ({}, "revise", "defaults: not approved, count 0 < max 1"),
    ],
)
def test_should_continue_to_revision(state, expected, why):
    assert should_continue_to_revision(state) == expected, why


# ---------------------------------------------------------------------------
# 4. End-to-end through the compiled graph, every agent mocked
# ---------------------------------------------------------------------------


async def _run_graph(monkeypatch, *, critiques, max_revisions=2):
    """Run the compiled graph once with all four agents mocked.

    ``critiques`` is the sequence of critic verdicts, so a caller can drive the
    revision loop by handing back `not approved` first.
    """
    monkeypatch.delenv("AIX_CORRECTIVE_RAG_ENABLED", raising=False)

    planner = AsyncMock()
    planner.plan = AsyncMock(return_value=_plan())

    retriever = AsyncMock()
    retriever.retrieve = AsyncMock(return_value=_retrieval_result())

    writer = AsyncMock()
    writer.write = AsyncMock(return_value="# Lezione\n\nBozza.")
    writer.revise = AsyncMock(return_value="# Lezione\n\nBozza rivista.")

    critic = AsyncMock()
    critic.critique = AsyncMock(side_effect=[_critique(a) for a in critiques])

    state = create_initial_state(
        query="Crea una lezione sulla memoria di lavoro",
        domain="neuro",
        language="it",
        max_revisions=max_revisions,
    )

    with (
        patch.object(graph_nodes, "get_planner", return_value=planner),
        patch.object(graph_nodes, "get_retriever", return_value=retriever),
        patch.object(graph_nodes, "get_writer", return_value=writer),
        patch.object(graph_nodes, "get_critic", return_value=critic),
    ):
        final = await build_lesson_planner_graph().ainvoke(state)

    return final, SimpleNamespace(
        planner=planner, retriever=retriever, writer=writer, critic=critic
    )


async def test_end_to_end_approved_on_first_pass(monkeypatch):
    final, agents = await _run_graph(monkeypatch, critiques=[True])

    assert not final.get("error")
    assert final["approved"] is True
    assert final["final_lesson_plan"] == "# Lezione\n\nBozza."
    assert final["revision_count"] == 0

    # Every stage ran exactly once, and the writer was never asked to revise.
    agents.planner.plan.assert_awaited_once()
    agents.retriever.retrieve.assert_awaited_once()
    agents.writer.write.assert_awaited_once()
    agents.writer.revise.assert_not_awaited()
    agents.critic.critique.assert_awaited_once()

    # State really flowed between nodes: the retriever's output is visible in
    # the final state, which is what a per-node unit test cannot demonstrate.
    assert final["retrieved_nodes"] == [{"name": "Memoria di lavoro", "labels": ["Concept"]}]


async def test_end_to_end_revision_loop_runs_then_stops_at_the_ceiling(monkeypatch):
    # Rejected twice with max_revisions=1: one real revision, then the ceiling.
    final, agents = await _run_graph(monkeypatch, critiques=[False, False], max_revisions=1)

    assert not final.get("error")
    assert final["approved"] is False
    assert final["revision_count"] == 2
    assert agents.writer.write.await_count == 1
    assert agents.writer.revise.await_count == 1, "the loop went back through the writer"
    assert agents.critic.critique.await_count == 2
    assert final["lesson_plan_draft"] == "# Lezione\n\nBozza rivista."


async def test_second_turn_reuses_the_first_turn_retrieval(monkeypatch):
    """The real pipeline, two turns, one thread: the corpus survives.

    ``test_state_partition.py`` proves the mechanism on a synthetic graph; this
    proves it where it has to hold — through plan → retrieve → write → critique
    with the actual nodes. It is the precondition for skipping the Knowledge
    Graph search on a follow-up turn: without it there is nothing to reuse.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    monkeypatch.delenv("AIX_CORRECTIVE_RAG_ENABLED", raising=False)
    monkeypatch.setattr(
        "aix.agent.graph.checkpointer.get_checkpointer",
        AsyncMock(return_value=InMemorySaver()),
    )

    planner = AsyncMock(plan=AsyncMock(return_value=_plan()))
    retriever = AsyncMock(retrieve=AsyncMock(return_value=_retrieval_result()))
    writer = AsyncMock(
        write=AsyncMock(return_value="# Bozza"), revise=AsyncMock(return_value="# Rivista")
    )
    critic = AsyncMock(critique=AsyncMock(side_effect=[_critique(True), _critique(True)]))

    graph = await get_compiled_graph()
    config = {"configurable": {"thread_id": "lesson-under-test"}}

    with (
        patch.object(graph_nodes, "get_planner", return_value=planner),
        patch.object(graph_nodes, "get_retriever", return_value=retriever),
        patch.object(graph_nodes, "get_writer", return_value=writer),
        patch.object(graph_nodes, "get_critic", return_value=critic),
    ):
        await graph.ainvoke(
            create_initial_state(query="Crea una lezione", max_revisions=2), config=config
        )
        second = await graph.ainvoke(
            create_initial_state(query="riscrivila più corta", max_revisions=2), config=config
        )

    assert second["retrieved_nodes"] == [{"name": "Memoria di lavoro", "labels": ["Concept"]}], (
        "turn 1's retrieval must still be in state at the start of turn 2"
    )
    assert second["revision_count"] == 0, "per-run counters restart"
    assert second["teacher_query"] == "riscrivila più corta"


async def test_max_revisions_is_the_number_of_revisions(monkeypatch):
    """``max_revisions=N`` permits N revisions.

    It permitted N-1 until 2026-09-29: ``critique_node`` counts the rejection
    before ``should_continue_to_revision`` compares, and the comparison was
    ``>=``. When the default became 1 on 2026-09-25 -- meant as "one revision" --
    no draft was revised any more, and a test here documented it as the
    behaviour "as it stands" instead of as the bug it was.
    """
    final, agents = await _run_graph(monkeypatch, critiques=[False, True], max_revisions=1)

    assert final["approved"] is True
    assert agents.writer.revise.await_count == 1, "max_revisions=1 is one revision"

    final, agents = await _run_graph(monkeypatch, critiques=[False], max_revisions=0)

    assert final["approved"] is False
    assert agents.writer.revise.await_count == 0, "max_revisions=0 is none"
