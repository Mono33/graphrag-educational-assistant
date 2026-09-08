"""Graph topology and branch-routing coverage.

Before this file nothing compiled the graph, and the two functions that carry
every routing decision — :func:`should_continue_to_revision` and
:func:`should_retry_retrieval` — had no tests at all. Every edge in the
pipeline was therefore unverified, including the revision loop and the
corrective-RAG retry ceiling.

What is locked in here:

1. **Default topology** (``AIX_CORRECTIVE_RAG_ENABLED`` unset/false) — the exact
   node set and edge set, so a stray ``add_node`` / ``add_edge`` shows up as a
   failing test rather than as behaviour drift.
2. **Corrective-RAG topology** — a smoke test only. The feature is off by
   default and its cost/benefit is still unmeasured, so it is deliberately not
   on the critical path: we assert it still *compiles* and wires its retry
   edge, nothing more.
3. **Branch functions** — table-driven, including the precedence of ``error``
   over every other signal and both ceilings (``max_revisions`` and
   ``AIX_CORRECTIVE_RAG_MAX_ATTEMPTS``).
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

from aix.agent.agents.planner_agent import RetrievalPlan
from aix.agent.graph import nodes as graph_nodes
from aix.agent.graph.lesson_planner_graph import build_lesson_planner_graph
from aix.agent.graph.nodes import should_continue_to_revision, should_retry_retrieval
from aix.agent.graph.state import create_initial_state

pytestmark = pytest.mark.unit


def _edges(compiled) -> set[tuple[str, str]]:
    """The compiled graph's edges as ``{(source, target)}``."""
    return {(e.source, e.target) for e in compiled.get_graph().edges}


def _conditional_edges(compiled) -> set[tuple[str, str]]:
    return {(e.source, e.target) for e in compiled.get_graph().edges if e.conditional}


# ---------------------------------------------------------------------------
# 1. Default topology — plan → retrieve → write → critique → [write | END]
# ---------------------------------------------------------------------------


def test_default_topology_has_exactly_the_four_agent_nodes(monkeypatch):
    monkeypatch.delenv("AIX_CORRECTIVE_RAG_ENABLED", raising=False)
    compiled = build_lesson_planner_graph()

    assert set(compiled.get_graph().nodes) == {
        "__start__",
        "plan",
        "retrieve",
        "write",
        "critique",
        "__end__",
    }


def test_default_topology_wires_the_linear_path_and_revision_loop(monkeypatch):
    monkeypatch.delenv("AIX_CORRECTIVE_RAG_ENABLED", raising=False)
    compiled = build_lesson_planner_graph()

    assert _edges(compiled) == {
        ("__start__", "plan"),
        ("plan", "retrieve"),
        ("retrieve", "write"),  # direct edge — the grader is not in this topology
        ("write", "critique"),
        ("critique", "write"),  # the revision loop
        ("critique", "__end__"),
    }
    # Only the critique fan-out is a decision; everything else is unconditional.
    assert _conditional_edges(compiled) == {
        ("critique", "write"),
        ("critique", "__end__"),
    }


# ---------------------------------------------------------------------------
# 2. Corrective RAG — smoke only, deliberately off the critical path
# ---------------------------------------------------------------------------


def test_corrective_rag_topology_still_compiles_and_wires_its_retry_edge(monkeypatch):
    """The flag is off in production and the feature's value is unmeasured.

    This asserts only that turning it on still produces a valid graph with the
    grader node and its retry edge back into the retriever — enough to catch a
    change that silently breaks the flagged path, without paying to keep a
    second topology fully covered.
    """
    monkeypatch.setenv("AIX_CORRECTIVE_RAG_ENABLED", "true")
    compiled = build_lesson_planner_graph()

    assert "grade_retrieval" in compiled.get_graph().nodes
    edges = _edges(compiled)
    assert ("retrieve", "grade_retrieval") in edges
    assert ("grade_retrieval", "retrieve") in edges  # the retry loop
    assert ("grade_retrieval", "write") in edges
    # The direct retrieve → write edge is replaced by the grader hop.
    assert ("retrieve", "write") not in edges


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
            {"approved": False, "revision_count": 0, "max_revisions": 2},
            "revise",
            "budget left, so loop back to the writer",
        ),
        (
            {"approved": False, "revision_count": 2, "max_revisions": 2},
            "finish",
            "ceiling reached — ship the draft rather than loop",
        ),
        (
            {"approved": False, "revision_count": 5, "max_revisions": 2},
            "finish",
            "past the ceiling behaves like the ceiling",
        ),
        (
            {"approved": False, "revision_count": 0, "max_revisions": 0},
            "finish",
            "a zero budget disables the revision loop entirely",
        ),
        ({}, "revise", "defaults: not approved, count 0 < max 1"),
    ],
)
def test_should_continue_to_revision(state, expected, why):
    assert should_continue_to_revision(state) == expected, why


# ---------------------------------------------------------------------------
# 3b. should_retry_retrieval
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "state, expected, why",
    [
        ({"error": "boom"}, "error", "an upstream failure short-circuits"),
        (
            {"error": "boom", "retrieval_grade": "irrelevant"},
            "error",
            "error outranks a retry-worthy grade",
        ),
        ({"retrieval_grade": "relevant"}, "continue", "good retrieval goes to the writer"),
        ({}, "continue", "no grade defaults to relevant — the feature is opt-in"),
        (
            {"retrieval_grade": "irrelevant", "retrieval_attempts": 0},
            "retry",
            "budget left, so re-enter the retriever",
        ),
        (
            {"retrieval_grade": "ambiguous", "retrieval_attempts": 1},
            "retry",
            "ambiguous is retry-worthy too",
        ),
        (
            {"retrieval_grade": "irrelevant", "retrieval_attempts": 2},
            "continue",
            "ceiling reached — write with retrieval_warning rather than loop",
        ),
        (
            {"retrieval_grade": "irrelevant", "retrieval_attempts": 99},
            "continue",
            "past the ceiling behaves like the ceiling",
        ),
    ],
)
def test_should_retry_retrieval(state, expected, why, monkeypatch):
    monkeypatch.delenv("AIX_CORRECTIVE_RAG_MAX_ATTEMPTS", raising=False)  # default 2
    assert should_retry_retrieval(state) == expected, why


@pytest.mark.parametrize(
    "raw, effective, why",
    [
        ("1", 1, "lower bound"),
        ("4", 4, "upper bound"),
        ("0", 1, "clamped up — zero attempts would make the node pointless"),
        ("99", 4, "clamped down — a flaky grader must not loop the pipeline"),
        ("not-a-number", 2, "unparseable falls back to the default"),
    ],
)
def test_corrective_rag_attempt_ceiling_is_clamped(raw, effective, why, monkeypatch):
    monkeypatch.setenv("AIX_CORRECTIVE_RAG_MAX_ATTEMPTS", raw)
    assert graph_nodes._corrective_rag_max_attempts() == effective, why

    # And the router honours that ceiling: one below retries, at it continues.
    below = {"retrieval_grade": "irrelevant", "retrieval_attempts": effective - 1}
    at = {"retrieval_grade": "irrelevant", "retrieval_attempts": effective}
    assert should_retry_retrieval(below) == "retry"
    assert should_retry_retrieval(at) == "continue"


# ---------------------------------------------------------------------------
# 4. End-to-end through the compiled graph, every agent mocked
# ---------------------------------------------------------------------------


def _plan() -> RetrievalPlan:
    return RetrievalPlan(
        query_intent="lesson_creation",
        key_concepts=["memoria di lavoro"],
        search_queries=["memoria di lavoro"],
        lesson_type="full_lesson",
        target_grade=None,
        special_needs=[],
        time_constraints=None,
        intent_confidence="HIGH",
        reasoning="mock",
        scope_status="in_scope",
        scope_confidence=1.0,
        subject_concepts=["memoria di lavoro"],
        pedagogy_concepts=[],
        response_language="it",
        language_confidence="HIGH",
    )


def _retrieval_result():
    node = {"name": "Memoria di lavoro", "labels": ["Concept"]}
    return SimpleNamespace(
        is_hybrid=False,
        search_results=[SimpleNamespace(nodes=[node], relationships=[])],
        nodes=[node],
        relationships=[],
        recommendations=[{"name": "Chunking"}],
        confidence="HIGH",
        curated_media=None,
        external_resources=None,
    )


def _critique(approved: bool):
    return SimpleNamespace(
        summary="ok" if approved else "manca la valutazione",
        average_score=0.9 if approved else 0.4,
        approved=approved,
        scores={"coerenza": 0.9},
        strengths=["chiara"],
        revision_instructions=None if approved else "aggiungi la valutazione",
    )


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
    # Rejected twice with max_revisions=2: one real revision, then the ceiling.
    final, agents = await _run_graph(monkeypatch, critiques=[False, False], max_revisions=2)

    assert not final.get("error")
    assert final["approved"] is False
    assert final["revision_count"] == 2
    assert agents.writer.write.await_count == 1
    assert agents.writer.revise.await_count == 1, "the loop went back through the writer"
    assert agents.critic.critique.await_count == 2
    assert final["lesson_plan_draft"] == "# Lezione\n\nBozza rivista."


async def test_max_revisions_1_disables_the_revision_loop_entirely(monkeypatch):
    """``max_revisions=N`` permits N-1 actual revisions, not N.

    ``critique_node`` increments ``revision_count`` *before*
    ``should_continue_to_revision`` compares it against the ceiling, so at
    ``max_revisions=1`` the first rejection already trips ``count >= max`` and
    the run finishes without the writer ever revising.

    This matters beyond the arithmetic: ``LessonPlannerPipeline`` defaults to
    ``AIX_MAX_REVISIONS=1``, so on a default deployment the Critic can reject a
    draft but the Writer is never given the chance to act on it. The test
    documents the behaviour as it stands — change the default or the
    comparison, and it fails loudly rather than drifting.
    """
    final, agents = await _run_graph(monkeypatch, critiques=[False], max_revisions=1)

    assert final["approved"] is False
    assert final["revision_count"] == 1
    assert agents.critic.critique.await_count == 1
    assert agents.writer.revise.await_count == 0, "no revision happens at max_revisions=1"
