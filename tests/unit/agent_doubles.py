"""Stand-ins for the four agents, shared by the graph tests.

These used to live in ``test_graph_topology`` and be imported from the other
test modules. That worked under ``python -m pytest``, which puts the working
directory on ``sys.path``, and failed under plain ``pytest``, which does not —
so the whole thing collapsed the first time CI ran it. They live in a module of
their own now, and one test module no longer imports another.

The shapes here are the ones the nodes actually read: ``plan_node`` uses
sixteen attributes of ``RetrievalPlan``, ``retrieve_node`` unpacks
``search_results`` alongside the flat node list, and ``critique_node`` branches
on ``approved`` to decide between finishing and asking for a revision.
"""

from __future__ import annotations

from types import SimpleNamespace

from aix.agent.agents.planner_agent import RetrievalPlan


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
