"""The validator inside the compiled graph (Q1 · 1.3 + 1.4).

``test_constraint_validator`` pins what the rules decide. This pins what the
graph does with that decision: an infeasible draft goes back to the Writer with
only the remediation, a feasible one goes straight to the Critic, and the budget
that bounds the loop is the validator's own rather than the Critic's.

The last point is the one worth stating. Spending a *revision* on a lesson that
cannot be delivered would leave the teacher with fewer attempts at a better
lesson — two different failures, two different budgets.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from tests.unit.agent_doubles import _critique, _plan, _retrieval_result

from aix.agent.graph import nodes as graph_nodes
from aix.agent.graph.lesson_planner_graph import build_lesson_planner_graph, reset_compiled_graph
from aix.agent.graph.state import create_initial_state

pytestmark = pytest.mark.unit

PROFILE = {
    "group": {"students_number": 25},
    "classroom": {"own_device": "NO", "has_wifi": False, "forniture_mobility": "NO"},
    "time_available_minutes": 60,
}

INFEASIBLE = """# Lezione
## Introduzione *(60 minuti)*
Gli studenti aprono Kahoot dal proprio smartphone.
"""

FEASIBLE = """# Lezione
## Introduzione *(60 minuti)*
Attività con schede cartacee.
"""


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("AIX_CORRECTIVE_RAG_ENABLED", raising=False)
    monkeypatch.delenv("AIX_CONDITIONAL_RETRIEVAL_ENABLED", raising=False)
    monkeypatch.setenv("AIX_VALIDATOR_ENABLED", "true")
    reset_compiled_graph()
    yield
    reset_compiled_graph()


async def _run(drafts: list[str], **initial):
    """Run the whole graph with a Writer that produces ``drafts`` in order."""
    writer = AsyncMock(
        write=AsyncMock(side_effect=drafts),
        revise=AsyncMock(side_effect=drafts),
    )
    planner = AsyncMock(plan=AsyncMock(return_value=_plan()))
    retriever = AsyncMock(retrieve=AsyncMock(return_value=_retrieval_result()))
    critic = AsyncMock(critique=AsyncMock(return_value=_critique(True)))

    with (
        patch.object(graph_nodes, "get_planner", return_value=planner),
        patch.object(graph_nodes, "get_retriever", return_value=retriever),
        patch.object(graph_nodes, "get_writer", return_value=writer),
        patch.object(graph_nodes, "get_critic", return_value=critic),
    ):
        final = await build_lesson_planner_graph().ainvoke(
            create_initial_state(query="Crea una lezione", **initial)
        )
    return final, writer, critic


# ---------------------------------------------------------------------------
# The feasible case
# ---------------------------------------------------------------------------


async def test_a_feasible_lesson_goes_straight_to_the_critic():
    final, writer, critic = await _run([FEASIBLE], educational_profile=PROFILE)

    assert writer.write.await_count == 1
    assert critic.critique.await_count == 1
    assert final["validation_passed"] is True
    assert final["validation_violations"] == []


async def test_without_a_profile_there_is_nothing_to_validate():
    """Most API callers pass no classroom; the graph must not change for them."""
    final, writer, _critic = await _run([FEASIBLE])

    assert writer.write.await_count == 1
    assert final["validation_passed"] is True


# ---------------------------------------------------------------------------
# The rewrite
# ---------------------------------------------------------------------------


async def test_an_infeasible_lesson_is_sent_back_to_the_writer():
    final, writer, critic = await _run([INFEASIBLE, FEASIBLE], educational_profile=PROFILE)

    assert writer.write.await_count == 2, "the first draft was not deliverable"
    assert critic.critique.await_count == 1, "the Critic only ever saw the fixed one"
    assert final["validation_passed"] is True
    assert final["final_lesson_plan"] == FEASIBLE


async def test_the_rewrite_carries_only_the_remediation():
    _final, writer, _critic = await _run([INFEASIBLE, FEASIBLE], educational_profile=PROFILE)

    instruction = writer.write.await_args.kwargs["refinement_instruction"]
    assert "Kahoot" in instruction
    assert "dispositivi personali" in instruction
    assert "## Classroom Constraints" not in instruction, (
        "the Writer already has the classroom; repeating it buries the fix"
    )


async def test_the_rewrite_does_not_spend_a_critic_revision():
    final, _writer, _critic = await _run([INFEASIBLE, FEASIBLE], educational_profile=PROFILE)

    assert final["validation_attempts"] == 1
    assert final["revision_count"] == 0, "the Critic's budget is a different budget"


# ---------------------------------------------------------------------------
# The budget
# ---------------------------------------------------------------------------


async def test_a_lesson_that_never_becomes_feasible_still_reaches_the_critic(monkeypatch):
    """Better to show the teacher the violations than to loop forever."""
    monkeypatch.setenv("AIX_VALIDATOR_MAX_RETRIES", "2")

    final, writer, critic = await _run(
        [INFEASIBLE, INFEASIBLE, INFEASIBLE], educational_profile=PROFILE
    )

    assert writer.write.await_count == 3  # first attempt plus two rewrites
    assert critic.critique.await_count == 1
    assert final["validation_passed"] is False
    # Two rules, honestly: a Kahoot quiz in this classroom needs both a device
    # the students do not have and a connection the room does not have.
    assert {v["rule"] for v in final["validation_violations"]} == {"byod_off", "no_wifi"}


async def test_the_budget_is_configurable(monkeypatch):
    monkeypatch.setenv("AIX_VALIDATOR_MAX_RETRIES", "0")

    _final, writer, critic = await _run([INFEASIBLE], educational_profile=PROFILE)

    assert writer.write.await_count == 1, "no retries means no rewrite"
    assert critic.critique.await_count == 1


# ---------------------------------------------------------------------------
# The flag
# ---------------------------------------------------------------------------


async def test_with_the_validator_off_the_run_is_the_old_one(monkeypatch):
    monkeypatch.setenv("AIX_VALIDATOR_ENABLED", "false")

    final, writer, critic = await _run([INFEASIBLE], educational_profile=PROFILE)

    assert writer.write.await_count == 1
    assert critic.critique.await_count == 1
    assert final["validation_violations"] is None
