"""A refinement chip acts on its own turn, and only on that one.

Three defects, found on 2026-10-02 while planning the light follow-up turns:

- the chip's instruction was popped from a *copy* of the profile, so it stayed
  saved and every later turn was simplified (or deepened) again;
- a chip posts no text, so the chat showed the previous request as the
  teacher's turn;
- a validator retry replaced the teacher's instruction with its own fix.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from tests.unit.agent_doubles import _critique, _plan, _retrieval_result

from aix.agent.graph import nodes as graph_nodes
from aix.agent.graph.lesson_planner_graph import build_lesson_planner_graph, reset_compiled_graph
from aix.agent.graph.state import create_initial_state
from aix.webui.agent.service import take_refinement
from aix.webui.lessons.routes import refinement_label

pytestmark = pytest.mark.unit


class _Lesson:
    def __init__(self, profile):
        self.educational_profile_json = profile


def test_the_instruction_is_taken_off_the_saved_profile():
    lesson = _Lesson(
        {"pedagogical_intent": "x", "__refinement__": "Riduci", "__graph_search__": False}
    )

    assert take_refinement(lesson) == "Riduci"
    assert lesson.educational_profile_json == {"pedagogical_intent": "x", "__graph_search__": False}
    assert take_refinement(lesson) is None, "the next turn is not refined again"


def test_without_a_chip_the_profile_is_left_alone():
    profile = {"pedagogical_intent": "x"}
    lesson = _Lesson(profile)

    assert take_refinement(lesson) is None
    assert lesson.educational_profile_json is profile
    assert take_refinement(_Lesson(None)) is None


@pytest.mark.parametrize(
    "code, detail, expected",
    [
        ("simplify", None, "Semplifica la lezione"),
        ("deepen", "sul capitolo 3", "Approfondisci la lezione: sul capitolo 3"),
        ("custom", "più giochi di ruolo", "più giochi di ruolo"),
        ("sconosciuto", None, "Rifinisci la lezione"),
    ],
)
def test_the_chat_shows_the_chip_the_teacher_pressed(code, detail, expected):
    assert refinement_label(code, detail) == expected


# ---------------------------------------------------------------------------
# The validator keeps the teacher's instruction
# ---------------------------------------------------------------------------

PROFILE = {
    "group": {"students_number": 25},
    "classroom": {"own_device": "NO", "has_wifi": False, "forniture_mobility": "NO"},
    "time_available_minutes": 60,
}
INFEASIBLE = (
    "# Lezione\n## Introduzione *(60 minuti)*\nGli studenti aprono Kahoot dal proprio smartphone.\n"
)
FEASIBLE = "# Lezione\n## Introduzione *(60 minuti)*\nAttività con schede cartacee.\n"


@pytest.fixture
def _validator(monkeypatch):
    monkeypatch.delenv("AIX_CORRECTIVE_RAG_ENABLED", raising=False)
    monkeypatch.setenv("AIX_CONDITIONAL_RETRIEVAL_ENABLED", "false")
    monkeypatch.setenv("AIX_VALIDATOR_ENABLED", "true")
    monkeypatch.setenv("AIX_VALIDATOR_MAX_RETRIES", "2")
    reset_compiled_graph()
    yield
    reset_compiled_graph()


async def test_a_retry_adds_the_fix_to_the_teacher_s_instruction(_validator):
    writer = AsyncMock(write=AsyncMock(side_effect=[INFEASIBLE, INFEASIBLE, FEASIBLE]))
    with (
        patch.object(
            graph_nodes, "get_planner", return_value=AsyncMock(plan=AsyncMock(return_value=_plan()))
        ),
        patch.object(
            graph_nodes,
            "get_retriever",
            return_value=AsyncMock(retrieve=AsyncMock(return_value=_retrieval_result())),
        ),
        patch.object(graph_nodes, "get_writer", return_value=writer),
        patch.object(
            graph_nodes,
            "get_critic",
            return_value=AsyncMock(critique=AsyncMock(return_value=_critique(True))),
        ),
    ):
        await build_lesson_planner_graph().ainvoke(
            create_initial_state(
                query="Crea una lezione",
                educational_profile=PROFILE,
                refinement_instruction="Semplifica il lessico",
            )
        )

    last = writer.write.await_args_list[-1].kwargs["refinement_instruction"]
    assert last.startswith("Semplifica il lessico")
    assert "Kahoot" in last
    assert last.count(graph_nodes._REMEDIATION_MARK) == 1, "two retries, one fix"
