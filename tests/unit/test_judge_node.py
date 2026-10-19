"""The judge in the chain: where a lesson goes after the Critic, and what the judge does to it."""

from __future__ import annotations

import pytest

from aix.agent.graph.nodes import (
    DOCUMENTS_UNUSED_INSTRUCTION,
    judge_node,
    should_continue_to_revision,
)
from aix.agent.validation import judge as judge_module
from aix.agent.validation.judge import Issue, JudgeReport

pytestmark = pytest.mark.unit

DRAFT = "# Le sillabe\n\n## Attività 1 (20 minuti)\n- La divisione di gatto è «ga-to».\n"


def _issue(quote, fix, severity="bloccante", kind="sostituzione"):
    return Issue(check="esempi_lingua", severity=severity, quote=quote, why="w", fix=fix, kind=kind)


def _report(*issues, error=""):
    return JudgeReport(issues=list(issues), model="m", cost=0.01, seconds=5.0, error=error)


@pytest.fixture
def judged(monkeypatch):
    """The judge returns the reports given, one per call."""
    answers: list[JudgeReport] = []

    async def fake(lesson, **kwargs):
        return answers.pop(0)

    monkeypatch.setattr(judge_module, "judge_lesson", fake)
    return answers


# --- the router -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "off", "on"),
    [
        ({"approved": True}, "finish", "judge"),
        ({"approved": False, "revision_count": 1, "max_revisions": 1}, "revise", "judge"),
        ({"approved": False, "revision_count": 2, "max_revisions": 1}, "finish", "judge"),
        (
            {
                "approved": False,
                "revision_count": 1,
                "max_revisions": 1,
                "revision_instructions": DOCUMENTS_UNUSED_INSTRUCTION,
            },
            "revise",
            "revise",  # a part is missing: nothing to fix in place
        ),
    ],
)
def test_with_the_judge_on_the_critic_no_longer_rewrites_the_lesson(monkeypatch, state, off, on):
    monkeypatch.delenv("AIX_JUDGE_ENABLED", raising=False)
    assert should_continue_to_revision(state) == off
    monkeypatch.setenv("AIX_JUDGE_ENABLED", "true")
    assert should_continue_to_revision(state) == on


# --- the node -------------------------------------------------------------------


async def test_a_wrong_example_is_fixed_in_place_and_the_minor_problems_stay_as_notes(judged):
    minor = _issue("La divisione di gatto", "-", severity="minore", kind="revisione")
    judged += [_report(_issue("«ga-to»", "«gat-to»"), minor), _report(minor)]

    out = await judge_node({"final_lesson_plan": DRAFT, "teacher_query": "sillabe"})

    assert out["final_lesson_plan"] == DRAFT.replace("«ga-to»", "«gat-to»")
    assert [n["severity"] for n in out["review_notes"]] == ["minore"]
    summary = out["judge_summary"]
    assert (summary["replaced"], summary["second_check"], summary["undone"]) == (1, True, False)
    assert summary["cost"] == pytest.approx(0.02)


async def test_a_repair_that_leaves_more_blocking_problems_is_undone(judged):
    judged += [
        _report(_issue("«ga-to»", "«gat-to»")),
        _report(_issue("«gat-to»", "x"), _issue("Attività 1", "y")),
    ]

    out = await judge_node({"final_lesson_plan": DRAFT})

    assert out["final_lesson_plan"] == DRAFT
    assert out["judge_summary"]["undone"] is True
    assert [n["quote"] for n in out["review_notes"]] == ["«ga-to»"]


async def test_a_judge_that_fails_leaves_the_lesson_as_it_was(judged):
    judged.append(_report(error="RuntimeError: provider down"))

    out = await judge_node({"final_lesson_plan": DRAFT})

    assert out["final_lesson_plan"] == DRAFT and out["review_notes"] == []
    assert "provider down" in out["judge_summary"]["error"]


async def test_nothing_found_nothing_changed(judged):
    judged.append(_report())

    out = await judge_node({"lesson_plan_draft": DRAFT})

    assert out["final_lesson_plan"] == DRAFT
    assert out["judge_summary"]["second_check"] is False


@pytest.mark.parametrize(("flag", "present"), [("false", False), ("true", True)])
def test_the_judge_is_in_the_chain_only_when_switched_on(monkeypatch, flag, present):
    from aix.agent.graph.lesson_planner_graph import build_lesson_planner_graph

    monkeypatch.setenv("AIX_JUDGE_ENABLED", flag)

    assert ("judge" in build_lesson_planner_graph().get_graph().nodes) is present
