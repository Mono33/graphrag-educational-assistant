"""The judge in the chain: where a lesson goes after the Critic, and what the judge does to it."""

from __future__ import annotations

import pytest

from aix.agent.graph.nodes import (
    DOCUMENTS_UNUSED_INSTRUCTION,
    judge_node,
    should_continue_to_revision,
)
from aix.agent.validation import judge as judge_module
from aix.agent.validation.judge import Issue, JudgeReport, Verification

pytestmark = pytest.mark.unit

DRAFT = "# Le sillabe\n\n## Attività 1 (20 minuti)\n- La divisione di gatto è «ga-to».\n"


def _issue(quote, fix, severity="bloccante", kind="sostituzione"):
    return Issue(check="esempi_lingua", severity=severity, quote=quote, why="w", fix=fix, kind=kind)


def _report(*issues, error=""):
    return JudgeReport(issues=list(issues), model="m", cost=0.01, seconds=5.0, error=error)


@pytest.fixture
def judged(monkeypatch):
    """The judge returns the reports given, one per call; the verification says
    «ok» to every change unless ``verdicts`` lists what to say."""
    answers: list[JudgeReport] = []
    verdicts: list[bool] = []

    async def fake(lesson, **kwargs):
        return answers.pop(0)

    async def fake_verify(changes, **kwargs):
        ok = verdicts[: len(changes)] or [True] * len(changes)
        return Verification(ok=ok, reasons=["sbagliata"] * len(changes), cost=0.002, seconds=1.0)

    monkeypatch.setattr(judge_module, "judge_lesson", fake)
    monkeypatch.setattr(judge_module, "verify_changes", fake_verify)
    return answers, verdicts


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
    answers, _ = judged
    minor = _issue("La divisione di gatto", "-", severity="minore", kind="revisione")
    answers.append(_report(_issue("«ga-to»", "«gat-to»"), minor))

    out = await judge_node({"final_lesson_plan": DRAFT, "teacher_query": "sillabe"})

    assert out["final_lesson_plan"] == DRAFT.replace("«ga-to»", "«gat-to»")
    assert [n["severity"] for n in out["review_notes"]] == ["minore"]
    summary = out["judge_summary"]
    assert (summary["replaced"], summary["verified"], summary["reverted"]) == (1, 1, 0)
    assert summary["cost"] == pytest.approx(0.012)


async def test_a_change_judged_wrong_is_put_back_and_becomes_a_note(judged):
    answers, verdicts = judged
    answers.append(_report(_issue("«ga-to»", "«g-atto»")))
    verdicts.append(False)

    out = await judge_node({"final_lesson_plan": DRAFT})

    assert out["final_lesson_plan"] == DRAFT
    assert out["judge_summary"]["reverted"] == 1
    assert [n["quote"] for n in out["review_notes"]] == ["«ga-to»"]


async def test_only_the_wrong_change_is_put_back(judged):
    """Two fixes: the right one stays, the wrong one goes back."""
    answers, verdicts = judged
    draft = DRAFT + "- La Bastiglia fu presa nel 1792.\n"
    answers.append(_report(_issue("«ga-to»", "«gat-to»"), _issue("nel 1792", "nel 1798")))
    verdicts.extend([True, False])

    out = await judge_node({"final_lesson_plan": draft})

    assert "«gat-to»" in out["final_lesson_plan"] and "nel 1792" in out["final_lesson_plan"]
    assert [n["quote"] for n in out["review_notes"]] == ["nel 1792"]


async def test_a_judge_that_fails_leaves_the_lesson_as_it_was(judged):
    answers, _ = judged
    answers.append(_report(error="RuntimeError: provider down"))

    out = await judge_node({"final_lesson_plan": DRAFT})

    assert out["final_lesson_plan"] == DRAFT and out["review_notes"] == []
    assert "provider down" in out["judge_summary"]["error"]


async def test_nothing_found_nothing_changed(judged):
    answers, _ = judged
    answers.append(_report())

    out = await judge_node({"lesson_plan_draft": DRAFT})

    assert out["final_lesson_plan"] == DRAFT
    assert out["judge_summary"]["verified"] == 0


@pytest.mark.parametrize(("flag", "present"), [("false", False), ("true", True)])
def test_the_judge_takes_the_critic_s_place_only_when_switched_on(monkeypatch, flag, present):
    from aix.agent.graph.lesson_planner_graph import build_lesson_planner_graph

    monkeypatch.setenv("AIX_JUDGE_ENABLED", flag)

    nodes = build_lesson_planner_graph().get_graph().nodes
    assert ("judge" in nodes) is present and ("check_documents" in nodes) is present
    assert ("critique" in nodes) is not present


async def test_without_the_critic_a_lesson_that_ignores_the_documents_goes_back_once():
    from aix.agent.graph.nodes import documents_check_node

    state = {
        "lesson_plan_draft": DRAFT,
        "material_context_items": [{"score": 0.8, "text": "passaggio"}],
        "revision_count": 0,
        "max_revisions": 1,
    }

    first = await documents_check_node(state)
    assert first["revision_instructions"] == DOCUMENTS_UNUSED_INSTRUCTION
    assert should_continue_to_revision({**state, **first}) == "revise"

    second = await documents_check_node({**state, **first})
    assert second["approved"] and second["final_lesson_plan"] == DRAFT


async def test_a_lesson_without_documents_goes_straight_to_the_judge(monkeypatch):
    from aix.agent.graph.nodes import documents_check_node

    monkeypatch.setenv("AIX_JUDGE_ENABLED", "true")
    state = {"lesson_plan_draft": DRAFT}

    out = await documents_check_node(state)

    assert should_continue_to_revision({**state, **out}) == "judge"


def test_the_review_notes_are_shown_under_the_lesson_only_when_there_are_some():
    from pathlib import Path

    from jinja2 import Environment, FileSystemLoader

    root = Path(__file__).resolve().parents[2] / "src" / "aix" / "webui" / "templates"
    template = Environment(loader=FileSystemLoader(str(root))).get_template(
        "partials/review_notes.html"
    )
    note = {
        "check": "esempi_lingua",
        "severity": "minore",
        "quote": "«foglia/moglie»",
        "why": "stesso suono",
        "fix": "«foglia/folla»",
        "kind": "sostituzione",
    }

    html = template.render(notes=[note])

    assert "Note di revisione (1)" in html and "Esempi di lingua" in html
    assert "«foglia/folla»" in html and "aix-review-note--minor" in html
    assert template.render(notes=[]).strip() == ""
