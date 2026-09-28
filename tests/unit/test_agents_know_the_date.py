"""The Writer and the Critic are told what day it is; the Critic asks for lesson changes.

A lesson on current affairs, 2026-09-25. The Critic asked whether the 2026
migration crisis in Ceuta was "a hypothetical scenario" and wanted "a more
robust source for a future event": the event was real, its source fetched that
day, and nothing had ever told the model the date. The same review asked the
Writer for "a self-assessment score" on each neuroscience criterion, and the
lesson grew a table in which the AI gave itself 5/5.

These tests check the prompts the models actually receive, not the templates
alone: a note that is written but never sent protects nothing.
"""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aix.agent.agents.critic_agent import CriticAgent
from aix.agent.agents.writer_agent import WriterAgent
from aix.agent.prompts import critic_prompt
from aix.agent.prompts.today import today_note

pytestmark = pytest.mark.unit


def _completion(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")]
    )


def test_the_note_says_the_date_in_italian_and_not_to_invent_news():
    note = today_note(date(2026, 9, 25))

    assert "25 settembre 2026" in note
    assert "non trattarli come ipotetici" in note
    assert "Non aggiungere però fatti recenti che le fonti non riportano" in note


async def test_the_writer_is_told_the_date():
    from tests.unit.agent_doubles import _plan, _retrieval_result

    captured: dict = {}

    async def _fake(_client, *, messages, **_kw):
        captured["messages"] = messages
        return _completion("# Lezione")

    with (
        patch.object(WriterAgent, "_get_client", return_value=object()),
        patch("aix.agent.agents.writer_agent.guarded_chat_completion", _fake),
    ):
        await WriterAgent().write(
            teacher_query="Crea una lezione sulla crisi migratoria",
            plan=_plan(),
            retrieval_result=_retrieval_result(),
        )

    prompt = next(m["content"] for m in captured["messages"] if m["role"] == "user")
    assert "## Data di oggi" in prompt
    assert str(date.today().year) in prompt


async def test_the_critic_is_told_the_date():
    from tests.unit.agent_doubles import _retrieval_result

    captured: dict = {}
    verdict = {
        "scores": {"structure": 4},
        "average_score": 4.0,
        "decision": "APPROVE",
        "strengths": [],
        "weaknesses": [],
        "revision_instructions": None,
        "summary": "ok",
    }

    async def _fake(_client, *, messages, **_kw):
        captured["messages"] = messages
        return _completion(json.dumps(verdict))

    with (
        patch.object(CriticAgent, "_get_client", return_value=object()),
        patch("aix.agent.agents.critic_agent.guarded_chat_completion", _fake),
    ):
        retrieval = _retrieval_result()
        retrieval.to_context_string = lambda *args, **kwargs: "Contesto recuperato."
        await CriticAgent().critique(
            lesson_plan="# Lezione sulla crisi di Ceuta del 2026",
            teacher_query="Crea una lezione di attualità",
            retrieval_result=retrieval,
        )

    prompt = next(m["content"] for m in captured["messages"] if m["role"] == "user")
    assert "## Data di oggi" in prompt


@pytest.mark.parametrize(
    "name", ["CRITIC_SYSTEM_PROMPT_LESSON", "CRITIC_SYSTEM_PROMPT_INFORMATIONAL"]
)
def test_the_critic_asks_for_changes_to_the_lesson_never_for_self_assessment(name):
    prompt = getattr(critic_prompt, name)

    assert "Never ask the Writer to add scores" in prompt
    assert "self-assessments" in prompt
    assert "in the language of the lesson" in prompt
    assert "auto-approve" not in prompt, "nothing is auto-approved at the limit"
