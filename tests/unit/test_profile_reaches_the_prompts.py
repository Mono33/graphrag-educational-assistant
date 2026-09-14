"""The rendered profile actually reaches both prompts (Q1 · 1.1).

``test_profile_renderer`` pins what the section says. These pin that it is used:
the Writer and the Planner are the two places the classroom has to arrive, and a
renderer nobody calls would pass every one of those tests while the lessons kept
asking for smartphones.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aix.agent.agents.planner_agent import PlannerAgent
from aix.agent.agents.writer_agent import WriterAgent

pytestmark = pytest.mark.unit

PROFILE = {
    "group": {"students_number": 25, "disabilities": ["DCGM", "DLDS"]},
    "classroom": {"own_device": "NO", "has_wifi": False},
    "time_available_minutes": 60,
}


def _completion(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")]
    )


async def _planner_prompt(**kwargs) -> str:
    captured: dict = {}

    async def _fake(_client, *, messages, **_kw):
        captured["messages"] = messages
        return _completion(
            json.dumps(
                {
                    "query_intent": "lesson_creation",
                    "key_concepts": ["memoria di lavoro"],
                    "search_queries": ["memoria di lavoro"],
                }
            )
        )

    with (
        patch.object(PlannerAgent, "_get_client", return_value=object()),
        patch("aix.agent.agents.planner_agent.guarded_chat_completion", _fake),
    ):
        await PlannerAgent().plan(query="Crea una lezione", **kwargs)

    return next(m["content"] for m in captured["messages"] if m["role"] == "user")


async def test_the_planner_is_told_the_classroom_constraints():
    prompt = await _planner_prompt(educational_profile=PROFILE)

    assert "do NOT have personal devices" in prompt
    assert "NO WiFi" in prompt


async def test_without_a_profile_the_planner_prompt_is_unchanged():
    assert await _planner_prompt() == await _planner_prompt(educational_profile=None)
    assert "Classroom constraints" not in await _planner_prompt()


async def _writer_prompt(profile) -> str:
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
            teacher_query="Crea una lezione",
            plan=_plan(),
            retrieval_result=_retrieval_result(),
            educational_profile=profile,
        )

    return next(m["content"] for m in captured["messages"] if m["role"] == "user")


async def test_the_writer_receives_the_whole_profile():
    prompt = await _writer_prompt(PROFILE)

    assert "do NOT have personal devices" in prompt
    assert "Class size: 25 students" in prompt
    assert "- Duration: 60 minutes" in prompt


@pytest.mark.parametrize("code", ["DCGM", "DLDS"])
async def test_the_writer_never_sees_the_raw_bes_codes(code):
    assert code not in await _writer_prompt(PROFILE)
