"""The light turn's lesson is resent every turn: marked for the provider's cache, the text unchanged."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aix.agent.prompts.light_prompt import LIGHT_USER_TEMPLATE
from aix.agent.prompts.profile_renderer import render_profile_section
from aix.agent.prompts.today import today_note
from aix.core import prompt_cache
from aix.webui.agent import light

pytestmark = pytest.mark.unit

LESSON = SimpleNamespace(
    educational_profile_json={"grade": "seconda media", "__internal": "x"},
    lesson_plan_md="# Lezione: la memoria di lavoro\n\n## Attività {con graffe}\n",
)


def _old_user(documents, history, instruction):
    """How the user message was written before the split, byte for byte."""
    profile = {"grade": "seconda media"}
    return LIGHT_USER_TEMPLATE.format(
        today=today_note().strip(),
        profile=render_profile_section(profile),
        lesson=LESSON.lesson_plan_md,
        graph="",
        documents=documents,
        history=history,
        instruction=instruction,
    )


def test_the_user_message_is_the_same_text_as_before():
    messages = light._messages(
        LESSON, "accorcia l'attività", "## Storia\n", "## Passaggi\n", answer_here=False
    )

    assert messages[1]["content"] == _old_user(
        "## Passaggi\n", "## Storia\n", "accorcia l'attività"
    )


def test_the_head_is_the_start_of_the_user_message_and_holds_the_lesson():
    head = light._head(LESSON)
    user = light._messages(LESSON, "q", "", "", answer_here=True)[1]["content"]

    assert user.startswith(head)
    assert LESSON.lesson_plan_md in head
    assert "## La richiesta della docente" not in head, "the request changes: it is not cached"


def test_with_the_cache_on_the_lesson_is_the_marked_block(monkeypatch):
    monkeypatch.setenv("AIX_PROMPT_CACHE_ENABLED", "true")
    messages = light._messages(LESSON, "q", "", "## Passaggi\n", answer_here=False)

    marked = prompt_cache.cacheable(
        messages, model="anthropic/claude-sonnet-5.5", stable_user_prefix=light._head(LESSON)
    )

    head, tail = marked[1]["content"]
    assert head["cache_control"] == {"type": "ephemeral"} and LESSON.lesson_plan_md in head["text"]
    assert "cache_control" not in tail and "## Passaggi" in tail["text"]
    assert head["text"] + tail["text"] == messages[1]["content"]
