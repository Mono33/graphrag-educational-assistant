"""A topic outside the Knowledge Graph still lets the Writer cite the teacher's documents.

The hybrid prompt (history, literature: subjects the graph does not cover) said
"use external sources for WHAT to teach" and "do NOT insert annotation
brackets anywhere in the lesson body". With a deck and a map attached, the
first draft cited neither, twice in a row, and only the revision -- whose
prompt has no such lines -- put the markers in (2026-09-30).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aix.agent.agents.writer_agent import WriterAgent
from aix.agent.prompts.writer_prompt import (
    WRITER_SYSTEM_PROMPT_HYBRID,
    WRITER_USER_TEMPLATE_HYBRID,
    get_writer_prompts,
)

pytestmark = pytest.mark.unit


def test_the_hybrid_prompts_make_the_documents_the_source_and_keep_their_markers():
    assert "[[material:…]]" in WRITER_SYSTEM_PROMPT_HYBRID
    assert "teacher's attached documents** as the primary source" in WRITER_SYSTEM_PROMPT_HYBRID
    assert "dai materiali allegati dalla docente" in WRITER_SYSTEM_PROMPT_HYBRID
    assert "Use the teacher's attached documents, when there are any, for WHAT" in (
        WRITER_USER_TEMPLATE_HYBRID
    )
    assert "cited in the body with their [[material:…]] markers" in WRITER_USER_TEMPLATE_HYBRID


def test_a_history_lesson_with_a_deck_gets_the_hybrid_prompt_and_the_passages():
    from tests.unit.agent_doubles import _plan, _retrieval_result

    captured: dict = {}

    async def _fake(_client, *, messages, **_kw):
        captured["messages"] = messages
        message = SimpleNamespace(content="# Lezione")
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

    plan = _plan()
    plan.scope_status = "partial_scope"
    passage = {
        "source_id": "v1",
        "source_label": "La Guerra Fredda.pptx",
        "attachment": 1,
        "score": 0.6,
        "text": "Giugno 1948: l'URSS chiude gli accessi a Berlino Ovest.",
        "citation": "[[material:v1#slide=4]]",
    }

    async def _run():
        with (
            patch.object(WriterAgent, "_get_client", return_value=object()),
            patch("aix.agent.agents.writer_agent.guarded_chat_completion", _fake),
        ):
            await WriterAgent().write(
                teacher_query="crea una lezione coinvolgente sfruttando le slide allegate",
                plan=plan,
                retrieval_result=_retrieval_result(),
                external_resources={"wikipedia": [{"title": "Guerra fredda", "summary": "..."}]},
                material_context_items=[passage],
            )

    import asyncio

    asyncio.run(_run())

    system = next(m["content"] for m in captured["messages"] if m["role"] == "system")
    user = next(m["content"] for m in captured["messages"] if m["role"] == "user")
    assert system.startswith(get_writer_prompts("lesson_creation", "partial_scope")[0][:60])
    assert "Create a lesson plan for a topic OUTSIDE the Knowledge Graph domain" in user
    assert "[[material:v1#slide=4]]" in user
    assert "Sono la fonte\nprincipale dei contenuti" in user
