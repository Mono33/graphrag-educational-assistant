"""A turn whose writer produced nothing must fail, and say why.

From the third real test, 2026-09-16. ``AIX_WRITER_MAX_TOKENS`` was 2000 and
the model is a reasoning one: with two attachments and a thread behind it, the
whole budget went on thinking and the writer returned zero characters. Twice,
four minutes, and what the teacher saw was not an error — the chat showed the
*previous* lesson again, because ``final_lesson_plan`` survives in the thread's
checkpoint and nothing checked whether this turn had written anything.

Three defences, one per test below: retry with a budget that can actually hold a
lesson, refuse to pass an empty draft on as a result, and never read a lesson out
of a state whose nodes reported a failure.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# The writer retries with a budget that can hold a lesson
# ---------------------------------------------------------------------------


async def test_the_zero_content_retry_raises_the_token_ceiling():
    from tests.unit.agent_doubles import _plan, _retrieval_result

    import aix.agent.agents.writer_agent as writer_module
    from aix.agent.agents.writer_agent import WriterAgent

    monkey_budget = 2000
    calls: list[int] = []

    async def _fake_stream(_self, _client, _messages, kwargs, _bus):
        calls.append(kwargs["max_tokens"])
        return ("# Lezione" if len(calls) > 1 else "", "stop")

    with (
        patch.object(writer_module, "_WRITER_MAX_TOKENS", monkey_budget),
        patch.object(writer_module, "_WRITER_MIN_CONTENT_TOKENS", 4000),
        patch.object(WriterAgent, "_get_client", return_value=object()),
        patch.object(WriterAgent, "_stream_completion", _fake_stream),
    ):
        import asyncio

        content = await WriterAgent().write(
            teacher_query="Crea una lezione",
            plan=_plan(),
            retrieval_result=_retrieval_result(),
            token_bus=asyncio.Queue(),
        )

    assert calls == [2000, 4000], "retrying under the same ceiling repeats the failure"
    assert content.strip()


# ---------------------------------------------------------------------------
# An empty draft is a failed turn, not a silent one
# ---------------------------------------------------------------------------


async def test_write_node_reports_an_empty_draft_as_an_error():
    from aix.agent.graph import nodes as graph_nodes

    writer = AsyncMock(write=AsyncMock(return_value="   \n  "))
    state = {
        "teacher_query": "amplia la lezione con il nuovo documento",
        "plan": {"query_intent": "lesson_creation"},
        "query_intent": "lesson_creation",
        "language": "it",
    }

    with patch.object(graph_nodes, "get_writer", return_value=writer):
        updates = await graph_nodes.write_node(state)

    assert updates["current_step"] == "error"
    assert "AIX_WRITER_MAX_TOKENS" in updates["error"], "the message says what to change"
    assert "lesson_plan_draft" not in updates


# ---------------------------------------------------------------------------
# The previous lesson is never served as the answer to a new request
# ---------------------------------------------------------------------------


def test_a_failed_turn_never_returns_the_previous_lesson():
    from aix.webui.agent.service import (
        _extract_lesson_plan_md,
        _fail_if_the_turn_produced_nothing,
    )

    # Exactly the shape of the 2026-09-16 run: the writer wrote nothing this
    # turn, and turn 1's approved lesson is still in the checkpoint.
    final_state = {
        "error": "Il modello non ha prodotto testo: aumenta AIX_WRITER_MAX_TOKENS.",
        "final_lesson_plan": "# La Guerra Fredda\n\nLezione del turno precedente.",
    }

    assert _extract_lesson_plan_md(final_state), "the stale lesson is there to be served"
    with pytest.raises(RuntimeError, match="AIX_WRITER_MAX_TOKENS"):
        _fail_if_the_turn_produced_nothing(final_state)


def test_a_healthy_run_passes_through_untouched():
    from aix.webui.agent.service import _fail_if_the_turn_produced_nothing

    for state in ({}, {"error": None}, {"error": "   "}, {"final_lesson_plan": "# Lezione"}):
        _fail_if_the_turn_produced_nothing(state)
