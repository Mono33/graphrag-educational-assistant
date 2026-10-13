"""The Writer's and the Planner's system prompts are marked for the provider's cache, unchanged.

They are the same for every teacher of a domain (Writer: intent, domain
knowledge, constraints, ~3-4k tokens; Planner: 14k characters of instructions and
examples) and were paid in full on every call.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit

SONNET = "anthropic/claude-sonnet-5.5"


@pytest.fixture
def cache_on(monkeypatch):
    monkeypatch.setenv("AIX_PROMPT_CACHE_ENABLED", "true")
    monkeypatch.delenv("AIX_PLANNER_MODEL", raising=False)


async def _writer_messages(model: str) -> list:
    from tests.unit.agent_doubles import _plan, _retrieval_result

    from aix.agent.agents.writer_agent import WriterAgent
    from aix.core.config import config

    seen: list = []

    async def _fake_stream(_self, _client, messages, _kwargs, _bus):
        seen.append(messages)
        return ("# Lezione\n\nTesto.", "stop")

    with (
        patch.object(config.openai, "model", model),
        patch.object(WriterAgent, "_get_client", return_value=object()),
        patch.object(WriterAgent, "_stream_completion", _fake_stream),
    ):
        await WriterAgent().write(
            teacher_query="Crea una lezione",
            plan=_plan(),
            retrieval_result=_retrieval_result(),
            token_bus=asyncio.Queue(),
        )
    return seen[0]


async def test_the_writer_system_prompt_is_one_marked_block(cache_on):
    messages = await _writer_messages(SONNET)

    (block,) = messages[0]["content"]
    assert block["cache_control"] == {"type": "ephemeral"}
    assert block["text"].strip(), "the same text, only wrapped"
    assert isinstance(messages[1]["content"], str), "the request itself is not cached"


async def test_the_planner_system_prompt_is_one_marked_block(cache_on):
    from aix.agent.agents.planner_agent import PlannerAgent
    from aix.agent.prompts.planner_prompt import PLANNER_SYSTEM_PROMPT
    from aix.core.config import config

    seen = {}

    async def create(**kwargs):
        seen.update(kwargs)
        plan = {"query_intent": "definition", "intent_confidence": "HIGH", "key_concepts": []}
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(plan), reasoning_content=None),
                    finish_reason="stop",
                )
            ]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with patch.object(config.openai, "model", SONNET):
        await PlannerAgent()._request_plan(client, "Cos'è la memoria?", model=None)

    (block,) = seen["messages"][0]["content"]
    assert block == {
        "type": "text",
        "text": PLANNER_SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }
