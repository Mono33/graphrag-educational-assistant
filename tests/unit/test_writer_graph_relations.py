"""The graph's relationships reach the Writer as a few lines, only when switched on."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aix.agent.agents.writer_agent import _format_relationships

pytestmark = pytest.mark.unit

RELATIONS = [
    {"source": "ADHD", "type": "SUGGESTS", "target": "Pause attive"},
    {"source": "ADHD", "type": "SUGGESTS", "target": "Pause attive"},
    {"source": "Dislessia", "type": "MITIGATED_BY", "target": "Font ad alta leggibilità"},
    {"source": "", "type": "X", "target": "senza origine"},
]


def test_relationships_become_unique_short_lines():
    text = _format_relationships(RELATIONS)

    assert "- ADHD —SUGGESTS→ Pause attive" in text
    assert text.count("Pause attive") == 1, "duplicates are dropped"
    assert "senza origine" not in text, "an incomplete triple says nothing"


def test_at_most_fifteen():
    many = [{"source": f"A{i}", "type": "SUGGESTS", "target": f"B{i}"} for i in range(40)]

    assert _format_relationships(many).count("\n- ") == 15


def test_none_is_nothing():
    assert _format_relationships([]) == ""


async def _user_prompt(monkeypatch, enabled: bool) -> str:
    from tests.unit.agent_doubles import _plan, _retrieval_result

    from aix.agent.agents.writer_agent import WriterAgent

    monkeypatch.setenv("AIX_WRITER_GRAPH_RELATIONS", "true" if enabled else "false")
    result = _retrieval_result()
    result.relationships = RELATIONS
    seen = []

    async def _fake_stream(_self, _client, messages, _kwargs, _bus):
        seen.append(messages)
        return ("# Lezione", "stop")

    with (
        patch.object(WriterAgent, "_get_client", return_value=object()),
        patch.object(WriterAgent, "_stream_completion", _fake_stream),
    ):
        await WriterAgent().write(
            teacher_query="Crea una lezione",
            plan=_plan(),
            retrieval_result=result,
            token_bus=asyncio.Queue(),
        )
    return seen[0][1]["content"]


async def test_switched_on_the_writer_sees_them(monkeypatch):
    assert "ADHD —SUGGESTS→ Pause attive" in await _user_prompt(monkeypatch, True)


async def test_switched_off_the_prompt_is_as_before(monkeypatch):
    assert "Relazioni nel Knowledge Graph" not in await _user_prompt(monkeypatch, False)
