"""A UDL lesson's footer says UDL, not neuroscience (staging, 2026-10-16)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aix.agent.agents.writer_agent import WriterAgent
from aix.agent.prompts.writer_prompt import apply_kg_basis, kg_basis

pytestmark = pytest.mark.unit


def _text(message) -> str:
    content = message["content"]
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content)
    return content


async def _hybrid_messages(domain: str) -> list[dict]:
    from tests.unit.agent_doubles import _plan

    plan = _plan()
    plan.scope_status = "partial_scope"
    result = SimpleNamespace(nodes=[], relationships=[], recommendations=[], confidence="HIGH")
    seen = []

    async def _fake_stream(_self, _client, messages, _kwargs, _bus):
        seen.append(messages)
        return ("# Lezione", "stop")

    with (
        patch.object(WriterAgent, "_get_client", return_value=object()),
        patch.object(WriterAgent, "_stream_completion", _fake_stream),
    ):
        await WriterAgent().write(
            teacher_query="Il ciclo dell'acqua",
            plan=plan,
            retrieval_result=result,
            external_resources={"wikipedia": [], "papers": []},
            domain=domain,
            token_bus=asyncio.Queue(),
        )
    return seen[0]


async def test_a_udl_lesson_names_the_udl_graph():
    system, user = (_text(m) for m in await _hybrid_messages("udl"))

    assert "basate sul Knowledge Graph UDL (Universal Design for Learning)" in system
    assert "neuroscien" not in system.split("## Domain Expert", 1)[0].lower()
    assert "Apply these UDL-based (Universal Design for Learning) strategies" in user
    assert "{kg_basis" not in system + user


async def test_a_neuro_lesson_keeps_its_wording():
    system, user = (_text(m) for m in await _hybrid_messages("neuro"))

    assert "Knowledge Graph di neuroscienze dell'apprendimento" in system
    assert "Apply these neuroscience-based strategies" in user


def test_unknown_domain_falls_back_to_neuro():
    assert kg_basis("boh") == kg_basis("neuro")
    assert (
        apply_kg_basis("{kg_basis_it}", "UDL")
        == "Knowledge Graph UDL (Universal Design for Learning)"
    )
