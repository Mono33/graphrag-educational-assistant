"""
CORE 2 #11a — JSON parse hardening regression coverage
=======================================================

Locks in three guarantees added by the #11a fix:

1. ``json_mode=True`` is forwarded to the OpenAI client on Planner and
   Critic completion calls so providers that honour it return strict JSON
   (eliminates the bulk of the silent "Approved due to parsing error"
   failures observed pre-#11a).
2. The legacy parse-error fallback in :class:`CriticAgent` and
   :class:`PlannerAgent` is preserved bit-for-bit by default — flipping
   ``AIX_CRITIC_PARSE_ERROR_BEHAVIOR`` is the only way to change it, so
   pre-#11a callers see *zero* behavioural difference.
3. When ``AIX_CRITIC_PARSE_ERROR_BEHAVIOR=revise``, the Critic forces a
   revision pass instead of silently approving, surfacing the failure to
   downstream observability and the writer-revise loop.

These are pure unit tests — the LLM is fully mocked, so they run in
milliseconds and are stable in CI.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aix.agent.agents.critic_agent import CriticAgent
from aix.agent.agents.planner_agent import PlannerAgent
from aix.agent.agents.retriever_agent import RetrievalResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_response(text: str) -> SimpleNamespace:
    """Mimic the bits of the OpenAI completion response that the agents read."""
    msg = SimpleNamespace(content=text, reasoning_content=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None)


def _patch_planner_client(planner: PlannerAgent, completion_text: str) -> MagicMock:
    """Install a fake AsyncOpenAI client on the planner; return the mock."""
    fake_create = AsyncMock(return_value=_fake_response(completion_text))
    fake_client = MagicMock()
    fake_client.chat = MagicMock()
    fake_client.chat.completions = MagicMock()
    fake_client.chat.completions.create = fake_create
    planner._client = fake_client
    return fake_create


def _patch_critic_client(critic: CriticAgent, completion_text: str) -> MagicMock:
    fake_create = AsyncMock(return_value=_fake_response(completion_text))
    fake_client = MagicMock()
    fake_client.chat = MagicMock()
    fake_client.chat.completions = MagicMock()
    fake_client.chat.completions.create = fake_create
    critic._client = fake_client
    return fake_create


# ---------------------------------------------------------------------------
# 1. json_mode forwarding
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_planner_forwards_json_mode_to_client():
    """The Planner must request ``response_format={"type": "json_object"}``
    on non-reasoning models (default ``gpt-4o``)."""
    planner = PlannerAgent(model="gpt-4o")
    fake_create = _patch_planner_client(
        planner,
        json.dumps({
            "query_intent": "lesson_creation",
            "key_concepts": ["a"],
            "search_queries": ["a"],
            "lesson_type": "full_lesson",
        }),
    )

    await planner.plan("test query", domain="neuro", language="it")

    fake_create.assert_awaited_once()
    kwargs = fake_create.await_args.kwargs
    assert kwargs.get("response_format") == {"type": "json_object"}, (
        f"Planner did not opt into json_mode (got response_format="
        f"{kwargs.get('response_format')!r})"
    )


@pytest.mark.asyncio
async def test_critic_forwards_json_mode_to_client():
    """Same guarantee for the Critic — historically this was the
    silent-failure path that auto-approved unparseable JSON."""
    critic = CriticAgent(model="gpt-4o")
    fake_create = _patch_critic_client(
        critic,
        json.dumps({
            "scores": {"a": 4},
            "average_score": 4.0,
            "decision": "APPROVE",
            "strengths": [],
            "weaknesses": [],
            "summary": "ok",
        }),
    )

    await critic.critique(
        lesson_plan="draft",
        teacher_query="query",
        retrieval_result=RetrievalResult(),
        revision_count=0,
        max_revisions=2,
        domain="neuro",
        language="it",
        query_intent="lesson_creation",
    )

    fake_create.assert_awaited_once()
    kwargs = fake_create.await_args.kwargs
    assert kwargs.get("response_format") == {"type": "json_object"}, (
        f"Critic did not opt into json_mode (got response_format="
        f"{kwargs.get('response_format')!r})"
    )


# ---------------------------------------------------------------------------
# 2. Default (legacy) parse-error fallback is preserved bit-for-bit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_critic_parse_error_default_path_is_legacy_approve(monkeypatch):
    """When the LLM returns garbage and ``AIX_CRITIC_PARSE_ERROR_BEHAVIOR``
    is unset, the Critic must keep the pre-#11a "auto-approve with summary
    'Approved due to parsing error'" behaviour for backward compatibility."""
    monkeypatch.delenv("AIX_CRITIC_PARSE_ERROR_BEHAVIOR", raising=False)
    critic = CriticAgent(model="gpt-4o")
    _patch_critic_client(critic, "this is not json at all <<<")

    result = await critic.critique(
        lesson_plan="draft",
        teacher_query="query",
        retrieval_result=RetrievalResult(),
        revision_count=0,
        max_revisions=2,
        domain="neuro",
        language="it",
        query_intent="lesson_creation",
    )

    assert result.decision == "APPROVE"
    assert result.summary == "Approved due to parsing error"
    assert result.approved is True


@pytest.mark.asyncio
async def test_planner_parse_error_default_path_is_legacy_fallback():
    """The Planner returns a degraded but legal RetrievalPlan on parse
    failure — same behaviour as pre-#11a."""
    planner = PlannerAgent(model="gpt-4o")
    _patch_planner_client(planner, "<<< not json >>>")

    plan = await planner.plan("query", domain="neuro", language="it")

    assert plan.query_intent == "lesson_creation"
    assert plan.lesson_type == "full_lesson"
    assert plan.intent_confidence == "LOW"
    assert plan.reasoning == "Fallback plan due to JSON parsing error"


# ---------------------------------------------------------------------------
# 3. Opt-in "revise" mode forces a revision pass instead of silent approve
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_critic_parse_error_revise_mode_forces_revision(monkeypatch):
    """With ``AIX_CRITIC_PARSE_ERROR_BEHAVIOR=revise`` the failure is
    surfaced into the revision loop with a typed ``[parse_error]`` marker
    in ``revision_instructions`` instead of being silently approved."""
    monkeypatch.setenv("AIX_CRITIC_PARSE_ERROR_BEHAVIOR", "revise")
    critic = CriticAgent(model="gpt-4o")
    _patch_critic_client(critic, "{ malformed json")

    result = await critic.critique(
        lesson_plan="draft",
        teacher_query="query",
        retrieval_result=RetrievalResult(),
        revision_count=0,
        max_revisions=2,
        domain="neuro",
        language="it",
        query_intent="lesson_creation",
    )

    assert result.decision == "REVISE"
    assert result.approved is False
    assert "[parse_error]" in (result.revision_instructions or "")


# ---------------------------------------------------------------------------
# 4. 5-run smoke: check that valid JSON never trips the fallback path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_critic_5_run_smoke_no_parse_error_when_response_is_valid(monkeypatch):
    """5 runs against a deterministic mocked client that always returns
    well-formed JSON. Locks in that no run accidentally enters the
    parse-error fallback (which would surface as
    summary='Approved due to parsing error')."""
    monkeypatch.delenv("AIX_CRITIC_PARSE_ERROR_BEHAVIOR", raising=False)

    valid_payload = json.dumps({
        "scores": {"clarity": 4, "evidence": 4},
        "average_score": 4.0,
        "decision": "APPROVE",
        "strengths": ["good"],
        "weaknesses": [],
        "summary": "looks fine",
    })

    for run_idx in range(5):
        critic = CriticAgent(model="gpt-4o")
        _patch_critic_client(critic, valid_payload)

        result = await critic.critique(
            lesson_plan=f"draft #{run_idx}",
            teacher_query="query",
            retrieval_result=RetrievalResult(),
            revision_count=0,
            max_revisions=2,
            domain="neuro",
            language="it",
            query_intent="lesson_creation",
        )

        # If this fires, the Critic accidentally drifted back into the
        # parse-error fallback even on valid JSON — that's a regression of
        # the #11a fix and must be investigated immediately.
        assert result.summary != "Approved due to parsing error", (
            f"run {run_idx}: critic fell into parse-error fallback on valid JSON"
        )
        assert result.decision == "APPROVE"
