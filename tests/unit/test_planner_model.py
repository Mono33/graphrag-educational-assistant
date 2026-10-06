"""A cheaper planner must never mean a worse plan.

The planner is a classification and an extraction, so a flash-lite model may do
it (19 s and $0.035 a call on Sonnet, 2026-10-06). What is pinned here is the net
under it: a plan the cheap model is unsure of, cannot give, or fails to deliver is
asked again of the default model, and with no ``AIX_PLANNER_MODEL`` nothing changes.
"""

from __future__ import annotations

import json

import pytest

from aix.agent.agents import planner_agent
from aix.agent.agents.planner_agent import PlannerAgent, weak_plan_reason

pytestmark = pytest.mark.unit

CHEAP = "google/gemini-3.5-flash-lite"


def _plan(**over):
    base = {
        "response_language": "it",
        "language_confidence": "HIGH",
        "query_intent": "definition",
        "intent_confidence": "HIGH",
        "scope_status": "in_scope",
        "scope_confidence": 0.9,
        "key_concepts": ["neuroplasticity"],
        "search_queries": ["neuroplasticity definition"],
    }
    base.update(over)
    return base


class _Script:
    """``_request_plan`` replaced by what each model would answer, in order."""

    def __init__(self, cheap, default):
        self.answers = {CHEAP: list(cheap), None: list(default)}
        self.models = []

    async def __call__(self, client, user_prompt, *, model):
        self.models.append(model)
        answer = self.answers[model].pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.delenv("AIX_PLANNER_MODEL", raising=False)
    monkeypatch.delenv("AIX_PLANNER_ESCALATE", raising=False)
    planner = PlannerAgent()
    monkeypatch.setattr(planner, "_get_client", lambda: object())
    return planner


def _use(monkeypatch, agent, script):
    monkeypatch.setattr(agent, "_request_plan", script)
    return script


async def test_with_no_planner_model_only_the_default_model_is_asked(agent, monkeypatch):
    script = _use(monkeypatch, agent, _Script(cheap=[], default=[_plan()]))

    plan = await agent.plan("Cos'è la neuroplasticità?")

    assert script.models == [None]
    assert plan.query_intent == "definition"


async def test_a_good_plan_from_the_cheap_model_is_taken_and_costs_one_call(agent, monkeypatch):
    monkeypatch.setenv("AIX_PLANNER_MODEL", CHEAP)
    script = _use(monkeypatch, agent, _Script(cheap=[_plan()], default=[]))

    plan = await agent.plan("Cos'è la neuroplasticità?")

    assert script.models == [CHEAP]
    assert plan.key_concepts == ["neuroplasticity"]


@pytest.mark.parametrize(
    "weak",
    [
        _plan(intent_confidence="LOW"),
        _plan(language_confidence="low"),
        _plan(response_language="de"),
        _plan(scope_confidence=0.3),
        _plan(scope_confidence="boh"),
        {"key_concepts": []},
        json.JSONDecodeError("the plan is truncated", "{", 0),
        RuntimeError("provider unavailable"),
    ],
    ids=[
        "intent LOW",
        "language LOW",
        "unsupported language",
        "scope unsure",
        "scope unreadable",
        "no intent",
        "unparseable",
        "call failed",
    ],
)
async def test_a_weak_or_missing_plan_is_asked_again_of_the_default_model(agent, monkeypatch, weak):
    monkeypatch.setenv("AIX_PLANNER_MODEL", CHEAP)
    script = _use(
        monkeypatch, agent, _Script(cheap=[weak], default=[_plan(key_concepts=["from default"])])
    )

    plan = await agent.plan("Cos'è la neuroplasticità?")

    assert script.models == [CHEAP, None]
    assert plan.key_concepts == ["from default"], "the plan used is the default model's"


async def test_with_escalation_off_a_weak_plan_is_kept_and_a_failure_falls_back(agent, monkeypatch):
    monkeypatch.setenv("AIX_PLANNER_MODEL", CHEAP)
    monkeypatch.setenv("AIX_PLANNER_ESCALATE", "false")

    script = _use(monkeypatch, agent, _Script(cheap=[_plan(intent_confidence="LOW")], default=[]))
    plan = await agent.plan("q")
    assert script.models == [CHEAP] and plan.intent_confidence == "LOW"

    script = _use(
        monkeypatch,
        agent,
        _Script(cheap=[json.JSONDecodeError("truncated", "{", 0)], default=[]),
    )
    plan = await agent.plan("q")
    assert script.models == [CHEAP]
    assert plan.reasoning == "Fallback plan due to JSON parsing error"


def test_the_planner_model_is_empty_unless_set(monkeypatch):
    monkeypatch.delenv("AIX_PLANNER_MODEL", raising=False)
    assert planner_agent.planner_model() is None

    monkeypatch.setenv("AIX_PLANNER_MODEL", f"  {CHEAP} ")
    assert planner_agent.planner_model() == CHEAP


def test_a_sound_plan_is_not_weak():
    assert weak_plan_reason(_plan()) is None
    assert weak_plan_reason(_plan(response_language="IT-it")) is None
