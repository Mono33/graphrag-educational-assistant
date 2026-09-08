"""Timeout and retry budget for the LLM calls.

The budget used to be entirely implicit. The OpenAI client was built with no
``timeout`` and no ``max_retries``, so the SDK defaults applied: a 600-second
read timeout and two retries. One hung call could therefore hold a lesson for
half an hour while the run registry's heartbeat kept reporting it healthy.

It is now split deliberately in two halves, and these tests pin both:

* the **client** carries the inner half — an explicit timeout and one retry,
  configured in one place so the whole budget is readable;
* the **node** carries the outer half — a ``RetryPolicy`` that re-enters the
  node, rebuilding the prompt and re-parsing the response rather than replaying
  the same HTTP request.

Two attempts on each side bounds a node at four calls. The half that makes the
outer retry possible at all is the re-raise in the nodes: they catch everything
by default, so without it LangGraph would never see a transient failure.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

from aix.agent.graph import nodes as graph_nodes
from aix.agent.graph.lesson_planner_graph import (
    LLM_RETRY_POLICY,
    build_lesson_planner_graph,
    reset_compiled_graph,
)
from aix.agent.graph.nodes import TRANSIENT_LLM_ERRORS
from aix.agent.graph.state import create_initial_state
from aix.core.config import OpenAIConfig

pytestmark = pytest.mark.unit

_REQUEST = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")


def _rate_limit() -> RateLimitError:
    return RateLimitError("slow down", response=httpx.Response(429, request=_REQUEST), body=None)


@pytest.fixture(autouse=True)
def _clean_graph_singleton():
    reset_compiled_graph()
    yield
    reset_compiled_graph()


# ---------------------------------------------------------------------------
# 1. The client half of the budget
# ---------------------------------------------------------------------------


def test_client_no_longer_inherits_the_sdk_ten_minute_timeout(monkeypatch):
    monkeypatch.delenv("AIX_LLM_TIMEOUT_S", raising=False)
    monkeypatch.delenv("AIX_LLM_MAX_RETRIES", raising=False)

    client = OpenAIConfig(api_key="k").get_async_client()

    assert client.timeout == 120.0, "the SDK default is 600s, which is not an interactive budget"
    assert client.max_retries == 1, "the other retry is spent by the node-level policy"


@pytest.mark.parametrize(
    "raw, expected, why",
    [
        ("30", 30.0, "plain value"),
        ("0.5", 5.0, "clamped up — below 5s nothing would ever complete"),
        ("not-a-number", 120.0, "unparseable falls back to the default"),
    ],
)
def test_timeout_env_override(raw, expected, why, monkeypatch):
    monkeypatch.setenv("AIX_LLM_TIMEOUT_S", raw)
    assert OpenAIConfig(api_key="k").get_async_client().timeout == expected, why


@pytest.mark.parametrize(
    "raw, expected, why",
    [
        ("0", 0, "retries can be turned off entirely"),
        ("3", 3, "plain value"),
        ("99", 5, "clamped — the node policy multiplies whatever is set here"),
        ("", 1, "unparseable falls back to the default"),
    ],
)
def test_max_retries_env_override(raw, expected, why, monkeypatch):
    monkeypatch.setenv("AIX_LLM_MAX_RETRIES", raw)
    assert OpenAIConfig(api_key="k").get_async_client().max_retries == expected, why


def test_sync_client_gets_the_same_budget(monkeypatch):
    monkeypatch.setenv("AIX_LLM_TIMEOUT_S", "45")
    monkeypatch.setenv("AIX_LLM_MAX_RETRIES", "2")
    cfg = OpenAIConfig(api_key="k")

    assert cfg.get_client().timeout == cfg.get_async_client().timeout
    assert cfg.get_client().max_retries == cfg.get_async_client().max_retries


# ---------------------------------------------------------------------------
# 2. The policy, and what it is attached to
# ---------------------------------------------------------------------------


def test_policy_retries_exactly_what_the_nodes_re_raise():
    """One tuple, used by both halves, so they cannot disagree."""
    assert LLM_RETRY_POLICY.retry_on is TRANSIENT_LLM_ERRORS
    assert set(TRANSIENT_LLM_ERRORS) == {
        RateLimitError,
        APITimeoutError,
        APIConnectionError,
        InternalServerError,
    }


def test_policy_budget_is_bounded():
    assert LLM_RETRY_POLICY.max_attempts == 2, (
        "two here on top of the client's two bounds a node at four calls"
    )
    assert LLM_RETRY_POLICY.jitter is True


@pytest.mark.parametrize("node", ["plan", "write", "critique"])
def test_llm_nodes_carry_the_policy(node, monkeypatch):
    monkeypatch.delenv("AIX_CORRECTIVE_RAG_ENABLED", raising=False)
    compiled = build_lesson_planner_graph()
    assert tuple(compiled.nodes[node].retry_policy) == (LLM_RETRY_POLICY,)


def test_retriever_does_not_carry_the_llm_policy(monkeypatch):
    """The retriever talks to Neo4j, not the LLM — a different failure mode."""
    monkeypatch.delenv("AIX_CORRECTIVE_RAG_ENABLED", raising=False)
    compiled = build_lesson_planner_graph()
    assert not compiled.nodes["retrieve"].retry_policy


def test_grader_is_deliberately_left_without_a_policy(monkeypatch):
    """It already degrades to grade=relevant rather than failing the run.

    Retrying an optional quality gate spends latency on a node designed never
    to matter that much — so it keeps its fail-safe instead of a retry budget.
    """
    monkeypatch.setenv("AIX_CORRECTIVE_RAG_ENABLED", "true")
    compiled = build_lesson_planner_graph()
    assert not compiled.nodes["grade_retrieval"].retry_policy


# ---------------------------------------------------------------------------
# 3. The re-raise, without which the policy would never fire
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        _rate_limit(),
        APITimeoutError(request=_REQUEST),
        APIConnectionError(request=_REQUEST),
        InternalServerError("upstream", response=httpx.Response(503, request=_REQUEST), body=None),
    ],
    ids=["429", "timeout", "connection", "5xx"],
)
async def test_plan_node_re_raises_transient_failures(exc):
    planner = AsyncMock(plan=AsyncMock(side_effect=exc))
    with patch.object(graph_nodes, "get_planner", return_value=planner):
        with pytest.raises(type(exc)):
            await graph_nodes.plan_node(create_initial_state(query="q"))


async def test_plan_node_still_swallows_everything_else():
    """A bad prompt or a broken response must end the run cleanly, not crash it."""
    planner = AsyncMock(plan=AsyncMock(side_effect=ValueError("malformed plan")))
    with patch.object(graph_nodes, "get_planner", return_value=planner):
        result = await graph_nodes.plan_node(create_initial_state(query="q"))

    assert result["current_step"] == "error"
    assert "malformed plan" in result["error"]


# ---------------------------------------------------------------------------
# 4. End to end: one transient failure is absorbed, the run still finishes
# ---------------------------------------------------------------------------


async def test_a_single_rate_limit_no_longer_fails_the_run(monkeypatch):
    """The whole chain: node re-raises, policy retries, second attempt succeeds."""
    monkeypatch.delenv("AIX_CORRECTIVE_RAG_ENABLED", raising=False)

    from tests.unit.test_graph_topology import _critique, _plan, _retrieval_result

    planner = AsyncMock(plan=AsyncMock(side_effect=[_rate_limit(), _plan()]))
    retriever = AsyncMock(retrieve=AsyncMock(return_value=_retrieval_result()))
    writer = AsyncMock(write=AsyncMock(return_value="# Lezione"))
    critic = AsyncMock(critique=AsyncMock(return_value=_critique(True)))

    with (
        patch.object(graph_nodes, "get_planner", return_value=planner),
        patch.object(graph_nodes, "get_retriever", return_value=retriever),
        patch.object(graph_nodes, "get_writer", return_value=writer),
        patch.object(graph_nodes, "get_critic", return_value=critic),
    ):
        final = await build_lesson_planner_graph().ainvoke(
            create_initial_state(query="Crea una lezione", max_revisions=2)
        )

    assert planner.plan.await_count == 2, "the planner was re-entered after the 429"
    assert not final.get("error")
    assert final["final_lesson_plan"] == "# Lezione"


async def test_exhausting_the_budget_surfaces_the_failure(monkeypatch):
    """Better than the empty 'done' an error state used to produce."""
    monkeypatch.delenv("AIX_CORRECTIVE_RAG_ENABLED", raising=False)

    planner = AsyncMock(plan=AsyncMock(side_effect=_rate_limit()))

    with patch.object(graph_nodes, "get_planner", return_value=planner):
        with pytest.raises(RateLimitError):
            await build_lesson_planner_graph().ainvoke(create_initial_state(query="q"))

    assert planner.plan.await_count == LLM_RETRY_POLICY.max_attempts


async def test_pipeline_run_reports_exhaustion_as_a_failed_result(monkeypatch):
    """The caller contract does not change: ``run()`` returns, it does not raise.

    Letting transients escape the node changes what reaches the graph's caller,
    so this pins the two entry points that have to absorb it. ``run()`` keeps
    returning a result dict; the WebUI's streaming path has its own handler that
    marks the lesson errored and emits an ``error`` event.
    """
    from aix.agent.graph.lesson_planner_graph import LessonPlannerPipeline

    monkeypatch.delenv("AIX_CORRECTIVE_RAG_ENABLED", raising=False)
    monkeypatch.setattr(
        "aix.agent.graph.checkpointer.get_checkpointer", AsyncMock(return_value=None)
    )

    planner = AsyncMock(plan=AsyncMock(side_effect=_rate_limit()))
    with patch.object(graph_nodes, "get_planner", return_value=planner):
        result = await LessonPlannerPipeline(domain="neuro").run("q")

    assert result["success"] is False
    assert result["lesson_plan"] is None
    assert "slow down" in result["error"]
