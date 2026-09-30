"""How many times the Critic may send a draft back: one source, read when needed.

``AIX_MAX_REVISIONS`` was documented with a default of 1 and never counted:
the Orchestrator hard-coded ``max_revisions=2`` in its signature and passed it
down, and the pipeline's own default was evaluated once, at import. A real run
on 2026-09-25 spent 63 s on a second draft for four points of the Critic's
score and was still not approved; one revision is the default now.
"""

from __future__ import annotations

import pytest

from aix.agent.graph.lesson_planner_graph import LessonPlannerPipeline, default_max_revisions
from aix.agent.orchestrator import AgentOrchestrator

pytestmark = pytest.mark.unit


def test_one_revision_by_default(monkeypatch):
    monkeypatch.delenv("AIX_MAX_REVISIONS", raising=False)

    assert default_max_revisions() == 1
    assert AgentOrchestrator().max_revisions == 1
    assert AgentOrchestrator().get_pipeline().max_revisions == 1


def test_the_variable_is_read_when_the_pipeline_is_built_not_at_import(monkeypatch):
    monkeypatch.setenv("AIX_MAX_REVISIONS", "3")

    assert LessonPlannerPipeline().max_revisions == 3
    assert AgentOrchestrator().max_revisions == 3, "the Orchestrator no longer overrides it"


def test_an_explicit_value_still_wins(monkeypatch):
    monkeypatch.setenv("AIX_MAX_REVISIONS", "3")

    assert AgentOrchestrator(max_revisions=0).max_revisions == 0


@pytest.mark.parametrize("raw", ["", "due", "-4"])
def test_a_bad_value_falls_back_to_something_sane(monkeypatch, raw):
    monkeypatch.setenv("AIX_MAX_REVISIONS", raw)

    assert default_max_revisions() in (0, 1)


# ---------------------------------------------------------------------------
# At the limit: the last draft is reviewed, and the cards say what happens
# ---------------------------------------------------------------------------


async def test_the_draft_written_at_the_limit_is_still_reviewed():
    """It used to be approved unread, with an invented 4.0 (2026-09-30)."""
    import json
    from types import SimpleNamespace
    from unittest.mock import patch

    from aix.agent.agents.critic_agent import CriticAgent
    from tests.unit.agent_doubles import _retrieval_result

    calls = []
    verdict = {"scores": {"structure": 3}, "average_score": 3.0, "decision": "REVISE",
               "strengths": [], "weaknesses": ["x"], "revision_instructions": "y", "summary": "z"}

    async def _fake(_client, *, messages, **_kw):
        calls.append(messages)
        content = json.dumps(verdict)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content),
                                                        finish_reason="stop")])

    retrieval = _retrieval_result()
    retrieval.to_context_string = lambda *args, **kwargs: "Contesto."
    with (
        patch.object(CriticAgent, "_get_client", return_value=object()),
        patch("aix.agent.agents.critic_agent.guarded_chat_completion", _fake),
    ):
        result = await CriticAgent().critique(
            lesson_plan="# Bozza rivista",
            teacher_query="Crea una lezione",
            retrieval_result=retrieval,
            revision_count=1,
            max_revisions=1,
        )

    assert calls, "the model reads the revised draft"
    assert result.approved is False, "and its verdict is the real one"


@pytest.mark.parametrize(
    "revision_count, limit_shown",
    [(1, False), (2, True)],  # 1: the one permitted revision follows; 2: none left
)
def test_the_critic_card_says_the_limit_only_when_no_revision_follows(revision_count, limit_shown):
    from jinja2 import Environment, FileSystemLoader

    html = (
        Environment(loader=FileSystemLoader("src/aix/webui/templates"))
        .get_template("partials/chat_critic_card.html")
        .render(payload={"approved": False, "revision_count": revision_count, "max_revisions": 1,
                         "score_pct": 80, "revision_instructions": "Usa i documenti."})
    )

    assert ("Limite revisioni raggiunto" in html) is limit_shown
