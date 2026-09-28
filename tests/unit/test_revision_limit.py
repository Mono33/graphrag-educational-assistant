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
