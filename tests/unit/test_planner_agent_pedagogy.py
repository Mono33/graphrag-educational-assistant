"""The graph is searched for teaching, never for the subject (2026-10-20)."""

from __future__ import annotations

import pytest

from aix.agent.agents.planner_agent import keep_pedagogy

pytestmark = pytest.mark.unit

SUBJECT = ["Cold War", "Yalta Conference 1945", "Iron Curtain", "nuclear deterrence and arms race"]


def test_the_queries_about_the_subject_go_and_the_ones_about_teaching_stay():
    queries = [
        "Yalta Conference and division of Europe",
        "Iron Curtain and spheres of influence",
        "nuclear deterrence and arms race",
        "propaganda and communication during the Cold War",
        "visual aids and interactive whiteboard teaching",
        "strategies to stimulate situational interest in students",
        "strategies to teach the Cold War through debate",
        "Disabilità cognitiva media",
    ]

    assert keep_pedagogy(queries, SUBJECT) == [
        "visual aids and interactive whiteboard teaching",
        "strategies to stimulate situational interest in students",
        "strategies to teach the Cold War through debate",
        "Disabilità cognitiva media",
    ]


def test_without_a_subject_nothing_is_dropped():
    assert keep_pedagogy(["working memory", "ADHD"], None) == ["working memory", "ADHD"]
