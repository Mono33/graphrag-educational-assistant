"""Accepting a turn, as both the WebUI and the API do it (2026-10-14).

The WebUI's behaviour is pinned by its own tests (``test_light_turns.py``,
``test_refinement_one_turn.py``, ``test_intent_and_planner_card.py``…), which
did not change when the logic moved here. These pin what the API relies on: a
refusal carries a stable code, and a first turn can skip the intent card.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select
from tests.unit.test_light_turns import ANNA, LESSON, _env, maker  # noqa: F401 — fixtures

from aix.core import concurrency
from aix.webui.agent.turns import TurnAccepted, TurnRefused, accept_turn
from aix.webui.lessons.models import Lesson, LessonMessage

pytestmark = pytest.mark.unit


async def _accept(maker, form, *, lesson_id=LESSON, ask_intent=True, status=None):
    async with maker() as session:
        lesson = await session.get(Lesson, lesson_id)
        if status is not None:
            lesson.status = status
        outcome = await accept_turn(
            session, lesson, form, rate_key=f"test:{uuid.uuid4()}", ask_intent=ask_intent
        )
        rows = (
            (await session.execute(select(LessonMessage).order_by(LessonMessage.turn_index)))
            .scalars()
            .all()
        )
        return outcome, lesson, rows


async def test_a_question_on_a_finished_lesson_is_a_light_turn(maker):
    outcome, lesson, rows = await _accept(maker, {"query": "sì"})

    assert isinstance(outcome, TurnAccepted)
    assert (outcome.mode, outcome.turn_index) == ("light", 2)
    assert lesson.status == "running"
    assert rows[-1].content_md == "sì"


@pytest.mark.parametrize(
    ("status", "code", "http"),
    [("running", "already_running", 409)],
)
async def test_a_refusal_carries_a_code_a_machine_can_read(maker, status, code, http):
    outcome, _, _ = await _accept(maker, {"query": "ancora"}, status=status)

    assert isinstance(outcome, TurnRefused)
    assert (outcome.code, outcome.status_code) == (code, http)
    assert outcome.message


async def test_an_empty_request_on_a_lesson_without_a_topic_is_refused(maker):
    async with maker() as session:
        session.add(Lesson(id=uuid.uuid4(), owner_id=ANNA, domain="neuro", status="draft"))
        await session.commit()
        new = (await session.execute(select(Lesson).where(Lesson.status == "draft"))).scalar_one()

    outcome, _, _ = await _accept(maker, {}, lesson_id=new.id)

    assert isinstance(outcome, TurnRefused)
    assert outcome.code == "empty_query"


async def test_the_rate_limit_says_when_to_retry(maker, monkeypatch):
    def _limited(_key):
        raise concurrency.RateLimited(4.2, limit=5, window_s=60)

    monkeypatch.setattr("aix.webui.agent.turns.check_user_rate_limit", _limited)

    outcome, _, _ = await _accept(maker, {"query": "ancora"})

    assert (outcome.code, outcome.status_code, outcome.retry_after) == ("rate_limited", 429, 5)


async def test_without_the_intent_card_a_first_turn_writes_its_message_and_runs(maker):
    async with maker() as session:
        lesson = Lesson(id=uuid.uuid4(), owner_id=ANNA, domain="udl", status="draft")
        session.add(lesson)
        await session.commit()

    outcome, lesson, rows = await _accept(
        maker, {"query": "Crea una lezione sulla cellula"}, lesson_id=lesson.id, ask_intent=False
    )

    assert isinstance(outcome, TurnAccepted)
    assert (outcome.mode, outcome.ask_intent) == ("new", False)
    assert "__ask_intent__" not in (lesson.educational_profile_json or {})
    assert [r.content_md for r in rows if r.lesson_id == lesson.id] == [
        "Crea una lezione sulla cellula"
    ]


async def test_with_the_intent_card_a_first_turn_asks_first(maker):
    async with maker() as session:
        lesson = Lesson(id=uuid.uuid4(), owner_id=ANNA, domain="udl", status="draft")
        session.add(lesson)
        await session.commit()

    outcome, lesson, _ = await _accept(maker, {"query": "Crea"}, lesson_id=lesson.id)

    assert outcome.ask_intent is True
    assert lesson.educational_profile_json["__ask_intent__"] is True
