"""What the teacher is asked, and what the planner card shows.

Both pinned here after the second real test on 2026-09-16:

* the teacher had chosen «Correggere un'idea errata» in the profile and the
  chat asked for the intent anyway, stopping the generation and spending a
  whole Planner call (78 s) before the run it was going to do regardless;
* the planner card listed only the pedagogy concepts that go to the Knowledge
  Graph, so a teacher who had attached a Cold War handout, and whose profile
  said «guerra fredda», saw «UDL, engagement, scaffolding» and concluded the
  agent had ignored the subject. The Planner had not: its subject concepts
  were Yalta, the Marshall Plan and the iron curtain, and nothing showed them.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from aix.webui.auth.dependencies import optional_current_user
from aix.webui.auth.models import User
from aix.webui.db import Base, get_async_session
from aix.webui.lessons.models import Lesson

pytestmark = pytest.mark.unit

ANNA = uuid.UUID("11111111-1111-1111-1111-111111111111")
LESSON = uuid.UUID("44444444-4444-4444-4444-444444444444")

PROFILE = {
    "subject_area": "Storia",
    "specific_topic": "guerra fredda",
    "classroom": {"has_lim": True, "has_wifi": True, "own_device": "YES"},
    "time_available_minutes": 60,
}


def _user() -> User:
    return User(
        id=ANNA,
        email="anna@scuola.it",
        hashed_password="x",
        is_active=True,
        is_superuser=False,
        is_verified=True,
    )


@pytest.fixture
async def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("WEBUI_CSRF_ENABLED", "false")

    from aix.api.main import app

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{(tmp_path / 'intent.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    maker = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with maker() as session:
        session.add(_user())
        await session.commit()

    async def _session():
        async with maker() as session:
            yield session

    app.dependency_overrides[get_async_session] = _session
    app.dependency_overrides[optional_current_user] = _user
    try:
        yield app, maker
    finally:
        app.dependency_overrides.clear()
        await engine.dispose()


@pytest.fixture
def client(_app):
    app, _maker = _app
    return TestClient(app, follow_redirects=False)


@pytest.fixture
def maker(_app):
    _app_obj, session_maker = _app
    return session_maker


async def _draft(maker, profile: dict) -> None:
    async with maker() as session:
        session.add(
            Lesson(
                id=LESSON,
                owner_id=ANNA,
                domain="udl",
                status="draft",
                educational_profile_json=profile,
            )
        )
        await session.commit()


async def _profile_after_run(maker, client) -> dict:
    response = client.post(
        f"/webui/lesson/{LESSON}/run",
        data={"query": "crea lezione basandoti sul documento allegato"},
    )
    assert response.status_code == 200
    async with maker() as session:
        lesson = await session.get(Lesson, LESSON)
        assert lesson.status == "running"
        return lesson.educational_profile_json or {}


# ---------------------------------------------------------------------------
# The intent card
# ---------------------------------------------------------------------------


async def test_an_intent_already_in_the_profile_is_not_asked_again(maker, client):
    await _draft(maker, {**PROFILE, "pedagogical_intent": "misconception"})

    profile = await _profile_after_run(maker, client)

    assert "__planner_only__" not in profile, "the run goes straight through"
    assert profile["pedagogical_intent"] == "misconception", "and the choice is kept"


@pytest.mark.parametrize("intent", [None, "", "   "], ids=["absent", "empty", "blank"])
async def test_without_an_intent_the_card_is_still_shown(maker, client, intent):
    profile = dict(PROFILE)
    if intent is not None:
        profile["pedagogical_intent"] = intent
    await _draft(maker, profile)

    assert (await _profile_after_run(maker, client))["__planner_only__"] is True


# ---------------------------------------------------------------------------
# The planner card
# ---------------------------------------------------------------------------


def _planner_card(payload: dict) -> str:
    from aix.webui.lessons.routes import templates

    return templates.get_template("partials/chat_planner_card.html").render(payload=payload)


def test_the_card_shows_the_subject_apart_from_the_teaching_strategies():
    from aix.webui.agent.service import _build_planner_payload

    payload = _build_planner_payload(
        {
            "query_intent": "lesson_creation",
            "scope_status": "partial_scope",
            "key_concepts": ["misconception correction"],
            "subject_concepts": ["Guerra Fredda", "Conferenza di Yalta 1945"],
            "search_queries": ["misconception correction strategies"],
        }
    )
    html = _planner_card(payload)

    assert payload["subject_concepts"] == ["Guerra Fredda", "Conferenza di Yalta 1945"]
    assert "Argomento della lezione" in html
    assert "Guerra Fredda" in html
    assert "Strategie didattiche cercate nel Knowledge Graph" in html
    assert "misconception correction" in html


def test_a_lesson_with_no_subject_concepts_renders_as_before():
    html = _planner_card({"key_concepts": ["scaffolding"], "subject_concepts": []})

    assert "Argomento della lezione" not in html
    assert "scaffolding" in html
