"""The final lesson card says what the Critic actually decided.

On 2026-09-25 a lesson the Critic sent back twice and never approved -- the
log said ``approved=False revisions=2`` -- was shown to the teacher as
"approvata dal critic · 0 revisioni". The card endpoint was handed a meta
written by hand, ``{"approved": status == "complete", "revision_count": 0}``,
so every finished lesson carried an approval nobody had given.

The verdict is recorded with the answer, in the assistant message; the card
reads it from there. Where there is no record, there is no badge.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from aix.api.main import app
from aix.webui.auth.dependencies import optional_current_user
from aix.webui.auth.models import User
from aix.webui.db import Base, get_async_session
from aix.webui.lessons.models import Lesson, LessonMessage

pytestmark = pytest.mark.unit

ANNA = uuid.UUID("11111111-1111-1111-1111-111111111111")
REJECTED = uuid.UUID("33333333-3333-3333-3333-333333333333")
APPROVED = uuid.UUID("55555555-5555-5555-5555-555555555555")
UNRECORDED = uuid.UUID("66666666-6666-6666-6666-666666666666")


@pytest.fixture
def client(tmp_path, monkeypatch):
    import asyncio

    monkeypatch.setenv("WEBUI_CSRF_ENABLED", "false")
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{(tmp_path / 'card.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    maker = async_sessionmaker(engine, expire_on_commit=False)

    async def _seed():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with maker() as s:
            s.add(User(id=ANNA, email="anna@scuola.it", hashed_password="x",
                       is_active=True, is_superuser=False, is_verified=True))
            for lesson_id in (REJECTED, APPROVED, UNRECORDED):
                s.add(Lesson(id=lesson_id, owner_id=ANNA, domain="neuro", status="complete",
                             title="Crisi migratoria", lesson_plan_md="# Lezione\n\nTesto."))
            for lesson_id, meta in (
                (REJECTED, {"approved": False, "revision_count": 2, "duration_seconds": 170.3}),
                (APPROVED, {"approved": True, "revision_count": 1, "duration_seconds": 95.0}),
            ):
                s.add(LessonMessage(lesson_id=lesson_id, role="user", content_md="crea",
                                    turn_index=1))
                s.add(LessonMessage(lesson_id=lesson_id, role="assistant", content_md="# Lezione",
                                    turn_index=1, agent_kind="writer", meta_json=meta))
            await s.commit()

    asyncio.run(_seed())

    async def _session():
        async with maker() as session:
            yield session

    app.dependency_overrides[get_async_session] = _session
    app.dependency_overrides[optional_current_user] = lambda: User(
        id=ANNA, email="anna@scuola.it", hashed_password="x",
        is_active=True, is_superuser=False, is_verified=True,
    )
    try:
        yield TestClient(app, follow_redirects=False)
    finally:
        app.dependency_overrides.clear()
        asyncio.run(engine.dispose())


def _card(client, lesson_id) -> str:
    response = client.get(f"/webui/lesson/{lesson_id}/card-fragment")
    assert response.status_code == 200
    return response.text


def test_a_lesson_the_critic_never_approved_does_not_say_approved(client):
    html = _card(client, REJECTED)

    assert "approvata dal critic" not in html
    assert "limite revisioni raggiunto" in html
    assert "<strong>2</strong> revisioni" in html


def test_an_approved_lesson_still_says_so(client):
    """The fix must not simply remove the badge: an approval that happened shows."""
    html = _card(client, APPROVED)

    assert "approvata dal critic" in html
    assert "<strong>1</strong> revisioni" in html


def test_without_a_recorded_verdict_there_is_no_badge_at_all(client):
    html = _card(client, UNRECORDED)

    assert "approvata dal critic" not in html
    assert "limite revisioni raggiunto" not in html
