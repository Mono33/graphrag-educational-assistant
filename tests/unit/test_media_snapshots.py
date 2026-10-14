from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from aix.api.main import app
from aix.webui.agent.service import _persist_assistant_turn
from aix.webui.auth.dependencies import optional_current_user
from aix.webui.auth.models import User
from aix.webui.db import Base, get_async_session
from aix.webui.lessons.media_snapshots import (
    latest_display_snapshot,
    sanitize_curated_media,
    save_curated_snapshot,
    save_live_snapshot,
)
from aix.webui.lessons.models import Lesson, LessonMessage

pytestmark = pytest.mark.unit

ANNA = uuid.UUID("11111111-1111-1111-1111-111111111111")
BRUNO = uuid.UUID("22222222-2222-2222-2222-222222222222")
LESSON = uuid.UUID("33333333-3333-3333-3333-333333333333")


@pytest.fixture
async def database(tmp_path, monkeypatch):
    monkeypatch.setenv("WEBUI_CSRF_ENABLED", "false")
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{(tmp_path / 'media.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        session.add_all(
            [
                User(
                    id=ANNA,
                    email="anna@scuola.it",
                    hashed_password="x",
                    is_active=True,
                    is_superuser=False,
                    is_verified=True,
                ),
                User(
                    id=BRUNO,
                    email="bruno@scuola.it",
                    hashed_password="x",
                    is_active=True,
                    is_superuser=False,
                    is_verified=True,
                ),
                Lesson(
                    id=LESSON,
                    owner_id=ANNA,
                    domain="neuro",
                    status="complete",
                    title="Memoria",
                    educational_profile_json={},
                    lesson_plan_md="# Piano",
                    teacher_query="memoria di lavoro",
                ),
                LessonMessage(
                    lesson_id=LESSON,
                    role="user",
                    content_md="memoria di lavoro",
                    turn_index=1,
                ),
            ]
        )
        await session.commit()

    try:
        yield maker
    finally:
        await engine.dispose()


def test_snapshot_keeps_only_rendered_bounded_fields():
    media = {
        "videos": [
            {"title": f"Video {index}", "url": f"https://example.test/{index}", "secret": "x"}
            for index in range(10)
        ],
        "unknown_bucket": [{"title": "not rendered"}],
    }

    snapshot = sanitize_curated_media(media)

    assert len(snapshot["videos"]) == 6
    assert "secret" not in snapshot["videos"][0]
    assert "unknown_bucket" not in snapshot


async def test_live_and_curated_writes_merge_without_overwriting(database):
    async with database() as session:
        await save_live_snapshot(
            session,
            lesson_id=LESSON,
            owner_id=ANNA,
            turn_index=1,
            media={"wikipedia": [{"title": "Memoria", "url": "https://it.wikipedia.org/"}]},
        )
        await session.commit()

    async with database() as session:
        await save_curated_snapshot(
            session,
            lesson_id=LESSON,
            owner_id=ANNA,
            turn_index=1,
            media={"videos": [{"title": "Lezione", "url": "https://video.test/1"}]},
        )
        await session.commit()

    async with database() as session:
        snapshot = await latest_display_snapshot(
            session,
            lesson_id=LESSON,
            owner_id=ANNA,
        )
        assert snapshot is not None
        assert snapshot.curated_json["videos"][0]["title"] == "Lezione"
        assert snapshot.live_json["wikipedia"][0]["title"] == "Memoria"
        assert snapshot.live_fetched is True


async def test_snapshot_lookup_is_owner_scoped(database):
    async with database() as session:
        await save_curated_snapshot(
            session,
            lesson_id=LESSON,
            owner_id=ANNA,
            turn_index=1,
            media={},
        )
        await session.commit()

    async with database() as session:
        assert (
            await latest_display_snapshot(
                session,
                lesson_id=LESSON,
                owner_id=BRUNO,
            )
            is None
        )


async def test_missing_snapshot_table_never_rolls_back_the_assistant_message(database):
    async with database() as session:
        await session.execute(text("DROP TABLE lesson_media_snapshot"))
        await session.commit()
        await _persist_assistant_turn(
            session,
            LESSON,
            ANNA,
            "Risposta conservata",
            {},
            {"videos": [{"title": "Non persistibile"}]},
        )

    async with database() as session:
        reply = await session.scalar(
            select(LessonMessage).where(
                LessonMessage.lesson_id == LESSON,
                LessonMessage.role == "assistant",
            )
        )
        assert reply is not None
        assert reply.content_md == "Risposta conservata"


@pytest.fixture
async def client(database):
    async with database() as session:
        await save_curated_snapshot(
            session,
            lesson_id=LESSON,
            owner_id=ANNA,
            turn_index=1,
            media={"videos": [{"title": "Video persistito", "url": "https://video.test/1"}]},
        )
        await save_live_snapshot(
            session,
            lesson_id=LESSON,
            owner_id=ANNA,
            turn_index=1,
            media={
                "wikipedia": [
                    {"title": "Voce persistita", "url": "https://it.wikipedia.org/wiki/Memoria"}
                ]
            },
        )
        await session.commit()

    async def session_override():
        async with database() as session:
            yield session

    app.dependency_overrides[get_async_session] = session_override
    app.dependency_overrides[optional_current_user] = lambda: User(
        id=ANNA,
        email="anna@scuola.it",
        hashed_password="x",
        is_active=True,
        is_superuser=False,
        is_verified=True,
    )
    try:
        yield TestClient(app, follow_redirects=False)
    finally:
        app.dependency_overrides.clear()


def test_reload_replays_curated_and_live_without_a_new_search(client):
    page = client.get(f"/webui/lesson/{LESSON}")
    assert page.status_code == 200
    assert "Video persistito" in page.text
    assert f"/webui/lesson/{LESSON}/media-live" in page.text

    live = client.get(f"/webui/lesson/{LESSON}/media-live")
    assert live.status_code == 200
    assert "Voce persistita" in live.text
