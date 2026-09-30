"""Attachments keep the order they were added in, and the turn they arrived with.

On a later turn a teacher says «use the new file» or «only the second
document». The Writer can resolve that only if it is told the order.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from aix.webui.auth.models import User
from aix.webui.db import Base
from aix.webui.lessons.models import Lesson, LessonMessage
from aix.webui.materials.models import AssetBlob, LessonMaterialBinding, Material, MaterialVersion
from aix.webui.materials.retrieval import attachments_for_lesson

pytestmark = pytest.mark.unit

ANNA = uuid.UUID("00000000-0000-0000-0000-00000000a111")
LESSON = uuid.UUID("00000000-0000-0000-0000-00000000c111")
T0 = datetime(2026, 9, 30, 10, 0, tzinfo=timezone.utc)


@pytest.fixture
async def session(tmp_path):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{(tmp_path / 'order.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        session.add(
            User(
                id=ANNA,
                email="anna@scuola.it",
                hashed_password="x",
                is_active=True,
                is_superuser=False,
                is_verified=True,
            )
        )
        session.add(Lesson(id=LESSON, owner_id=ANNA, domain="neuro", status="complete"))
        await session.flush()
        yield session
    await engine.dispose()


async def _attach(session, name: str, minutes: int, *, enabled: bool = True):
    blob = AssetBlob(
        owner_id=ANNA,
        storage_provider="local",
        object_key=f"k-{name}",
        sha256=uuid.uuid4().hex * 2,
        size_bytes=1,
        verified_media_type="application/pdf",
    )
    material = Material(owner_id=ANNA, display_name=name)
    session.add_all([blob, material])
    await session.flush()
    version = MaterialVersion(
        material_id=material.id,
        version_number=1,
        blob_id=blob.id,
        source_sha256=blob.sha256,
        media_type="application/pdf",
        created_by=ANNA,
        status="ready",
    )
    session.add(version)
    await session.flush()
    session.add(
        LessonMaterialBinding(
            lesson_id=LESSON,
            material_id=material.id,
            owner_id=ANNA,
            selected_version_id=version.id,
            created_by=ANNA,
            enabled=enabled,
            created_at=T0 + timedelta(minutes=minutes),
        )
    )
    await session.flush()
    return str(version.id).lower()


async def _ask(session, turn: int, minutes: int):
    session.add(
        LessonMessage(
            lesson_id=LESSON,
            role="user",
            content_md=f"turno {turn}",
            turn_index=turn,
            created_at=T0 + timedelta(minutes=minutes),
        )
    )
    await session.flush()


async def test_attachments_come_in_the_order_they_were_added_with_their_turn(session):
    slides = await _attach(session, "slide.pptx", 0)
    carta = await _attach(session, "mappa.webp", 1)
    await _ask(session, 1, 2)
    await _ask(session, 2, 10)
    libro = await _attach(session, "libro.pdf", 20)  # added while writing turn 3
    await _ask(session, 3, 21)

    attachments = await attachments_for_lesson(session, LESSON, ANNA)

    assert [(a["source_id"], a["position"], a["turn"], a["new"]) for a in attachments] == [
        (slides, 1, 1, False),
        (carta, 2, 1, False),
        (libro, 3, 3, True),
    ]
    assert attachments[2]["source_label"] == "libro.pdf"


async def test_a_removed_attachment_leaves_the_list(session):
    await _attach(session, "vecchio.pdf", 0, enabled=False)
    kept = await _attach(session, "nuovo.pdf", 1)
    await _ask(session, 1, 2)

    attachments = await attachments_for_lesson(session, LESSON, ANNA)

    assert [a["source_id"] for a in attachments] == [kept]
    assert attachments[0]["position"] == 1


async def test_on_the_first_turn_nothing_is_called_new(session):
    await _attach(session, "slide.pptx", 0)
    await _ask(session, 1, 1)

    assert (await attachments_for_lesson(session, LESSON, ANNA))[0]["new"] is False


def test_the_writer_reads_the_numbered_attachments_and_which_one_is_new():
    from aix.agent.agents.writer_agent import teacher_material_appendix

    overview = [
        {"source_id": "a", "source_label": "slide.pptx", "position": 1, "turn": 1, "new": False},
        {
            "source_id": "b",
            "source_label": "libro.pdf",
            "position": 2,
            "turn": 3,
            "new": True,
            "summary": "Un libro sulla Guerra Fredda.",
        },
    ]
    passages = [
        {
            "source_id": "a",
            "source_label": "slide.pptx",
            "attachment": 1,
            "score": 0.6,
            "text": "Yalta, 1945.",
            "citation": "[[material:a#slide=1]]",
        },
        {
            "source_id": "b",
            "source_label": "libro.pdf",
            "attachment": 2,
            "score": 0.5,
            "text": "Il ponte aereo.",
            "citation": "[[material:b#page=4]]",
        },
    ]

    appendix = teacher_material_appendix(passages, None, phase="write", material_maps=overview)

    assert "**Allegato 1 · slide.pptx** — aggiunto al turno 1" in appendix
    assert "**Allegato 2 · libro.pdf** — aggiunto al turno 3, in questo turno" in appendix
    assert "### Allegato 2 · libro.pdf — aggiunto al turno" not in appendix
    assert "### Allegato 2 · libro.pdf" in appendix
    assert "«nuovo» allegato" in appendix
