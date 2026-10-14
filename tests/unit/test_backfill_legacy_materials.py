from __future__ import annotations

import uuid

import pytest
from scripts.ops.backfill_legacy_materials import backfill
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from aix.webui.auth.models import User
from aix.webui.db import Base
from aix.webui.lessons.models import Lesson
from aix.webui.materials.models import IngestionJob, Material

pytestmark = pytest.mark.unit

OWNER = uuid.UUID("11111111-1111-1111-1111-111111111111")
LESSON = uuid.UUID("22222222-2222-2222-2222-222222222222")


async def test_backfill_is_dry_run_by_default_atomic_and_idempotent(tmp_path, monkeypatch):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{(tmp_path / 'backfill.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    upload_root = tmp_path / "uploads"
    lesson_dir = upload_root / str(LESSON)
    lesson_dir.mkdir(parents=True)
    (lesson_dir / "legacy.txt").write_bytes(b"Contesto didattico precedente")

    async with maker() as session:
        session.add(
            User(
                id=OWNER,
                email="docente@scuola.it",
                hashed_password="x",
                is_active=True,
                is_superuser=False,
                is_verified=True,
            )
        )
        session.add(
            Lesson(
                id=LESSON,
                owner_id=OWNER,
                domain="neuro",
                status="complete",
                educational_profile_json={},
                uploaded_files_json=[
                    {
                        "id": "legacy-1",
                        "filename": "legacy.txt",
                        "stored_name": "legacy.txt",
                        "mime": "text/plain",
                    }
                ],
            )
        )
        await session.commit()

    monkeypatch.setattr("aix.webui.db.async_session_maker", maker)
    monkeypatch.setattr("aix.webui.lessons.uploads.UPLOAD_ROOT", upload_root)
    monkeypatch.setenv("AIX_STORAGE_PROVIDER", "local")
    monkeypatch.setenv("AIX_STORAGE_LOCAL_ROOT", str(tmp_path / "objects"))

    assert await backfill(apply=False) == (1, 0)
    async with maker() as session:
        lesson = await session.get(Lesson, LESSON)
        assert "material_id" not in lesson.uploaded_files_json[0]
        assert await session.scalar(select(func.count(Material.id))) == 0

    assert await backfill(apply=True) == (1, 0)
    assert await backfill(apply=True) == (0, 0)
    async with maker() as session:
        lesson = await session.get(Lesson, LESSON)
        assert lesson.uploaded_files_json[0]["material_id"]
        assert await session.scalar(select(func.count(Material.id))) == 1
        assert await session.scalar(select(func.count(IngestionJob.id))) == 1

    await engine.dispose()
