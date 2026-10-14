from __future__ import annotations

import uuid

import pytest
from scripts.ops.audit_material_storage import audit
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from aix.webui.auth.models import User
from aix.webui.db import Base
from aix.webui.lessons.models import Lesson
from aix.webui.materials.repository import MaterialService
from aix.webui.materials.storage import LocalDurableStorage

pytestmark = pytest.mark.unit

OWNER = uuid.UUID("11111111-1111-1111-1111-111111111111")
LESSON = uuid.UUID("22222222-2222-2222-2222-222222222222")


async def test_audit_detects_missing_legacy_and_durable_bytes(tmp_path):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{(tmp_path / 'audit.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    upload_root = tmp_path / "uploads"
    storage = LocalDurableStorage(tmp_path / "objects")

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
        lesson = Lesson(
            id=LESSON,
            owner_id=OWNER,
            domain="neuro",
            status="draft",
            educational_profile_json={},
        )
        session.add(lesson)
        await session.flush()
        material, _version = await MaterialService(session, storage).store_new_material(
            b"materiale",
            owner_id=OWNER,
            display_name="materiale.txt",
            media_type="text/plain",
        )
        legacy_dir = upload_root / str(LESSON)
        legacy_dir.mkdir(parents=True)
        legacy_file = legacy_dir / "file-1.txt"
        legacy_file.write_bytes(b"legacy")
        lesson.uploaded_files_json = [
            {
                "id": "file-1",
                "stored_name": legacy_file.name,
                "material_id": str(material.id),
            }
        ]
        await session.commit()

    async with maker() as session:
        healthy = await audit(session, upload_root=upload_root, storage=storage)
        assert healthy.healthy
        assert healthy.legacy_files_checked == 1
        assert healthy.durable_blobs_checked == 1

    legacy_file.unlink()
    for path in (tmp_path / "objects").rglob("*"):
        if path.is_file():
            path.unlink()

    async with maker() as session:
        broken = await audit(session, upload_root=upload_root, storage=storage)
        assert not broken.healthy
        assert len(broken.missing_legacy_files) == 1
        assert len(broken.missing_durable_blobs) == 1

    await engine.dispose()
