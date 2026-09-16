"""The two things that only PostgreSQL can answer (Fase 0 gate, both dialects).

Everything else in this suite runs on SQLite, which is the developer database.
Production is PostgreSQL, and two pieces of this system are written *for* it:

* the migrations — ``material_chunk`` creates the ``vector`` extension, converts
  a column to ``vector(1024)`` and adds a generated ``tsvector``, none of which
  SQLite ever executes. A migration that fails there fails on the deploy;
* the job claim — the dispatcher uses ``FOR UPDATE SKIP LOCKED`` on PostgreSQL
  and a plain conditional update on SQLite. The path that runs in production is
  therefore the one no test was covering (Louis, 2026-09-15).

Skipped unless ``AIX_TEST_POSTGRES_URL`` is set, so the ordinary run is
unchanged. To run them::

    docker run -d --name aix-pg-test -e POSTGRES_PASSWORD=test \\
        -e POSTGRES_DB=aixtest -p 55432:5432 pgvector/pgvector:pg16
    AIX_TEST_POSTGRES_URL=postgresql+psycopg://postgres:test@localhost:55432/aixtest \\
        pytest tests/integration/test_postgres_dialect.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid

import pytest

# Windows only, and a property of the driver rather than of this code: psycopg's
# async mode refuses the default Proactor loop. Set at import, before
# pytest-asyncio builds a loop for the first test. On Linux — CI and the VM —
# this is a no-op.
if sys.platform == "win32":  # pragma: no cover - developer machines only
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

pytestmark = pytest.mark.integration

POSTGRES_URL = os.getenv("AIX_TEST_POSTGRES_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not POSTGRES_URL, reason="AIX_TEST_POSTGRES_URL is not set"),
]

ANNA = uuid.UUID("11111111-1111-1111-1111-111111111111")


@pytest.fixture
async def engine():
    """A clean schema per test: everything dropped and rebuilt."""
    engine = create_async_engine(POSTGRES_URL)
    async with engine.begin() as conn:
        await conn.execute(text("DROP SCHEMA public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))
    try:
        yield engine
    finally:
        await engine.dispose()


async def _migrate(engine) -> None:
    from aix.webui.db import _migrate_or_adopt

    async with engine.begin() as conn:
        await conn.run_sync(_migrate_or_adopt)


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------


async def test_every_migration_applies_on_postgres(engine):
    await _migrate(engine)

    async with engine.begin() as conn:
        names = await conn.run_sync(lambda c: set(inspect(c).get_table_names()))

    assert {"user", "lesson", "material", "material_version", "ingestion_job"} <= names
    assert {"document_ir", "material_chunk"} <= names


async def test_the_chunk_column_is_a_real_vector_with_its_index(engine):
    """The parts SQLite never runs: pgvector's type, the Italian FTS, the index."""
    await _migrate(engine)

    async with engine.begin() as conn:
        column_type = await conn.scalar(
            text(
                "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                "WHERE attrelid = 'material_chunk'::regclass AND attname = 'embedding'"
            )
        )
        fts_type = await conn.scalar(
            text(
                "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                "WHERE attrelid = 'material_chunk'::regclass AND attname = 'fts'"
            )
        )
        indexes = await conn.scalars(
            text("SELECT indexname FROM pg_indexes WHERE tablename = 'material_chunk'")
        )

    assert column_type == "vector(1024)"
    assert fts_type == "tsvector"
    assert "ix_material_chunk_fts" in set(indexes)


async def test_a_vector_can_be_written_and_read_back(engine):
    """Proof the ORM type and the column agree — the thing a dialect split hides."""
    from aix.webui.auth.models import User
    from aix.webui.materials.models import AssetBlob, Material, MaterialChunk, MaterialVersion

    await _migrate(engine)
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
        blob = AssetBlob(
            owner_id=ANNA,
            storage_provider="local",
            object_key="k",
            sha256="0" * 64,
            size_bytes=1,
            verified_media_type="text/plain",
        )
        material = Material(owner_id=ANNA, display_name="Programma")
        session.add_all([blob, material])
        await session.flush()

        version = MaterialVersion(
            material_id=material.id,
            version_number=1,
            blob_id=blob.id,
            source_sha256="0" * 64,
            media_type="text/plain",
            created_by=ANNA,
        )
        session.add(version)
        await session.flush()

        session.add(
            MaterialChunk(
                owner_id=ANNA,
                material_version_id=version.id,
                ordinal=0,
                text="La Guerra Fredda oppose USA e URSS.",
                text_hash="1" * 64,
                locator_json={"page": 1, "block_ids": ["b1"]},
                embedding=[0.25] * 1024,
                embedding_model="baai/bge-m3",
                chunker_version="chunk@1",
            )
        )
        await session.commit()

    async with maker() as session:
        from sqlalchemy import select

        chunk = (await session.execute(select(MaterialChunk))).scalar_one()
        assert len(chunk.embedding) == 1024
        assert chunk.embedding[0] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# The job claim, on the path production actually takes
# ---------------------------------------------------------------------------


async def test_two_workers_never_claim_the_same_job_on_postgres(engine):
    """``FOR UPDATE SKIP LOCKED`` is PostgreSQL-only, so this is its only test."""
    from aix.webui.auth.models import User
    from aix.webui.materials.jobs import JobDispatcher
    from aix.webui.materials.repository import MaterialService
    from aix.webui.materials.storage import LocalDurableStorage

    await _migrate(engine)
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
        await session.commit()

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        async with maker() as session:
            service = MaterialService(session, LocalDurableStorage(Path(tmp)))
            _material, version = await service.store_new_material(
                b"Programma di classe",
                owner_id=ANNA,
                display_name="Programma",
                media_type="text/plain",
            )
            await session.commit()

    dispatcher = JobDispatcher(maker)
    await dispatcher.enqueue(
        owner_id=ANNA, material_version_id=version.id, parser_profile_version="ir@1"
    )

    first = await dispatcher.claim(owner="worker-a")
    second = await dispatcher.claim(owner="worker-b")

    assert first is not None
    assert second is None, "one job, one worker"
