"""Bringing an existing database under migration control, without guessing.

Startup now applies migrations, because the workflow that ships this code
builds an image and stops there — there is no deploy step in this repository
where ``alembic upgrade head`` could live. That makes the startup path the only
one that reaches production on its own, and also the one that must not damage
anything.

Three cases, and the third is why this file exists:

1. **Fresh database** — the baseline creates the schema.
2. **Existing schema that matches the models** — adopted with ``stamp``, no DDL.
3. **Existing schema that does *not* match** — left strictly alone.

Case 3 is production's shape. Its schema was built by ``create_all`` plus a
hot-patch list guarded by ``if not sqlite: return``, so on Postgres some columns
may never have been applied. Running ``upgrade`` there fails on tables that
already exist; stamping there is worse, because it records the database as up to
date while columns are still missing — and no later migration would ever add
them. The only safe answer is to change nothing and say exactly what is missing.
"""

from __future__ import annotations

import logging

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine

from aix.webui.auth import models as _auth_models  # noqa: F401
from aix.webui.db import Base, _migrate_or_adopt, _read_revision, _schema_gaps
from aix.webui.lessons import models as _lesson_models  # noqa: F401

pytestmark = pytest.mark.unit

# Columns the SQLite-only hot-patch list used to add. A Postgres database old
# enough to predate them never received them, which is exactly case 3.
HOTPATCHED = ["lesson_plan_md", "error_message", "teacher_query", "uploaded_files_json"]


@pytest.fixture
def db_url(tmp_path):
    return f"sqlite+aiosqlite:///{(tmp_path / 'webui.db').as_posix()}"


async def _run(db_url, fn):
    engine = create_async_engine(db_url)
    try:
        async with engine.begin() as conn:
            return await conn.run_sync(fn)
    finally:
        await engine.dispose()


async def _seed_legacy_schema(db_url, *, drop: list[str] | None = None):
    """Build the schema the way ``create_all`` used to, optionally incomplete."""
    engine = create_async_engine(db_url)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            for column in drop or []:
                await conn.execute(text(f"ALTER TABLE lesson DROP COLUMN {column}"))
    finally:
        await engine.dispose()


def _tables(sync_conn) -> set[str]:
    return set(inspect(sync_conn).get_table_names())


def _lesson_columns(sync_conn) -> set[str]:
    return {c["name"] for c in inspect(sync_conn).get_columns("lesson")}


# ---------------------------------------------------------------------------
# 1. Fresh database
# ---------------------------------------------------------------------------


async def test_fresh_database_is_created_from_the_migrations(db_url):
    await _run(db_url, _migrate_or_adopt)

    tables = await _run(db_url, _tables)
    assert {"user", "lesson", "lesson_message", "saved_profile", "agent_run"} <= tables
    assert "alembic_version" in tables

    assert await _run(db_url, _read_revision) is not None


async def test_fresh_database_includes_the_formerly_hot_patched_columns(db_url):
    """The baseline is the model state, so the four columns are simply in it."""
    await _run(db_url, _migrate_or_adopt)

    columns = await _run(db_url, _lesson_columns)
    assert set(HOTPATCHED) <= columns


async def test_running_twice_is_a_no_op(db_url):
    await _run(db_url, _migrate_or_adopt)
    first = await _run(db_url, _read_revision)

    await _run(db_url, _migrate_or_adopt)

    assert await _run(db_url, _read_revision) == first


# ---------------------------------------------------------------------------
# 2. Existing schema that matches — adopt it
# ---------------------------------------------------------------------------


async def test_matching_legacy_schema_is_adopted_without_ddl(db_url):
    await _seed_legacy_schema(db_url)
    before = await _run(db_url, _lesson_columns)

    assert await _run(db_url, _read_revision) is None, "no version table yet"

    await _run(db_url, _migrate_or_adopt)

    assert await _run(db_url, _read_revision) is not None, "adopted"
    assert await _run(db_url, _lesson_columns) == before, "stamp must not touch the schema"


async def test_adoption_does_not_re_create_tables(db_url):
    """``upgrade`` on an existing schema would fail; adoption must not try."""
    await _seed_legacy_schema(db_url)

    engine = create_async_engine(db_url)
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    'INSERT INTO "user" (id, email, hashed_password, is_active, '
                    "is_superuser, is_verified) VALUES ('u1', 'a@b.c', 'x', 1, 0, 0)"
                )
            )
    finally:
        await engine.dispose()

    await _run(db_url, _migrate_or_adopt)

    engine = create_async_engine(db_url)
    try:
        async with engine.connect() as conn:
            rows = (await conn.execute(text('SELECT count(*) FROM "user"'))).scalar()
    finally:
        await engine.dispose()

    assert rows == 1, "existing data survived untouched"


# ---------------------------------------------------------------------------
# 3. Existing schema that does not match — change nothing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing", HOTPATCHED)
async def test_incomplete_legacy_schema_is_left_alone(db_url, missing, caplog):
    """Production's shape: tables present, a hot-patched column never applied."""
    await _seed_legacy_schema(db_url, drop=[missing])
    before = await _run(db_url, _lesson_columns)

    with caplog.at_level(logging.ERROR):
        await _run(db_url, _migrate_or_adopt)

    assert await _run(db_url, _read_revision) is None, (
        "stamping here would record the database as up to date while a column "
        "is missing, and no later migration would ever add it"
    )
    assert await _run(db_url, _lesson_columns) == before, "schema untouched"
    assert missing in caplog.text, "the gap has to be named, not merely refused"


async def test_gap_report_names_table_and_columns(db_url):
    await _seed_legacy_schema(db_url, drop=["teacher_query", "error_message"])

    gaps = await _run(db_url, _schema_gaps)

    assert set(gaps) == {"lesson"}
    assert set(gaps["lesson"]) == {"teacher_query", "error_message"}


async def test_no_gaps_reported_for_a_matching_schema(db_url):
    await _seed_legacy_schema(db_url)
    assert await _run(db_url, _schema_gaps) == {}
