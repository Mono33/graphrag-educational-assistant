"""Bringing an existing database under migration control, without guessing.

Startup now applies migrations, because the workflow that ships this code
builds an image and stops there — there is no deploy step in this repository
where ``alembic upgrade head`` could live. That makes the startup path the only
one that reaches production on its own, and also the one that must not damage
anything.

Three cases, and the third is why this file exists:

1. **Fresh database** — the baseline creates the schema.
2. **Existing schema that is the baseline** — stamped at the baseline revision
   and then migrated forward, so everything added since actually gets created.
3. **Existing schema that is something else** — left strictly alone.

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
    """Build a database exactly as it was *before* this branch, optionally incomplete.

    The baseline migration, then ``alembic_version`` removed: a database that
    predates migration control, which is what production is.

    It used to be ``Base.metadata.create_all``, and that was the flaw Louis's
    review found on 2026-09-15. The models had grown the material tables, so the
    "legacy" database in the test already contained them — the one shape that
    could never occur in production, and the only one the adoption code handled.
    Every test passed while a real database was refused adoption and left
    outside migration control.
    """
    from alembic import command

    from aix.webui.db import _alembic_config

    engine = create_async_engine(db_url)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(
                lambda sync_conn: command.upgrade(_alembic_config(sync_conn), "c8d88f37e59c")
            )
            await conn.execute(text("DROP TABLE alembic_version"))
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
    assert {
        "user",
        "lesson",
        "lesson_message",
        "lesson_media_snapshot",
        "saved_profile",
        "saved_resource",
        "api_idempotency_key",
        "agent_run",
    } <= tables
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


async def test_new_resource_migrations_downgrade_and_upgrade_cleanly(db_url):
    from alembic import command

    from aix.webui.db import _alembic_config

    await _run(db_url, _migrate_or_adopt)
    assert {"lesson_media_snapshot", "saved_resource"} <= await _run(db_url, _tables)

    await _run(
        db_url,
        lambda conn: command.downgrade(_alembic_config(conn), "3b8e6d1f4a27"),
    )
    assert not ({"lesson_media_snapshot", "saved_resource"} & await _run(db_url, _tables))

    await _run(
        db_url,
        lambda conn: command.upgrade(_alembic_config(conn), "head"),
    )
    assert {"lesson_media_snapshot", "saved_resource"} <= await _run(db_url, _tables)


# ---------------------------------------------------------------------------
# 2. Existing schema that matches — adopt it
# ---------------------------------------------------------------------------


async def test_a_real_pre_feature_database_reaches_head_with_its_new_tables(db_url):
    """The production case, and the one the old test could not express.

    A database from before this branch has ``lesson`` and ``user`` and nothing
    else. Adoption must put it under migration control *at the baseline* and
    then migrate it forward, so the tables added since — material, the parsed
    IR, the private chunks — actually get created. Stamping head instead would
    declare it up to date with none of them, permanently.
    """
    from alembic.script import ScriptDirectory

    from aix.webui.db import _alembic_config

    await _seed_legacy_schema(db_url)
    assert "material" not in await _run(db_url, _tables), "the legacy shape has none"

    await _run(db_url, _migrate_or_adopt)

    tables = await _run(db_url, _tables)
    assert {
        "material",
        "material_version",
        "ingestion_job",
        "document_ir",
        "lesson_media_snapshot",
        "saved_resource",
    } <= tables
    assert "material_chunk" in tables

    head = await _run(
        db_url,
        lambda conn: ScriptDirectory.from_config(_alembic_config(conn)).get_current_head(),
    )
    assert await _run(db_url, _read_revision) == head, "and it is fully up to date"


async def test_the_teacher_s_data_survives_adoption(db_url):
    """Adoption migrates a schema; it must not touch a single row."""
    await _seed_legacy_schema(db_url)

    engine = create_async_engine(db_url)
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    'INSERT INTO "user" (id, email, hashed_password, is_active, '
                    "is_superuser, is_verified) VALUES ('u1', 'anna@scuola.it', 'x', 1, 0, 1)"
                )
            )
    finally:
        await engine.dispose()

    await _run(db_url, _migrate_or_adopt)

    rows = await _run(
        db_url, lambda conn: conn.execute(text('SELECT email FROM "user"')).fetchall()
    )
    assert [r[0] for r in rows] == ["anna@scuola.it"]


async def test_adoption_leaves_the_existing_tables_exactly_as_they_were(db_url):
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
