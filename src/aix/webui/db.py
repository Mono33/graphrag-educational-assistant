"""
WebUI database layer (CORE 2 #6.6 P1).

Async SQLAlchemy 2.x core for the webui's authoritative store of users,
lesson submissions, and (later) chat history. Backed by a SQLite file in
development; the same code path runs against managed Postgres in CORE 6
deployment by changing only ``WEBUI_DATABASE_URL``.

Why this lives under ``aix.webui`` and not ``aix.api``:
    - The public DEV-facing JSON API (``aix.api``) is *stateless* by design —
      it returns GraphRAG context for a query and that contract is frozen.
    - The webui has its own per-user state (auth tokens, saved lessons, chat
      history) that the public API must not depend on. Keeping the DB local
      to the webui package preserves that boundary.

DB URL precedence:
    1. ``WEBUI_DATABASE_URL`` env var (e.g. ``postgresql+asyncpg://…`` in prod)
    2. Default: ``sqlite+aiosqlite:///<repo_root>/data/webui/webui.db``

Schema bootstrap:
    ``init_db()`` is awaited from the FastAPI lifespan in ``aix.api.main``.
    It runs ``Base.metadata.create_all`` so the DB is usable on a fresh
    checkout without an Alembic step. Migrations land in P5/CORE 6 once the
    schema starts evolving.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

from sqlalchemy import event

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

logger = logging.getLogger(__name__)


# Resolve the default SQLite path relative to the package root, so the DB
# lives in <repo>/data/webui/webui.db regardless of the cwd uvicorn was
# launched from. This mirrors the strategy used in ``webui.routes`` for
# the templates directory.
_PACKAGE_DIR = Path(__file__).resolve().parent  # …/src/aix/webui
_REPO_ROOT = _PACKAGE_DIR.parents[2]  # …/graphaixlearning
_DEFAULT_DB_PATH = _REPO_ROOT / "data" / "webui" / "webui.db"
_DEFAULT_DB_URL = f"sqlite+aiosqlite:///{_DEFAULT_DB_PATH.as_posix()}"


def _resolve_db_url() -> str:
    """Return the configured async DB URL, defaulting to dev SQLite."""
    return os.getenv("WEBUI_DATABASE_URL", _DEFAULT_DB_URL)


def resolve_db_url() -> str:
    """Public accessor for the configured async DB URL.

    Alembic's ``env.py`` needs the same URL this module connects with. Reading
    it from here rather than from ``alembic.ini`` keeps one source of truth: a
    deployment that sets ``WEBUI_DATABASE_URL`` migrates the database it is
    actually going to use, with no second place to keep in sync.
    """
    return _DB_URL


def _safe_db_label(url: str) -> str:
    """Return a log-safe DB label without leaking credentials."""
    if url.startswith("sqlite"):
        return url.split("///", 1)[-1]

    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return "<configured>"

    host = parts.hostname or ""
    port = f":{parts.port}" if parts.port is not None else ""
    auth = f"{parts.username}:***@" if parts.username else ""
    netloc = f"{auth}{host}{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


# DeclarativeBase shared by every webui model (User, Lesson, …).
class Base(DeclarativeBase):
    """Common SQLAlchemy declarative base for all webui tables."""


# Engine and session factory are created at import time (cheap; no I/O until
# the first connect), so dependencies that ask for ``get_async_session``
# don't pay an init cost on every request.
_DB_URL = _resolve_db_url()

# echo=False keeps the uvicorn logs clean; flip via WEBUI_DB_ECHO=1 when
# debugging a query.
_ECHO = os.getenv("WEBUI_DB_ECHO", "0") in ("1", "true", "True")

engine: AsyncEngine = create_async_engine(
    _DB_URL,
    echo=_ECHO,
    # SQLite-specific: required when sharing an aiosqlite connection across
    # asyncio tasks (FastAPI dependency injection does this).
    connect_args={"check_same_thread": False} if _DB_URL.startswith("sqlite") else {},
)

if _DB_URL.startswith("sqlite"):

    @event.listens_for(engine.sync_engine, "connect")
    def _sqlite_pragmas(dbapi_connection, _record) -> None:  # pragma: no cover - driver glue
        """WAL, and a long wait for the lock.

        SQLite in its default ``journal_mode=delete`` has one writer and no
        concurrent reader: a reader blocks the writer and the writer blocks the
        reader. That is a fine trade for a database a single request touches,
        and a bad one here, because the ingestion worker runs **inside the same
        process** as the routes and writes hundreds of chunk rows in one
        transaction.

        On 2026-09-22 a teacher attached a 114-page magazine. The worker took
        seventeen seconds to index it; the upload request, trying to annotate
        the lesson's manifest in the meantime, never returned at all — no
        response, no chip, no error. The file was indexed and the page looked
        broken, which is the worst of both. It took reading the access log to
        notice the POST had no reply.

        WAL lets the reader carry on while the writer works. ``busy_timeout``
        covers what WAL does not — two writers — and 30s is chosen against the
        length of an ingestion transaction, not against a human's patience: the
        alternative to waiting is an error on an upload that is actually fine.
        """
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.execute("PRAGMA synchronous=NORMAL")
        finally:
            cursor.close()


async_session_maker: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine,
    expire_on_commit=False,
    autoflush=False,
)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency yielding a per-request ``AsyncSession``.

    Use this everywhere the webui needs DB access:

        async def my_route(session: AsyncSession = Depends(get_async_session)):
            ...

    Sessions are automatically closed when the request finishes; no manual
    ``await session.close()`` is needed.
    """
    async with async_session_maker() as session:
        yield session


# Arbitrary but fixed application-defined key for the first-boot DDL advisory
# lock. Any constant works; it only has to be identical across workers so they
# contend on the *same* lock. ('AIX_DB' as bytes → bigint, well within the
# signed 64-bit range Postgres advisory locks accept.)
_INIT_DB_ADVISORY_LOCK_KEY = 0x4149585F4442


def _init_ddl_lock_stmt(dialect_name: str):
    """Return the statement that serialises concurrent first-boot DDL, or None.

    With ``uvicorn --workers N`` (Phase B #39) every worker runs ``init_db`` at
    once. ``create_all`` is check-then-CREATE *without* ``IF NOT EXISTS``, so on
    a fresh Postgres DB two workers can both decide a table is missing and race
    the ``CREATE`` → one raises ``DuplicateTable``. A transaction-scoped
    ``pg_advisory_xact_lock`` makes exactly one worker do the DDL while the rest
    wait, then find the tables already present (clean no-op). The lock releases
    automatically when the ``engine.begin()`` transaction commits.

    Postgres-only by design: SQLite dev is single-writer and single-worker, so
    it keeps its original lock-free path (backward compatible).
    """
    if dialect_name == "postgresql":
        from sqlalchemy import text

        return text("SELECT pg_advisory_xact_lock(:key)")
    return None


def _auto_migrate_enabled() -> bool:
    """Whether startup applies migrations. Default on, and deliberately so.

    The GitHub workflow that ships this code builds an image and stops there —
    there is no deploy step in this repository where ``alembic upgrade head``
    could be added. Migrations that travel inside the image therefore reach
    production on the next push; migrations that need a deploy step would wait
    on a pipeline the AI team does not own.

    The switch exists for the day that changes, and for one scenario where
    startup is the wrong place: a migration long enough to outlast the
    container healthcheck's start period. Set ``AIX_DB_AUTO_MIGRATE=false`` and
    run ``alembic upgrade head`` out of band.
    """
    return (os.getenv("AIX_DB_AUTO_MIGRATE") or "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _alembic_config(connection):
    """Alembic config bound to an already-open connection.

    env.py uses this connection instead of opening its own, so the migration
    runs inside the transaction that holds the advisory lock.
    """
    from alembic.config import Config

    cfg = Config(str(_REPO_ROOT / "alembic.ini"))
    cfg.attributes["connection"] = connection
    return cfg


#: The first migration: the WebUI schema as it was before any of this work.
#: A database that predates migrations is compared against *this*, never
#: against the current models — see ``_baseline_schema``.
_BASELINE_REVISION = "c8d88f37e59c"


def _baseline_schema() -> dict[str, set[str]]:
    """Tables and columns the baseline migration creates.

    Built by running that migration against a throwaway in-memory SQLite, so it
    cannot drift from the migration it describes — a hand-written list would be
    correct exactly until someone edited the baseline.

    Comparing against the *models* instead was the bug this replaces: the models
    have grown ``material``, ``document_ir`` and ``material_chunk``, so every
    real production database looked "incomplete", was refused adoption, and
    stayed outside migration control with ``alembic_version`` empty — while the
    tests passed, because they seeded their legacy database from those same
    models.
    """
    from alembic import command
    from sqlalchemy import create_engine, inspect

    engine = create_engine("sqlite://")
    try:
        with engine.begin() as connection:
            command.upgrade(_alembic_config(connection), _BASELINE_REVISION)
            inspector = inspect(connection)
            return {
                name: {c["name"] for c in inspector.get_columns(name)}
                for name in inspector.get_table_names()
                if name != "alembic_version"
            }
    finally:
        engine.dispose()


def _schema_gaps(sync_conn) -> dict[str, list[str]]:
    """What the baseline has and the live schema does not.

    Empty means the database is the one the baseline describes, so it can be
    adopted at that revision and then migrated forward normally. Anything listed
    means it is some *other* schema, and nothing is touched.
    """
    from sqlalchemy import inspect

    inspector = inspect(sync_conn)
    live_tables = set(inspector.get_table_names())
    gaps: dict[str, list[str]] = {}

    for table_name, expected_columns in _baseline_schema().items():
        if table_name not in live_tables:
            gaps[table_name] = ["<table absent>"]
            continue
        live_cols = {c["name"] for c in inspector.get_columns(table_name)}
        absent = sorted(expected_columns - live_cols)
        if absent:
            gaps[table_name] = absent

    return gaps


def _migrate_or_adopt(sync_conn) -> None:
    """Bring the database under migration control without ever guessing.

    Three cases, and the third is the one that needs care:

    1. ``alembic_version`` exists — ordinary upgrade.
    2. No application tables — a fresh database; the baseline creates them.
    3. Tables but no ``alembic_version`` — a database that predates migrations.
       This is production: its schema was built by ``create_all`` plus a
       hot-patch list that only ever ran on SQLite, so on Postgres some columns
       may never have been applied.

       Running ``upgrade`` here would fail trying to re-create existing tables.
       Stamping blindly is worse: it would record the database as up to date
       while columns are still missing, and no future migration would ever add
       them. So the schema is compared against the models first, and adopted
       only if it already matches. If it does not, nothing is changed and the
       exact gap is logged — the schema stays exactly as it was.
    """
    from alembic import command
    from sqlalchemy import inspect

    live_tables = set(inspect(sync_conn).get_table_names())
    cfg = _alembic_config(sync_conn)

    if "alembic_version" in live_tables:
        command.upgrade(cfg, "head")
        return

    if not (live_tables & {t.name for t in Base.metadata.sorted_tables}):
        logger.info("[db] empty database — creating the schema from migrations")
        command.upgrade(cfg, "head")
        return

    gaps = _schema_gaps(sync_conn)
    if gaps:
        logger.error(
            "[db] existing schema does NOT match the migration baseline and was "
            "left untouched. Missing: %s. Resolve it before migrations can run: "
            "add what is missing, then `alembic stamp %s` once. This is the "
            "case the SQLite-only hot-patch list used to leave behind.",
            {t: cols for t, cols in gaps.items()},
            _BASELINE_REVISION,
        )
        return

    # Stamp at the baseline, then migrate forward. Stamping *head* here would
    # record a database as up to date while every table added after the
    # baseline — material, document_ir, material_chunk — was still missing, and
    # no later migration would ever create them.
    logger.info(
        "[db] existing schema is the baseline — adopting it (stamp %s) and migrating forward",
        _BASELINE_REVISION,
    )
    command.stamp(cfg, _BASELINE_REVISION)
    command.upgrade(cfg, "head")


def _read_revision(sync_conn) -> Optional[str]:
    from alembic.runtime.migration import MigrationContext

    return MigrationContext.configure(sync_conn).get_current_revision()


async def current_db_revision() -> Optional[str]:
    """Alembic revision the database is at, or ``None`` if unknown.

    Surfaced by ``/api/v1/health`` so "which schema is live?" is answerable
    from outside the VM, which matters now that a deploy can migrate.
    """
    try:
        async with engine.connect() as conn:
            return await conn.run_sync(_read_revision)
    except Exception:  # noqa: BLE001
        return None


async def init_db() -> None:
    """Bring the WebUI schema up to date, under migration control.

    Replaces the previous ``create_all`` plus hand-written hot-patch list. That
    list was guarded by ``if not sqlite: return``, so a Postgres deployment
    received none of it — the defect these migrations exist to end.

    Multi-worker safe: on Postgres the work runs under a transaction-scoped
    advisory lock, so concurrent uvicorn workers do not migrate at once.

    Never raises out of the migration itself; the caller in the API lifespan
    already treats a WebUI database problem as non-fatal, so that the public
    ``/api/v1`` surface — which does not use this database — still comes up.
    """
    # Importing the model modules registers their tables against ``Base``.
    from aix.webui.auth import models as _auth_models  # noqa: F401
    from aix.webui.lessons import models as _lesson_models  # noqa: F401
    from aix.webui.materials import models as _material_models  # noqa: F401

    if _DB_URL.startswith("sqlite"):
        _DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not _auto_migrate_enabled():
        logger.info(
            "[db] AIX_DB_AUTO_MIGRATE is off — schema left untouched; "
            "run `alembic upgrade head` out of band"
        )
        return

    async with engine.begin() as conn:
        lock_stmt = _init_ddl_lock_stmt(engine.dialect.name)
        if lock_stmt is not None:
            await conn.execute(lock_stmt, {"key": _INIT_DB_ADVISORY_LOCK_KEY})
        await conn.run_sync(_migrate_or_adopt)

    logger.info(
        "✅ WebUI DB ready (%s) at revision %s",
        _safe_db_label(_DB_URL),
        await current_db_revision(),
    )
