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
from urllib.parse import urlsplit, urlunsplit

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


async def init_db() -> None:
    """
    Create all webui tables if they don't yet exist, and apply the small
    set of dev-only column hot-adds we accumulate between phases.

    Imported lazily inside the function so importing ``aix.webui.db`` doesn't
    transitively import every model package (avoids circular imports during
    test collection).

    Idempotent: ``create_all`` is a no-op if the tables already exist, and
    the column hot-adds skip already-present columns. This is *strictly*
    a dev convenience while we don't have Alembic — it keeps the SQLite
    schema in sync with model changes without forcing a wipe of the dev DB.
    Production migrations land in P5 / CORE 6 with Alembic.

    Multi-worker safe (#39): on Postgres the DDL runs under a transaction-scoped
    advisory lock, so concurrent ``uvicorn --workers`` processes booting against
    a fresh DB don't race ``create_all`` (see :func:`_init_ddl_lock_stmt`).
    """
    # Importing the model modules registers their classes against ``Base``
    # so ``Base.metadata.create_all`` knows about every table.
    from aix.webui.auth import models as _auth_models  # noqa: F401
    from aix.webui.lessons import models as _lesson_models  # noqa: F401

    # Make sure the SQLite parent directory exists before connecting.
    if _DB_URL.startswith("sqlite"):
        _DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    async with engine.begin() as conn:
        # Serialise DDL across concurrent workers on Postgres (#39). No-op on
        # SQLite. Taken inside the transaction so it is held across create_all
        # and released on commit.
        _lock_stmt = _init_ddl_lock_stmt(engine.dialect.name)
        if _lock_stmt is not None:
            await conn.execute(_lock_stmt, {"key": _INIT_DB_ADVISORY_LOCK_KEY})
        await conn.run_sync(Base.metadata.create_all)
        # Dev-only schema patch-up (no-op once Alembic lands).
        await _apply_dev_schema_hotpatches(conn)

    logger.info("✅ WebUI DB ready (%s)", _safe_db_label(_DB_URL))


async def _apply_dev_schema_hotpatches(conn) -> None:
    """
    Add columns introduced after the initial ``create_all`` snapshot was
    written to disk. SQLite is forgiving: ``ALTER TABLE … ADD COLUMN`` is
    cheap and accepts any new nullable column without rewriting rows.

    We can't rely on ``Base.metadata.create_all`` for this — it only creates
    tables that don't exist; it never touches an existing table's columns.

    Each entry is a (table, column, DDL) triple. We probe ``PRAGMA table_info``
    first and only run the ALTER when the column is absent, so this is safe
    to call on every startup.

    On non-SQLite dialects this is a no-op — Postgres/asyncpg deployments
    will move to Alembic before they accumulate schema drift.
    """
    from sqlalchemy import text

    if not _DB_URL.startswith("sqlite"):
        return

    hotpatches = [
        # CORE 2 #6.6 P2 phase 1 — agent run lifecycle columns
        ("lesson", "lesson_plan_md", "ALTER TABLE lesson ADD COLUMN lesson_plan_md TEXT"),
        ("lesson", "error_message", "ALTER TABLE lesson ADD COLUMN error_message VARCHAR(500)"),
        # CORE 2 #6.6 P2 phase 2 — chat workspace
        # Persists the teacher's original (synthesized OR free-text) query so
        # we can render it as the user's first chat bubble on a fresh GET,
        # without re-deriving from the profile. Also the substrate for #10
        # Conversation Memory (multi-turn refinement).
        ("lesson", "teacher_query", "ALTER TABLE lesson ADD COLUMN teacher_query TEXT"),
        # CORE 2 #6.6 P3 — chat attachments manifest. Files live on disk;
        # this column stores [{id, filename, mime, size, text_excerpt,
        # stored_name}, …]. The Writer reads the joined excerpts via
        # ``AgentState.teacher_provided_context`` — the KG is untouched.
        ("lesson", "uploaded_files_json", "ALTER TABLE lesson ADD COLUMN uploaded_files_json TEXT"),
    ]

    for table, column, ddl in hotpatches:
        # PRAGMA returns one row per column — if our target column isn't in
        # that list, we need to add it.
        existing_cols = {
            row[1] for row in (await conn.execute(text(f"PRAGMA table_info({table})"))).all()
        }
        if column in existing_cols:
            continue
        logger.info("🛠  Hot-patching dev schema: %s.%s", table, column)
        await conn.execute(text(ddl))
