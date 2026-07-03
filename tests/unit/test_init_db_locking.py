"""CORE 6 — Phase B #39 first-boot DDL concurrency guard.

Locks in the dialect-guarded advisory-lock behaviour that lets multiple
``uvicorn --workers`` processes run ``init_db`` at once against a fresh
Postgres DB without racing ``create_all`` (which is check-then-CREATE without
``IF NOT EXISTS``), while keeping the single-worker SQLite dev path lock-free
and byte-for-byte identical to before.

No live Postgres is required: we assert the *selection* logic (which dialect
gets a lock, and that it is the transaction-scoped variant) plus that the
real ``init_db`` still bootstraps a working SQLite schema.
"""

from __future__ import annotations

import pytest

from aix.webui.db import (
    _INIT_DB_ADVISORY_LOCK_KEY,
    _init_ddl_lock_stmt,
)

pytestmark = pytest.mark.unit


def test_sqlite_gets_no_ddl_lock() -> None:
    """SQLite dev is single-writer/single-worker — it must keep the original
    lock-free path so nothing about the dev workflow changes."""
    assert _init_ddl_lock_stmt("sqlite") is None


def test_postgres_gets_transaction_scoped_advisory_lock() -> None:
    """Postgres (multi-worker prod) must serialise DDL, and it must use the
    *transaction*-scoped lock so it auto-releases when ``init_db``'s
    ``engine.begin()`` transaction commits — the session-scoped
    ``pg_advisory_lock`` would leak and require a manual unlock."""
    stmt = _init_ddl_lock_stmt("postgresql")
    assert stmt is not None
    sql = str(stmt)
    assert "pg_advisory_xact_lock" in sql
    # Guard against a regression to the session-scoped variant.
    assert "pg_advisory_lock(" not in sql


def test_lock_key_is_within_signed_64_bit_range() -> None:
    """Postgres advisory-lock keys are signed 64-bit bigints; a value outside
    that range would raise at runtime on the very first prod boot."""
    assert -(2**63) <= _INIT_DB_ADVISORY_LOCK_KEY < 2**63


async def test_init_db_bootstraps_sqlite_and_is_idempotent(monkeypatch, tmp_path) -> None:
    """The real ``init_db`` must still create a usable schema on SQLite and be
    safe to call repeatedly (idempotent) — the backward-compatible dev path.

    We point the module's ``engine``/``_DB_URL`` at a temp SQLite file rather
    than reloading the module: reloading would rebind ``Base`` to fresh, empty
    metadata (the model classes stay registered on the original ``Base``), so
    ``create_all`` would create nothing. Monkeypatching keeps the real ``Base``
    with its registered tables while redirecting I/O to the temp DB.
    """
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    import aix.webui.db as db

    db_path = tmp_path / "webui_test.db"
    db_url = f"sqlite+aiosqlite:///{db_path.as_posix()}"
    temp_engine = create_async_engine(db_url, connect_args={"check_same_thread": False})

    monkeypatch.setattr(db, "engine", temp_engine)
    monkeypatch.setattr(db, "_DB_URL", db_url)
    try:
        await db.init_db()
        # Second call must be a clean no-op (tables already exist).
        await db.init_db()

        async with temp_engine.connect() as conn:
            rows = (
                await conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table'")
                )
            ).all()
        table_names = {r[0] for r in rows}
        # Core webui tables must exist after bootstrap.
        assert "user" in table_names
        assert "lesson" in table_names
        assert "agent_run" in table_names
    finally:
        await temp_engine.dispose()
