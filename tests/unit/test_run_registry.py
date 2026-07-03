"""CORE 6 — Phase B #37 run-registry coverage.

Locks in the behaviour that lets multiple workers safely share one "is this
lesson already generating?" answer, while keeping single-worker SQLite dev
byte-for-byte identical to the pre-#37 in-memory set.

Two backends are exercised:

* :class:`InMemoryRunRegistry` — the dev default. Must behave like the old
  ``set`` (claim/is_active/release, idempotent, per-lesson isolation).
* :class:`DbRunRegistry` — the prod backend. Exercised against a temp-file
  SQLite DB (separate connections per op, like separate workers) to prove the
  atomic claim, cross-"worker" contention, heartbeat-TTL staleness takeover,
  and token-scoped release.

The DB backend's SQL is dialect-portable (plain INSERT + conditional UPDATE,
no PG-only upsert), so SQLite is a faithful stand-in for the claim semantics
without needing a live Postgres.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import aix.core.run_registry as rr
from aix.core.run_registry import (
    DbRunRegistry,
    InMemoryRunRegistry,
    get_run_registry,
    set_run_registry,
)
from aix.webui.lessons.models import AgentRun

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# In-memory backend — the SQLite dev default (must match the old set)
# ---------------------------------------------------------------------------


async def test_inmemory_claim_blocks_second_then_frees_on_release():
    reg = InMemoryRunRegistry()
    lid = uuid.uuid4()

    token = await reg.claim(lid)
    assert token is not None
    assert await reg.is_active(lid) is True

    # A second claim on the same lesson is refused while the first is live.
    assert await reg.claim(lid) is None

    await reg.release(lid, token)
    assert await reg.is_active(lid) is False
    # After release the lesson can be claimed again (e.g. a follow-up turn).
    assert await reg.claim(lid) is not None


async def test_inmemory_is_per_lesson_and_heartbeat_is_noop():
    reg = InMemoryRunRegistry()
    a, b = uuid.uuid4(), uuid.uuid4()

    ta = await reg.claim(a)
    assert await reg.is_active(a) is True
    assert await reg.is_active(b) is False  # distinct lesson unaffected

    await reg.heartbeat(a, ta)  # no-op, must not raise
    assert await reg.is_active(a) is True


# ---------------------------------------------------------------------------
# DB backend — the Postgres prod path (SQLite temp file stands in)
# ---------------------------------------------------------------------------


@pytest.fixture()
async def db_session_maker(tmp_path):
    """A real file-backed SQLite engine so each registry op opens its own
    connection yet shares state — mimicking independent workers/replicas."""
    url = f"sqlite+aiosqlite:///{(tmp_path / 'reg.db').as_posix()}"
    engine = create_async_engine(url, connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(AgentRun.__table__.create)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield maker
    finally:
        await engine.dispose()


async def _force_stale(maker, lesson_id: uuid.UUID) -> None:
    """Push a run's heartbeat far into the past so the TTL marks it stale —
    deterministic stand-in for 'the owning worker crashed'."""
    async with maker() as session:
        await session.execute(
            update(AgentRun)
            .where(AgentRun.lesson_id == lesson_id)
            .values(heartbeat_at=datetime.utcnow() - timedelta(hours=1))
        )
        await session.commit()


async def test_db_claim_is_atomic_across_workers(db_session_maker):
    worker_a = DbRunRegistry(db_session_maker)
    worker_b = DbRunRegistry(db_session_maker)  # a second "process"
    lid = uuid.uuid4()

    token_a = await worker_a.claim(lid, owner_id=uuid.uuid4())
    assert token_a is not None

    # Worker B sees A's live run (shared table) and is refused.
    assert await worker_b.is_active(lid) is True
    assert await worker_b.claim(lid) is None

    # Once A releases, B can claim.
    await worker_a.release(lid, token_a)
    assert await worker_b.is_active(lid) is False
    assert await worker_b.claim(lid) is not None


async def test_db_stale_run_is_taken_over(db_session_maker):
    reg = DbRunRegistry(db_session_maker)
    lid = uuid.uuid4()

    first = await reg.claim(lid)
    assert first is not None

    # Simulate the owner crashing: heartbeat goes stale.
    await _force_stale(db_session_maker, lid)

    # A stale run must not count as active, and must be claimable again.
    assert await reg.is_active(lid) is False
    second = await reg.claim(lid)
    assert second is not None and second != first


async def test_db_release_is_token_scoped(db_session_maker):
    """A worker that wrongly believes it still owns a run (after a stale
    takeover by someone else) must not delete the successor's row."""
    reg = DbRunRegistry(db_session_maker)
    lid = uuid.uuid4()

    stale_token = await reg.claim(lid)
    await _force_stale(db_session_maker, lid)
    new_token = await reg.claim(lid)  # takeover
    assert new_token != stale_token

    # The old owner's release targets its own (gone) run_id → no-op.
    await reg.release(lid, stale_token)
    assert await reg.is_active(lid) is True  # successor's row survived

    # The real owner's release clears it.
    await reg.release(lid, new_token)
    assert await reg.is_active(lid) is False


async def test_db_heartbeat_revives_a_nearly_stale_run(db_session_maker):
    reg = DbRunRegistry(db_session_maker)
    lid = uuid.uuid4()

    token = await reg.claim(lid)
    await _force_stale(db_session_maker, lid)
    assert await reg.is_active(lid) is False  # would look crashed…

    await reg.heartbeat(lid, token)  # …but the owner is alive and refreshes
    assert await reg.is_active(lid) is True


# ---------------------------------------------------------------------------
# Backend selection (the backward-compat switch)
# ---------------------------------------------------------------------------


def test_get_run_registry_memory_mode(monkeypatch):
    set_run_registry(None)
    monkeypatch.setenv("AIX_RUN_REGISTRY", "memory")
    try:
        reg = get_run_registry()
        assert isinstance(reg, InMemoryRunRegistry)
        assert reg.backend == "memory"
        # Singleton: same instance on subsequent calls.
        assert get_run_registry() is reg
    finally:
        set_run_registry(None)


def test_get_run_registry_db_mode(monkeypatch):
    set_run_registry(None)
    monkeypatch.setenv("AIX_RUN_REGISTRY", "db")
    try:
        reg = get_run_registry()
        assert isinstance(reg, DbRunRegistry)
        assert reg.backend == "postgres"
    finally:
        set_run_registry(None)


def test_ttl_and_interval_env_overrides(monkeypatch):
    monkeypatch.setenv("AIX_RUN_HEARTBEAT_TTL_S", "42")
    monkeypatch.setenv("AIX_RUN_HEARTBEAT_INTERVAL_S", "7")
    assert rr.ttl_s() == 42.0
    assert rr.heartbeat_interval_s() == 7.0
