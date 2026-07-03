"""Cross-worker in-flight run registry (CORE 6 — Phase B #37).

Why this module exists
----------------------
The WebUI SSE endpoint must answer one question before it starts streaming a
lesson generation: *"is this lesson already being generated right now?"* — so a
duplicate browser tab or an aggressive auto-reconnect doesn't kick off a second,
concurrent pipeline for the same ``lesson.id`` (double LLM cost, duplicated SSE,
racing writes to the same thread).

Until #37 that question was answered by a **per-process** ``set[uuid]``
(``_ACTIVE_RUNS`` in the lessons routes). That set is invisible to other
processes, so the moment we run more than one uvicorn worker / replica (#39)
worker A can't see that worker B is already driving a lesson. This module
replaces that set with a shared registry so the answer is correct across every
worker.

Two backends, chosen at runtime (backward-compatible by design)
---------------------------------------------------------------
* :class:`InMemoryRunRegistry` — the *exact* old behaviour: an in-process set.
  Selected automatically when the WebUI DB is SQLite (i.e. local single-worker
  dev). Nothing about the dev workflow changes.
* :class:`DbRunRegistry` — a row per in-flight run in the shared Postgres DB
  (reusing the WebUI SQLAlchemy engine — no new pool, no new dependency).
  Selected automatically when ``WEBUI_DATABASE_URL`` is Postgres (i.e. the FEM
  production / multi-worker deployment).

Override the auto-selection with ``AIX_RUN_REGISTRY=auto|memory|db`` (a
kill-switch: force ``memory`` to instantly revert to the old behaviour without
a redeploy).

Crash recovery (preserves the old "lost on restart" semantic)
-------------------------------------------------------------
The old set vanished on restart, which was *intentional*: a stale "running"
marker must never permanently lock a lesson out of being re-run after a crash.
The DB backend reproduces this with a heartbeat: the owning worker refreshes
``heartbeat_at`` every :func:`heartbeat_interval_s` seconds; a row whose
heartbeat is older than :func:`ttl_s` is considered **stale** (its worker died
mid-run) and is ignored by ``is_active`` and taken over by ``claim``. So a crash
self-heals after at most one TTL instead of blocking forever.

Lifecycle (mirrors the old set ops)
-----------------------------------
::

    registry = get_run_registry()
    token = await registry.claim(lesson_id, owner_id=...)   # set.add → atomic
    if token is None:
        ...  # someone else owns it — surface "già in corso"
    try:
        # ... drive the run; for long runs refresh the heartbeat:
        hb = asyncio.create_task(heartbeat_loop(registry, lesson_id, token))
        ...
    finally:
        hb.cancel()
        await registry.release(lesson_id, token)            # set.discard
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables (env-configurable, sane defaults)
# ---------------------------------------------------------------------------

_DEFAULT_TTL_S = 60.0
_DEFAULT_HEARTBEAT_INTERVAL_S = 15.0


def _float_env(name: str, default: float, *, lo: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return max(lo, float(raw))
    except (TypeError, ValueError):
        return default


def ttl_s() -> float:
    """Seconds after the last heartbeat before a run row is considered stale.

    A stale row means the owning worker crashed mid-run; ``is_active`` ignores
    it and ``claim`` may take it over, so a crash never permanently blocks a
    lesson from being re-run.
    """
    return _float_env("AIX_RUN_HEARTBEAT_TTL_S", _DEFAULT_TTL_S, lo=1.0)


def heartbeat_interval_s() -> float:
    """How often the owning worker refreshes ``heartbeat_at`` for a live run.

    Kept well below :func:`ttl_s` (default 15s vs 60s TTL) so a healthy run is
    never mistaken for a crashed one.
    """
    return _float_env(
        "AIX_RUN_HEARTBEAT_INTERVAL_S", _DEFAULT_HEARTBEAT_INTERVAL_S, lo=1.0
    )


def _worker_id() -> str:
    """Best-effort identity of the owning process (host:pid) for diagnostics
    and the #40 concurrency dashboard."""
    try:
        host = socket.gethostname()
    except Exception:  # noqa: BLE001 — never fail a claim over a hostname lookup
        host = "unknown"
    return f"{host}:{os.getpid()}"


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class InMemoryRunRegistry:
    """Per-process registry — the pre-#37 behaviour, unchanged.

    Correct for single-worker dev (SQLite). Not shared across processes, so it
    is deliberately NOT used when Postgres is configured. No heartbeat/TTL is
    needed: a single process always runs the ``release`` in its ``finally``,
    and a restart simply empties the set (the old crash-recovery semantic).
    """

    backend = "memory"

    def __init__(self) -> None:
        self._active: set[uuid.UUID] = set()

    async def claim(
        self, lesson_id: uuid.UUID, *, owner_id: Optional[uuid.UUID] = None
    ) -> Optional[str]:
        if lesson_id in self._active:
            return None
        self._active.add(lesson_id)
        return uuid.uuid4().hex

    async def heartbeat(self, lesson_id: uuid.UUID, token: str) -> None:  # noqa: ARG002
        return None

    async def release(
        self, lesson_id: uuid.UUID, token: Optional[str] = None  # noqa: ARG002
    ) -> None:
        self._active.discard(lesson_id)

    async def is_active(self, lesson_id: uuid.UUID) -> bool:
        return lesson_id in self._active

    async def active_runs(self) -> list[dict[str, Any]]:
        return [{"lesson_id": str(x)} for x in self._active]


class DbRunRegistry:
    """Shared registry backed by the WebUI Postgres DB (one row per in-flight
    run). Reuses the WebUI ``async_sessionmaker`` — no extra pool/dependency.

    All operations open their own short-lived session rather than borrowing the
    request's session, because the heartbeat task outlives any single request
    scope and must not interfere with the route's transaction.
    """

    backend = "postgres"

    def __init__(self, session_maker: Any) -> None:
        self._session_maker = session_maker

    async def claim(
        self, lesson_id: uuid.UUID, *, owner_id: Optional[uuid.UUID] = None
    ) -> Optional[str]:
        """Atomically register ``lesson_id`` as in-flight.

        Returns an opaque token on success, or ``None`` if another (live) run
        already owns the lesson. Portable, race-safe strategy:

        1. Try a plain INSERT. The ``lesson_id`` primary key makes this atomic —
           under concurrent claims exactly one INSERT wins; the rest raise
           ``IntegrityError``.
        2. On conflict, attempt a conditional takeover UPDATE gated on the
           existing row being stale (``heartbeat_at < now - ttl``). The row lock
           serialises racing takeovers, so again exactly one wins; a fresh row
           yields ``rowcount == 0`` → we correctly report "owned by someone
           else".
        """
        from sqlalchemy import update
        from sqlalchemy.exc import IntegrityError

        from aix.webui.lessons.models import AgentRun

        now = datetime.utcnow()
        token = uuid.uuid4()
        worker = _worker_id()

        async with self._session_maker() as session:
            try:
                session.add(
                    AgentRun(
                        lesson_id=lesson_id,
                        run_id=token,
                        owner_id=owner_id,
                        worker_id=worker,
                        started_at=now,
                        heartbeat_at=now,
                    )
                )
                await session.commit()
                return token.hex
            except IntegrityError:
                await session.rollback()

            # A row already exists. Take it over only if it is stale.
            cutoff = now - timedelta(seconds=ttl_s())
            result = await session.execute(
                update(AgentRun)
                .where(AgentRun.lesson_id == lesson_id, AgentRun.heartbeat_at < cutoff)
                .values(
                    run_id=token,
                    owner_id=owner_id,
                    worker_id=worker,
                    started_at=now,
                    heartbeat_at=now,
                )
            )
            await session.commit()
            if result.rowcount == 1:
                logger.info(
                    "[run-registry] took over stale run for lesson_id=%s", lesson_id
                )
                return token.hex
            return None

    async def heartbeat(self, lesson_id: uuid.UUID, token: str) -> None:
        from sqlalchemy import update

        from aix.webui.lessons.models import AgentRun

        async with self._session_maker() as session:
            await session.execute(
                update(AgentRun)
                .where(
                    AgentRun.lesson_id == lesson_id,
                    AgentRun.run_id == uuid.UUID(hex=token),
                )
                .values(heartbeat_at=datetime.utcnow())
            )
            await session.commit()

    async def release(self, lesson_id: uuid.UUID, token: Optional[str] = None) -> None:
        from sqlalchemy import delete

        from aix.webui.lessons.models import AgentRun

        stmt = delete(AgentRun).where(AgentRun.lesson_id == lesson_id)
        if token is not None:
            # Only delete OUR row — never a successor that took over after we
            # were (wrongly) considered stale.
            stmt = stmt.where(AgentRun.run_id == uuid.UUID(hex=token))
        async with self._session_maker() as session:
            await session.execute(stmt)
            await session.commit()

    async def is_active(self, lesson_id: uuid.UUID) -> bool:
        from sqlalchemy import select

        from aix.webui.lessons.models import AgentRun

        cutoff = datetime.utcnow() - timedelta(seconds=ttl_s())
        async with self._session_maker() as session:
            result = await session.execute(
                select(AgentRun.lesson_id).where(
                    AgentRun.lesson_id == lesson_id,
                    AgentRun.heartbeat_at >= cutoff,
                )
            )
            return result.first() is not None

    async def active_runs(self) -> list[dict[str, Any]]:
        from sqlalchemy import select

        from aix.webui.lessons.models import AgentRun

        cutoff = datetime.utcnow() - timedelta(seconds=ttl_s())
        async with self._session_maker() as session:
            result = await session.execute(
                select(AgentRun).where(AgentRun.heartbeat_at >= cutoff)
            )
            return [
                {
                    "lesson_id": str(row.lesson_id),
                    "owner_id": str(row.owner_id) if row.owner_id else None,
                    "worker_id": row.worker_id,
                    "started_at": row.started_at.isoformat(),
                    "heartbeat_at": row.heartbeat_at.isoformat(),
                }
                for row in result.scalars().all()
            ]


# ---------------------------------------------------------------------------
# Selection (process singleton) + heartbeat helper
# ---------------------------------------------------------------------------

# Union type for either backend. Both expose the same async surface.
RunRegistry = Any

_REGISTRY: Optional[RunRegistry] = None


def get_run_registry() -> RunRegistry:
    """Return the process-singleton registry, choosing the backend from config.

    * ``AIX_RUN_REGISTRY=memory`` → always in-memory (kill-switch).
    * ``AIX_RUN_REGISTRY=db``     → always DB-backed.
    * ``auto`` (default)          → DB-backed iff the WebUI DB is Postgres,
                                    else in-memory.
    """
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY

    mode = os.getenv("AIX_RUN_REGISTRY", "auto").strip().lower()

    if mode == "memory":
        _REGISTRY = InMemoryRunRegistry()
    elif mode == "db":
        from aix.webui.db import async_session_maker

        _REGISTRY = DbRunRegistry(async_session_maker)
    else:  # auto — match the WebUI DB dialect
        from aix.webui.db import async_session_maker, engine

        if engine.dialect.name == "postgresql":
            _REGISTRY = DbRunRegistry(async_session_maker)
        else:
            _REGISTRY = InMemoryRunRegistry()

    logger.info("[run-registry] backend=%s (mode=%s)", _REGISTRY.backend, mode)
    return _REGISTRY


def set_run_registry(registry: Optional[RunRegistry]) -> None:
    """Override / reset the singleton. For tests and explicit wiring only."""
    global _REGISTRY
    _REGISTRY = registry


async def heartbeat_loop(
    registry: RunRegistry, lesson_id: uuid.UUID, token: str
) -> None:
    """Refresh the run's heartbeat until cancelled.

    Run as a background task for the duration of a long generation. On the
    in-memory backend this is a cheap no-op loop; on the DB backend it keeps the
    row from being mistaken for a crashed (stale) run. Best-effort: a transient
    heartbeat failure is logged, not raised, so it never tears down a live run.
    """
    interval = heartbeat_interval_s()
    try:
        while True:
            await asyncio.sleep(interval)
            try:
                await registry.heartbeat(lesson_id, token)
            except Exception:  # noqa: BLE001 — heartbeat must not kill the run
                logger.warning(
                    "[run-registry] heartbeat failed for lesson_id=%s", lesson_id
                )
    except asyncio.CancelledError:
        raise
