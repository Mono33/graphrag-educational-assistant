"""
LangGraph checkpointer singleton (CORE 2 #10.2 + CORE 4 #15.a).

Why this module exists
----------------------
LangGraph's checkpointer abstraction (``BaseCheckpointSaver``) is what makes
multi-turn conversation memory possible. Every ``ainvoke`` / ``astream`` call
that carries a ``thread_id`` config writes the post-step ``AgentState`` to
the saver, and reads it back on the next call with the same ``thread_id``.
Time-travel (``aget_state_history``), branching (``aupdate_state``), and
human-in-the-loop interrupts (CORE 4 #19) all build on this primitive.

Backend selection (dev vs production)
-------------------------------------
The saver class is chosen **at runtime from the environment**, because
``BaseCheckpointSaver`` is backend-agnostic — the agent graph never knows
which backend it talks to:

* **Postgres (production).** If ``LANGGRAPH_DATABASE_URL`` (or
  ``LANGGRAPH_CHECKPOINTER_URL``) is a ``postgres://`` / ``postgresql://``
  URL, we open an ``AsyncPostgresSaver`` backed by a
  ``psycopg_pool.AsyncConnectionPool``. The pool gives us concurrent
  writers, multi-replica-safe shared state, and durable storage on a
  dedicated volume — everything SQLite cannot do under real load.

* **SQLite (dev default).** Otherwise we fall back to ``AsyncSqliteSaver``
  on a local file (``data/agent_threads.db``). Zero setup, perfect for a
  laptop, but single-writer and not shareable across processes — hence
  dev-only.

We deliberately do **not** silently fall back from Postgres to SQLite: if a
Postgres URL was configured but init fails, we log an error and return
``None`` (single-turn, no persistence) rather than quietly writing thread
state to an ephemeral container file. Silent fallback would mask a prod
misconfiguration and split memory across two backends.

Design choices
--------------
1. **Process-level singleton.** The agent pipeline is invoked from many
   places (webui SSE, public JSON+SSE API, MCP server, CLI), and they
   should all share one saver instance — anything else fragments the
   thread store. We initialise lazily on first access and keep the
   reference for the process lifetime.

2. **Lazy first-access init.** A FastAPI lifespan hook would be cleaner,
   but the lazy pattern keeps the agent layer self-contained and avoids a
   startup ordering dependency on ``aix.api.main``. ``close_checkpointer()``
   is provided for explicit teardown (tests, graceful shutdown) and closes
   the Postgres pool / SQLite connection.

3. **Graceful degradation if a backend package is missing.**
   ``get_checkpointer()`` returns ``None`` instead of crashing — the graph
   compiles without persistence and runs continue in single-turn mode
   (the pre-#10 behaviour). The relevant package is named in the warning.

4. **Configurable via env, no code change for ops moves.**
   ``LANGGRAPH_DATABASE_URL`` (preferred, shared with the webui DB naming)
   and ``LANGGRAPH_CHECKPOINTER_URL`` (legacy) both work. Postgres pool
   size is tunable via ``LANGGRAPH_PG_POOL_MAX`` (default 20).

Usage
-----
::

    from aix.agent.graph.checkpointer import get_checkpointer

    saver = await get_checkpointer()
    if saver is not None:
        graph = workflow.compile(checkpointer=saver)
    else:
        graph = workflow.compile()  # single-turn fallback

The caller passes the per-invocation thread_id config — the saver doesn't
know which thread you mean::

    async for chunk in graph.astream(
        initial_state,
        config={"configurable": {"thread_id": str(lesson.id)}},
        stream_mode="updates",
    ):
        ...

Windows note
------------
psycopg's async mode requires the Selector event loop. On Windows, set
``asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())``
*before* the loop starts if you run the app against Postgres locally.
Production runs on Linux (Docker), where this is a no-op.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level singleton state. Plain globals are fine here because:
#   • we serialise initial-access through ``_INIT_LOCK``,
#   • we never re-init after the first attempt (success OR failure),
#   • uvicorn workers each get their own copy (no cross-worker drift).
# ---------------------------------------------------------------------------

_CHECKPOINTER: Optional[Any] = None
_INIT_ATTEMPTED: bool = False
_INIT_LOCK: asyncio.Lock = asyncio.Lock()
# SQLite: the AsyncSqliteSaver context manager whose ``__aexit__`` releases
# the aiosqlite connection. Postgres: ``None`` (the pool is held in _POOL).
_CONTEXT_MANAGER: Optional[Any] = None
# Postgres: the psycopg_pool.AsyncConnectionPool we own for the process
# lifetime. SQLite: ``None``.
_POOL: Optional[Any] = None
# Which backend is live: "postgres" | "sqlite" | None (not initialised /
# closed / degraded). Read via active_backend().
_BACKEND: Optional[str] = None


def active_backend() -> Optional[str]:
    """
    Return the name of the live checkpointer backend (``"postgres"`` or
    ``"sqlite"``), or ``None`` if the checkpointer is not initialised, was
    closed, or degraded to single-turn mode.

    Used by ops smoke tests and ``/healthz``-style diagnostics to confirm
    the *intended* backend is actually the one running.
    """
    return _BACKEND


# ---------------------------------------------------------------------------
# Backend URL resolution
# ---------------------------------------------------------------------------

_PG_SCHEMES = ("postgres://", "postgresql://", "postgresql+", "postgres+")


def _is_postgres_url(url: str) -> bool:
    return url.startswith(_PG_SCHEMES)


def _normalize_pg_url(url: str) -> str:
    """
    psycopg (which backs AsyncPostgresSaver) wants a plain libpq URL like
    ``postgresql://user:pass@host:5432/db``. Strip any SQLAlchemy-style
    driver suffix (e.g. ``postgresql+asyncpg://`` → ``postgresql://``) so a
    ``WEBUI_DATABASE_URL``-shaped value can be reused safely.
    """
    scheme, sep, rest = url.partition("://")
    if sep and "+" in scheme:
        base = scheme.split("+", 1)[0]
        return f"{base}://{rest}"
    return url


def _resolve_postgres_url() -> Optional[str]:
    """
    Return a normalised Postgres URL if one is configured, else ``None``.

    Preference order:
        1. ``LANGGRAPH_DATABASE_URL`` (production canonical — see compose).
        2. ``LANGGRAPH_CHECKPOINTER_URL`` (legacy override), but only if it
           is itself a Postgres URL (it may instead carry a sqlite path).
    """
    for var in ("LANGGRAPH_DATABASE_URL", "LANGGRAPH_CHECKPOINTER_URL"):
        val = os.getenv(var)
        if val and _is_postgres_url(val):
            return _normalize_pg_url(val)
    return None


def _resolve_db_path() -> Path:
    """
    SQLite fallback path. Default: ``<repo_root>/data/agent_threads.db``.

    Override via ``LANGGRAPH_CHECKPOINTER_URL`` for ops relocations or
    staging. The URL is parsed liberally: ``sqlite:///<abs_path>``,
    ``sqlite+aiosqlite:///<abs_path>``, and bare paths all work (we strip
    the scheme prefix).
    """
    override = os.getenv("LANGGRAPH_CHECKPOINTER_URL")
    if override and not _is_postgres_url(override):
        for prefix in ("sqlite+aiosqlite:///", "sqlite:///"):
            if override.startswith(prefix):
                return Path(override[len(prefix) :])
        return Path(override)

    # Default location, mirroring the webui DB convention. Resolve relative
    # to the package root so the path is stable regardless of cwd uvicorn
    # was launched from.
    package_dir = Path(__file__).resolve().parents[3]  # …/src/aix → …/src
    repo_root = package_dir.parent  # …/graphaixlearning
    return repo_root / "data" / "agent_threads.db"


# ---------------------------------------------------------------------------
# Backend initialisers
# ---------------------------------------------------------------------------


async def _init_postgres(url: str) -> Optional[Any]:
    """
    Open an ``AsyncPostgresSaver`` over a process-lifetime connection pool.

    Returns the saver on success, ``None`` on graceful failure (missing
    package or connection error). On failure we do NOT fall back to SQLite
    — see the module docstring.
    """
    global _POOL, _CHECKPOINTER, _BACKEND

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError as exc:
        logger.warning(
            "[checkpointer] langgraph-checkpoint-postgres not installed (%s); "
            "Postgres was requested via LANGGRAPH_DATABASE_URL but cannot be used. "
            "Install with: pip install 'langgraph-checkpoint-postgres>=2.0,<3.0'",
            exc,
        )
        return None

    try:
        from psycopg_pool import AsyncConnectionPool
    except ImportError as exc:
        logger.warning(
            "[checkpointer] psycopg-pool not installed (%s); cannot open Postgres pool. "
            "Install with: pip install 'psycopg[binary,pool]>=3.1'",
            exc,
        )
        return None

    pool: Optional[Any] = None
    try:
        # autocommit=True is required by AsyncPostgresSaver (DDL in setup()
        # + pipeline writes). prepare_threshold=0 avoids server-side prepared
        # statement churn across pooled connections.
        connection_kwargs = {"autocommit": True, "prepare_threshold": 0}
        max_size = int(os.getenv("LANGGRAPH_PG_POOL_MAX", "20"))
        # open=False + explicit open() avoids psycopg_pool's "opened in the
        # constructor" warning and keeps pool creation on the running loop.
        pool = AsyncConnectionPool(
            conninfo=url,
            max_size=max_size,
            kwargs=connection_kwargs,
            open=False,
        )
        await pool.open(wait=True)
        saver = AsyncPostgresSaver(pool)
        # Idempotent: CREATE TABLE IF NOT EXISTS for the 3 checkpoint tables.
        await saver.setup()
    except Exception as exc:  # noqa: BLE001 — surface any backend init failure
        logger.exception(
            "[checkpointer] failed to open AsyncPostgresSaver: %s", exc
        )
        if pool is not None:
            try:
                await pool.close()
            except Exception:  # noqa: BLE001
                logger.exception("[checkpointer] error closing partial Postgres pool")
        return None

    _POOL = pool
    _CHECKPOINTER = saver
    _BACKEND = "postgres"
    logger.info(
        "[checkpointer] AsyncPostgresSaver ready (pool max_size=%s, multi-turn enabled)",
        pool.max_size,
    )
    return _CHECKPOINTER


async def _init_sqlite() -> Optional[Any]:
    """
    Open an ``AsyncSqliteSaver`` on a local file. Returns the saver on
    success, ``None`` if ``langgraph-checkpoint-sqlite`` is missing or the
    DB cannot be opened (graceful degradation).
    """
    global _CONTEXT_MANAGER, _CHECKPOINTER, _BACKEND

    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except ImportError as exc:
        logger.warning(
            "[checkpointer] langgraph-checkpoint-sqlite not installed (%s); "
            "agent runs will not persist conversation state. "
            "Install with: pip install 'langgraph-checkpoint-sqlite>=3.0,<4.0'",
            exc,
        )
        return None

    db_path = _resolve_db_path()
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning(
            "[checkpointer] cannot create parent dir for %s: %s; skipping checkpointer init",
            db_path,
            exc,
        )
        return None

    try:
        # ``from_conn_string`` returns an async context manager that owns the
        # underlying aiosqlite connection. We ``__aenter__`` it for the
        # process lifetime; ``close_checkpointer()`` runs the matching exit.
        cm = AsyncSqliteSaver.from_conn_string(str(db_path))
        saver = await cm.__aenter__()
        # Idempotent: ensure the checkpoint tables exist before first write.
        await saver.setup()
    except Exception as exc:  # noqa: BLE001 — surface any backend init failure
        logger.exception(
            "[checkpointer] failed to open AsyncSqliteSaver at %s: %s", db_path, exc
        )
        return None

    _CONTEXT_MANAGER = cm
    _CHECKPOINTER = saver
    _BACKEND = "sqlite"
    logger.info(
        "[checkpointer] AsyncSqliteSaver ready at %s (multi-turn enabled)", db_path
    )
    return _CHECKPOINTER


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def get_checkpointer() -> Optional[Any]:
    """
    Return the lazily-initialised checkpointer singleton, choosing the
    backend from the environment (Postgres if ``LANGGRAPH_DATABASE_URL`` is
    a postgres URL, else SQLite). Returns ``None`` on graceful failure.

    First-access semantics:
        1. Acquire ``_INIT_LOCK`` so concurrent first-callers don't race.
        2. Re-check the singleton + ``_INIT_ATTEMPTED`` inside the lock.
        3. Resolve the backend and delegate to the matching initialiser.

    A failed attempt is sticky: we return ``None`` on subsequent calls
    rather than retrying every invocation.
    """
    global _INIT_ATTEMPTED

    # Fast path — already initialised. Avoids the lock for hot calls.
    if _CHECKPOINTER is not None:
        return _CHECKPOINTER
    if _INIT_ATTEMPTED:
        return None

    async with _INIT_LOCK:
        # Re-check inside the lock (another coroutine may have won the race).
        if _CHECKPOINTER is not None:
            return _CHECKPOINTER
        if _INIT_ATTEMPTED:
            return None
        _INIT_ATTEMPTED = True

        pg_url = _resolve_postgres_url()
        if pg_url is not None:
            saver = await _init_postgres(pg_url)
            if saver is None:
                # Postgres was explicitly requested. Do NOT fall back to a
                # local SQLite file — that would split thread state and hide
                # a production misconfiguration. Degrade to single-turn.
                logger.error(
                    "[checkpointer] Postgres requested but unavailable; "
                    "running WITHOUT persistence (single-turn). Fix the DB "
                    "config rather than relying on a silent SQLite fallback."
                )
            return saver

        return await _init_sqlite()


async def close_checkpointer() -> None:
    """
    Best-effort shutdown hook for the singleton. Safe to call multiple
    times; safe to call when the checkpointer was never initialised.

    Closes the SQLite connection (``__aexit__``) and/or the Postgres pool,
    and resets all singleton state so a subsequent ``get_checkpointer()``
    re-initialises cleanly. Tests can call this between cases.
    """
    global _CHECKPOINTER, _INIT_ATTEMPTED, _CONTEXT_MANAGER, _POOL, _BACKEND

    cm = _CONTEXT_MANAGER
    pool = _POOL
    _CHECKPOINTER = None
    _CONTEXT_MANAGER = None
    _POOL = None
    _INIT_ATTEMPTED = False
    _BACKEND = None

    if cm is not None:
        try:
            await cm.__aexit__(None, None, None)
        except Exception:  # noqa: BLE001
            logger.exception("[checkpointer] error closing AsyncSqliteSaver")

    if pool is not None:
        try:
            await pool.close()
        except Exception:  # noqa: BLE001
            logger.exception("[checkpointer] error closing Postgres pool")


def thread_config(thread_id: str) -> dict:
    """
    Helper to build the ``config`` dict that LangGraph's checkpointer
    requires on every ``ainvoke`` / ``astream`` call.

    Centralised so we have ONE place to add ``checkpoint_ns`` / additional
    keys later (e.g., for namespaced multi-tenant deployments in CORE 6).
    """
    return {"configurable": {"thread_id": str(thread_id)}}
