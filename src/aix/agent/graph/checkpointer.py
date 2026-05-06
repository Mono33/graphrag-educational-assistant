"""
LangGraph checkpointer singleton (CORE 2 #10.2).

Why this module exists
----------------------
LangGraph's checkpointer abstraction (``BaseCheckpointSaver``) is what makes
multi-turn conversation memory possible. Every ``ainvoke`` / ``astream`` call
that carries a ``thread_id`` config writes the post-step ``AgentState`` to
the saver, and reads it back on the next call with the same ``thread_id``.
Time-travel (``aget_state_history``), branching (``aupdate_state``), and
human-in-the-loop interrupts (CORE 4 #19) all build on this primitive.

We use ``AsyncSqliteSaver`` in dev and switch to ``AsyncPostgresSaver`` in
production (CORE 4 #15). Both speak the same protocol — only the saver
class instantiation changes.

Design choices
--------------
1. **Process-level singleton.** The agent pipeline is invoked from many
   places (webui SSE, public JSON+SSE API, MCP server, CLI), and they
   should all share one saver instance — anything else fragments the
   thread store. We initialise lazily on first access and keep the
   reference for the process lifetime.

2. **Lazy first-access init (not lifespan).** A FastAPI lifespan hook would
   be cleaner, but it would require touching ``aix.api.main`` and adding
   a startup ordering dependency between the agent module and the API
   surface. The lazy pattern keeps the agent layer self-contained for V1.
   #15 (Postgres migration) is the right place to move to lifespan-managed
   pooling — at that point we want explicit setup/teardown for the
   ``psycopg_pool.AsyncConnectionPool``, and lifespan is the natural fit.

3. **Graceful degradation if not installed.** ``langgraph-checkpoint-sqlite``
   is pinned in ``requirements.txt`` but tests / CI / older checkouts may
   not have it yet. ``get_checkpointer()`` returns ``None`` instead of
   crashing — the graph compiles without persistence, and runs continue
   to work in single-turn mode (the pre-#10 behaviour).

4. **DB path under ``data/agent_threads.db``** — same convention as the
   webui's SQLite (``data/webui/webui.db``). Configurable via the
   ``LANGGRAPH_CHECKPOINTER_URL`` env var so #15's Postgres swap and ops
   relocations don't need a code change.

5. **Connection lifetime is the process lifetime.** We deliberately don't
   close the connection on app shutdown in V1 — uvicorn's process exit
   releases the file handle, and SQLite's WAL is crash-safe. #15 will
   add explicit close hooks when the connection becomes a Postgres pool
   that benefits from graceful shutdown.

Usage
-----
::

    from aix.agent.graph.checkpointer import get_checkpointer

    # Anywhere on the async path (e.g., inside a node, a service helper,
    # or graph compile setup).
    saver = await get_checkpointer()
    if saver is not None:
        graph = workflow.compile(checkpointer=saver)
    else:
        graph = workflow.compile()  # single-turn fallback

The caller is responsible for passing the per-invocation thread_id
config — the saver doesn't know which thread you mean::

    async for chunk in graph.astream(
        initial_state,
        config={"configurable": {"thread_id": str(lesson.id)}},
        stream_mode="updates",
    ):
        ...
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
# We intentionally hold a reference to the AsyncSqliteSaver context manager
# so its ``__aexit__`` can run on interpreter shutdown if anyone bothers
# to call ``close_checkpointer()``. We don't auto-close otherwise — see
# design choice #5 above.
_CONTEXT_MANAGER: Optional[Any] = None


# ---------------------------------------------------------------------------
# DB path resolution
# ---------------------------------------------------------------------------

def _resolve_db_path() -> Path:
    """
    Default: ``<repo_root>/data/agent_threads.db``.

    Override via ``LANGGRAPH_CHECKPOINTER_URL`` env var for ops relocations
    or staging environments. The URL is parsed liberally:
    ``sqlite:///<abs_path>``, ``sqlite+aiosqlite:///<abs_path>``, and bare
    paths all work (we strip the scheme prefix).
    """
    override = os.getenv("LANGGRAPH_CHECKPOINTER_URL")
    if override:
        # Strip any scheme prefix; AsyncSqliteSaver.from_conn_string takes
        # a plain filesystem path (or ``:memory:``).
        for prefix in ("sqlite+aiosqlite:///", "sqlite:///"):
            if override.startswith(prefix):
                return Path(override[len(prefix):])
        return Path(override)

    # Default location, mirroring the webui DB convention. Resolve relative
    # to the package root so the path is stable regardless of cwd uvicorn
    # was launched from.
    package_dir = Path(__file__).resolve().parents[3]   # …/src/aix → …/src
    repo_root = package_dir.parent                       # …/graphaixlearning
    return repo_root / "data" / "agent_threads.db"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def get_checkpointer() -> Optional[Any]:
    """
    Return the lazily-initialised AsyncSqliteSaver singleton, or ``None``
    if ``langgraph-checkpoint-sqlite`` is not installed (graceful
    degradation — see design choice #3).

    First-access semantics:
        1. Acquire ``_INIT_LOCK`` so concurrent first-callers don't race.
        2. Re-check the singleton (another caller may have won the race).
        3. Re-check ``_INIT_ATTEMPTED`` — if a previous attempt failed,
           short-circuit to ``None`` rather than retrying every call.
        4. Import the saver class. ``ImportError`` → log + return None.
        5. Create the parent directory, open the saver via ``from_conn_string``,
           call ``setup()`` (idempotent — creates the 3 checkpoint tables),
           and stash the result.

    Returns the saver on success, ``None`` on graceful failure.
    """
    global _CHECKPOINTER, _INIT_ATTEMPTED, _CONTEXT_MANAGER

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

        try:
            # Local import — keeps the agent module importable when
            # langgraph-checkpoint-sqlite is missing (test envs, fresh
            # checkouts before pip install).
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
                "[checkpointer] cannot create parent dir for %s: %s; "
                "skipping checkpointer init",
                db_path, exc,
            )
            return None

        try:
            # ``from_conn_string`` returns an async context manager that
            # owns the underlying aiosqlite connection. We deliberately
            # ``__aenter__`` it without a matching ``__aexit__`` for the
            # process lifetime — see design choice #5.
            cm = AsyncSqliteSaver.from_conn_string(str(db_path))
            saver = await cm.__aenter__()
        except Exception as exc:  # noqa: BLE001 — surface any backend init failure
            logger.exception(
                "[checkpointer] failed to open AsyncSqliteSaver at %s: %s",
                db_path, exc,
            )
            return None

        _CONTEXT_MANAGER = cm
        _CHECKPOINTER = saver
        logger.info(
            "[checkpointer] AsyncSqliteSaver ready at %s (multi-turn enabled)",
            db_path,
        )
        return _CHECKPOINTER


async def close_checkpointer() -> None:
    """
    Best-effort shutdown hook for the singleton. Safe to call multiple
    times; safe to call when the checkpointer was never initialised.

    Not wired into FastAPI lifespan today — see design choice #5. Tests
    that want a clean reset can call this between cases.
    """
    global _CHECKPOINTER, _INIT_ATTEMPTED, _CONTEXT_MANAGER

    cm = _CONTEXT_MANAGER
    _CHECKPOINTER = None
    _CONTEXT_MANAGER = None
    _INIT_ATTEMPTED = False

    if cm is None:
        return
    try:
        await cm.__aexit__(None, None, None)
    except Exception:  # noqa: BLE001
        logger.exception("[checkpointer] error closing AsyncSqliteSaver")


def thread_config(thread_id: str) -> dict:
    """
    Helper to build the ``config`` dict that LangGraph's checkpointer
    requires on every ``ainvoke`` / ``astream`` call.

    Centralised so we have ONE place to add ``checkpoint_ns`` / additional
    keys later (e.g., for namespaced multi-tenant deployments in CORE 6).
    """
    return {"configurable": {"thread_id": str(thread_id)}}
