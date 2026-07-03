#!/usr/bin/env python
"""
Local dev launcher for the GraphRAG API — Windows + Postgres safe.

Why this exists
---------------
LangGraph's Postgres checkpointer is backed by ``psycopg``'s async driver,
which **refuses** to run on Windows' default ``ProactorEventLoop`` and requires
``WindowsSelectorEventLoopPolicy``.

``aix.api.main`` already switches that policy at import time, but that is not
enough for modern Uvicorn on Windows. Uvicorn 0.36+ passes an explicit loop
factory to its runner; for a plain single-process Windows server that factory
returns ``asyncio.ProactorEventLoop`` directly, ignoring the global event-loop
policy. So even if the app logs that the selector policy was activated, the
live loop can still be a ProactorEventLoop. psycopg then can't connect and the
checkpointer dies with a 30s ``PoolTimeout`` (multi-turn memory degrades to
single-turn).

This launcher sets the selector policy, builds a ``uvicorn.Server`` manually,
and runs ``server.serve()`` inside our own ``asyncio.run()``. That bypasses
Uvicorn's Windows Proactor loop factory while preserving normal Uvicorn server
behaviour, so psycopg connects cleanly and you get
``[checkpointer] AsyncPostgresSaver ready ... multi-turn enabled``.

It is a no-op beyond calling uvicorn on Linux/macOS (production), so the exact
same command works everywhere. Production on Linux never needed this — the
default loop already supports psycopg async.

Usage (PowerShell, from the graphaixlearning/ root)
---------------------------------------------------
    # 1) Start the local dev Postgres (separate from prod):
    docker compose -f deploy/docker-compose.dev-postgres.yml up -d

    # 2) Point the app at it (current shell session only). Use 127.0.0.1, NOT
    #    localhost: on Windows libpq tries IPv6 ::1 first, but Docker publishes
    #    the port on IPv4 127.0.0.1 only, so localhost stalls psycopg's pool.
    #    (The checkpointer also auto-rewrites localhost->127.0.0.1 on Windows
    #    as a safety net, but setting it here keeps both drivers fast.)
    $env:WEBUI_DATABASE_URL     = "postgresql+asyncpg://aix:aixdevpass@127.0.0.1:5432/aix"
    $env:LANGGRAPH_DATABASE_URL = "postgresql://aix:aixdevpass@127.0.0.1:5432/aix"

    # 3) Launch via this script (NOT `python -m uvicorn`):
    python scripts/ops/run_local.py                 # 127.0.0.1:8765
    python scripts/ops/run_local.py --port 9000     # override host/port/log-level

SQLite dev (no Postgres env vars) keeps working unchanged: the policy switch
only activates on Windows, and the default Proactor loop is fine for the
SQLite checkpointer.

Note on --reload
----------------
``--reload`` is intentionally NOT supported here. Uvicorn's reloader/multiprocess
paths own their child-process event-loop setup, which can re-introduce the exact
Windows loop issue this launcher fixes. Restart the process manually after code
changes when validating against Postgres.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Make ``aix`` importable when run from any cwd / without an editable install.
# The launcher lives at <repo>/scripts/ops/run_local.py, so the source root is
# <repo>/src. Mirrors the sys.path insertion aix.api.main does for itself.
_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _maybe_set_windows_selector_policy() -> bool:
    """Switch to the selector loop on Windows so psycopg async works.

    Must be called BEFORE uvicorn creates the event loop. Returns ``True`` if
    the policy was changed (Windows), ``False`` otherwise (no-op on
    Linux/macOS, where the default loop already supports psycopg).
    """
    if sys.platform != "win32":
        return False
    selector_policy = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if selector_policy is None:  # extremely old/odd Python build — safety net
        return False
    asyncio.set_event_loop_policy(selector_policy())
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the GraphRAG API locally with the Windows selector event-loop "
            "policy set before uvicorn starts (required for the Postgres "
            "checkpointer on Windows)."
        )
    )
    parser.add_argument("--host", default=os.getenv("AIX_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("AIX_PORT", "8765")))
    parser.add_argument(
        "--log-level", default=os.getenv("AIX_LOG_LEVEL", "info")
    )
    args = parser.parse_args()

    switched = _maybe_set_windows_selector_policy()

    import uvicorn

    if switched:
        print(
            "[run_local] Windows detected -> WindowsSelectorEventLoopPolicy set "
            "BEFORE uvicorn; psycopg async checkpointer can connect."
        )
    else:
        print(
            "[run_local] Non-Windows platform -> default event loop "
            "(no policy change needed)."
        )

    config = uvicorn.Config(
        "aix.api.main:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
    server = uvicorn.Server(config)

    async def _serve() -> None:
        loop_name = type(asyncio.get_running_loop()).__name__
        print(f"[run_local] Running uvicorn.Server on {loop_name}.")
        await server.serve()

    asyncio.run(_serve())


if __name__ == "__main__":
    main()
