"""
End-to-end smoke for the LangGraph Postgres checkpointer (CORE 4 #15.a).

Spins the existing ``get_checkpointer()`` lazy singleton against the URL
in ``LANGGRAPH_DATABASE_URL`` and verifies:

  1. The right backend was opened (``active_backend() == 'postgres'``).
  2. ``saver.setup()`` created the 3 expected tables (``checkpoints``,
     ``checkpoint_blobs``, ``checkpoint_writes``) -- confirmed by
     querying ``information_schema.tables`` directly via psycopg.
  3. A real ``aput`` -> ``aget_tuple`` round-trip on a synthetic thread.
  4. ``close_checkpointer()`` releases the connection cleanly.

Plain-ASCII output only, since the Windows console default codepage is
cp1252 and would otherwise mojibake non-ASCII text. Intended to be
deleted after the migration smoke passes -- kept as a file rather than
inline so it can be re-run if anything regresses.
"""

import asyncio
import logging
import os
import sys
import uuid

# psycopg's async mode rejects Windows' default ProactorEventLoop. Linux
# is unaffected -- set the policy only on Windows. Must be set BEFORE
# asyncio.run() is called.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("smoke_postgres_checkpointer")


async def main() -> int:
    pg_url = os.environ.get("LANGGRAPH_DATABASE_URL")
    if not pg_url or "postgres" not in pg_url:
        print("ERROR: set LANGGRAPH_DATABASE_URL to a postgres URL first.")
        return 2

    # Reset the singleton to be safe in case this is re-run.
    from aix.agent.graph import checkpointer as cp_mod

    cp_mod._CHECKPOINTER = None
    cp_mod._INIT_ATTEMPTED = False
    cp_mod._CONTEXT_MANAGER = None
    cp_mod._BACKEND = None

    print("\n=== STEP 1: get_checkpointer() ===")
    saver = await cp_mod.get_checkpointer()
    backend = cp_mod.active_backend()
    assert saver is not None, "saver should not be None"
    assert backend == "postgres", f"expected backend='postgres', got {backend!r}"
    print(f"   [OK] backend={backend}, saver class={type(saver).__name__}")

    print("\n=== STEP 2: verify 3 tables exist via psycopg ===")
    import psycopg

    expected_tables = {"checkpoints", "checkpoint_blobs", "checkpoint_writes"}
    async with await psycopg.AsyncConnection.connect(pg_url) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' "
                "AND table_name IN ('checkpoints','checkpoint_blobs','checkpoint_writes')"
            )
            rows = await cur.fetchall()
    found = {r[0] for r in rows}
    print(f"   tables found: {sorted(found)}")
    missing = expected_tables - found
    assert not missing, f"missing tables: {missing}"
    print("   [OK] all 3 LangGraph checkpoint tables exist")

    print("\n=== STEP 3: aput -> aget_tuple round-trip ===")
    from langgraph.checkpoint.base import (
        Checkpoint,
        CheckpointMetadata,
        empty_checkpoint,
    )

    thread_id = f"smoke-{uuid.uuid4()}"
    # LangGraph 1.x AsyncPostgresSaver requires checkpoint_ns + checkpoint_id
    # in configurable. checkpoint_ns is "" for the default namespace.
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
        }
    }

    try:
        ckpt: Checkpoint = empty_checkpoint()
    except Exception:
        ckpt = {
            "v": 1,
            "id": str(uuid.uuid4()),
            "ts": "2026-05-16T00:00:00Z",
            "channel_values": {},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }

    ckpt["channel_values"]["smoke_marker"] = "hello-postgres"

    metadata: CheckpointMetadata = {
        "source": "smoke",
        "step": 0,
        "writes": {},
    }
    new_versions = {}

    config_after_put = await saver.aput(config, ckpt, metadata, new_versions)
    cfg_keys = sorted(config_after_put["configurable"].keys())
    print(f"   put OK   -> config_after_put.configurable keys = {cfg_keys}")

    fetched = await saver.aget_tuple(config_after_put)
    assert fetched is not None, "aget_tuple returned None -- round-trip broken"
    fetched_marker = fetched.checkpoint["channel_values"].get("smoke_marker")
    print(f"   get OK   -> fetched marker = {fetched_marker!r}")
    assert fetched_marker == "hello-postgres", "marker did not survive round-trip"
    print("   [OK] checkpoint persisted + retrieved correctly")

    print("\n=== STEP 4: close_checkpointer() ===")
    await cp_mod.close_checkpointer()
    print(f"   active_backend() after close: {cp_mod.active_backend()!r}")
    assert cp_mod.active_backend() is None
    print("   [OK] closed cleanly")

    print("\n*** ALL POSTGRES SMOKE STEPS PASSED ***")
    return 0


if __name__ == "__main__":
    rc = asyncio.run(main())
    sys.exit(rc)
