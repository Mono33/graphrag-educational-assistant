#!/usr/bin/env python3
"""Copy the WebUI's users and lessons from SQLite into Postgres.

Why this exists
---------------
Until October 2026 the deployed app ran without ``WEBUI_DATABASE_URL``, so the
WebUI kept its data in SQLite inside the container
(``/graphrag-aixlearning/data/webui/webui.db``). The move to Postgres with
pgvector starts from an empty database. This carries over what a teacher would
miss: accounts, saved class profiles, lessons and their conversations.

What it copies, in this order: ``user``, ``saved_profile``, ``lesson``,
``lesson_message``. Nothing else: the in-flight run registry is empty by
definition, the LangGraph checkpoints only speed up a follow-up turn (a
lesson's history is in ``lesson_message``), and the old version had no
materials.

It is safe to run twice. A row whose id is already in Postgres is skipped,
never overwritten.

Procedure (see ``docs/product/Deploy_Staging_Checklist.md``):

1. before replacing the old container, copy its SQLite file out:
   ``docker cp <old>:/graphrag-aixlearning/data/webui/webui.db ./webui.db``;
2. start the new image with ``WEBUI_DATABASE_URL`` on Postgres. It creates the
   schema at startup; ``/api/v1/health`` shows ``db_revision``;
3. copy the file in and run this:
   ``docker cp ./webui.db <new>:/tmp/webui.db``
   ``docker exec <new> python scripts/ops/migrate_sqlite_to_postgres.py /tmp/webui.db``
   (dry run), then the same with ``--apply``.

The target is ``WEBUI_DATABASE_URL`` -- the container already has it -- or
``--target``.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

#: Parents before children: the foreign keys must find their row.
TABLES = ("user", "saved_profile", "lesson", "lesson_message")
BATCH = 500


def _aware(value):
    """SQLite keeps timestamps without a zone; they were written in UTC."""
    if isinstance(value, datetime) and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


async def _source_columns(engine, table: str) -> set[str]:
    from sqlalchemy import inspect

    async with engine.connect() as conn:
        return await conn.run_sync(lambda c: {col["name"] for col in inspect(c).get_columns(table)})


async def migrate(sqlite_path: Path, target_url: str, *, apply: bool) -> int:
    from sqlalchemy import select, text
    from sqlalchemy.dialects.postgresql import insert
    from sqlalchemy.ext.asyncio import create_async_engine

    from aix.webui.auth import models as _auth_models  # noqa: F401 — registers the tables
    from aix.webui.db import Base, head_revision
    from aix.webui.lessons import models as _lesson_models  # noqa: F401
    from aix.webui.materials import models as _material_models  # noqa: F401

    if not sqlite_path.is_file():
        print(f"File SQLite non trovato: {sqlite_path}")
        return 2
    if not target_url.startswith("postgresql"):
        print("La destinazione deve essere Postgres (WEBUI_DATABASE_URL o --target).")
        return 2

    source = create_async_engine(f"sqlite+aiosqlite:///{sqlite_path.as_posix()}")
    target = create_async_engine(target_url)
    try:
        async with target.connect() as conn:
            live = await conn.scalar(text("select version_num from alembic_version"))
    except Exception:  # noqa: BLE001 — no table yet: the schema was never created
        live = None
    if live != head_revision():
        print(
            f"Il database Postgres non ha ancora lo schema (revisione {live!r}, attesa "
            f"{head_revision()!r}). Avvia prima la nuova immagine su questo database: "
            "crea le tabelle all'avvio."
        )
        await source.dispose()
        await target.dispose()
        return 2

    total_new = 0
    for name in TABLES:
        table = Base.metadata.tables[name]
        present = await _source_columns(source, name)
        if not present:
            print(f"  {name:16} assente nel file SQLite, saltata")
            continue
        columns = [c for c in table.columns if c.name in present]
        async with source.connect() as conn:
            rows = [
                {col.name: _aware(value) for col, value in zip(columns, row, strict=True)}
                for row in (await conn.execute(select(*columns))).all()
            ]
        async with target.connect() as conn:
            known = set((await conn.execute(select(table.c.id))).scalars())
        fresh = [row for row in rows if row["id"] not in known]
        print(
            f"  {name:16} {len(rows):6} nel file SQLite, {len(rows) - len(fresh):6} già in "
            f"Postgres, {len(fresh):6} da copiare"
        )
        if apply and fresh:
            async with target.begin() as conn:
                for start in range(0, len(fresh), BATCH):
                    await conn.execute(
                        insert(table).on_conflict_do_nothing(index_elements=["id"]),
                        fresh[start : start + BATCH],
                    )
        total_new += len(fresh)

    await source.dispose()
    await target.dispose()
    if apply:
        print(f"Copiate {total_new} righe.")
    else:
        print(f"Prova a vuoto: {total_new} righe da copiare. Ripeti con --apply per copiarle.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sqlite", type=Path, help="the old webui.db")
    parser.add_argument("--target", default=os.getenv("WEBUI_DATABASE_URL", ""))
    parser.add_argument("--apply", action="store_true", help="copy; without it, only count")
    args = parser.parse_args()
    return asyncio.run(migrate(args.sqlite, args.target, apply=args.apply))


if __name__ == "__main__":
    raise SystemExit(main())
