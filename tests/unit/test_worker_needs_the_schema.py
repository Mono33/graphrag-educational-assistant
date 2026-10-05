"""The ingestion worker starts only on a database at the latest migration.

Found by the rehearsal of the staging upgrade (2026-10-05): on a Postgres
without pgvector the migrations roll back -- by design, the API still serves --
but the worker, now on by default, started anyway and failed every three
seconds on ``ingestion_job``, a table that did not exist. With error reporting
on, that is twelve hundred reports an hour. It now says so once and waits.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from aix.webui import db

pytestmark = pytest.mark.unit


def test_the_head_is_a_migration_shipped_in_the_image():
    head = db.head_revision()

    shipped = [p.name for p in Path("alembic/versions").glob("*.py")]
    assert any(f"-{head}_" in name for name in shipped)


@pytest.mark.parametrize(
    "live, expected",
    [("HEAD", True), ("c8d88f37e59c", False), (None, False)],
)
async def test_current_only_at_the_head(live, expected, monkeypatch):
    revision = db.head_revision() if live == "HEAD" else live
    monkeypatch.setattr(db, "current_db_revision", AsyncMock(return_value=revision))

    assert await db.schema_is_current() is expected


async def test_an_unreadable_database_is_not_current(monkeypatch):
    monkeypatch.setattr(db, "current_db_revision", AsyncMock(side_effect=RuntimeError("down")))

    assert await db.schema_is_current() is False
