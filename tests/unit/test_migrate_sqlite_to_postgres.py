"""The move of the WebUI's data from SQLite to Postgres refuses what it cannot do.

The copy itself needs a real Postgres: see
``tests/integration/test_postgres_dialect.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SPEC = importlib.util.spec_from_file_location(
    "migrate_sqlite_to_postgres", Path("scripts/ops/migrate_sqlite_to_postgres.py")
)
script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(script)


async def test_a_missing_file_is_refused(tmp_path, capsys):
    code = await script.migrate(tmp_path / "nope.db", "postgresql+asyncpg://x", apply=True)

    assert code == 2
    assert "non trovato" in capsys.readouterr().out


async def test_the_target_must_be_postgres(tmp_path, capsys):
    source = tmp_path / "webui.db"
    source.write_bytes(b"")

    code = await script.migrate(source, "sqlite+aiosqlite:///other.db", apply=True)

    assert code == 2
    assert "Postgres" in capsys.readouterr().out
