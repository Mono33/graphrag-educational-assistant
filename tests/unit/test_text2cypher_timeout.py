"""A graph query has its own short limit (2026-10-20: one took 99 s and held a lesson)."""

from __future__ import annotations

import pytest

from aix.core.config import text2cypher_timeout

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(("raw", "expected"), [(None, 20.0), ("8", 8.0), ("1", 3.0), ("x", 20.0)])
def test_the_graph_query_limit(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("AIX_TEXT2CYPHER_TIMEOUT_S", raising=False)
    else:
        monkeypatch.setenv("AIX_TEXT2CYPHER_TIMEOUT_S", raw)

    assert text2cypher_timeout() == expected
