"""
Strictly-additive REST surface guard for CORE 5 #20.

#20 promised to add the ``/mcp/`` Streamable HTTP mount WITHOUT changing
any existing public route (``/api/v1/*``, ``/auth/*``, ``/webui/*``). The
``/mcp/`` mount is a Starlette ASGI sub-app, so it does NOT appear in
``/openapi.json`` — meaning the live OpenAPI spec must be a *strict
superset* of the baseline captured before #20 landed.

Baseline file: ``data/diagnostic/openapi_before_p20.txt``
Generator:     ``scripts/diagnostic/capture_openapi_baseline.py``

If a route disappears, this test fails with the missing list.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_BASELINE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "diagnostic" / "openapi_before_p20.txt"
)


def _live_routes(http_client) -> set[str]:
    """Snapshot the live ``METHOD path`` set from /openapi.json."""
    spec = http_client.get("/openapi.json").json()
    return {
        f"{m.upper()} {p}"
        for p, methods in spec.get("paths", {}).items()
        for m in methods
        if m.lower() in {"get", "post", "put", "patch", "delete"}
    }


def _baseline_routes() -> list[str]:
    """Parse the baseline file's ``METHOD path`` lines."""
    if not _BASELINE_PATH.exists():
        pytest.skip(f"baseline missing at {_BASELINE_PATH}")
    expected: list[str] = []
    with _BASELINE_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2 and parts[0] in {
                "GET",
                "POST",
                "PUT",
                "PATCH",
                "DELETE",
            }:
                expected.append(f"{parts[0]} {parts[1]}")
    return expected


#: Routes deliberately withdrawn since the baseline, each with the reason.
#: The baseline file stays a factual record of what the surface was; this is
#: where a considered removal is declared, so it is a decision in a diff rather
#: than a line quietly edited out of the record.
WITHDRAWN: dict[str, str] = {
    "GET /auth/logout": (
        "Logging out changes state; as a GET any third-party page could fire it "
        "at a teacher mid-lesson. Now POST /auth/logout, under CSRF "
        "(review of 2026-09-15)."
    ),
}


def test_p20_strictly_additive(http_client):
    """Every route present before #20 must still be present today."""
    expected = _baseline_routes()
    assert expected, (
        f"baseline at {_BASELINE_PATH} contains no parseable routes — "
        "regenerate via scripts/diagnostic/capture_openapi_baseline.py"
    )

    live = _live_routes(http_client)
    missing = sorted(set(expected) - live - WITHDRAWN.keys())
    assert not missing, (
        "Strictly-additive regression: routes present before CORE 5 #20 are "
        "now MISSING from the live spec. This is a backward-compat break.\n"
        "  Missing routes:\n    - " + "\n    - ".join(missing)
    )


def test_mcp_path_is_not_in_openapi(http_client):
    """``/mcp/`` is an ASGI mount — it MUST NOT appear in /openapi.json.

    If it shows up, FastMCP started exposing OpenAPI metadata for the mount
    point and we'd need to update the baseline parsing logic. Catching
    this drift early avoids confusing future regressions.
    """
    spec = http_client.get("/openapi.json").json()
    paths = set(spec.get("paths", {}).keys())
    assert "/mcp" not in paths and "/mcp/" not in paths, (
        f"Unexpected: /mcp surfaced in /openapi.json. Spec paths: {sorted(paths)}"
    )
