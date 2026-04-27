"""Quick check that the MCP HTTP app is built and mounted at /mcp/.

Run with the project's venv Python:
    python scripts/diagnostic/inspect_mcp_mount.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

print("=" * 70)
print("STEP 1: Import aix.mcp.http_app directly")
print("=" * 70)
try:
    from aix.mcp.http_app import build_mcp_http_app, MCP_MOUNT_PATH

    print(f"OK — MCP_MOUNT_PATH = {MCP_MOUNT_PATH}")
except Exception as exc:
    print(f"FAIL — {type(exc).__name__}: {exc}")
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 70)
print("STEP 2: Call build_mcp_http_app() directly")
print("=" * 70)
try:
    direct_app = build_mcp_http_app()
    print(f"OK — direct_app = {type(direct_app).__name__ if direct_app else 'None'}")
except Exception as exc:
    print(f"FAIL — {type(exc).__name__}: {exc}")
    traceback.print_exc()

print()
print("=" * 70)
print("STEP 3: Import aix.api.main (full FastAPI app)")
print("=" * 70)
try:
    from aix.api.main import app, _mcp_http_app

    print(f"OK — _mcp_http_app = {type(_mcp_http_app).__name__ if _mcp_http_app else 'None'}")
except Exception as exc:
    print(f"FAIL — {type(exc).__name__}: {exc}")
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 70)
print("STEP 4: Inspect routes")
print("=" * 70)
for route in app.routes:
    path = getattr(route, "path", "") or ""
    if any(k in path for k in ("/mcp", "/auth", "/api/v1")):
        kind = type(route).__name__
        target = (
            getattr(route, "app", None)
            or getattr(route, "endpoint", None)
            or "<router>"
        )
        target_label = type(target).__name__ if hasattr(target, "__class__") else repr(target)
        print(f"  {kind:14s} {path:35s} -> {target_label}")
