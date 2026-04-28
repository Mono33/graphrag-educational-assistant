"""
Snapshot the live OpenAPI inventory of the public FastAPI app to a text file.

Used to refresh ``data/diagnostic/openapi_*.txt`` baselines whenever a phase
boundary lands and we want to lock the current REST surface as the reference
for the next regression-guard test.

Usage:
    # In-process — no uvicorn required:
    python scripts/diagnostic/capture_openapi_baseline.py \
        --out data/diagnostic/openapi_before_p20.txt \
        --label "captured before CORE 5 #20 Phase 6 baseline"

The output format mirrors ``openapi_before_p7.txt`` so existing regression
tests (``test_agent_routes.test_openapi_inventory_strictly_additive``) and
new ones can share the same parser.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple


def _import_app():
    """Import the live FastAPI app with safe defaults so it boots without .env."""
    os.environ.setdefault("WEBUI_AUTH_SECRET", "dev-fallback-for-baseline-capture")
    os.environ.setdefault("WEBUI_CORS_ALLOW_ORIGINS", "*")
    os.environ.setdefault("AIX_MCP_REQUIRE_AUTH", "1")

    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from aix.api.main import app  # type: ignore  # noqa: E402

    return app


def _collect_paths(app) -> Tuple[str, str, List[Tuple[str, str, List[str]]]]:
    """Return (title, version, [(method, path, tags)]) sorted alphabetically."""
    spec = app.openapi()
    title = str(spec.get("info", {}).get("title", "<unknown>"))
    version = str(spec.get("info", {}).get("version", "<unknown>"))
    rows: List[Tuple[str, str, List[str]]] = []
    for path, methods in spec.get("paths", {}).items():
        for method, op in methods.items():
            if method.lower() not in {"get", "post", "put", "patch", "delete"}:
                continue
            tags = list(op.get("tags") or [])
            rows.append((method.upper(), path, tags))
    rows.sort(key=lambda r: (r[1], r[0]))
    return title, version, rows


def _format(title: str, version: str, rows, label: str) -> str:
    width_path = max((len(p) for _, p, _ in rows), default=20) + 2
    lines: List[str] = [
        f"TITLE  : {title}",
        f"VERSION: {version}",
    ]
    if label:
        lines.append(f"BASELINE: {label}")
    lines.append("-" * 60)
    for method, path, tags in rows:
        tag_str = ",".join(tags) if tags else "-"
        lines.append(f"{method:<6} {path:<{width_path}}[tags: {tag_str}]")
    lines.append("-" * 60)
    lines.append(f"Total operations: {len(rows)}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output text file path")
    parser.add_argument(
        "--label",
        default="",
        help="Free-text annotation (date, phase, intent) embedded in the header.",
    )
    args = parser.parse_args()

    app = _import_app()
    title, version, rows = _collect_paths(app)
    text = _format(title, version, rows, args.label)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")

    print(f"[OK] wrote {len(rows)} operations to {out_path}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
