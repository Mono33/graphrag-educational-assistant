"""One-shot helper: print every path advertised in the live OpenAPI spec.

Used to answer the question "what does ``/docs`` already expose?" when
investigating CORE 2 #7 scope (FastAPI JSON+SSE Agent Endpoint).

Usage:
    python scripts/diagnostic/list_openapi_paths.py
    python scripts/diagnostic/list_openapi_paths.py --base http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8765",
                        help="API base URL (default: %(default)s)")
    args = parser.parse_args()

    url = f"{args.base.rstrip('/')}/openapi.json"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            spec = json.load(resp)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not fetch {url} — {exc}", file=sys.stderr)
        return 1

    info = spec.get("info", {})
    print(f"TITLE  : {info.get('title')}")
    print(f"VERSION: {info.get('version')}")
    print("-" * 60)

    paths = spec.get("paths", {})
    rows = []
    for path, methods in sorted(paths.items()):
        for method, op in methods.items():
            if method.lower() not in {"get", "post", "put", "patch", "delete"}:
                continue
            tags = ",".join(op.get("tags", [])) or "-"
            rows.append((method.upper(), path, tags))

    width = max(len(p) for _, p, _ in rows) if rows else 0
    for method, path, tags in rows:
        print(f"{method:6s} {path:<{width}}  [tags: {tags}]")

    print("-" * 60)
    print(f"Total operations: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
