#!/usr/bin/env python3
"""Export the /api contract to docs/api/openapi.json, the file the developers use.

    python scripts/ops/export_openapi.py

Refuses when the code breaks the published contract (a field, route or value
gone, a type changed, a request that needs more): the contract only grows. Run
it after any addition; ``tests/unit/test_api_contract.py`` fails until you do.

    python scripts/ops/export_openapi.py --breaking "why, and who agreed"

exports anyway. Only with the developers' agreement, or for a new /api/v2: a
client built on the published file will fail where it lists.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--breaking", metavar="REASON", help="export a breaking change anyway")
    args = parser.parse_args()
    os.environ.setdefault("AIX_TRACING_ENABLED", "false")
    from aix.api.contract import PUBLISHED, compatibility_problems, integration_contract
    from aix.api.main import app

    current = integration_contract(app)
    published = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    problems = compatibility_problems(published, current)
    if problems and args.breaking:
        print(f"Breaking change exported ({args.breaking}):", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
    elif problems:
        print("Not exported: the code breaks the published contract.", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    with PUBLISHED.open("w", encoding="utf-8", newline="\r\n") as out:
        out.write(json.dumps(current, indent=2, ensure_ascii=False) + "\n")
    print(f"Exported {len(current['paths'])} routes to {PUBLISHED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
