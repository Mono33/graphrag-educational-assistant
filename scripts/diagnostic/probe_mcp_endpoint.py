"""Static probes for the /mcp/ Streamable HTTP mount.

Asserts:
  1. GET /api/v1/health is 200 (existing behaviour intact).
  2. GET /docs is 200 (Swagger UI still mounts).
  3. POST /mcp/ without auth returns 401 (auth gating works).

Does NOT log in — credentials-free smoke. Run after starting uvicorn:
    uvicorn aix.api.main:app --port 8765 --app-dir src
"""

from __future__ import annotations

import sys

import urllib.error
import urllib.request


BASE = "http://127.0.0.1:8765"


def _probe(method: str, path: str, body: bytes | None = None, headers: dict[str, str] | None = None) -> tuple[int, str]:
    req = urllib.request.Request(
        f"{BASE}{path}",
        method=method,
        data=body,
        headers=headers or {},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read(200).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, (exc.read(200).decode("utf-8", "replace") if exc.fp else "")
    except Exception as exc:
        return -1, f"{type(exc).__name__}: {exc}"


def main() -> int:
    failures: list[str] = []

    print(f"Probing {BASE}")
    print("-" * 60)

    code, body = _probe("GET", "/api/v1/health")
    print(f"GET  /api/v1/health           -> {code}")
    if code != 200:
        failures.append(f"  /api/v1/health expected 200, got {code}: {body}")

    code, body = _probe("GET", "/docs")
    print(f"GET  /docs                    -> {code}")
    if code != 200:
        failures.append(f"  /docs expected 200, got {code}")

    code, body = _probe("GET", "/openapi.json")
    print(f"GET  /openapi.json            -> {code}")
    if code != 200:
        failures.append(f"  /openapi.json expected 200, got {code}")

    json_rpc = b'{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
    code, body = _probe(
        "POST",
        "/mcp/",
        body=json_rpc,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
    )
    print(f"POST /mcp/  (no auth)         -> {code}")
    print(f"     body[:160] = {body[:160]!r}")
    if code != 401:
        failures.append(f"  /mcp/ unauth expected 401, got {code}: {body[:200]}")

    print("-" * 60)
    if failures:
        print("FAIL:")
        for f in failures:
            print(f)
        return 1
    print("PASS — Phase 5 static probes all green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
