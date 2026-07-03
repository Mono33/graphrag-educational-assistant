"""
Integration tests for the public Agent JSON+SSE API (CORE 2 #7).

These tests use FastAPI's ``TestClient`` against the real app instance
(``aix.api.main:app``) so we exercise the entire request/response chain
exactly the way Swagger UI / curl / the AixLearning embed would. We do
**not** mock the HTTP transport.

What we *do* mock: the agent runtime itself
(``aix.webui.agent.service.stream_agent_events``). A real agent run
takes 60-120s, depends on OpenRouter + Neo4j + a populated KG, and is
not a unit test — it's an end-to-end smoke. The smoke is covered by the
webui (you click "Invia" in /webui/lesson/{id}); the test below locks
the *public contract shape* (auth, schema validation, status codes,
serialisation) so any future agent-layer refactor cannot silently
break partner integrations.

Coverage:
    * ``test_agent_run_requires_auth`` — 401 without a token.
    * ``test_agent_run_validates_payload`` — 422 on bad request body.
    * ``test_agent_run_happy_path`` — 200, full ``AgentRunResponse``
      shape, planner + retriever + meta populated when the underlying
      stream emits ``planner / retriever / done`` events.
    * ``test_agent_run_propagates_runtime_error`` — agent ``error``
      event surfaces as HTTP 502 with the message preserved.
    * ``test_agent_stream_serialises_events`` — SSE stream emits one
      ``event:`` line + ``data:`` JSON line per ``StreamEvent``.
    * ``test_openapi_inventory_strictly_additive`` — the /openapi.json
      surface still contains every route present before #7 landed
      (regression guard around the strict-backward-compat promise made
      in the PR).

Auth: tests register a fresh user via fastapi-users' ``UserManager`` so
they don't depend on test fixtures or seeded data. The same JWT secret
backs cookie + Bearer transports, so a token minted via the UserManager
+ JWTStrategy is valid on either.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Optional

import pytest

# Ensure dev defaults so the tests can boot ``aix.api.main`` without a
# fully-populated .env. We set these BEFORE importing the app.
os.environ.setdefault("WEBUI_AUTH_SECRET", "test-secret-for-pytest")
os.environ.setdefault("WEBUI_CORS_ALLOW_ORIGINS", "*")
# Make sure the DB lives in a tmp location so the test never collides with
# the dev SQLite. The webui module reads WEBUI_DB_PATH at import.
os.environ.setdefault(
    "WEBUI_DB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "webui", "test_p7.db"),
)


# ---------------------------------------------------------------------------
# Lazy app import — done inside the fixture so env vars above stick.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app():
    """Import the live FastAPI app once per test module."""
    # ``importlib.reload`` is intentionally avoided — the module-level
    # state (DB engine, fastapi-users wiring) is fine to share across
    # tests inside this module since each test owns its own user.
    from aix.api.main import app as live_app
    return live_app


@pytest.fixture
def client(app):
    """A regular httpx-based TestClient. Starlette's lifespan is exercised."""
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Auth helper — register + login via fastapi-users primitives.
# ---------------------------------------------------------------------------


@pytest.fixture
async def fresh_user_token(app) -> str:
    """
    Register a fresh user and return a Bearer JWT.

    Uses the same UserManager + JWTStrategy the live routes use, so the
    resulting token is interchangeable with a token minted via
    ``POST /auth/jwt/login`` against the same app.
    """
    from aix.webui.auth.backend import get_jwt_strategy
    from aix.webui.auth.dependencies import fastapi_users  # noqa: F401 — ensure side effects
    from aix.webui.auth.manager import UserManager, get_user_manager
    from aix.webui.auth.schemas import UserCreate
    from aix.webui.db import get_async_session

    email = f"pytest+{uuid.uuid4().hex[:8]}@example.com"
    password = "test-password-1234"

    async def _make_token() -> str:
        # Walk the dependency chain manually — same plumbing the route
        # uses, just without going through HTTP.
        async for session in get_async_session():
            from fastapi_users_db_sqlalchemy import SQLAlchemyUserDatabase

            from aix.webui.auth.models import User as UserModel

            user_db = SQLAlchemyUserDatabase(session, UserModel)
            manager = UserManager(user_db)
            user = await manager.create(
                UserCreate(email=email, password=password, display_name="pytest user"),
                safe=True,
            )
            strategy = get_jwt_strategy()
            return await strategy.write_token(user)

        raise RuntimeError("get_async_session yielded nothing")

    return await _make_token()


# ---------------------------------------------------------------------------
# Stream-event mocking helper
# ---------------------------------------------------------------------------


def _patch_stream(monkeypatch, events: list[dict[str, Any]]) -> None:
    """
    Replace ``stream_agent_events`` with a generator that emits the given
    pre-canned events. Each event dict is converted to a real
    ``StreamEvent`` so the route layer's serialisation paths are tested
    end-to-end (we only mock the LangGraph engine, not the route logic).
    """
    from aix.webui.agent.service import StreamEvent

    async def _fake(**_kwargs) -> AsyncIterator[StreamEvent]:
        for e in events:
            yield StreamEvent(
                kind=e["kind"],
                payload=e.get("payload", {}),
                lesson_plan_md=e.get("lesson_plan_md"),
                error=e.get("error"),
                meta=e.get("meta", {}),
            )

    # Patch in BOTH the source module AND the route module so import
    # binding doesn't matter.
    import aix.api.routes.agent as agent_route_mod
    import aix.webui.agent.service as service_mod

    monkeypatch.setattr(service_mod, "stream_agent_events", _fake)
    monkeypatch.setattr(agent_route_mod, "stream_agent_events", _fake)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_agent_run_requires_auth(client):
    """No Authorization header / cookie → 401."""
    resp = client.post(
        "/api/v1/agent/run",
        json={"query": "Crea una lezione su attenzione", "domain": "neuro"},
    )
    assert resp.status_code == 401, resp.text


def test_agent_stream_requires_auth(client):
    """No Authorization header / cookie → 401 on the SSE route too."""
    resp = client.post(
        "/api/v1/agent/stream",
        json={"query": "Crea una lezione su attenzione", "domain": "neuro"},
    )
    assert resp.status_code == 401, resp.text


@pytest.mark.asyncio
async def test_agent_run_validates_payload(client, fresh_user_token):
    """Missing required fields / out-of-enum domain → 422."""
    headers = {"Authorization": f"Bearer {fresh_user_token}"}

    # Missing required ``query``.
    resp = client.post(
        "/api/v1/agent/run",
        headers=headers,
        json={"domain": "neuro"},
    )
    assert resp.status_code == 422, resp.text

    # Bad enum on ``domain``.
    resp = client.post(
        "/api/v1/agent/run",
        headers=headers,
        json={"query": "test", "domain": "not_a_real_domain"},
    )
    assert resp.status_code == 422, resp.text

    # ``query`` too short.
    resp = client.post(
        "/api/v1/agent/run",
        headers=headers,
        json={"query": "x", "domain": "neuro"},
    )
    assert resp.status_code == 422, resp.text


@pytest.mark.asyncio
async def test_agent_run_happy_path(client, fresh_user_token, monkeypatch):
    """
    With a stubbed agent emitting planner→retriever→writer→critic→done,
    the route returns a 200 with the full ``AgentRunResponse``.
    """
    _patch_stream(monkeypatch, [
        {"kind": "planner", "payload": {
            "intent": "lesson_creation",
            "intent_label": "Creazione lezione",
            "scope": "in_scope",
            "scope_label": "Nel Knowledge Graph",
            "key_concepts": ["attenzione", "memoria"],
            "search_queries": ["strategie attenzione"],
        }},
        {"kind": "retriever", "payload": {
            "nodes_count": 7,
            "relationships_count": 12,
            "recommendations_count": 3,
            "media_counts": {"videos": 2, "articles": 1, "oer": 4},
            "media": {"videos": [], "resources": []},
            "top_concepts": ["Attenzione sostenuta"],
            "retrieval_confidence": "HIGH",
        }},
        {"kind": "writer_pending", "payload": {"revision": 1}},
        {"kind": "writer", "payload": {"revision": 1},
         "lesson_plan_md": "# Bozza"},
        {"kind": "critic", "payload": {"approved": True, "score": 4.5}},
        {"kind": "done",
         "lesson_plan_md": "# Lezione finale\n\nContenuto.",
         "meta": {
             "duration_seconds": 12.3,
             "approved": True,
             "revision_count": 0,
             "scores": {"average_score": 4.5},
             "nodes_count": 7,
             "recommendations_count": 3,
             "media_counts": {"videos": 2, "articles": 1, "oer": 4},
             "search_queries_count": 1,
         }},
    ])

    resp = client.post(
        "/api/v1/agent/run",
        headers={"Authorization": f"Bearer {fresh_user_token}"},
        json={"query": "Crea una lezione su attenzione", "domain": "neuro"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Top-level shape locked.
    assert set(body.keys()) >= {"lesson_plan_md", "meta", "planner", "retriever"}
    assert body["lesson_plan_md"] == "# Lezione finale\n\nContenuto."
    assert body["meta"]["approved"] is True
    assert body["meta"]["duration_seconds"] == pytest.approx(12.3)
    assert body["meta"]["media_counts"] == {"videos": 2, "articles": 1, "oer": 4}

    # Planner explainability flowed through.
    assert body["planner"]["intent"] == "lesson_creation"
    assert body["planner"]["key_concepts"] == ["attenzione", "memoria"]

    # Retriever explainability flowed through.
    assert body["retriever"]["nodes_count"] == 7
    assert body["retriever"]["retrieval_confidence"] == "HIGH"


@pytest.mark.asyncio
async def test_agent_run_propagates_runtime_error(client, fresh_user_token, monkeypatch):
    """An ``error`` StreamEvent → HTTP 502 with the message preserved."""
    _patch_stream(monkeypatch, [
        {"kind": "error", "error": "Knowledge Graph unreachable"},
    ])

    resp = client.post(
        "/api/v1/agent/run",
        headers={"Authorization": f"Bearer {fresh_user_token}"},
        json={"query": "Crea una lezione", "domain": "neuro"},
    )
    assert resp.status_code == 502, resp.text
    assert resp.json()["detail"] == "Knowledge Graph unreachable"


@pytest.mark.asyncio
async def test_agent_stream_serialises_events(client, fresh_user_token, monkeypatch):
    """
    SSE stream emits one ``event:`` + ``data:`` pair per StreamEvent, with
    the JSON ``data`` payload matching the public envelope shape.
    """
    _patch_stream(monkeypatch, [
        {"kind": "planner", "payload": {"intent": "lesson_creation"}},
        {"kind": "done",
         "lesson_plan_md": "# Done",
         "meta": {"duration_seconds": 1.0, "approved": True,
                  "revision_count": 0, "nodes_count": 0,
                  "recommendations_count": 0,
                  "media_counts": {"videos": 0, "articles": 0, "oer": 0},
                  "search_queries_count": 0}},
    ])

    with client.stream(
        "POST",
        "/api/v1/agent/stream",
        headers={"Authorization": f"Bearer {fresh_user_token}"},
        json={"query": "Crea una lezione", "domain": "neuro"},
    ) as resp:
        assert resp.status_code == 200, resp.read()
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = resp.read().decode("utf-8")

    # Parse the SSE frames. We expect at least one ``event: planner`` and
    # one ``event: done``. Frames are separated by blank lines.
    frames: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                frames.append(current)
                current = {}
            continue
        if line.startswith(":"):
            continue  # SSE comment / ping
        if ":" in line:
            key, _, value = line.partition(":")
            current[key.strip()] = value.lstrip()
    if current:
        frames.append(current)

    kinds = [f.get("event") for f in frames if "event" in f]
    assert "planner" in kinds, kinds
    assert "done" in kinds, kinds

    # Each ``data`` line is JSON and matches the public envelope shape.
    for frame in frames:
        if "data" not in frame:
            continue
        payload = json.loads(frame["data"])
        assert set(payload.keys()) >= {"kind", "data"}


def test_openapi_inventory_strictly_additive(client):
    """
    Regression guard for the strict-backward-compat promise made by #7.
    Every route present in ``data/diagnostic/openapi_before_p7.txt``
    must still be present in the live spec.
    """
    spec = client.get("/openapi.json").json()
    live_routes = {
        f"{m.upper()} {p}"
        for p, methods in spec["paths"].items()
        for m in methods
        if m.lower() in {"get", "post", "put", "patch", "delete"}
    }

    baseline_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "data",
        "diagnostic",
        "openapi_before_p7.txt",
    )
    if not os.path.exists(baseline_path):
        pytest.skip(f"baseline inventory missing at {baseline_path}")

    expected: list[str] = []
    with open(baseline_path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2 and parts[0] in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
                expected.append(f"{parts[0]} {parts[1]}")

    missing = sorted(set(expected) - live_routes)
    assert not missing, (
        "Backward-compat regression: routes present before #7 are now missing: "
        + ", ".join(missing)
    )


# ---------------------------------------------------------------------------
# Asyncio plumbing — pytest-asyncio is not in requirements.txt yet, so we
# fall back to a tiny event-loop-per-async-test shim. If pytest-asyncio
# *is* installed (some devs have it via -dev requirements) the real
# decorator wins; otherwise this shim makes ``@pytest.mark.asyncio``
# a no-op and we manage the loop ourselves below.
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(config, items):
    """Run async tests on a fresh event loop if pytest-asyncio is missing."""
    try:
        import pytest_asyncio  # noqa: F401
        return
    except ImportError:
        pass

    for item in items:
        if not asyncio.iscoroutinefunction(item.function):
            continue
        original = item.function

        def _runner(*args, _orig=original, **kwargs):
            return asyncio.run(_orig(*args, **kwargs))

        item.function = _runner
        item.obj = _runner
