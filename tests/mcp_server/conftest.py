"""
Shared fixtures for the CORE 5 #20 Phase 6 MCP test suite.

Design goals:
* No external services (no Neo4j, no OpenRouter, no live uvicorn).
* Reuse the *same* JWT secret + audience the real ``aix.webui.auth`` stack
  uses, so the Bearer token we mint is interchangeable with one from
  ``POST /auth/jwt/login``.
* Boot the FastAPI app exactly once per test session and share a single
  in-memory ``FastMCP`` instance across all MCP tests — this keeps the
  suite at <10s wall clock.

Fixtures provided:
    * ``mcp_server``          — the shared ``FastMCP`` instance
                                (built once via ``build_mcp_server``).
    * ``mcp_client``          — a connected FastMCP ``Client`` using
                                in-memory transport (no HTTP, no stdio).
    * ``app``                 — the live ``aix.api.main:app``.
    * ``http_client``         — Starlette ``TestClient`` against ``app``
                                (lifespan exercised — MCP HTTP mount alive).
    * ``fresh_user_token``    — newly registered user → Bearer JWT.
    * ``invalid_jwt_token``   — a syntactically valid JWT signed with the
                                wrong secret — used to assert auth rejects it.
"""

from __future__ import annotations

import os
import sys
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Environment defaults — set BEFORE importing any aix.* modules so they
# pick up our test-friendly values rather than whatever the dev .env has.
# ---------------------------------------------------------------------------
os.environ.setdefault("WEBUI_AUTH_SECRET", "test-secret-for-pytest-mcp-phase6")
os.environ.setdefault("WEBUI_CORS_ALLOW_ORIGINS", "*")
os.environ.setdefault("AIX_MCP_REQUIRE_AUTH", "1")
os.environ.setdefault(
    "WEBUI_DB_PATH",
    str(
        Path(__file__).resolve().parents[2]
        / "data"
        / "webui"
        / "test_p20_mcp.db"
    ),
)

# Ensure ``src/`` is on sys.path even if pip-install -e wasn't run (CI safety).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


# ---------------------------------------------------------------------------
# FastMCP server / client (in-memory)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def mcp_server():
    """Build and cache the shared ``FastMCP`` instance.

    We call ``build_mcp_server()`` once per test session — it is idempotent
    (guarded by a ``_REGISTERED`` flag inside the module) so subsequent test
    fixtures can share it without paying the cold-start cost again.
    """
    from aix.mcp.server import build_mcp_server

    return build_mcp_server()


@pytest.fixture
async def mcp_client(mcp_server) -> AsyncIterator[object]:
    """A connected FastMCP ``Client`` using in-memory transport.

    ``Client(mcp_server)`` skips HTTP / stdio entirely — the client talks to
    the FastMCP instance via direct in-process function calls. Perfect for
    locking the tool/resource/prompt contract without a network stack.
    """
    from fastmcp import Client

    async with Client(mcp_server) as client:
        yield client


# ---------------------------------------------------------------------------
# FastAPI app + HTTP client
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def app():
    """Import the live ``aix.api.main:app`` once per session."""
    from aix.api.main import app as live_app

    return live_app


@pytest.fixture
def http_client(app):
    """Starlette ``TestClient`` — exercises the lifespan (MCP HTTP mount).

    Use this whenever a test needs to hit a real HTTP endpoint (``/mcp/``,
    ``/api/v1/health``, ``/openapi.json``) instead of the in-memory MCP
    transport.
    """
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
@pytest.fixture
async def fresh_user_token(app) -> str:
    """Register a fresh user and return a Bearer JWT.

    This is the same plumbing used by ``tests/api/test_agent_routes.py`` —
    we walk the fastapi-users dependency chain manually so the test does
    not need to do an HTTP round-trip.
    """
    from aix.webui.auth.backend import get_jwt_strategy
    from aix.webui.auth.dependencies import fastapi_users  # noqa: F401  (side effects)
    from aix.webui.auth.manager import UserManager
    from aix.webui.auth.schemas import UserCreate
    from aix.webui.db import get_async_session

    email = f"mcp-pytest+{uuid.uuid4().hex[:8]}@example.com"
    password = "test-password-1234"

    async for session in get_async_session():
        from fastapi_users_db_sqlalchemy import SQLAlchemyUserDatabase

        from aix.webui.auth.models import User as UserModel

        user_db = SQLAlchemyUserDatabase(session, UserModel)
        manager = UserManager(user_db)
        user = await manager.create(
            UserCreate(email=email, password=password, display_name="mcp-pytest"),
            safe=True,
        )
        strategy = get_jwt_strategy()
        return await strategy.write_token(user)

    raise RuntimeError("get_async_session yielded nothing")


@pytest.fixture
def invalid_jwt_token() -> str:
    """A JWT signed with the WRONG secret — must be rejected by /mcp/.

    We construct it with the *correct* algorithm + audience so we test that
    the verifier really validates the signature (not just the header).
    """
    import jwt

    payload = {
        "sub": "attacker@example.com",
        "aud": "fastapi-users:auth",
    }
    return jwt.encode(payload, "wrong-secret-not-the-real-one", algorithm="HS256")
