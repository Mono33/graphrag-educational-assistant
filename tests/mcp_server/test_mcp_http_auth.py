"""
Lock the JWT Bearer auth gate on ``/mcp/``.

This is the production security boundary: the MCP HTTP endpoint is mounted
inside the public FastAPI app, so it's reachable by anyone who can hit the
host. Without auth the entire agent + KG surface would be exposed.

Coverage:
    * test_mcp_unauthenticated_returns_401 — no Authorization header → 401
    * test_mcp_with_fake_token_returns_401  — token signed with wrong secret → 401
    * test_mcp_with_valid_jwt_returns_2xx   — fresh fastapi-users token → 2xx
    * test_mcp_health_endpoint_unprotected  — /api/v1/health stays public
"""

from __future__ import annotations

import json

import pytest

# ---------------------------------------------------------------------------
# Minimal MCP "initialize" envelope — the very first request a real client
# sends. We use it as our auth probe because:
#   * Wrong / missing token → 401 *before* any routing logic runs.
#   * Valid token → 2xx (FastMCP returns 200 + initialize result, or 202 if
#     it queues the response on the SSE channel).
# Either 200 or 202 means "auth passed", which is what we're asserting.
# ---------------------------------------------------------------------------
_INITIALIZE_BODY = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "pytest-phase6", "version": "0.0.1"},
    },
}

_REQUIRED_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


def test_mcp_unauthenticated_returns_401(http_client):
    """No Authorization header → 401 Unauthorized."""
    resp = http_client.post(
        "/mcp/",
        headers=_REQUIRED_HEADERS,
        content=json.dumps(_INITIALIZE_BODY),
    )
    assert resp.status_code == 401, (
        f"Expected 401 for unauthenticated /mcp/, got {resp.status_code}: "
        f"{resp.text[:200]}"
    )


def test_mcp_with_fake_token_returns_401(http_client, invalid_jwt_token):
    """A JWT signed with the wrong secret must be rejected.

    This is the test that proves the verifier *actually* validates the
    signature — not just the algorithm header or the audience claim.
    """
    headers = {
        **_REQUIRED_HEADERS,
        "Authorization": f"Bearer {invalid_jwt_token}",
    }
    resp = http_client.post(
        "/mcp/",
        headers=headers,
        content=json.dumps(_INITIALIZE_BODY),
    )
    assert resp.status_code == 401, (
        f"Expected 401 for wrong-secret JWT, got {resp.status_code}: "
        f"{resp.text[:200]}"
    )


@pytest.mark.asyncio
async def test_mcp_with_valid_jwt_returns_2xx(http_client, fresh_user_token):
    """A valid fastapi-users-issued Bearer token unlocks /mcp/.

    We accept any 2xx as "auth passed" — FastMCP's Streamable HTTP transport
    sometimes responds 200 (immediate JSON) and sometimes 202 (queued via
    SSE), depending on the negotiated stream mode. Either is fine — the
    point is that the request was *not* rejected at the auth layer.
    """
    headers = {
        **_REQUIRED_HEADERS,
        "Authorization": f"Bearer {fresh_user_token}",
    }
    resp = http_client.post(
        "/mcp/",
        headers=headers,
        content=json.dumps(_INITIALIZE_BODY),
    )
    assert 200 <= resp.status_code < 300, (
        f"Expected 2xx for valid Bearer token on /mcp/, got "
        f"{resp.status_code}: {resp.text[:200]}"
    )


def test_mcp_health_endpoint_unprotected(http_client):
    """Sanity check: /api/v1/health is still public.

    We added /mcp/ behind auth — but the existing public endpoints must
    not have been gated by accident. This is a coupling-check between
    Phases 5 and the rest of the API.
    """
    resp = http_client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json().get("status") in {"healthy", "ok", "degraded"}
