"""
CORE 5 #20 Phase 5 — Streamable HTTP mount for the Aix MCP server.

This module is the *HTTP-side* counterpart of ``aix.mcp.stdio_main``. Where
``stdio_main`` runs the MCP server over stdin/stdout for local clients
(Claude Desktop, Cursor IDE), this module returns a Starlette ASGI app that
the public FastAPI surface (``aix.api.main``) mounts at ``/mcp/``.

Both transports share the *same* singleton ``mcp`` instance from
``aix.mcp.server``, so every tool / resource / prompt registered there is
automatically reachable on both stdio AND remote HTTP — zero duplication.

Why a single FastMCP instance is safe for both transports:
    The auth provider (``mcp.auth``) is only consulted by the HTTP middleware
    in ``create_streamable_http_app`` — see fastmcp.server.http. The stdio
    transport never enters that middleware, so setting ``mcp.auth`` here
    does not affect the stdio entry. In production each transport runs in a
    separate process (uvicorn for HTTP, ``python -m aix.mcp.stdio_main`` for
    stdio), so there is no cross-talk.

Auth design (CORE 5 #20 Phase 5):
    * Default: JWT Bearer enforced. Tokens are issued by
      ``POST /auth/jwt/login`` (mounted from fastapi-users in
      ``aix.api.main``). Same secret + same audience as the public
      ``/api/v1/agent/*`` endpoints, so a single Bearer token works on both.
    * Escape hatch: ``AIX_MCP_REQUIRE_AUTH=0`` disables the verifier so a
      developer can smoke-test ``/mcp/`` without minting a token. NEVER set
      this in any non-local environment — leaving the MCP endpoint open
      exposes the full agent + KG surface unauthenticated.

Lifespan note:
    FastMCP's ``http_app()`` returns a Starlette app whose lifespan
    initialises an internal ``StreamableHTTPSessionManager``. That lifespan
    MUST be entered before the first request, otherwise
    ``StreamableHTTPSessionManager.handle_request`` raises
    "Task group is not initialized". The parent FastAPI app must therefore
    enter ``mcp_app.lifespan(app)`` inside its own ``lifespan`` — see
    ``aix.api.main`` where this is done with an ``AsyncExitStack``.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public path constants
# ---------------------------------------------------------------------------
# Mount point of the MCP HTTP endpoint inside the public FastAPI app.
#
# The MCP spec recommends a single canonical path that handles both POST
# (client → server requests) and GET (server-initiated SSE for resumable
# notifications). Using "/mcp/" as a mount point with ``path="/"`` on the
# FastMCP side keeps the URL short and avoids double-prefixing.
MCP_MOUNT_PATH: str = "/mcp"


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
def _auth_required() -> bool:
    """Whether the HTTP endpoint should enforce JWT Bearer authentication.

    Defaults to ``True``. Setting ``AIX_MCP_REQUIRE_AUTH=0`` (or ``false``,
    ``no``) disables it — DEV ONLY.
    """
    raw = os.getenv("AIX_MCP_REQUIRE_AUTH", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _build_jwt_verifier():
    """Construct a FastMCP ``JWTVerifier`` aligned with fastapi-users tokens.

    fastapi-users' ``JWTStrategy`` defaults:
        * algorithm = HS256 (symmetric, shared secret)
        * audience  = "fastapi-users:auth"
        * secret    = ``WEBUI_AUTH_SECRET`` env var (fallback in dev)

    The verifier we hand to FastMCP must mirror those exactly so a token
    minted by ``POST /auth/jwt/login`` validates here. See backend.py:

        bearer_backend = AuthenticationBackend(
            name="jwt-bearer",
            transport=BearerTransport(tokenUrl="auth/jwt/login"),
            get_strategy=get_jwt_strategy,  # → secret=_AUTH_SECRET, HS256
        )

    Returns ``None`` if the underlying secret cannot be resolved (extremely
    unlikely — the dev fallback always exists), letting the caller fall
    back gracefully.
    """
    try:
        from fastmcp.server.auth.providers.jwt import JWTVerifier
        from aix.webui.auth.manager import _AUTH_SECRET
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[MCP] Could not build JWTVerifier: %s", exc)
        return None

    return JWTVerifier(
        public_key=_AUTH_SECRET,
        algorithm="HS256",
        audience="fastapi-users:auth",
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_mcp_http_app():
    """Build the Streamable HTTP ASGI app for the Aix MCP server.

    Returns a Starlette ASGI app ready to be mounted via FastAPI's
    ``app.mount(...)``. Returns ``None`` on any failure so the caller can
    keep the rest of the API surface up — a broken MCP mount should never
    take down ``/api/v1/*`` or ``/webui/*``.

    The returned app exposes a single endpoint at ``path="/"`` (so when
    mounted at ``/mcp`` the public URL is ``http://host/mcp/``). It serves
    POST requests for MCP RPC calls and GET requests for SSE notifications
    over the same connection — the canonical Streamable HTTP transport.
    """
    try:
        from aix.mcp.server import build_mcp_server
    except Exception as exc:
        logger.warning("[MCP] HTTP app build skipped — server import failed: %s", exc)
        return None

    try:
        mcp = build_mcp_server()
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("[MCP] HTTP app build failed inside build_mcp_server: %s", exc)
        return None

    if _auth_required():
        verifier = _build_jwt_verifier()
        if verifier is None:
            logger.error(
                "[MCP] AIX_MCP_REQUIRE_AUTH is set but JWT verifier could not "
                "be constructed; refusing to mount the HTTP app."
            )
            return None
        # Setting auth on the shared instance is safe because stdio doesn't
        # consult mcp.auth — see the module docstring above.
        mcp.auth = verifier
        logger.info(
            "[MCP] HTTP app: JWT Bearer auth ENABLED "
            "(HS256, aud=fastapi-users:auth, mount=%s)",
            MCP_MOUNT_PATH,
        )
    else:
        logger.warning(
            "[MCP] HTTP app: AIX_MCP_REQUIRE_AUTH is OFF — endpoint is "
            "UNAUTHENTICATED. DEV ONLY. Set AIX_MCP_REQUIRE_AUTH=1 (or "
            "leave unset) for any deployed environment."
        )

    try:
        # path="/" because the parent FastAPI app mounts us at MCP_MOUNT_PATH;
        # leaving path empty here would double the prefix.
        # transport="http" is the alias for "streamable-http".
        app = mcp.http_app(path="/", transport="http")
    except Exception as exc:
        logger.exception("[MCP] HTTP app build failed in mcp.http_app: %s", exc)
        return None

    logger.info(
        "[MCP] HTTP app built — mount target=%s, transport=streamable-http",
        MCP_MOUNT_PATH,
    )
    return app


__all__ = ["build_mcp_http_app", "MCP_MOUNT_PATH"]
