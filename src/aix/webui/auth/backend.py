"""
FastAPI-Users authentication backend (CORE 2 #6.6 P1).

A "backend" combines two pieces:
    1. Transport — how the JWT travels between server and client.
       We use ``CookieTransport`` so the JWT lives in an HttpOnly cookie,
       not in JS-accessible localStorage. That defends against XSS token theft
       and is the right default for a server-rendered HTML app.
    2. Strategy — how the JWT is created and verified.
       ``JWTStrategy`` signs short-lived tokens with the same secret used by
       the UserManager (see manager.py).

Cookie attributes:
    - HttpOnly  — JS cannot read the cookie (XSS defense)
    - Secure    — only sent over HTTPS in prod (configurable via env)
    - SameSite=Lax — sent on top-level navigations but blocked on cross-site
                     POST, which is the right balance for a self-contained app
    - max_age   — token lifetime (default 24h, refreshable on each login)

When we deploy in CORE 6 behind Caddy/Traefik, set ``WEBUI_COOKIE_SECURE=1``
in the environment; the dev default leaves it False so localhost works without
HTTPS.
"""

from __future__ import annotations

import os

from fastapi_users.authentication import (
    AuthenticationBackend,
    BearerTransport,
    CookieTransport,
    JWTStrategy,
)

from aix.webui.auth.manager import _AUTH_SECRET

# Cookie name. Distinct from the API session names (none yet) so we don't
# collide with a future SSO cookie shipped by the public /api/v1 surface.
_COOKIE_NAME = os.getenv("WEBUI_COOKIE_NAME", "aix_webui_auth")

# 24h default; ``WEBUI_TOKEN_LIFETIME_SECONDS`` overrides for short-lived
# tokens during development or longer-lived tokens for trusted environments.
_TOKEN_LIFETIME_SECONDS = int(os.getenv("WEBUI_TOKEN_LIFETIME_SECONDS", "86400"))

# In dev (HTTP localhost) Secure must be False or the browser ignores the
# cookie. Flip via env in production.
_COOKIE_SECURE = os.getenv("WEBUI_COOKIE_SECURE", "0") in ("1", "true", "True")


cookie_transport = CookieTransport(
    cookie_name=_COOKIE_NAME,
    cookie_max_age=_TOKEN_LIFETIME_SECONDS,
    cookie_secure=_COOKIE_SECURE,
    cookie_httponly=True,
    cookie_samesite="lax",
)


def get_jwt_strategy() -> JWTStrategy:
    """JWT signer/verifier — secret-pinned per environment."""
    return JWTStrategy(secret=_AUTH_SECRET, lifetime_seconds=_TOKEN_LIFETIME_SECONDS)


# The backend ties transport + strategy together. ``name`` matters because
# fastapi-users mounts route paths under it (e.g. /auth/{name}/login when
# the JSON API router is used). For the HTML routes we call backend methods
# directly so the name is mostly cosmetic.
auth_backend = AuthenticationBackend(
    name="cookie",
    transport=cookie_transport,
    get_strategy=get_jwt_strategy,
)


# ---------------------------------------------------------------------------
# Bearer transport for the public JSON API (CORE 2 #7).
#
# Why a *second* backend instead of replacing the cookie one:
#     The webui (server-rendered HTML) needs the cookie for a frictionless
#     UX — set-once, sent automatically with every browser navigation.
#     The public /api/v1/agent/* endpoints are designed for non-browser
#     clients (curl, Postman, the AixLearning embed, future mobile app)
#     where ``Authorization: Bearer <jwt>`` is the universal idiom.
#
# Why both backends share ``get_jwt_strategy``:
#     The same JWT secret signs both transports' tokens, so a token
#     issued by one is valid on the other. That gives a clean Swagger
#     UX in /docs: a teacher can hit ``POST /auth/jwt/login`` (mounted
#     by the API in main.py), receive a Bearer token, and use it to
#     test ``/api/v1/agent/run`` from the same Swagger session — no
#     parallel cookie / Bearer secrets to keep in sync.
#
# Backward-compat note:
#     The existing ``auth_backend`` symbol is kept (unchanged) so any
#     module still importing it (templates, tests, lessons routes) is
#     unaffected. We only ADD ``bearer_backend`` and update the
#     ``FastAPIUsers`` registration in ``dependencies.py`` to accept
#     either transport on protected routes.
#
# ``tokenUrl`` controls only OpenAPI metadata — it tells Swagger UI's
# "Authorize" dialog where to send password-flow login attempts. We
# point it at the public mount ``/auth/jwt/login`` (mounted from
# fastapi-users in aix/api/main.py).
# ---------------------------------------------------------------------------

bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")


bearer_backend = AuthenticationBackend(
    name="jwt-bearer",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)
