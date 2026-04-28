"""
Reusable FastAPI dependencies for webui auth (CORE 2 #6.6 P1).

This module exposes:
    fastapi_users         — the ``FastAPIUsers`` instance, used by other auth
                            modules to issue routers if/when needed
    current_active_user   — REQUIRED auth: routes that demand a logged-in user
    optional_current_user — OPTIONAL auth: routes that show different content
                            depending on whether a user is logged in (e.g. the
                            navbar partial). Returns ``None`` for anonymous.

Why two flavors:
    The HTML pages have different needs than a JSON API. A protected JSON
    endpoint should 401 on missing auth; a protected HTML page should
    *redirect* the user to /auth/login. We solve that by wrapping
    ``current_active_user`` in a route-level handler in ``routes.py`` rather
    than baking the redirect into the dependency itself — keeps this module
    framework-agnostic.
"""

from __future__ import annotations

import uuid

from fastapi_users import FastAPIUsers

from aix.webui.auth.backend import auth_backend, bearer_backend
from aix.webui.auth.manager import get_user_manager
from aix.webui.auth.models import User


# Generic FastAPIUsers instance keyed by our User model and UUID id type.
#
# Two backends are registered in parallel (CORE 2 #7):
#   * ``auth_backend`` — cookie transport, used by the server-rendered
#                        webui (HTML pages + htmx).
#   * ``bearer_backend`` — Authorization: Bearer transport, used by the
#                          public JSON API at /api/v1/agent/* (and any
#                          future non-browser consumer).
#
# Both share the same JWT signing strategy, so a token minted via either
# transport is interchangeable. When a user calls ``current_active_user``
# below, fastapi-users tries every registered backend's transport and
# accepts the request if ANY of them yields a valid token. This means:
#
#   * Existing webui flows (cookie-only) keep working byte-for-byte —
#     the cookie transport is still the first one tried, and the second
#     transport simply doesn't fire when the request lacks an
#     ``Authorization`` header.
#   * New API consumers send ``Authorization: Bearer <jwt>`` and the
#     second transport handles them, no cookie required.
#
# This is strictly additive behaviour: nothing that worked before fails
# now. See backend.py for the rationale on the dual-transport design.
fastapi_users: FastAPIUsers[User, uuid.UUID] = FastAPIUsers[User, uuid.UUID](
    get_user_manager,
    [auth_backend, bearer_backend],
)


# Required auth — returns the User, raises 401 if no valid cookie OR Bearer.
current_active_user = fastapi_users.current_user(active=True)

# Optional auth — returns the User if either transport yields a valid token,
# ``None`` otherwise. Used by navbar / public landing pages that adapt to
# auth state.
optional_current_user = fastapi_users.current_user(active=True, optional=True)
