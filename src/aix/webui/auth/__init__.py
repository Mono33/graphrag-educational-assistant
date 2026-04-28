"""
WebUI authentication subpackage (CORE 2 #6.6 P1).

Provides cookie-backed JWT auth for the webui via FastAPI-Users 14.x.

Public surface:
    router                 — APIRouter with HTML auth pages mounted under /auth
    current_active_user    — FastAPI dependency for required auth (HTML routes)
    optional_current_user  — FastAPI dependency for routes that show different
                             content for anon vs authenticated users (e.g. nav)

Why FastAPI-Users (vs rolling our own):
    - Battle-tested password hashing (argon2 + bcrypt fallback)
    - JWT + cookie transport already implemented and audited
    - Pluggable backends — we can later add OAuth2 (Google, Microsoft) without
      rewriting the user model
    - Pydantic v2 native, FastAPI 0.119 compatible

The JSON ``/api/v1/auth/...`` endpoints (login, register, /users/me) are NOT
mounted in P1 because the webui is the only consumer right now. They can be
mounted later (e.g. for a CLI client) by including the appropriate
``fastapi_users.get_*_router()`` from this module.
"""

from aix.webui.auth.backend import auth_backend, bearer_backend
from aix.webui.auth.dependencies import (
    current_active_user,
    fastapi_users,
    optional_current_user,
)
from aix.webui.auth.routes import router

__all__ = [
    "router",
    "current_active_user",
    "optional_current_user",
    # Re-exports needed by aix.api.main to mount the JSON Bearer login
    # router at /auth/jwt/* (CORE 2 #7).
    "auth_backend",
    "bearer_backend",
    "fastapi_users",
]
