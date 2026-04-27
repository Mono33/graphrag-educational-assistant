"""
FastAPI-Users ``UserManager`` (CORE 2 #6.6 P1).

The UserManager is the bridge between the HTTP layer and the User database
adapter. It holds the secrets used to sign verification / password-reset
tokens and provides hooks (``on_after_register``, ``on_after_login`` …)
where we can plug logging, audit events, or analytics.

Secrets:
    - ``WEBUI_AUTH_SECRET`` — used for verification + reset tokens.
      In dev, falls back to a deterministic-but-warned value so a fresh
      checkout works without manual setup. In any non-dev environment the
      env var MUST be set; we log a loud WARNING when the fallback is hit.

We deliberately do NOT issue verification emails in P1 — there is no SMTP
configured and the agent test surface only needs login. ``is_verified`` is
left at its default (False) and gates nothing in P1; we'll wire it up when
SMTP lands.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Optional

from fastapi import Depends, Request
from fastapi_users import BaseUserManager, UUIDIDMixin
from fastapi_users_db_sqlalchemy import SQLAlchemyUserDatabase
from sqlalchemy.ext.asyncio import AsyncSession

from aix.webui.auth.models import User
from aix.webui.db import get_async_session

logger = logging.getLogger(__name__)


# WARNING: this fallback must NEVER be used outside dev. In production the
# env var WEBUI_AUTH_SECRET is required and must be a long random string.
_DEV_FALLBACK_SECRET = "DEV_ONLY_aix_webui_change_me"
_AUTH_SECRET = os.getenv("WEBUI_AUTH_SECRET", _DEV_FALLBACK_SECRET)

if _AUTH_SECRET == _DEV_FALLBACK_SECRET:
    logger.warning(
        "⚠️  WEBUI_AUTH_SECRET not set — using dev fallback. "
        "Set a strong secret in .env before any non-local use."
    )


class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    """Manage user lifecycle: registration, login hooks, token signing."""

    reset_password_token_secret = _AUTH_SECRET
    verification_token_secret = _AUTH_SECRET

    async def on_after_register(
        self, user: User, request: Optional[Request] = None
    ) -> None:
        # Audit hook: useful in deployment when shipped to GlitchTip.
        # Keep PII minimal — log the user id, never the password or full email
        # (logged email is fine for dev; tighten via redactor in CORE 6 if
        # required by policy).
        logger.info(
            "👤 User registered: id=%s email=%s display_name=%s",
            user.id, user.email, user.display_name,
        )


# ----------------------------------------------------------------------------
# FastAPI dependencies
# ----------------------------------------------------------------------------

async def get_user_db(
    session: AsyncSession = Depends(get_async_session),
) -> SQLAlchemyUserDatabase:
    """DB adapter dependency — wraps the session in a fastapi-users adapter."""
    yield SQLAlchemyUserDatabase(session, User)


async def get_user_manager(
    user_db: SQLAlchemyUserDatabase = Depends(get_user_db),
) -> UserManager:
    """UserManager dependency — used by both auth backends and routes."""
    yield UserManager(user_db)
