"""Who owns what an API call creates (2026-10-14).

Every table the WebUI writes -- lessons, materials, chunks, bindings -- is keyed
on ``owner_id → user.id``, and every read is filtered by it; that filter is what
``tests/unit/test_idor.py`` pins. The API reuses it rather than inventing a
second authorization: an API caller acts **as a user**.

* A WebUI user calling with its own JWT is that user. What it creates through
  the API is its own, and shows up in its WebUI too.
* The service account (HTTP Basic, ``GRAPH_API_USERNAME``) is an application
  serving many people -- a chatbot platform. It names the person it acts for
  with an opaque ``end_user_id``, and each one becomes its own ``User`` row,
  provisioned on first use. Two end users of the same client, or the same id
  from two clients, never see each other's sessions or documents, because they
  are different owners.

The provisioned users cannot sign in to the WebUI: the address is under the
reserved ``.invalid`` domain, and the password is the hash of a random secret
that is never stored or shown. ``end_user_id`` is hashed into the address, so the
client's identifiers -- possibly personal -- are not kept in clear.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
from typing import Optional, Union

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from aix.webui.auth.dependencies import optional_current_user
from aix.webui.auth.models import User
from aix.webui.db import get_async_session

#: What ``_require_caller`` returns for the HTTP Basic service account.
SERVICE_ACCOUNT = "service_account"

#: RFC 2606: a domain that can never receive mail, so these are never real addresses.
API_EMAIL_DOMAIN = "api.invalid"

#: The id a client may send: anything printable, up to this long.
END_USER_ID_MAX = 200

_SLUG = re.compile(r"[^a-z0-9]+")


_basic_security = HTTPBasic(auto_error=False)


async def require_caller(
    basic: Optional[HTTPBasicCredentials] = Depends(_basic_security),
    user: Optional[User] = Depends(optional_current_user),
) -> User | str:
    """Authenticate via JWT/cookie (browser/webui) OR Basic Auth (service account)."""
    if basic is not None:
        svc_user = os.getenv("GRAPH_API_USERNAME", "")
        svc_pwd = os.getenv("GRAPH_API_PWD", "")
        if svc_user and svc_pwd:
            user_ok = secrets.compare_digest(basic.username.encode(), svc_user.encode())
            pwd_ok = secrets.compare_digest(basic.password.encode(), svc_pwd.encode())
            if user_ok and pwd_ok:
                return SERVICE_ACCOUNT
    if user is not None:
        return user
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


def _client_name() -> str:
    return os.getenv("GRAPH_API_USERNAME", "") or "client"


def api_email(client: str, end_user_id: Optional[str]) -> str:
    """The address of the user an API client acts as. Deterministic, and opaque."""
    slug = _SLUG.sub("-", client.lower()).strip("-")[:30] or "client"
    digest = hashlib.sha256(f"{client}\x00{end_user_id or ''}".encode()).hexdigest()[:20]
    return f"api-{slug}-{digest}@{API_EMAIL_DOMAIN}"


def check_end_user_id(end_user_id: Optional[str]) -> Optional[str]:
    """The id as given, or a 422 a client can act on."""
    if end_user_id is None:
        return None
    value = end_user_id.strip()
    if not value:
        return None
    if len(value) > END_USER_ID_MAX or not value.isprintable():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "invalid_end_user_id",
                "message": f"end_user_id: testo stampabile, al massimo {END_USER_ID_MAX} caratteri.",
            },
        )
    return value


def _unusable_password_hash() -> str:
    """A valid hash of a secret nobody knows: a login attempt is refused, not a crash."""
    from fastapi_users.password import PasswordHelper

    return PasswordHelper().hash(secrets.token_urlsafe(48))


async def owner_for(
    session: AsyncSession,
    caller: Union[User, str],
    end_user_id: Optional[str] = None,
) -> User:
    """The user this call acts as -- provisioned on the first call for an end user.

    A JWT caller is itself, whatever ``end_user_id`` says: a teacher cannot act
    as someone else by naming them.
    """
    if isinstance(caller, User):
        return caller
    if caller != SERVICE_ACCOUNT:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    email = api_email(_client_name(), check_end_user_id(end_user_id))
    existing = (await session.execute(select(User).where(User.email == email))).scalar_one_or_none()
    if existing is not None:
        return existing

    user = User(
        email=email,
        hashed_password=_unusable_password_hash(),
        is_active=True,
        is_superuser=False,
        is_verified=True,
    )
    session.add(user)
    try:
        await session.commit()
    except IntegrityError:
        # Two first calls for the same end user at once: the other one won.
        await session.rollback()
        return (await session.execute(select(User).where(User.email == email))).scalar_one()
    await session.refresh(user)
    return user


#: The header that names the end user a service account acts for.
END_USER_HEADER = "X-End-User-Id"


async def current_owner(
    caller: Union[User, str] = Depends(require_caller),
    end_user_id: Optional[str] = Header(default=None, alias=END_USER_HEADER),
    session: AsyncSession = Depends(get_async_session),
) -> User:
    """The user every resource route acts as: one dependency, one rule (see the module docstring)."""
    return await owner_for(session, caller, end_user_id)
