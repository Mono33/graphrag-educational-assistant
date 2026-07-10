"""
HTML auth routes for the webui (CORE 2 #6.6 P1).

Mounts under ``/auth`` and serves three classic flows as server-rendered
pages, with the JWT delivered to the browser as an HttpOnly cookie:

    GET  /auth/register   — render register form
    POST /auth/register   — create user, auto-login, redirect to ?next or /webui/
    GET  /auth/login      — render login form
    POST /auth/login      — authenticate, set cookie, redirect to ?next or /webui/
    GET  /auth/logout     — clear cookie, redirect to /webui/

We deliberately do NOT mount the JSON ``/auth/jwt/login`` etc. routes from
fastapi-users in P1 — see ``aix.webui.auth.__init__`` for rationale. The
HTML handlers below call the same UserManager / strategy primitives, so
adding the JSON routes later is purely additive.

``next`` query parameter:
    Both login and register accept an optional ``?next=/webui/...`` query
    param so that protected routes can redirect-with-bounce. We validate the
    target stays within the same origin (starts with ``/``) to prevent open
    redirect.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from fastapi import APIRouter, Depends, Form, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi_users.exceptions import (
    InvalidPasswordException,
    UserAlreadyExists,
)
from pydantic import ValidationError

from aix.webui.auth.backend import (
    cookie_transport,
    get_jwt_strategy,
)
from aix.webui.auth.dependencies import optional_current_user
from aix.webui.auth.manager import UserManager, get_user_manager
from aix.webui.auth.models import User
from aix.webui.auth.schemas import UserCreate

logger = logging.getLogger(__name__)


# Same template directory the rest of the webui uses.
_PACKAGE_DIR = Path(__file__).resolve().parents[1]  # …/src/aix/webui
_TEMPLATES_DIR = _PACKAGE_DIR / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


router = APIRouter(prefix="/auth", tags=["webui-auth"])


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _safe_next(next_url: Optional[str], default: str = "/webui/") -> str:
    """
    Validate the ``next`` query param so we never honour an off-site redirect.

    Accepts only same-origin relative paths (must start with a single ``/``
    and not ``//``). Anything else falls back to ``default``.
    """
    if not next_url:
        return default
    if not next_url.startswith("/") or next_url.startswith("//"):
        return default
    return next_url


def _set_auth_cookie(response: Response, token: str) -> None:
    """
    Set the auth cookie on ``response`` using the same attributes the
    fastapi-users CookieTransport would. We do this manually (rather than
    delegating to ``cookie_transport.get_login_response``) because we want a
    303 RedirectResponse, not a 204 NoContent.
    """
    response.set_cookie(
        key=cookie_transport.cookie_name,
        value=token,
        max_age=cookie_transport.cookie_max_age,
        path=cookie_transport.cookie_path,
        domain=cookie_transport.cookie_domain,
        secure=cookie_transport.cookie_secure,
        httponly=cookie_transport.cookie_httponly,
        samesite=cookie_transport.cookie_samesite,
    )


def _clear_auth_cookie(response: Response) -> None:
    """Clear the auth cookie — used by /auth/logout."""
    response.delete_cookie(
        key=cookie_transport.cookie_name,
        path=cookie_transport.cookie_path,
        domain=cookie_transport.cookie_domain,
    )


# ----------------------------------------------------------------------------
# GET pages
# ----------------------------------------------------------------------------


@router.get("/register", response_class=HTMLResponse, name="auth_register_get")
async def register_get(
    request: Request,
    next: Optional[str] = Query(default=None),
    user: Optional[User] = Depends(optional_current_user),
) -> Response:
    """Show the register form. Already-authed users are bounced to /webui/."""
    if user is not None:
        return RedirectResponse(_safe_next(next), status_code=303)
    return templates.TemplateResponse(
        request,

            "pages/auth_register.html",
        {
            "title": "Registrazione · AixLearning",
            "phase": "P1 — Auth",
            "next": _safe_next(next),
            "user": None,
            "form_errors": None,
            "form_values": {},
        },
    )


@router.get("/login", response_class=HTMLResponse, name="auth_login_get")
async def login_get(
    request: Request,
    next: Optional[str] = Query(default=None),
    user: Optional[User] = Depends(optional_current_user),
) -> Response:
    """Show the login form. Already-authed users are bounced to /webui/."""
    if user is not None:
        return RedirectResponse(_safe_next(next), status_code=303)
    return templates.TemplateResponse(
        request,

            "pages/auth_login.html",
        {
            "title": "Accesso · AixLearning",
            "phase": "P1 — Auth",
            "next": _safe_next(next),
            "user": None,
            "form_errors": None,
            "form_values": {},
        },
    )


# ----------------------------------------------------------------------------
# POST handlers
# ----------------------------------------------------------------------------


@router.post("/register", name="auth_register_post")
async def register_post(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    display_name: Optional[str] = Form(default=None),
    next: Optional[str] = Form(default=None),
    user_manager: UserManager = Depends(get_user_manager),
) -> Response:
    """
    Create a new user, auto-login by issuing a JWT cookie, and redirect.

    Re-renders the form with ``form_errors`` populated on validation /
    duplicate-email / weak-password failures so the user keeps their input.
    """
    redirect_to = _safe_next(next)
    form_values = {"email": email, "display_name": display_name or ""}

    try:
        user_create = UserCreate(
            email=email,
            password=password,
            display_name=(display_name.strip() if display_name else None),
        )
    except ValidationError as exc:
        return templates.TemplateResponse(
            request,

                "pages/auth_register.html",
            {
                "title": "Registrazione · AixLearning",
                "phase": "P1 — Auth",
                "next": redirect_to,
                "user": None,
                "form_errors": [err.get("msg", "Campo non valido") for err in exc.errors()],
                "form_values": form_values,
            },
            status_code=422,
        )

    try:
        user = await user_manager.create(user_create, safe=True, request=request)
    except UserAlreadyExists:
        return templates.TemplateResponse(
            request,

                "pages/auth_register.html",
            {
                "title": "Registrazione · AixLearning",
                "phase": "P1 — Auth",
                "next": redirect_to,
                "user": None,
                "form_errors": [
                    "Esiste già un account con questa email. "
                    "Prova ad accedere oppure usa un'altra email."
                ],
                "form_values": form_values,
            },
            status_code=409,
        )
    except InvalidPasswordException as exc:
        # fastapi-users wraps the reason in exc.reason
        reason = getattr(exc, "reason", "Password non valida.")
        return templates.TemplateResponse(
            request,

                "pages/auth_register.html",
            {
                "title": "Registrazione · AixLearning",
                "phase": "P1 — Auth",
                "next": redirect_to,
                "user": None,
                "form_errors": [str(reason)],
                "form_values": form_values,
            },
            status_code=422,
        )

    # Auto-login: write a token and set the cookie on a redirect response.
    strategy = get_jwt_strategy()
    token = await strategy.write_token(user)

    response = RedirectResponse(url=redirect_to, status_code=303)
    _set_auth_cookie(response, token)
    return response


@router.post("/login", name="auth_login_post")
async def login_post(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    next: Optional[str] = Form(default=None),
    user_manager: UserManager = Depends(get_user_manager),
) -> Response:
    """Authenticate against UserManager, set the auth cookie, redirect."""
    redirect_to = _safe_next(next)
    form_values = {"email": email}

    # UserManager.authenticate expects an OAuth2PasswordRequestForm-like
    # object (has .username and .password). We build a tiny attribute bag
    # rather than a nested class because Python class bodies don't see
    # enclosing function locals for names that are *also assigned* in the
    # body — `password = password` would raise NameError on the RHS lookup.
    # SimpleNamespace sidesteps that scoping wart entirely.
    credentials = SimpleNamespace(username=email, password=password)
    user = await user_manager.authenticate(credentials)
    if user is None or not user.is_active:
        return templates.TemplateResponse(
            request,

                "pages/auth_login.html",
            {
                "title": "Accesso · AixLearning",
                "phase": "P1 — Auth",
                "next": redirect_to,
                "user": None,
                "form_errors": ["Email o password non corretti, oppure account disattivato."],
                "form_values": form_values,
            },
            status_code=401,
        )

    strategy = get_jwt_strategy()
    token = await strategy.write_token(user)

    response = RedirectResponse(url=redirect_to, status_code=303)
    _set_auth_cookie(response, token)
    logger.info("🔐 User logged in: id=%s email=%s", user.id, user.email)
    return response


@router.get("/logout", name="auth_logout_get")
async def logout(
    user: Optional[User] = Depends(optional_current_user),
) -> Response:
    """Clear the auth cookie and bounce to the home page."""
    response = RedirectResponse(url="/webui/", status_code=303)
    _clear_auth_cookie(response)
    if user is not None:
        logger.info("🚪 User logged out: id=%s email=%s", user.id, user.email)
    return response
