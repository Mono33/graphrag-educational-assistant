"""CSRF protection for the WebUI's mutating requests (Fase 0 · PR 1).

Today nothing checks the origin of a state-changing request. The session
travels in a cookie, so any page the teacher has open can make their browser
POST to this application with their credentials attached — delete a lesson,
delete an uploaded file, start a run. ``SameSite=Lax`` on the auth cookie
blocks the simplest form of that, but it is one attribute on one cookie, it
does not cover a sibling subdomain, and it is not something the application
states or tests anywhere.

The token
---------
A signed double submit. The token travels in a cookie the page can read and
must be echoed back in a header (htmx) or a form field (the three plain
HTML forms). An attacker's page can make the browser *send* our cookie but cannot
*read* it, so it cannot produce the echo.

The signature is what makes the double submit sound. A plain random value
compared against itself is defeated by anyone who can write a cookie on the
parent domain — a sibling subdomain, say. Here the token carries an HMAC over
its own nonce **and the identity it was issued to**, so a value that was not
minted by this application with this secret is rejected, and one minted for a
different session is rejected too. That second half also closes login CSRF:
a token obtained while logged out cannot be replayed against a logged-in
session.

Where it is enforced
--------------------
One place: :class:`CSRFMiddleware`, over everything under ``/webui`` and
``/auth``. It fails closed, so a mutating route is protected the day it is
written and nobody has to remember to opt in — which is the whole reason to
put this in a middleware rather than a dependency per route.

It accepts the token from a header (htmx, the great majority) or from a form
field (the three ordinary ``<form method="post">`` submissions: create a
lesson, log in, register — a form cannot set a header). Serving both from one
place is why the middleware is written against ASGI directly; see the class.

Turning it off
--------------
``WEBUI_CSRF_ENABLED=false`` disables enforcement while still issuing tokens,
so the cookie and the templates can be verified in an environment before the
check starts rejecting. It is on by default; the flag exists for a rollout,
not as a supported configuration.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import time
from typing import Optional
from urllib.parse import parse_qs

from fastapi import Request, status
from starlette.responses import Response

from aix.webui.auth.backend import _COOKIE_NAME as _AUTH_COOKIE_NAME
from aix.webui.auth.backend import _COOKIE_SECURE
from aix.webui.auth.manager import _AUTH_SECRET

logger = logging.getLogger(__name__)

COOKIE_NAME = "aix_csrf"
HEADER_NAME = "X-CSRF-Token"
FORM_FIELD = "csrf_token"

#: Methods that must not change state, so they carry no token.
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})

#: Everything the middleware guards. ``/api/v1`` is deliberately absent: it
#: authenticates each request on its own credentials rather than on a browser
#: cookie, so there is no ambient authority for a third-party page to borrow.
PROTECTED_PREFIXES: tuple[str, ...] = ("/webui", "/auth")

#: A token older than this is refused even if it verifies. Longer than a
#: working session so a teacher does not lose a lesson mid-composition,
#: short enough that a token leaked into a log or a screenshot expires.
MAX_AGE_S = 12 * 3600

# Derived rather than reused: the same secret signs the session JWT, and a key
# that signs two different kinds of token invites a confusion between them.
_KEY = hashlib.sha256(b"aix-webui-csrf-v1|" + _AUTH_SECRET.encode("utf-8")).digest()


def _enabled() -> bool:
    return (os.getenv("WEBUI_CSRF_ENABLED") or "true").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _binding(request: Request) -> str:
    """What the token is tied to: the session it was issued for.

    Hashed rather than used raw so the session JWT never ends up inside a
    token that the page can read, and so the binding is a fixed short length.
    An anonymous visitor binds to the empty string, which is what lets the
    login form carry a token at all — and what makes that token useless once
    a session exists.
    """
    session = request.cookies.get(_AUTH_COOKIE_NAME) or ""
    return hashlib.sha256(session.encode("utf-8")).hexdigest()[:32]


def _sign(nonce: str, issued_at: int, binding: str) -> str:
    payload = f"{nonce}.{issued_at}.{binding}".encode()
    return hmac.new(_KEY, payload, hashlib.sha256).hexdigest()


def issue(request: Request) -> str:
    """Mint a token for this request's session."""
    nonce = secrets.token_urlsafe(16)
    issued_at = int(time.time())
    return f"{nonce}.{issued_at}.{_sign(nonce, issued_at, _binding(request))}"


def verify(request: Request, submitted: Optional[str]) -> bool:
    """True when ``submitted`` is a live token minted for this session.

    Both halves of the double submit are checked: the value must match the
    cookie the browser holds *and* carry our signature. Either alone is not
    enough — the cookie comparison catches a replayed token from another
    browser, the signature catches a cookie an attacker planted.
    """
    cookie = request.cookies.get(COOKIE_NAME)
    if not submitted or not cookie:
        return False
    if not hmac.compare_digest(submitted, cookie):
        return False

    parts = submitted.split(".")
    if len(parts) != 3:
        return False
    nonce, raw_issued_at, signature = parts

    try:
        issued_at = int(raw_issued_at)
    except ValueError:
        return False
    if issued_at > time.time() + 60:  # minted in the future — a forgery or a clock fault
        return False
    if time.time() - issued_at > MAX_AGE_S:
        return False

    return hmac.compare_digest(signature, _sign(nonce, issued_at, _binding(request)))


def _set_cookie(response: Response, token: str) -> None:
    """Readable by the page, unlike the session cookie — that is the point.

    The token is not a credential: it proves the request came from a page this
    application served, and the page has to be able to read it to say so.
    """
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=MAX_AGE_S,
        secure=_COOKIE_SECURE,
        httponly=False,
        samesite="lax",
        path="/",
    )


def token_for(request: Request) -> str:
    """The token a template should render, minting one if the request has none.

    Reads what the middleware already decided when it ran, so a page and the
    cookie it is served with never disagree.
    """
    existing = getattr(request.state, "csrf_token", None)
    if existing:
        return str(existing)
    return request.cookies.get(COOKIE_NAME) or issue(request)


class CSRFMiddleware:
    """Rejects unsigned mutating requests, and keeps the cookie fresh.

    Written against ASGI directly rather than as a ``BaseHTTPMiddleware``
    because it sometimes has to read the request body, and a
    ``BaseHTTPMiddleware`` that reads the body consumes the stream the route
    handler still needs. Here the body is buffered and replayed, so the handler
    sees exactly what the browser sent.

    That is what lets one mechanism cover both kinds of caller. htmx sends the
    token in a header; the three ordinary ``<form method="post">`` submissions
    (create a lesson, log in, register) cannot set a header and send a field
    instead. The alternative was a list of paths exempt from the header rule,
    checked by a dependency on each — and an exemption list is only correct
    while someone keeps it correct.

    The body is read only when there is no header *and* the content type is
    ``application/x-www-form-urlencoded``. File uploads are htmx, so they carry
    the header and are never buffered: nothing here holds an upload in memory.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        protected = request.url.path.startswith(PROTECTED_PREFIXES)
        unsafe = request.method.upper() not in SAFE_METHODS

        if protected and unsafe and _enabled():
            submitted = request.headers.get(HEADER_NAME)
            if not submitted and _is_urlencoded(request):
                body = await _drain(receive)
                receive = _replay(body)
                submitted = _token_from_form(body)

            if not verify(request, submitted):
                logger.warning(
                    "[CSRF] rejected %s %s — %s",
                    request.method,
                    request.url.path,
                    "no token submitted" if not submitted else "token did not verify",
                )
                response = Response(
                    "Richiesta non valida: ricarica la pagina e riprova.",
                    status_code=status.HTTP_403_FORBIDDEN,
                )
                await response(scope, receive, send)
                return

        # Decided on the way in, so a template rendered by this request and the
        # cookie set on its response always carry the same value.
        token = request.cookies.get(COOKIE_NAME)
        minted = not token or not verify(request, token)
        if minted:
            token = issue(request)
        scope.setdefault("state", {})["csrf_token"] = token

        if not minted:
            await self.app(scope, receive, send)
            return

        await self.app(scope, receive, _sending_cookie(send, token))


def _is_urlencoded(request: Request) -> bool:
    return request.headers.get("content-type", "").startswith("application/x-www-form-urlencoded")


async def _drain(receive) -> bytes:
    """Read the whole request body off the ASGI channel."""
    body = b""
    more = True
    while more:
        message = await receive()
        if message["type"] != "http.request":
            break
        body += message.get("body", b"")
        more = message.get("more_body", False)
    return body


def _replay(body: bytes):
    """Hand the body back to the application exactly once, then hold."""
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return receive


def _token_from_form(body: bytes) -> Optional[str]:
    values = parse_qs(body.decode("utf-8", errors="replace"))
    found = values.get(FORM_FIELD)
    return found[0] if found else None


def _sending_cookie(send, token: str):
    """Attach the refreshed token to whatever response the application sends."""

    async def wrapped(message):
        if message["type"] == "http.response.start":
            carrier = Response()
            _set_cookie(carrier, token)
            message = dict(message)
            message["headers"] = list(message.get("headers", [])) + [
                (b"set-cookie", carrier.headers["set-cookie"].encode("latin-1"))
            ]
        await send(message)

    return wrapped
