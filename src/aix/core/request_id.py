"""One id per request, from AIxLearning to our logs and Langfuse (2026-10-23).

A problem reported by the platform -- «the turn of 10:42 failed» -- had to be
found by time and guesswork. Now every request carries an id: the caller's
``X-Request-ID`` when it sends a usable one (a Dramatiq message id, say),
otherwise one made here. It comes back in the response's ``X-Request-ID``, is
written on every log line of the request (``[rid]``), and goes into the
metadata of the request's Langfuse trace (``core.tracing.start_run``).
"""

from __future__ import annotations

import contextvars
import logging
import re
import uuid
from typing import Any, Optional

HEADER = "X-Request-ID"
_HEADER_KEY = HEADER.lower().encode("latin-1")

#: What a caller's id may look like; anything else is replaced, not trusted.
_USABLE = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")

_current: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_id", default=None)


def current() -> Optional[str]:
    """The id of the request being served, if any."""
    return _current.get()


def _from_headers(headers: list[tuple[bytes, bytes]]) -> str:
    for key, value in headers:
        if key.lower() == _HEADER_KEY:
            candidate = value.decode("latin-1").strip()
            if _USABLE.match(candidate):
                return candidate
    return uuid.uuid4().hex


class RequestIdMiddleware:
    """Pure ASGI, so the id is set before the app runs and every task the
    request starts -- a stream's generator included -- inherits it."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        rid = _from_headers(scope.get("headers") or [])
        token = _current.set(rid)

        async def send_with_id(message: dict) -> None:
            if message.get("type") == "http.response.start":
                headers = [
                    (k, v) for k, v in message.get("headers") or [] if k.lower() != _HEADER_KEY
                ]
                headers.append((_HEADER_KEY, rid.encode("latin-1")))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            _current.reset(token)


class RequestIdFilter(logging.Filter):
    """Puts ``request_id`` on every record, ``-`` outside a request."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _current.get() or "-"
        return True
