"""CSRF protection on the WebUI's mutating requests (Fase 0 · PR 1).

Before this, nothing checked where a state-changing request came from. The
session lives in a cookie, so a page the teacher happened to have open could
make their browser delete a lesson or an uploaded file on their behalf, with
their credentials attached and no trace that it was not them.

What these tests pin is the whole shape of the defence, because each half is
useless alone:

  * an unsigned mutating request is refused, and a signed one is not;
  * both halves of the double submit are required — the cookie the browser
    holds *and* the echo the page produced, since only a page served by this
    application can read the cookie to echo it;
  * the signature is over the *session*, so a token minted while logged out
    cannot be replayed once a session exists (login CSRF);
  * ``/api/v1`` is deliberately untouched, because it authenticates per
    request rather than from ambient cookie authority.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest
from fastapi import FastAPI, Request
from fastapi import Form as FormField
from fastapi.testclient import TestClient

from aix.webui.auth.backend import _COOKIE_NAME as AUTH_COOKIE
from aix.webui.auth.csrf import (
    COOKIE_NAME,
    FORM_FIELD,
    HEADER_NAME,
    MAX_AGE_S,
    CSRFMiddleware,
    issue,
    verify,
)

pytestmark = pytest.mark.unit


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(CSRFMiddleware)

    @app.get("/webui/page")
    async def page(request: Request):
        return {"token": request.state.csrf_token}

    @app.post("/webui/lesson/{lesson_id}/upload")
    async def upload(lesson_id: str):
        return {"ok": True}

    @app.delete("/webui/lesson/{lesson_id}")
    async def delete(lesson_id: str):
        return {"ok": True}

    @app.post("/api/v1/agent/run")
    async def api_run():
        return {"ok": True}

    return app


@pytest.fixture
def client():
    with TestClient(_app()) as c:
        yield c


def _visit(client) -> str:
    """Load a page the way a browser would, and keep the cookie it sets."""
    return client.get("/webui/page").json()["token"]


# ---------------------------------------------------------------------------
# The middleware
# ---------------------------------------------------------------------------


def test_a_get_issues_a_token_and_the_cookie_that_carries_it(client):
    response = client.get("/webui/page")

    assert response.status_code == 200
    assert client.cookies[COOKIE_NAME] == response.json()["token"]


def test_the_cookie_is_readable_by_the_page():
    """Unlike the session cookie: the page has to echo it back to prove itself."""
    with TestClient(_app()) as c:
        header = c.get("/webui/page").headers["set-cookie"]

    assert "httponly" not in header.lower(), "a token the page cannot read cannot be echoed"
    assert "samesite=lax" in header.lower()


@pytest.mark.parametrize(
    "method, path",
    [
        ("post", "/webui/lesson/abc/upload"),
        ("delete", "/webui/lesson/abc"),
    ],
)
def test_a_mutating_request_without_a_token_is_refused(client, method, path):
    _visit(client)  # the teacher has a session and a cookie; the request has no echo

    assert getattr(client, method)(path).status_code == 403


@pytest.mark.parametrize(
    "method, path",
    [
        ("post", "/webui/lesson/abc/upload"),
        ("delete", "/webui/lesson/abc"),
    ],
)
def test_a_mutating_request_with_the_token_goes_through(client, method, path):
    token = _visit(client)

    response = getattr(client, method)(path, headers={HEADER_NAME: token})
    assert response.status_code == 200


def test_a_forged_token_is_refused(client):
    """The attacker guessed the format but not the key."""
    _visit(client)
    forged = f"nonce.{int(time.time())}.{'0' * 64}"

    assert client.post("/webui/lesson/abc/upload", headers={HEADER_NAME: forged}).status_code == 403


def test_a_token_from_another_browser_is_refused(client):
    """Signature alone is not enough — it must also match the cookie in hand."""
    _visit(client)
    with TestClient(_app()) as other:
        stolen = _visit(other)

    assert client.post("/webui/lesson/abc/upload", headers={HEADER_NAME: stolen}).status_code == 403


def test_reading_is_never_blocked(client):
    assert client.get("/webui/page").status_code == 200


def test_the_json_api_is_left_alone(client):
    """It authenticates per request, so there is no ambient authority to borrow."""
    assert client.post("/api/v1/agent/run").status_code == 200


def test_the_flag_turns_enforcement_off_but_still_issues(client):
    _visit(client)

    with patch("aix.webui.auth.csrf._enabled", return_value=False):
        assert client.post("/webui/lesson/abc/upload").status_code == 200


# ---------------------------------------------------------------------------
# The token itself
# ---------------------------------------------------------------------------


def _request(cookies: dict[str, str]) -> Request:
    raw = "; ".join(f"{k}={v}" for k, v in cookies.items()).encode()
    return Request({"type": "http", "headers": [(b"cookie", raw)] if raw else []})


def test_a_token_verifies_against_the_session_it_was_issued_for():
    request = _request({AUTH_COOKIE: "session-one"})
    token = issue(request)

    assert verify(_request({AUTH_COOKIE: "session-one", COOKIE_NAME: token}), token)


def test_a_token_minted_while_logged_out_cannot_be_used_once_logged_in():
    """Login CSRF: an attacker cannot pin a token onto a session they then wait for."""
    anonymous = _request({})
    token = issue(anonymous)

    assert verify(_request({COOKIE_NAME: token}), token), "valid while still anonymous"
    assert not verify(_request({AUTH_COOKIE: "session-one", COOKIE_NAME: token}), token)


def test_one_users_token_does_not_work_for_another():
    token = issue(_request({AUTH_COOKIE: "session-one"}))

    assert not verify(_request({AUTH_COOKIE: "session-two", COOKIE_NAME: token}), token)


def test_an_expired_token_is_refused():
    request = _request({AUTH_COOKIE: "session-one"})
    with patch("aix.webui.auth.csrf.time.time", return_value=time.time() - MAX_AGE_S - 1):
        stale = issue(request)

    assert not verify(_request({AUTH_COOKIE: "session-one", COOKIE_NAME: stale}), stale)


def test_a_token_from_the_future_is_refused():
    """A clock fault, or a forgery built to outlive the expiry check."""
    with patch("aix.webui.auth.csrf.time.time", return_value=time.time() + 3600):
        ahead = issue(_request({}))

    assert not verify(_request({COOKIE_NAME: ahead}), ahead)


@pytest.mark.parametrize(
    "submitted",
    ["", None, "not-a-token", "a.b", "a.b.c.d", "a.notanumber.c"],
    ids=["empty", "absent", "shapeless", "too few parts", "too many parts", "bad timestamp"],
)
def test_malformed_tokens_are_refused_without_raising(submitted):
    assert not verify(_request({COOKIE_NAME: "x"}), submitted)


def test_a_token_without_the_cookie_is_refused():
    """Half of a double submit is not a double submit."""
    token = issue(_request({}))

    assert not verify(_request({}), token)


# ---------------------------------------------------------------------------
# The form-field path — a plain <form> cannot set a header
# ---------------------------------------------------------------------------


def _form_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(CSRFMiddleware)

    @app.get("/webui/new")
    async def new(request: Request):
        return {"token": request.state.csrf_token}

    @app.post("/webui/lesson")
    async def create(title: str = FormField(...)):
        # Reading the body proves the middleware replayed what it consumed.
        return {"title": title}

    return app


@pytest.fixture
def form_client():
    with TestClient(_form_app()) as c:
        yield c


def test_a_form_post_without_the_field_is_refused(form_client):
    form_client.get("/webui/new")

    assert form_client.post("/webui/lesson", data={"title": "Memoria"}).status_code == 403


def test_a_form_post_with_the_field_goes_through(form_client):
    token = form_client.get("/webui/new").json()["token"]

    response = form_client.post("/webui/lesson", data={"title": "Memoria", FORM_FIELD: token})
    assert response.status_code == 200


def test_the_handler_still_receives_the_whole_body(form_client):
    """The middleware consumes the body to find the token; the route needs it too."""
    token = form_client.get("/webui/new").json()["token"]

    response = form_client.post(
        "/webui/lesson", data={"title": "Memoria di lavoro", FORM_FIELD: token}
    )
    assert response.json() == {"title": "Memoria di lavoro"}


def test_a_forged_field_is_refused(form_client):
    form_client.get("/webui/new")
    forged = f"nonce.{int(time.time())}.{'0' * 64}"

    assert (
        form_client.post("/webui/lesson", data={"title": "x", FORM_FIELD: forged}).status_code
        == 403
    )


def test_an_upload_is_never_buffered(form_client):
    """Multipart bodies can be large; only urlencoded forms are read."""
    form_client.get("/webui/new")

    with patch("aix.webui.auth.csrf._drain") as drained:
        form_client.post("/webui/lesson", files={"f": ("a.txt", b"x" * 1024)})

    drained.assert_not_called()
