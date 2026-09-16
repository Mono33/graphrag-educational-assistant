"""Logging out changes state, so it is a POST (Louis's review, 2026-09-15).

``GET /auth/logout`` was a link. Any page a teacher visits can fire a GET at
another site — an ``<img src>`` is enough — so a third-party page could log a
teacher out in the middle of writing a lesson. Nothing is stolen; the work in
progress is simply gone.

As a POST it sits under ``/auth``, which the CSRF middleware guards, so the
request has to come from a page this application served.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.unit


@pytest.fixture
def client():
    from aix.api.main import app

    return TestClient(app, follow_redirects=False)


def test_logout_is_not_reachable_with_a_get(client):
    """The shape of the fix: the old link no longer logs anyone out."""
    assert client.get("/auth/logout").status_code == 405


def test_logout_without_a_csrf_token_is_refused(client, monkeypatch):
    monkeypatch.setenv("WEBUI_CSRF_ENABLED", "true")

    assert client.post("/auth/logout").status_code == 403


def test_logout_clears_the_session_cookie(client, monkeypatch):
    monkeypatch.setenv("WEBUI_CSRF_ENABLED", "false")

    response = client.post("/auth/logout")

    assert response.status_code == 303
    assert response.headers["location"] == "/webui/"
    from aix.webui.auth.backend import _COOKIE_NAME as AUTH_COOKIE

    assert AUTH_COOKIE in response.headers.get("set-cookie", "")
