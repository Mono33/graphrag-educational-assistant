"""What the API shows to someone with no credentials (2026-10-20).

Staging serves the public WebUI, so there is no proxy authentication in front of
the API: what the application does not protect, anyone can call. Found by
calling it with no credentials: POST /api/v1/context answered (and spent model
calls), and the whole contract was readable at /docs and /openapi.json.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.unit


def test_the_context_route_needs_credentials(monkeypatch):
    monkeypatch.setenv("GRAPH_API_USERNAME", "chatbot-platform")
    monkeypatch.setenv("GRAPH_API_PWD", "segreto")
    from aix.api.main import app

    response = TestClient(app).post(
        "/api/v1/context", json={"query": "strategie per ADHD", "domain": "neuro"}
    )

    assert response.status_code == 401


@pytest.mark.parametrize(
    ("environment", "flag", "expected"),
    [
        ("staging", None, False),
        ("production", None, False),
        ("", None, True),
        ("development", None, True),
        ("staging", "true", True),
        ("", "false", False),
    ],
)
def test_the_api_docs_are_off_where_the_service_is_public(monkeypatch, environment, flag, expected):
    from aix.core.config import api_docs_enabled

    monkeypatch.setenv("ENVIRONMENT", environment)
    if flag is None:
        monkeypatch.delenv("AIX_API_DOCS_ENABLED", raising=False)
    else:
        monkeypatch.setenv("AIX_API_DOCS_ENABLED", flag)

    assert api_docs_enabled() is expected
