"""One error shape on the integration routes (2026-10-21).

A client of ``/sessions``, ``/materials`` and ``/catalog`` reads ``detail.code``
and ``detail.message`` on every error, whatever raised it. The older routes keep
the shapes their callers know.
"""

from __future__ import annotations

import pytest
from tests.unit.test_api_sessions import _as, _new_session, api  # noqa: F401 -- fixture

pytestmark = pytest.mark.unit

NO_CREDENTIALS = {"X-End-User-Id": "anna"}


def test_wrong_credentials_have_a_code(api):  # noqa: F811
    response = api.get("/api/v1/sessions", headers=NO_CREDENTIALS)

    assert response.status_code == 401
    assert response.json() == {
        "detail": {"code": "unauthorized", "message": "Credenziali mancanti o non valide."}
    }


def test_a_malformed_request_says_which_field(api):  # noqa: F811
    response = api.post("/api/v1/sessions", json={"domain": "storia"}, headers=_as())

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "invalid_request"
    assert detail["message"] == "La richiesta non è valida: domain."
    assert detail["errors"][0]["loc"] == ["body", "domain"]


def test_a_refusal_keeps_its_own_code(api):  # noqa: F811
    session_id = _new_session(api)

    response = api.post(
        f"/api/v1/sessions/{session_id}/turns", json={"message": "  "}, headers=_as()
    )

    assert response.json()["detail"]["code"] == "empty_message"


def test_an_unknown_address_under_the_routes_has_a_code(api):  # noqa: F811
    response = api.get("/api/v1/sessions/abc/nothing-here", headers=_as())

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "not_found"


def test_the_older_routes_keep_their_shape(api):  # noqa: F811
    response = api.post("/api/v1/context", json={"query": "x"}, headers=NO_CREDENTIALS)

    assert (response.status_code, response.json()) == (401, {"detail": "Unauthorized"})


def test_the_contract_documents_the_error_shape(api):  # noqa: F811
    spec = api.app.openapi()
    create = spec["paths"]["/api/v1/sessions"]["post"]["responses"]

    assert create["422"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ErrorResponse"
    }
    assert "401" in create
