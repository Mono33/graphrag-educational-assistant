"""Documents through the API: upload once, read the state, kept per end user (2026-10-14)."""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from tests.unit.test_shadow_upload import (  # noqa: F401 — fixtures and helpers
    DOCX_MIME,
    _docx,
    _drain_one,
    _rag_on,
)

from aix.webui.db import Base, get_async_session

pytestmark = pytest.mark.unit

AUTH = "Basic " + base64.b64encode(b"chatbot-platform:segreto").decode()


@pytest.fixture
async def api(tmp_path, monkeypatch, _rag_on):
    monkeypatch.setenv("GRAPH_API_USERNAME", "chatbot-platform")
    monkeypatch.setenv("GRAPH_API_PWD", "segreto")
    monkeypatch.setenv("AIX_MATERIALS_ENABLED", "true")
    monkeypatch.setenv("AIX_STORAGE_PROVIDER", "local")
    monkeypatch.setenv("AIX_STORAGE_LOCAL_ROOT", str(tmp_path / "objects"))

    from aix.api.main import app

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{(tmp_path / 'api.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    monkeypatch.setattr("aix.webui.db.async_session_maker", maker)

    async def _session():
        async with maker() as session:
            yield session

    app.dependency_overrides[get_async_session] = _session
    try:
        yield TestClient(app), maker
    finally:
        app.dependency_overrides.clear()
        await engine.dispose()


def _as(user: str) -> dict[str, str]:
    return {"Authorization": AUTH, "X-End-User-Id": user}


def _upload(client, user="anna", name="programma.docx", content=None, mime=DOCX_MIME):
    return client.post(
        "/api/v1/materials",
        files={"file": (name, content if content is not None else _docx(), mime)},
        headers=_as(user),
    )


async def test_an_upload_is_processing_then_ready_with_its_passages(api, tmp_path):
    client, maker = api

    response = _upload(client)

    assert response.status_code == 202, response.text
    material = response.json()
    assert (material["name"], material["status"]) == ("programma.docx", "processing")
    assert (
        client.get(f"/api/v1/materials/{material['material_id']}", headers=_as("anna")).json()[
            "status"
        ]
        == "processing"
    )

    await _drain_one(maker, tmp_path)

    ready = client.get(f"/api/v1/materials/{material['material_id']}", headers=_as("anna")).json()
    assert ready["status"] == "ready"
    assert ready["passages"] > 0
    assert ready["quality"]["parser"]


async def test_an_end_user_never_sees_another_s_documents(api):
    client, _ = api
    material_id = _upload(client, user="anna").json()["material_id"]

    assert client.get(f"/api/v1/materials/{material_id}", headers=_as("bruno")).status_code == 404
    assert client.get("/api/v1/materials", headers=_as("bruno")).json() == []
    assert (
        client.delete(f"/api/v1/materials/{material_id}", headers=_as("bruno")).status_code == 404
    )
    assert [
        m["material_id"] for m in client.get("/api/v1/materials", headers=_as("anna")).json()
    ] == [material_id]


async def test_a_file_the_webui_refuses_is_refused_with_its_code(api):
    client, _ = api

    response = _upload(client, name="programma.pdf", content=b"MZ\x90\x00", mime="application/pdf")

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "magic_mismatch"


async def test_with_the_pipeline_off_the_answer_says_so(api, monkeypatch):
    client, _ = api
    monkeypatch.setenv("AIX_MATERIALS_ENABLED", "false")

    response = _upload(client)

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "materials_disabled"


async def test_a_deleted_document_is_gone_from_the_library(api):
    client, _ = api
    material_id = _upload(client).json()["material_id"]

    assert client.delete(f"/api/v1/materials/{material_id}", headers=_as("anna")).status_code == 204
    assert client.get(f"/api/v1/materials/{material_id}", headers=_as("anna")).status_code == 404


def test_without_credentials_there_is_no_library(api):
    client, _ = api

    assert client.get("/api/v1/materials").status_code == 401
