"""One service account per platform instance (2026-10-16).

AIxLearning runs production, corporate and staging, each with numeric user ids
of its own. With one shared account, user 42 of corporate and user 42 of
production were the same owner here. ``AIX_API_CLIENTS`` gives each instance
its credentials, and the client is part of every end user's identity.
"""

from __future__ import annotations

import base64
import json

import pytest
from fastapi.security import HTTPBasicCredentials
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from aix.api.principals import (
    SERVICE_ACCOUNT,
    ServiceCaller,
    api_email,
    owner_for,
    require_caller,
    service_clients,
)
from aix.webui.db import Base

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clients(monkeypatch):
    monkeypatch.setenv("GRAPH_API_USERNAME", "aiforl-graph")
    monkeypatch.setenv("GRAPH_API_PWD", "storico")
    monkeypatch.setenv(
        "AIX_API_CLIENTS", json.dumps({"aixl-corporate": "corp-segreto", "aixl-staging": "stg"})
    )


@pytest.fixture
async def session(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 't.db').as_posix()}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as s:
        yield s
    await engine.dispose()


async def _login(name, password):
    return await require_caller(HTTPBasicCredentials(username=name, password=password), None)


async def test_each_instance_signs_in_with_its_own_credentials():
    corporate = await _login("aixl-corporate", "corp-segreto")
    historic = await _login("aiforl-graph", "storico")

    assert corporate == SERVICE_ACCOUNT and corporate.client == "aixl-corporate"
    assert historic.client == "aiforl-graph"


async def test_one_instance_s_password_does_not_open_another():
    with pytest.raises(Exception) as refused:
        await _login("aixl-corporate", "storico")

    assert getattr(refused.value, "status_code", None) == 401


async def test_user_42_of_two_instances_is_two_people(session):
    production = await owner_for(session, ServiceCaller("aiforl-graph"), "42")
    corporate = await owner_for(session, ServiceCaller("aixl-corporate"), "42")

    assert production.id != corporate.id


async def test_the_historic_account_keeps_the_identities_it_already_made(session):
    """Before this change the client was always ``GRAPH_API_USERNAME``."""
    owner = await owner_for(session, ServiceCaller("aiforl-graph"), "42")

    assert owner.email == api_email("aiforl-graph", "42")
    assert (await owner_for(session, SERVICE_ACCOUNT, "42")).id == owner.id


@pytest.mark.parametrize("raw", ["non json", '["a", "b"]', '{"vuoto": ""}'])
def test_a_malformed_client_list_adds_nobody_and_keeps_the_historic_one(monkeypatch, raw):
    monkeypatch.setenv("AIX_API_CLIENTS", raw)

    assert service_clients() == {"aiforl-graph": "storico"}


def test_the_platform_s_instances_reach_the_api_with_their_own_owner(monkeypatch):
    """End to end: the same ``X-End-User-Id`` from two instances, two libraries."""
    from fastapi.testclient import TestClient

    from aix.api.main import app

    client = TestClient(app)

    def headers(name, password, user="42"):
        token = base64.b64encode(f"{name}:{password}".encode()).decode()
        return {"Authorization": f"Basic {token}", "X-End-User-Id": user}

    assert client.get("/api/v1/catalog", headers=headers("aixl-staging", "stg")).status_code == 200
    assert client.get("/api/v1/catalog", headers=headers("aixl-staging", "x")).status_code == 401
