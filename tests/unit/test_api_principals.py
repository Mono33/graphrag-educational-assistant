"""An API caller acts as a user, and end users of a client are kept apart (2026-10-14)."""

from __future__ import annotations

import uuid

import pytest
from fastapi import HTTPException
from fastapi_users.password import PasswordHelper
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from aix.api.principals import SERVICE_ACCOUNT, api_email, owner_for
from aix.webui.auth.models import User
from aix.webui.db import Base

pytestmark = pytest.mark.unit


@pytest.fixture
async def session(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_API_USERNAME", "chatbot-platform")
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'p.db').as_posix()}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as s:
        yield s
    await engine.dispose()


async def test_the_same_end_user_is_the_same_owner_every_time(session):
    first = await owner_for(session, SERVICE_ACCOUNT, "studente-42")
    again = await owner_for(session, SERVICE_ACCOUNT, "studente-42")

    assert first.id == again.id


async def test_two_end_users_are_two_owners(session):
    anna = await owner_for(session, SERVICE_ACCOUNT, "anna")
    bruno = await owner_for(session, SERVICE_ACCOUNT, "bruno")
    nobody = await owner_for(session, SERVICE_ACCOUNT, None)

    assert len({anna.id, bruno.id, nobody.id}) == 3


async def test_the_same_id_from_two_clients_is_two_owners(session, monkeypatch):
    here = await owner_for(session, SERVICE_ACCOUNT, "anna")
    monkeypatch.setenv("GRAPH_API_USERNAME", "another-platform")
    there = await owner_for(session, SERVICE_ACCOUNT, "anna")

    assert here.id != there.id


async def test_a_provisioned_user_cannot_sign_in_and_hides_the_client_s_id(session):
    user = await owner_for(session, SERVICE_ACCOUNT, "mario.rossi@scuola.it")

    assert user.email.endswith("@api.invalid")
    assert "mario" not in user.email
    for guess in ("", "mario.rossi@scuola.it", "password"):
        verified, _ = PasswordHelper().verify_and_update(guess, user.hashed_password)
        assert not verified


async def test_a_webui_user_is_itself_whatever_end_user_it_names(session):
    teacher = User(
        id=uuid.uuid4(),
        email="anna@scuola.it",
        hashed_password="x",
        is_active=True,
        is_superuser=False,
        is_verified=True,
    )

    assert await owner_for(session, teacher, "someone-else") is teacher


@pytest.mark.parametrize("bad", ["x" * 201, "riga\nnuova"])
async def test_an_unusable_end_user_id_is_a_422(session, bad):
    with pytest.raises(HTTPException) as refused:
        await owner_for(session, SERVICE_ACCOUNT, bad)

    assert refused.value.status_code == 422
    assert refused.value.detail["code"] == "invalid_end_user_id"


def test_the_address_is_deterministic_and_per_client():
    assert api_email("platform", "anna") == api_email("platform", "anna")
    assert api_email("platform", "anna") != api_email("other", "anna")
