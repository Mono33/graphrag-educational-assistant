"""A retried request gets the first answer, not a second run (2026-10-16).

The platform's Dramatiq worker retries a failed job up to three times. With an
``Idempotency-Key`` a retry of a turn that already finished is answered with
that turn's result, without a second lesson and a second bill.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException
from sqlalchemy import update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from tests.unit.test_api_sessions import (  # noqa: F401 — the fixture
    _as,
    _fake_chain,
    _new_session,
    api,
)

from aix.api import idempotency
from aix.webui.agent.service import StreamEvent
from aix.webui.auth.models import User
from aix.webui.db import Base

pytestmark = pytest.mark.unit

KEY = {"Idempotency-Key": "job-4711-turn-1"}


@pytest.fixture
def runs(monkeypatch):
    """How many times the chain really ran."""
    count = {"n": 0}

    async def counted(lesson, session):
        count["n"] += 1
        async for event in _fake_chain(lesson, session):
            yield event

    monkeypatch.setattr("aix.webui.agent.turns.run_agent_stream", counted)
    return count


def _turn(client, session_id, headers=KEY, message="Crea una lezione sulla cellula"):
    # ``mode: lesson``: every turn goes to the (counted) chain, none to a fast answer.
    return client.post(
        f"/api/v1/sessions/{session_id}/turns",
        json={"message": message, "mode": "lesson"},
        headers={**_as(), **headers},
    )


def test_a_retried_turn_is_answered_with_the_first_result(api, runs):  # noqa: F811
    session_id = _new_session(api)

    first = _turn(api, session_id)
    again = _turn(api, session_id)

    assert first.status_code == again.status_code == 200
    assert again.json() == first.json()
    assert again.headers["idempotent-replayed"] == "true"
    assert "idempotent-replayed" not in first.headers
    assert runs["n"] == 1, "one lesson, one bill"


def test_without_a_key_every_request_is_a_new_turn(api, runs):  # noqa: F811
    session_id = _new_session(api)

    _turn(api, session_id, headers={})
    _turn(api, session_id, headers={}, message="Rifalla per la terza media")

    assert runs["n"] == 2


def test_the_same_key_for_another_request_is_refused(api, runs):  # noqa: F811
    session_id = _new_session(api)
    _turn(api, session_id)

    reused = _turn(api, session_id, message="Un'altra lezione")

    assert reused.status_code == 422
    assert reused.json()["detail"]["code"] == "idempotency_key_reused"


def test_a_failed_turn_frees_its_key(api, monkeypatch):  # noqa: F811
    attempts = {"n": 0}

    async def flaky(lesson, session):
        attempts["n"] += 1
        if attempts["n"] == 1:
            lesson.status = "error"
            await session.commit()
            yield StreamEvent(kind="error", error="provider down")
            return
        async for event in _fake_chain(lesson, session):
            yield event

    monkeypatch.setattr("aix.webui.agent.turns.run_agent_stream", flaky)
    session_id = _new_session(api)

    assert _turn(api, session_id).status_code == 502
    retried = _turn(api, session_id)

    assert retried.status_code == 200 and retried.json()["kind"] == "lesson"
    assert "idempotent-replayed" not in retried.headers


def test_the_stream_replays_a_turn_the_blocking_call_made(api, runs):  # noqa: F811
    """A client may retry ``/turns`` through ``/turns/stream``: one key, one turn."""
    session_id = _new_session(api)
    first = _turn(api, session_id).json()

    with api.stream(
        "POST",
        f"/api/v1/sessions/{session_id}/turns/stream",
        json={"message": "Crea una lezione sulla cellula", "mode": "lesson"},
        headers={**_as(), **KEY},
    ) as response:
        assert response.headers["idempotent-replayed"] == "true"
        frames = [
            json.loads(line[len("data: ") :])
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]

    assert frames == [{"kind": "result", "result": first}]
    assert runs["n"] == 1


def test_a_retried_session_is_the_same_session(api):  # noqa: F811
    body = {"domain": "udl", "educational_profile": {"pedagogical_intent": "intro"}}
    headers = {**_as(), "Idempotency-Key": "job-4711-session"}

    first = api.post("/api/v1/sessions", json=body, headers=headers)
    again = api.post("/api/v1/sessions", json=body, headers=headers)

    assert (first.status_code, again.status_code) == (201, 201)
    assert again.json()["session_id"] == first.json()["session_id"]
    assert len(api.get("/api/v1/sessions", headers=_as()).json()) == 1


def test_keys_are_per_end_user(api, runs):  # noqa: F811
    anna = _new_session(api, user="anna")
    bruno = _new_session(api, user="bruno")

    _turn(api, anna)
    other = api.post(
        f"/api/v1/sessions/{bruno}/turns",
        json={"message": "Crea una lezione sulla cellula", "mode": "lesson"},
        headers={**_as("bruno"), **KEY},
    )

    assert other.status_code == 200 and "idempotent-replayed" not in other.headers
    assert runs["n"] == 2


# ---------------------------------------------------------------------------
# The record itself: in progress, orphaned, invalid
# ---------------------------------------------------------------------------


@pytest.fixture
async def db(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'i.db').as_posix()}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        owner = User(
            id=uuid.uuid4(),
            email="api-x@api.invalid",
            hashed_password="x",
            is_active=True,
            is_superuser=False,
            is_verified=True,
        )
        session.add(owner)
        await session.commit()
        yield session, owner.id
    await engine.dispose()


async def _begin(session, owner_id, key="k1", fingerprint="f1"):
    return await idempotency.begin(
        session, owner_id=owner_id, scope="turn:x", key=key, fingerprint=fingerprint
    )


async def test_a_request_still_running_says_so_and_when_to_retry(db):
    session, owner_id = db
    assert isinstance(await _begin(session, owner_id), idempotency.Claim)

    with pytest.raises(HTTPException) as busy:
        await _begin(session, owner_id)

    assert busy.value.status_code == 409
    assert busy.value.detail["code"] == "idempotency_in_progress"
    assert busy.value.headers["Retry-After"] == str(idempotency.RETRY_AFTER_SECONDS)


async def test_a_key_orphaned_by_a_crash_is_taken_over(db):
    session, owner_id = db
    await _begin(session, owner_id)
    long_ago = datetime.now(timezone.utc) - idempotency.STALE_AFTER - timedelta(minutes=1)
    await session.execute(update(idempotency.ApiIdempotencyKey).values(created_at=long_ago))
    await session.commit()

    assert isinstance(await _begin(session, owner_id), idempotency.Claim)


async def test_a_finished_key_replays_and_expires_after_its_ttl(db):
    session, owner_id = db
    claim = await _begin(session, owner_id)
    await idempotency.finish(session, claim, 200, {"kind": "answer"})

    replay = await _begin(session, owner_id)
    assert replay == idempotency.Replay(200, {"kind": "answer"})

    expired = datetime.now(timezone.utc) - idempotency.TTL - timedelta(minutes=1)
    await session.execute(update(idempotency.ApiIdempotencyKey).values(created_at=expired))
    await session.commit()
    assert isinstance(await _begin(session, owner_id), idempotency.Claim)


@pytest.mark.parametrize("bad", ["x" * 201, "riga\nnuova"])
async def test_an_unusable_key_is_a_422(db, bad):
    session, owner_id = db

    with pytest.raises(HTTPException) as refused:
        await _begin(session, owner_id, key=bad)

    assert refused.value.detail["code"] == "invalid_idempotency_key"
