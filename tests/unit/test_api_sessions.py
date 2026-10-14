"""Conversations through the API: lesson, answer, hand-off, kept per end user (2026-10-14).

The whole chain is faked (it has its own tests); the light turn is the real one
with a fake model, as in ``test_light_turns.py``.
"""

from __future__ import annotations

import base64
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from tests.unit.test_light_turns import _Client

from aix.core.config import config as app_config
from aix.webui.agent.service import StreamEvent
from aix.webui.db import Base, get_async_session
from aix.webui.lessons.models import LessonMessage

pytestmark = pytest.mark.unit

AUTH = "Basic " + base64.b64encode(b"chatbot-platform:segreto").decode()
LESSON_MD = "# La cellula\n\n## Introduzione *(10 minuti)*\n\nLa cellula è l'unità della vita."


def _as(user: str = "anna") -> dict[str, str]:
    return {"Authorization": AUTH, "X-End-User-Id": user}


async def _fake_chain(lesson, session):
    """What ``run_agent_stream`` does, without a model: phases, the lesson, persisted."""
    yield StreamEvent(kind="planner", payload={"intent": "lesson_creation", "scope": "in_scope"})
    yield StreamEvent(kind="retriever", payload={"nodes_count": 12})
    yield StreamEvent(kind="critic", payload={"approved": True, "score": 4.2})
    turn = max([m.turn_index for m in (await session.run_sync(lambda s: lesson.messages))] or [1])
    lesson.lesson_plan_md = LESSON_MD
    lesson.status = "complete"
    session.add(
        LessonMessage(
            lesson_id=lesson.id,
            role="assistant",
            content_md=LESSON_MD,
            turn_index=turn,
            agent_kind="writer",
            meta_json={"approved": True},
        )
    )
    await session.commit()
    yield StreamEvent(kind="done", lesson_plan_md=LESSON_MD, meta={"duration_seconds": 1.0})


@pytest.fixture
async def api(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_API_USERNAME", "chatbot-platform")
    monkeypatch.setenv("GRAPH_API_PWD", "segreto")
    monkeypatch.setenv("AIX_LIGHT_TURNS_ENABLED", "true")
    monkeypatch.setenv("AIX_TRACING_ENABLED", "false")
    monkeypatch.setattr("aix.webui.agent.turns.run_agent_stream", _fake_chain)

    from aix.api.main import app

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{(tmp_path / 'sessions.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async def _session():
        async with maker() as session:
            yield session

    app.dependency_overrides[get_async_session] = _session
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()
        await engine.dispose()


def _new_session(client, user="anna"):
    response = client.post(
        "/api/v1/sessions",
        json={
            "domain": "udl",
            "educational_profile": {"pedagogical_intent": "first_introduction"},
        },
        headers=_as(user),
    )
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


def _turn(client, session_id, user="anna", **body):
    return client.post(f"/api/v1/sessions/{session_id}/turns", json=body, headers=_as(user))


def test_the_first_turn_writes_the_lesson(api):
    session_id = _new_session(api)

    response = _turn(api, session_id, message="Crea una lezione sulla cellula")

    assert response.status_code == 200, response.text
    result = response.json()
    assert (result["kind"], result["turn_index"]) == ("lesson", 1)
    assert result["markdown"].startswith("# La cellula")
    assert set(result["phases"]) == {"planner", "retriever", "critic"}
    assert result["ai_disclosure"]
    assert response.headers["x-ai-generated"] == "true"

    detail = api.get(f"/api/v1/sessions/{session_id}", headers=_as()).json()
    assert detail["status"] == "complete"
    assert detail["lesson_markdown"].startswith("# La cellula")
    assert [(t["role"], t["kind"]) for t in detail["turns"]] == [
        ("user", "message"),
        ("assistant", "lesson"),
    ]


def test_a_question_after_the_lesson_is_answered_in_a_few_seconds(api, monkeypatch):
    session_id = _new_session(api)
    _turn(api, session_id, message="Crea una lezione sulla cellula")
    monkeypatch.setattr(
        app_config.openai, "get_async_client", lambda: _Client(["Perché ", "la cellula vive."])
    )

    result = _turn(api, session_id, message="sì").json()

    assert (result["kind"], result["turn_index"]) == ("answer", 2)
    assert result["markdown"] == "Perché la cellula vive."
    detail = api.get(f"/api/v1/sessions/{session_id}", headers=_as()).json()
    assert detail["turns"][-1]["kind"] == "answer"
    assert detail["lesson_markdown"].startswith("# La cellula"), "an answer leaves the lesson"


def test_a_request_for_a_new_lesson_comes_back_as_a_handoff(api, monkeypatch):
    session_id = _new_session(api)
    _turn(api, session_id, message="Crea una lezione sulla cellula")
    monkeypatch.setattr(
        app_config.openai, "get_async_client", lambda: _Client(["<<<NUOVA_", "LEZIONE>>>"])
    )

    result = _turn(api, session_id, message="Fammi una lezione sui vulcani").json()

    assert result["kind"] == "handoff"
    assert result["handoff_query"] == "Fammi una lezione sui vulcani"
    turns = api.get(f"/api/v1/sessions/{session_id}", headers=_as()).json()["turns"]
    assert len(turns) == 2, "the handed-off request is not left in the history unanswered"

    again = _turn(api, session_id, message=result["handoff_query"], mode="lesson").json()
    assert again["kind"] == "lesson"


def test_an_answer_needs_a_lesson_first(api):
    session_id = _new_session(api)

    response = _turn(api, session_id, message="Perché?", mode="answer")

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "no_lesson_yet"


def test_an_empty_turn_is_refused_with_a_code(api):
    session_id = _new_session(api)

    response = _turn(api, session_id)

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "empty_message"


def test_another_end_user_cannot_see_or_use_the_session(api):
    session_id = _new_session(api, user="anna")

    assert api.get(f"/api/v1/sessions/{session_id}", headers=_as("bruno")).status_code == 404
    assert _turn(api, session_id, user="bruno", message="ciao").status_code == 404
    assert api.get("/api/v1/sessions", headers=_as("bruno")).json() == []
    assert [s["session_id"] for s in api.get("/api/v1/sessions", headers=_as()).json()] == [
        session_id
    ]


def test_a_rate_limited_turn_says_when_to_retry(api, monkeypatch):
    from aix.core import concurrency

    def _limited(_key):
        raise concurrency.RateLimited(9.0, limit=5, window_s=60)

    monkeypatch.setattr("aix.webui.agent.turns.check_user_rate_limit", _limited)
    session_id = _new_session(api)

    response = _turn(api, session_id, message="Crea")

    assert response.status_code == 429
    assert response.json()["detail"]["code"] == "rate_limited"
    assert response.headers["retry-after"] == "10"


def test_documents_need_the_pipeline_on(api):
    session_id = _new_session(api)

    response = api.post(
        f"/api/v1/sessions/{session_id}/materials",
        json={"material_ids": ["00000000-0000-0000-0000-000000000001"]},
        headers=_as(),
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "materials_disabled"


def test_the_stream_ends_with_the_result(api):
    session_id = _new_session(api)

    with api.stream(
        "POST",
        f"/api/v1/sessions/{session_id}/turns/stream",
        json={"message": "Crea una lezione sulla cellula"},
        headers=_as(),
    ) as response:
        frames = [
            json.loads(line[len("data: ") :])
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]

    assert [f["kind"] for f in frames][:3] == ["planner", "retriever", "critic"]
    assert frames[-1]["kind"] == "result"
    assert frames[-1]["result"]["kind"] == "lesson"
    assert frames[-1]["result"]["markdown"].startswith("# La cellula")


def test_a_session_can_be_deleted(api):
    session_id = _new_session(api)

    assert api.delete(f"/api/v1/sessions/{session_id}", headers=_as()).status_code == 204
    assert api.get(f"/api/v1/sessions/{session_id}", headers=_as()).status_code == 404
