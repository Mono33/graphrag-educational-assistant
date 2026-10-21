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
            "educational_profile": {"pedagogical_intent": "intro"},
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
    assert result["planner"]["intent"] == "lesson_creation"
    assert result["retriever"]["nodes_count"] == 12
    assert result["critic"]["approved"] is True
    assert result["lesson_meta"]["duration_seconds"] == 1.0
    assert result["structure"]["sections"][0] == {
        "title": "Introduzione (10 minuti)",
        "level": 2,
        "minutes": 10,
    }
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
    assert result["answer_meta"]["passages"] == 0
    assert result["planner"] is None, "an answer has no planner card"
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


def test_the_media_panel_is_empty_when_the_live_search_is_off(api, monkeypatch):
    monkeypatch.setenv("AIX_MEDIA_LIVE_ENABLED", "false")
    session_id = _new_session(api)

    response = api.get(f"/api/v1/sessions/{session_id}/media", headers=_as())

    assert response.status_code == 200
    assert response.json() == {
        "turn_index": None,
        "curated": {},
        "videos": [],
        "citations": [],
        "wikipedia": [],
    }


def test_the_catalog_has_the_webui_s_choices(api):
    catalog = api.get("/api/v1/catalog", headers=_as()).json()

    assert {d["code"] for d in catalog["domains"]} == {"neuro", "udl"}
    assert any(i["code"] == "intro" for i in catalog["pedagogical_intents"])
    assert {r["code"] for r in catalog["refinements"]} >= {"simplify", "custom"}
    assert catalog["profile"]["disabilities"]
    assert ".pdf" in catalog["materials"]["extensions"]


@pytest.mark.parametrize(
    ("fmt", "media_type", "magic"),
    [
        ("md", "text/markdown", b"# La cellula"),
        ("pdf", "application/pdf", b"%PDF"),
        ("docx", "application/vnd.openxmlformats-officedocument", b"PK"),
        ("outline", "text/plain", None),
    ],
)
def test_the_lesson_downloads_in_every_format(api, fmt, media_type, magic):
    session_id = _new_session(api)
    _turn(api, session_id, message="Crea una lezione sulla cellula")

    response = api.get(f"/api/v1/sessions/{session_id}/export?format={fmt}", headers=_as())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(media_type)
    assert response.headers["x-ai-generated"] == "true"
    if magic:
        assert response.content.startswith(magic)


def test_no_lesson_no_download_and_a_bad_format_says_which_exist(api):
    session_id = _new_session(api)

    empty = api.get(f"/api/v1/sessions/{session_id}/export?format=pdf", headers=_as())
    assert (empty.status_code, empty.json()["detail"]["code"]) == (404, "no_lesson_yet")

    _turn(api, session_id, message="Crea una lezione sulla cellula")
    bad = api.get(f"/api/v1/sessions/{session_id}/export?format=odt", headers=_as())
    assert (bad.status_code, bad.json()["detail"]["code"]) == (400, "unsupported_format")


async def _fake_chain_with_the_judge(lesson, session):
    """The chain with AIX_JUDGE_ENABLED: no Critic, the judge's card instead."""
    async for event in _fake_chain(lesson, session):
        if event.kind == "critic":
            yield StreamEvent(kind="judge_pending", payload={"write_revision": 1})
            yield StreamEvent(
                kind="judge",
                payload={
                    "found": 2,
                    "replaced": 1,
                    "fixed": [
                        {
                            "check": "fatti",
                            "why": "data sbagliata",
                            "before": "1792",
                            "after": "1789",
                            "kind": "sostituzione",
                            "teacher_material": True,
                        }
                    ],
                    "points": [{"check": "classe", "severity": "minore", "quote": "q", "why": "w"}],
                    "notes": 1,
                    "write_revision": 1,
                },
            )
            continue
        yield event


def test_with_the_judge_on_the_turn_returns_its_corrections_typed(api, monkeypatch):
    monkeypatch.setattr("aix.webui.agent.turns.run_agent_stream", _fake_chain_with_the_judge)
    session_id = _new_session(api)

    result = _turn(api, session_id, message="Crea una lezione sulla cellula").json()

    assert result["critic"] is None
    judge = result["judge"]
    assert (judge["found"], judge["replaced"], judge["notes"]) == (2, 1, 1)
    assert judge["fixed"][0] == {
        "check": "fatti",
        "why": "data sbagliata",
        "before": "1792",
        "after": "1789",
        "kind": "sostituzione",
        "teacher_material": True,
    }
    assert judge["points"][0]["quote"] == "q"


@pytest.mark.parametrize(
    ("status", "plan", "code"),
    [("running", "# L", "already_running"), ("draft", "", "no_lesson_yet")],
)
def test_an_answer_during_a_turn_is_already_running_not_no_lesson_yet(status, plan, code):
    """Two turns at once: the second was refused as «no_lesson_yet» (replay, 2026-10-20)."""
    from types import SimpleNamespace

    from fastapi import HTTPException

    from aix.api.routes.sessions import _form
    from aix.api.schemas.sessions import TurnRequest

    lesson = SimpleNamespace(status=status, lesson_plan_md=plan)

    with pytest.raises(HTTPException) as refused:
        _form(TurnRequest(message="E per l'Attività 2?", mode="answer"), lesson)

    assert (refused.value.status_code, refused.value.detail["code"]) == (409, code)


# --- What a lesson keeps (2026-10-21): its cards and its resources, readable later


CARDS = {
    "planner": {"intent": "lesson_creation", "subject_concepts": ["cellula"]},
    "retriever": {"nodes_count": 12, "top_concepts": ["Mitocondrio"]},
    "judge": {
        "found": 1,
        "replaced": 1,
        "fixed": [{"check": "fatti", "before": "a", "after": "b"}],
    },
}
CURATED = {"videos": [{"title": "La cellula in 5 minuti", "url": "https://example.org/v"}]}
LIVE = {"wikipedia": [{"title": "Cellula", "url": "https://it.wikipedia.org/wiki/Cellula"}]}


def _fake_chain_that_keeps(meta):
    async def chain(lesson, session):
        from aix.webui.lessons.models import LessonMediaSnapshot

        turn = max(
            [m.turn_index for m in (await session.run_sync(lambda s: lesson.messages))] or [1]
        )
        lesson.lesson_plan_md = LESSON_MD
        lesson.status = "complete"
        session.add(
            LessonMessage(
                lesson_id=lesson.id,
                role="assistant",
                content_md=LESSON_MD,
                turn_index=turn,
                agent_kind="writer",
                meta_json=meta,
            )
        )
        session.add(
            LessonMediaSnapshot(
                lesson_id=lesson.id,
                turn_index=turn,
                owner_id=lesson.owner_id,
                curated_json=CURATED,
                live_json=LIVE,
                live_fetched=True,
            )
        )
        await session.commit()
        yield StreamEvent(kind="done", lesson_plan_md=LESSON_MD, meta={"duration_seconds": 1.0})

    return chain


def test_a_lesson_read_later_still_has_its_cards(api, monkeypatch):
    meta = {"approved": True, "scores": {"average_score": 4.0}, "cards": CARDS, "kg_digest": "…"}
    monkeypatch.setattr("aix.webui.agent.turns.run_agent_stream", _fake_chain_that_keeps(meta))
    session_id = _new_session(api)
    _turn(api, session_id, message="Crea una lezione sulla cellula")

    turns = api.get(f"/api/v1/sessions/{session_id}", headers=_as()).json()["turns"]

    question, lesson = turns
    assert question["planner"] is None
    assert lesson["planner"]["subject_concepts"] == ["cellula"]
    assert lesson["retriever"]["top_concepts"] == ["Mitocondrio"]
    assert lesson["judge"]["fixed"][0]["after"] == "b"
    assert lesson["critic"] is None
    # The cards are typed above; the graph digest is the light turns' material.
    assert lesson["meta"] == {"approved": True, "scores": {"average_score": 4.0}}


def test_a_lesson_saved_before_the_cards_still_has_its_judge(api, monkeypatch):
    meta = {
        "approved": True,
        "judge": {"found": 2, "blocking": 1, "fixed": [{"check": "fatti", "after": "b"}]},
        "review_notes": [{"check": "classe", "severity": "bloccante", "quote": "q", "why": "w"}],
    }
    monkeypatch.setattr("aix.webui.agent.turns.run_agent_stream", _fake_chain_that_keeps(meta))
    session_id = _new_session(api)
    _turn(api, session_id, message="Crea una lezione sulla cellula")

    lesson = api.get(f"/api/v1/sessions/{session_id}", headers=_as()).json()["turns"][1]

    assert lesson["planner"] is None
    judge = lesson["judge"]
    assert (judge["found"], judge["notes"], judge["notes_blocking"]) == (2, 1, 1)
    assert judge["points"][0]["quote"] == "q"


def test_the_media_has_the_curated_resources_and_an_earlier_lesson_s(api, monkeypatch):
    monkeypatch.setattr("aix.webui.agent.turns.run_agent_stream", _fake_chain_that_keeps({}))

    async def no_live_search(lesson):
        raise AssertionError("what was shown is replayed, not searched again")

    monkeypatch.setattr("aix.api.routes.sessions.live_media_for_lesson", no_live_search)
    session_id = _new_session(api)
    _turn(api, session_id, message="Crea una lezione sulla cellula")

    latest = api.get(f"/api/v1/sessions/{session_id}/media", headers=_as()).json()
    earlier = api.get(f"/api/v1/sessions/{session_id}/media?turn_index=1", headers=_as()).json()
    missing = api.get(f"/api/v1/sessions/{session_id}/media?turn_index=9", headers=_as()).json()

    assert latest["turn_index"] == 1
    assert latest["curated"]["videos"][0]["title"] == "La cellula in 5 minuti"
    assert latest["wikipedia"][0]["title"] == "Cellula"
    assert earlier == latest
    assert missing["curated"] == {} and missing["turn_index"] is None


def test_a_lesson_turn_keeps_the_cards_the_webui_showed():
    from aix.webui.agent.service import lesson_cards

    state = {
        "query_intent": "lesson_creation",
        "retrieved_nodes": [{"name": "Mitocondrio"}],
        "curated_media": {"videos": [{"title": "v"}]},
        "judge_summary": {"found": 1, "fixed": []},
        "review_notes": [],
    }

    with_judge = lesson_cards(state)
    with_critic = lesson_cards({**state, "judge_summary": None})

    assert set(with_judge) == {"planner", "retriever", "judge"}
    assert "media" not in with_judge["retriever"]  # kept in the media snapshot
    assert with_judge["retriever"]["top_concepts"] == ["Mitocondrio"]
    assert set(with_critic) == {"planner", "retriever", "critic"}
    json.dumps(with_judge)  # saved as JSON with the message
