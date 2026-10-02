"""Light follow-up turns: one streamed answer instead of the whole chain.

A follow-up cost 50–70 s because every request ran planner, Knowledge Graph,
Writer and Critic again, and the Writer rewrote the whole lesson to answer
«perché hai scelto questa attività?». A light turn reads the lesson, the
conversation and the documents and answers in one call; the lesson does not
change. A request for a new lesson is handed back to the teacher.
"""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from aix.agent.graph import write_stream
from aix.api.main import app
from aix.core.config import config as app_config
from aix.webui.agent import light
from aix.webui.auth.dependencies import optional_current_user
from aix.webui.auth.models import User
from aix.webui.db import Base, get_async_session
from aix.webui.lessons.models import Lesson, LessonMessage

pytestmark = pytest.mark.unit

ANNA = uuid.UUID("00000000-0000-0000-0000-00000000a222")
LESSON = uuid.UUID("00000000-0000-0000-0000-00000000c222")
#: Longer than the 2 000 characters the full chain's history used to keep.
LESSON_MD = "# La fotosintesi\n\n## Attività 2\n" + "Gli studenti osservano una foglia. " * 120


def _anna() -> User:
    return User(
        id=ANNA,
        email="anna@scuola.it",
        hashed_password="x",
        is_active=True,
        is_superuser=False,
        is_verified=True,
    )


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("AIX_LIGHT_TURNS_ENABLED", "true")
    monkeypatch.setenv("AIX_TRACING_ENABLED", "false")
    monkeypatch.setenv("WEBUI_CSRF_ENABLED", "false")


@pytest.fixture
def maker(tmp_path):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{(tmp_path / 'light.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    maker = async_sessionmaker(engine, expire_on_commit=False)

    async def _seed():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with maker() as s:
            s.add(_anna())
            s.add(
                Lesson(
                    id=LESSON,
                    owner_id=ANNA,
                    domain="neuro",
                    status="complete",
                    teacher_query="Crea una lezione sulla fotosintesi",
                    lesson_plan_md=LESSON_MD,
                    educational_profile_json={"pedagogical_intent": "x"},
                )
            )
            s.add(LessonMessage(lesson_id=LESSON, role="user", content_md="Crea", turn_index=1))
            s.add(
                LessonMessage(
                    lesson_id=LESSON,
                    role="assistant",
                    content_md=LESSON_MD,
                    turn_index=1,
                    agent_kind="writer",
                    meta_json={"approved": True, "revision_count": 0},
                )
            )
            await s.commit()

    asyncio.run(_seed())
    yield maker
    asyncio.run(engine.dispose())


# ---------------------------------------------------------------------------
# Which turns are light
# ---------------------------------------------------------------------------


def _lesson(status="complete", plan=LESSON_MD):
    return SimpleNamespace(status=status, lesson_plan_md=plan)


def test_a_follow_up_with_a_message_is_light():
    assert light.wants_light_turn(_lesson(), {"query": "perché?"}, has_query=True)


@pytest.mark.parametrize(
    "lesson, form, has_query",
    [
        (_lesson(status="draft", plan=None), {}, True),  # the first lesson
        (_lesson(status="error"), {}, True),  # a failed turn is retried in full
        (_lesson(), {}, False),  # a button with no text
        (_lesson(), {"full_run": "1"}, True),  # «Lezione completa»
        (_lesson(), {"refinement_code": "simplify"}, True),  # chips rewrite the lesson
        (_lesson(), {"intent_confirmed": "1"}, True),
    ],
)
def test_the_full_chain_keeps_what_needs_it(lesson, form, has_query):
    assert not light.wants_light_turn(lesson, form, has_query=has_query)


def test_switched_off_every_turn_runs_the_full_chain(monkeypatch):
    monkeypatch.setenv("AIX_LIGHT_TURNS_ENABLED", "false")
    assert not light.wants_light_turn(_lesson(), {}, has_query=True)


def test_the_flag_is_for_one_turn():
    lesson = SimpleNamespace(educational_profile_json={"x": 1, "__light__": {"answer_here": True}})

    assert light.take_light_turn(lesson) == {"answer_here": True}
    assert lesson.educational_profile_json == {"x": 1}
    assert light.take_light_turn(lesson) is None


# ---------------------------------------------------------------------------
# The hand-off marker never reaches the chat
# ---------------------------------------------------------------------------


def _through(pieces):
    gate = light.EscalationGate()
    shown = "".join(gate.feed(p) for p in pieces) + gate.flush()
    return shown, gate.escalated


def test_an_answer_streams_as_it_comes():
    assert _through(["La ", "foglia ", "è verde."]) == ("La foglia è verde.", False)


def test_the_marker_in_pieces_is_caught_and_never_shown():
    assert _through(["  <<<NUO", "VA_LEZ", "IONE>>>", "testo"]) == ("", True)


def test_an_answer_that_starts_like_the_marker_is_not_lost():
    assert _through(["<<", "x"]) == ("<<x", False)
    assert _through(["<<"]) == ("<<", False)


# ---------------------------------------------------------------------------
# The turn
# ---------------------------------------------------------------------------


class _Client:
    """A model that streams ``pieces`` and records what it was asked."""

    def __init__(self, pieces, *, fail=False):
        self.pieces, self.fail, self.calls = pieces, fail, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("provider down")

        async def stream():
            for piece in self.pieces:
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))]
                )

        return stream()


async def _turn(maker, monkeypatch, client, text="Perché hai scelto l'attività 2?", **flags):
    monkeypatch.setattr(app_config.openai, "get_async_client", lambda: client)
    buses = []
    register = write_stream.register
    monkeypatch.setattr(
        write_stream, "register", lambda sid: buses.append(register(sid)) or buses[-1]
    )
    async with maker() as session:
        session.add(LessonMessage(lesson_id=LESSON, role="user", content_md=text, turn_index=2))
        lesson = await session.get(Lesson, LESSON)
        lesson.status = "running"
        await session.commit()
        events = [e async for e in light.run_light_turn(lesson, session, **flags)]
    async with maker() as session:
        lesson = await session.get(Lesson, LESSON)
        rows = (
            (await session.execute(select(LessonMessage).where(LessonMessage.turn_index == 2)))
            .scalars()
            .all()
        )
    tokens = []
    while buses and not buses[0].empty():
        tokens.append(buses[0].get_nowait())
    return events, lesson, rows, tokens


async def test_an_answer_is_streamed_and_saved_and_the_lesson_is_untouched(maker, monkeypatch):
    client = _Client(["L'attività 2 ", "fa osservare."])

    events, lesson, rows, tokens = await _turn(maker, monkeypatch, client)

    assert [e.kind for e in events] == ["light_pending", "light_done"]
    assert tokens == [("content", "L'attività 2 "), ("content", "fa osservare."), None]
    answer = next(r for r in rows if r.role == "assistant")
    assert (answer.agent_kind, answer.content_md) == ("tutor", "L'attività 2 fa osservare.")
    assert answer.meta_json["kind"] == "answer"
    assert lesson.status == "complete"
    assert lesson.lesson_plan_md == LESSON_MD
    assert lesson.teacher_query == "Crea una lezione sulla fotosintesi"


async def test_the_model_reads_the_whole_lesson_and_the_request(maker, monkeypatch):
    client = _Client(["Ok."])

    await _turn(maker, monkeypatch, client)

    prompt = client.calls[0]["messages"][1]["content"]
    assert LESSON_MD.strip() in prompt, "the whole lesson, not its first 2 000 characters"
    assert "Perché hai scelto l'attività 2?" in prompt
    assert client.calls[0]["stream"] is True
    # The lesson is the working document; the history does not repeat it.
    assert prompt.count("Gli studenti osservano una foglia.") == 120


async def test_a_new_lesson_is_handed_back_to_the_teacher(maker, monkeypatch):
    client = _Client(["<<<NUOVA_", "LEZIONE>>>"])

    events, lesson, rows, tokens = await _turn(
        maker, monkeypatch, client, text="Fammi una lezione sulla rivoluzione francese"
    )

    assert [e.kind for e in events] == ["light_pending", "light_escalate"]
    assert events[-1].payload["query"] == "Fammi una lezione sulla rivoluzione francese"
    assert rows == [], "the request goes back out with the card, not into the history"
    assert tokens == [None]
    assert lesson.status == "complete"


async def test_answer_here_never_hands_off(maker, monkeypatch):
    client = _Client(["Ecco una proposta."])

    await _turn(maker, monkeypatch, client, answer_here=True)

    system = client.calls[0]["messages"][0]["content"]
    assert "NUOVA_LEZIONE" not in system
    assert "la lezione salvata non cambia" in system


async def test_a_failed_answer_leaves_the_lesson_as_it_was(maker, monkeypatch):
    events, lesson, rows, _ = await _turn(maker, monkeypatch, _Client([], fail=True))

    assert [e.kind for e in events] == ["light_pending", "light_error"]
    assert lesson.status == "complete"
    assert lesson.lesson_plan_md == LESSON_MD
    assert [r.role for r in rows] == ["user"]


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------


@pytest.fixture
def client(maker):
    async def _session():
        async with maker() as session:
            yield session

    app.dependency_overrides[get_async_session] = _session
    app.dependency_overrides[optional_current_user] = _anna
    try:
        yield TestClient(app, follow_redirects=False)
    finally:
        app.dependency_overrides.clear()


async def _rows(maker):
    async with maker() as session:
        lesson = await session.get(Lesson, LESSON)
        rows = (
            (await session.execute(select(LessonMessage).order_by(LessonMessage.turn_index)))
            .scalars()
            .all()
        )
    return lesson, rows


def test_a_light_request_keeps_the_lesson_s_topic(client, maker):
    response = client.post(f"/webui/lesson/{LESSON}/run", data={"query": "Perché l'attività 2?"})

    assert response.status_code == 200
    lesson, rows = asyncio.run(_rows(maker))
    assert lesson.educational_profile_json["__light__"] == {"answer_here": False}
    assert lesson.teacher_query == "Crea una lezione sulla fotosintesi"
    assert rows[-1].content_md == "Perché l'attività 2?"


def test_lezione_completa_runs_the_full_chain_with_the_graph(client, maker):
    client.post(
        f"/webui/lesson/{LESSON}/run", data={"query": "Rifalla per la terza", "full_run": "1"}
    )

    lesson, _ = asyncio.run(_rows(maker))
    assert "__light__" not in lesson.educational_profile_json
    assert lesson.educational_profile_json["__graph_search__"] is True
    assert lesson.teacher_query == "Rifalla per la terza"


def test_the_final_card_is_the_answer_and_the_lesson_card_ignores_it(client, maker):
    async def _answer():
        async with maker() as s:
            s.add(LessonMessage(lesson_id=LESSON, role="user", content_md="Perché?", turn_index=2))
            s.add(
                LessonMessage(
                    lesson_id=LESSON,
                    role="assistant",
                    content_md="Perché **osservare** aiuta.",
                    turn_index=2,
                    agent_kind="tutor",
                    meta_json={"kind": "answer", "seconds": 4.2},
                )
            )
            await s.commit()

    asyncio.run(_answer())
    html = client.get(f"/webui/lesson/{LESSON}/card-fragment").text

    assert "<strong>osservare</strong>" in html
    assert "la lezione non è cambiata" in html
    assert "Piano della lezione" not in html

    page = client.get(f"/webui/lesson/{LESSON}").text
    assert "Perché <strong>osservare</strong> aiuta." in page
    assert "Lezione completa (con Knowledge Graph)" in page


def test_the_hand_off_card_replaces_the_pending_answer_and_resends_the_request():
    from aix.webui.agent.service import StreamEvent
    from aix.webui.lessons.routes import _stream_event_to_sse

    lesson = SimpleNamespace(
        id=LESSON, status="complete", educational_profile_json={}, uploaded_files_json=[]
    )
    query = "Una lezione sull'«Illuminismo» <b>nuova</b>"
    sse = _stream_event_to_sse(
        SimpleNamespace(), lesson, StreamEvent(kind="light_escalate", payload={"query": query})
    )

    html = sse["data"]
    assert 'hx-swap-oob="outerHTML:#lesson-card-loading"' in html
    assert '"full_run": "1"' in html and '"light_answer": "1"' in html
    assert "<b>nuova</b>" not in html, "the request is text, not markup"
    # An apostrophe in the request must not close the single-quoted hx-vals.
    assert "sull\\u0027" in html
