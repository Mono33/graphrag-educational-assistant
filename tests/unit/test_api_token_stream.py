"""The text of a turn, streamed with its phases (2026-10-16).

AIxLearning shows the answer while it is written; the session stream now carries
the Writer's and the fast answer's text in ``kind: text`` frames, with a draft
number that grows when the Critic asks for a revision, and without the internal
citation markers. See ``aix.api.token_stream``.
"""

from __future__ import annotations

import json

import pytest
from tests.unit.test_api_sessions import (  # noqa: F401 — the fixture
    LESSON_MD,
    _as,
    _new_session,
    _turn,
    api,
)
from tests.unit.test_light_turns import _Client

from aix.agent.graph import write_stream
from aix.api.token_stream import PreviewText
from aix.core.config import config as app_config
from aix.webui.agent.service import StreamEvent
from aix.webui.lessons.models import LessonMessage

pytestmark = pytest.mark.unit


def _frames(client, session_id, message):
    with client.stream(
        "POST",
        f"/api/v1/sessions/{session_id}/turns/stream",
        json={"message": message},
        headers=_as(),
    ) as response:
        return [
            json.loads(line[len("data: ") :])
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]


def _text(frames, draft):
    return "".join(f["text"] for f in frames if f["kind"] == "text" and f["draft"] == draft)


async def _chain_with_a_revision(lesson, session):
    """``run_agent_stream`` without a model: a first draft, a revision, the lesson."""
    bus = write_stream.register(str(lesson.id))
    try:
        yield StreamEvent(kind="planner", payload={"intent": "lesson_creation"})
        yield StreamEvent(kind="writer_pending", payload={})
        for piece in ("# Bozza", " uno [[mate", "rial:abc#p3]] fine"):
            bus.put_nowait(("content", piece))
        bus.put_nowait(("think", "ragiono…"))
        bus.put_nowait(None)
        yield StreamEvent(kind="critic", payload={"approved": False, "score": 2.5})
        for piece in ("# La ", "cellula"):
            bus.put_nowait(("content", piece))
        bus.put_nowait(None)
        yield StreamEvent(kind="critic", payload={"approved": True, "score": 4.2})
        lesson.lesson_plan_md = LESSON_MD
        lesson.status = "complete"
        session.add(
            LessonMessage(
                lesson_id=lesson.id,
                role="assistant",
                content_md=LESSON_MD,
                turn_index=1,
                agent_kind="writer",
                meta_json={"approved": True},
            )
        )
        await session.commit()
        yield StreamEvent(kind="done", lesson_plan_md=LESSON_MD, meta={"duration_seconds": 1.0})
    finally:
        write_stream.deregister(str(lesson.id))


def test_a_lesson_s_text_arrives_draft_by_draft_before_the_result(api, monkeypatch):  # noqa: F811
    monkeypatch.setattr("aix.webui.agent.turns.run_agent_stream", _chain_with_a_revision)
    session_id = _new_session(api)

    frames = _frames(api, session_id, "Crea una lezione sulla cellula")
    kinds = [f["kind"] for f in frames]

    assert _text(frames, 1) == "# Bozza uno  fine", "no marker, no reasoning"
    assert _text(frames, 2) == "# La cellula"
    assert [f["draft"] for f in frames if f["kind"] == "draft_end"] == [1, 2]
    # The draft comes before the critic that judged it, the result last of all.
    assert kinds.index("draft_end") < kinds.index("critic")
    assert kinds[-1] == "result"
    assert frames[-1]["result"]["markdown"].startswith("# La cellula")


def test_a_fast_answer_streams_its_text_too(api, monkeypatch):  # noqa: F811
    session_id = _new_session(api)
    _turn(api, session_id, message="Crea una lezione sulla cellula")
    monkeypatch.setattr(
        app_config.openai, "get_async_client", lambda: _Client(["Perché ", "la cellula vive."])
    )

    frames = _frames(api, session_id, "sì")

    assert _text(frames, 1) == "Perché la cellula vive."
    assert frames[-1]["result"]["kind"] == "answer"
    assert frames[-1]["result"]["markdown"] == "Perché la cellula vive."


def test_the_turn_without_streaming_is_unchanged(api, monkeypatch):  # noqa: F811
    monkeypatch.setattr("aix.webui.agent.turns.run_agent_stream", _chain_with_a_revision)
    session_id = _new_session(api)

    result = _turn(api, session_id, message="Crea una lezione sulla cellula").json()

    assert result["kind"] == "lesson" and result["markdown"].startswith("# La cellula")


@pytest.mark.parametrize(
    ("deltas", "shown"),
    [
        (["Testo [[material:", "x#p1]] dopo"], "Testo  dopo"),
        (["Testo [", "[material:x]] dopo"], "Testo  dopo"),
        (["Un [link", "](https://e.it) qui"], "Un [link](https://e.it) qui"),
        (["Fine [[material:tronc"], "Fine "),
    ],
)
def test_the_preview_never_shows_a_citation_marker(deltas, shown):
    preview = PreviewText()

    text = "".join(preview.feed(d) for d in deltas) + preview.flush()

    assert text == shown
