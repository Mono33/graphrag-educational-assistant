"""One id per request, and no teacher's words in the staging logs (2026-10-23)."""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from aix.core import request_id
from aix.core.redact import private
from aix.webui.lessons.live_media import _live_media_concepts
from aix.webui.lessons.models import Lesson

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def client():
    from aix.api.main import app

    return TestClient(app)


def test_a_request_without_an_id_gets_one(client):
    response = client.get("/api/v1/catalog")

    assert len(response.headers[request_id.HEADER]) == 32


def test_the_platform_s_id_comes_back(client):
    response = client.get("/api/v1/catalog", headers={"X-Request-ID": "dramatiq-7f3a:turn-2"})

    assert response.headers[request_id.HEADER] == "dramatiq-7f3a:turn-2"


def test_an_unusable_id_is_replaced_not_trusted(client):
    response = client.get("/api/v1/catalog", headers={"X-Request-ID": "x" * 65})

    assert response.headers[request_id.HEADER] != "x" * 65


async def test_every_log_line_of_a_request_carries_its_id():
    seen: list[str] = []

    async def app(scope, receive, send):
        record = logging.LogRecord("t", logging.INFO, __file__, 1, "inside", None, None)
        request_id.RequestIdFilter().filter(record)
        seen.append(record.request_id)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "headers": [(b"x-request-id", b"abc-123")]}
    await request_id.RequestIdMiddleware(app)(scope, None, send)

    outside = logging.LogRecord("t", logging.INFO, __file__, 1, "outside", None, None)
    request_id.RequestIdFilter().filter(outside)
    assert seen == ["abc-123"] and outside.request_id == "-"
    assert (b"x-request-id", b"abc-123") in sent[0]["headers"]


@pytest.mark.parametrize(
    ("environment", "flag", "shows_text"),
    [
        ("staging", "", False),
        ("production", "", False),
        ("development", "", True),
        ("staging", "true", True),
        ("development", "false", False),
    ],
)
def test_the_teacher_s_words_stay_out_of_staging_logs(monkeypatch, environment, flag, shows_text):
    monkeypatch.setenv("ENVIRONMENT", environment)
    monkeypatch.setenv("AIX_LOG_TEACHER_TEXT", flag)
    message = "una classe con due bambini con ADHD e Marco che non parla italiano"

    logged = private(message)

    assert ("Marco" in logged) is shows_text
    if not shows_text:
        assert logged == private(message) and logged.startswith(f"<{len(message)} car. #")


def _lesson(query: str, **profile) -> Lesson:
    return Lesson(teacher_query=query, educational_profile_json=profile, status="complete")


def test_live_media_searches_the_planner_s_subjects_not_the_request():
    request = (
        "ho una classe della primaria e vorrei una lezione sul ciclo dell'acqua; "
        "ci sono due bambini con ADHD e due di madrelingua spagnola"
    )

    with_subjects = _live_media_concepts(_lesson(request), ["ciclo dell'acqua", "evaporazione"])
    without = _live_media_concepts(_lesson(request, subject_area="Scienze"))

    assert with_subjects == ["ciclo dell'acqua", "evaporazione"]
    assert without == ["Scienze"]  # a whole message is never sent to a search engine


def test_a_short_request_is_still_the_topic_searched():
    """The teacher's Italian words find the Italian Wikipedia; the planner's English ones do not."""
    lesson = _lesson("crea una lezione sui vulcani")

    assert _live_media_concepts(lesson) == ["vulcani"]
    assert _live_media_concepts(lesson, ["volcanoes", "eruptions"]) == ["vulcani"]
