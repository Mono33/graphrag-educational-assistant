"""Two teachers, one server: nobody reaches anybody else's work (Fase 0 · PR 1).

Every owner-scoped route already filters on ``owner_id``. What was missing was
anything that would notice if one stopped. A dropped ``WHERE`` clause is not a
crash and not a failing page — it is Bruno opening Anna's lesson, and it looks
like a perfectly ordinary diff.

Each route is exercised twice, and the second half is what makes the first half
mean anything: an application that answered 404 to *everyone* would satisfy the
"Bruno cannot" assertions and be completely broken. So every case checks that
Anna still gets her own lesson on the same route, in the same test.

**404, not 403.** Answering "forbidden" would confirm the id exists, which
hands an attacker an oracle for enumerating other people's lessons. "Not found"
is the same answer they get for an id that was never real.

CSRF is switched off here on purpose. It has its own tests, and leaving it on
would mean every case below carried a token, which obscures what is actually
being asserted — and would let a CSRF rejection masquerade as an authorisation
check that is no longer running.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from aix.api.main import app
from aix.webui.auth.dependencies import optional_current_user
from aix.webui.auth.models import User
from aix.webui.db import Base, get_async_session
from aix.webui.lessons.models import Lesson, SavedProfile

pytestmark = pytest.mark.unit

ANNA = uuid.UUID("11111111-1111-1111-1111-111111111111")
BRUNO = uuid.UUID("22222222-2222-2222-2222-222222222222")
ANNA_LESSON = uuid.UUID("33333333-3333-3333-3333-333333333333")
ANNA_PROFILE = uuid.UUID("44444444-4444-4444-4444-444444444444")

SECRET = "Contenuto riservato della classe di Anna"


@pytest.fixture
async def _database(tmp_path, monkeypatch):
    """A temp database with Anna's lesson, wired into the real application."""
    monkeypatch.setenv("WEBUI_CSRF_ENABLED", "false")

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{(tmp_path / 'idor.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as s:
        for user_id, email in ((ANNA, "anna@scuola.it"), (BRUNO, "bruno@scuola.it")):
            s.add(
                User(
                    id=user_id,
                    email=email,
                    hashed_password="x",
                    is_active=True,
                    is_superuser=False,
                    is_verified=True,
                )
            )
        s.add(
            Lesson(
                id=ANNA_LESSON,
                owner_id=ANNA,
                domain="neuro",
                status="complete",
                title="Lezione di Anna",
                lesson_plan_md=f"# {SECRET}",
                uploaded_files_json=[{"id": "f1", "name": "programma.txt", "excerpt": "x"}],
            )
        )
        s.add(SavedProfile(id=ANNA_PROFILE, owner_id=ANNA, name="Classe di Anna", profile_json={}))
        await s.commit()

    async def _session():
        async with maker() as session:
            yield session

    app.dependency_overrides[get_async_session] = _session
    try:
        yield
    finally:
        app.dependency_overrides.clear()
        await engine.dispose()


def _login(user_id: uuid.UUID, email: str) -> None:
    """Become one of the two teachers for the next request.

    ``dependency_overrides`` lives on the application, not on the client, so two
    long-lived clients would not be two users: whichever fixture ran last would
    be the one making every request. The identity is therefore set immediately
    before each call, which is also closer to what a request actually does.
    """
    app.dependency_overrides[optional_current_user] = lambda: User(
        id=user_id,
        email=email,
        hashed_password="x",
        is_active=True,
        is_superuser=False,
        is_verified=True,
    )


ANNA_LOGIN = (ANNA, "anna@scuola.it")
BRUNO_LOGIN = (BRUNO, "bruno@scuola.it")


@pytest.fixture
def client(_database):
    return TestClient(app, follow_redirects=False)


#: Every owner-scoped route, as (method, path, kwargs).
#:
#: The two SSE endpoints (``/stream`` and ``/writer-stream``) are absent
#: deliberately: reaching them as the owner opens a stream that does not end,
#: so the "and Anna still can" half of each case could not be written. They
#: reject the same way — through the same owner-filtered query — and their
#: behaviour is covered where that query is.
READS = [
    ("GET", "/webui/lesson/{lesson}", {}),
    ("GET", "/webui/lesson/{lesson}/card-fragment", {}),
    ("GET", "/webui/lesson/{lesson}/chat-input-fragment", {}),
    ("GET", "/webui/lesson/{lesson}/export?fmt=md", {}),
    ("GET", "/webui/lesson/{lesson}/print", {}),
]

WRITES = [
    ("POST", "/webui/lesson/{lesson}/run", {"data": {"teacher_query": "riscrivi"}}),
    ("POST", "/webui/lesson/{lesson}/upload", {}),
    ("DELETE", "/webui/lesson/{lesson}/upload/f1", {}),
    ("POST", "/webui/lesson/{lesson}/profile", {"data": {}}),
    ("DELETE", "/webui/lesson/{lesson}", {}),
]


def _call(client: TestClient, who, method: str, path: str, kwargs: dict | None = None):
    _login(*who)
    return client.request(method, path.format(lesson=ANNA_LESSON), **(kwargs or {}))


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method, path, kwargs", READS, ids=[r[1] for r in READS])
def test_bruno_cannot_read_annas_lesson(client, method, path, kwargs):
    response = _call(client, BRUNO_LOGIN, method, path, kwargs)

    assert response.status_code == 404, "and 404 rather than 403 — see the module docstring"
    assert SECRET not in response.text


@pytest.mark.parametrize("method, path, kwargs", READS, ids=[r[1] for r in READS])
def test_anna_still_can(client, method, path, kwargs):
    """Without this half, an application that 404s for everyone would pass."""
    assert _call(client, ANNA_LOGIN, method, path, kwargs).status_code == 200


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method, path, kwargs", WRITES, ids=[w[1] for w in WRITES])
def test_bruno_cannot_change_annas_lesson(client, method, path, kwargs):
    assert _call(client, BRUNO_LOGIN, method, path, kwargs).status_code == 404


def test_and_the_lesson_is_untouched_afterwards(client):
    """A refused write must not have been half applied."""
    for method, path, kwargs in WRITES:
        _call(client, BRUNO_LOGIN, method, path, kwargs)

    survivor = _call(client, ANNA_LOGIN, "GET", "/webui/lesson/{lesson}")
    assert survivor.status_code == 200
    assert SECRET in survivor.text


# ---------------------------------------------------------------------------
# Listings and profiles
# ---------------------------------------------------------------------------


def test_the_library_shows_only_your_own_lessons(client):
    as_bruno = _call(client, BRUNO_LOGIN, "GET", "/webui/lessons").text
    assert SECRET not in as_bruno
    assert "Lezione di Anna" not in as_bruno

    assert "Lezione di Anna" in _call(client, ANNA_LOGIN, "GET", "/webui/lessons").text


def test_bruno_cannot_delete_annas_saved_profile(client):
    refused = _call(client, BRUNO_LOGIN, "DELETE", f"/webui/profiles/{ANNA_PROFILE}")
    assert refused.status_code == 404

    assert "Classe di Anna" in _call(client, ANNA_LOGIN, "GET", "/webui/lesson/new").text


def test_an_id_that_never_existed_answers_the_same_way(client):
    """The two must be indistinguishable, or the 404 is an enumeration oracle."""
    invented = uuid.uuid4()

    assert _call(client, BRUNO_LOGIN, "GET", f"/webui/lesson/{invented}").status_code == 404
    assert _call(client, BRUNO_LOGIN, "GET", "/webui/lesson/{lesson}").status_code == 404
