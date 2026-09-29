"""The private material path, end to end, with only the provider faked.

Why this file exists
--------------------
On 2026-09-16 the private RAG ran for the first time against real documents and
did nothing at all. Three defects, none of which 700+ unit tests could see:

* ``embedding._client()`` imported ``aix.config.settings``, a module that does
  not exist — every unit test passed a fake client, so that line had never run;
* the ingestion job was completed *before* indexing, so the failure could not be
  recorded: both jobs sat ``ready`` with zero chunks and were never retried;
* the lesson fell back to pasting the whole document, which is the behaviour the
  feature exists to replace, and only a log line said so.

The common cause is a test habit, not a coding mistake: each unit test replaced
exactly the piece that was broken. So this test fakes **one** thing — the
provider accessor ``aix.core.config.config.openai.get_async_client`` — and runs
everything else for real: the route, the intake, the parser, the chunker, the
ingestion worker, the repository, the retriever, and the Writer's prompt
assembly. Patching the accessor and not our own modules is deliberate: it leaves
``embedding._client()`` to import what it really imports, which is what would
have caught the first defect.

No network: the fake embedder is a deterministic bag-of-words hash, so "la
fotosintesi" really is closer to the photosynthesis paragraph than to the Cold
War one, and the ranking assertions below mean something.
"""

from __future__ import annotations

import hashlib
import io
import math
import re
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from aix.webui.auth.dependencies import optional_current_user
from aix.webui.auth.models import User
from aix.webui.db import Base, get_async_session
from aix.webui.lessons.models import Lesson
from aix.webui.materials.models import LessonMaterialBinding, MaterialChunk, MaterialVersion

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

ANNA = uuid.UUID("11111111-1111-1111-1111-111111111111")
LESSON = uuid.UUID("55555555-5555-5555-5555-555555555555")

DIMENSION = 1024

#: Each blank line starts a page. Every page carries real prose on purpose: a
#: page holding only a heading falls under the parser's 25-character threshold,
#: counts as having no text layer, and the chip would warn about OCR — true
#: behaviour, but not what these tests are about.
STORIA = """Le origini della guerra fredda. Dalla Seconda guerra mondiale
emersero due sole grandi potenze, USA e URSS, subito avversarie.

A Yalta l'Europa fu divisa in due sfere d'influenza, separate dalla linea che
Churchill chiamò cortina di ferro.

Il Piano Marshall. Fra il 1947 e il 1952 il governo americano finanziò la
ricostruzione dell'Europa occidentale con denaro, macchinari e merci.
"""

SCIENZE = """La fotosintesi clorofilliana

La fotosintesi clorofilliana trasforma anidride carbonica e acqua in glucosio
grazie alla luce solare captata dalla clorofilla nei cloroplasti.
"""


# ---------------------------------------------------------------------------
# The one thing that is faked: the provider
# ---------------------------------------------------------------------------


def _bag_of_words_vector(text: str) -> list[float]:
    """A deterministic embedding with real lexical behaviour.

    Each word lands in a bucket by hash and the vector is normalised, so cosine
    similarity behaves the way an embedding should for these tests: documents
    sharing vocabulary with the query score higher. No network, same numbers on
    every machine.
    """
    vector = [0.0] * DIMENSION
    for word in re.findall(r"\w+", text.lower()):
        bucket = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16) % DIMENSION
        vector[bucket] += 1.0

    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else vector


class _FakeEmbeddings:
    def __init__(self) -> None:
        self.calls = 0
        self.texts: list[str] = []

    async def create(self, *, model, input):  # noqa: A002 — the provider's name
        self.calls += 1
        self.texts.extend(input)
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=i, embedding=_bag_of_words_vector(text))
                for i, text in enumerate(input)
            ]
        )


@pytest.fixture
def provider(monkeypatch):
    """Replace only ``config.openai.get_async_client``.

    Not ``aix.webui.materials.embedding._client``, and not ``embed_texts``:
    those are the seams the unit tests already replace, and the defect of
    2026-09-16 lived between them.
    """
    from aix.core.config import config

    embeddings = _FakeEmbeddings()
    monkeypatch.setattr(
        config.openai,
        "get_async_client",
        lambda: SimpleNamespace(embeddings=embeddings),
        raising=True,
    )
    return embeddings


@pytest.fixture
def flags(monkeypatch, tmp_path):
    monkeypatch.setenv("WEBUI_CSRF_ENABLED", "false")
    monkeypatch.setenv("AIX_UPLOAD_PARSER_ENABLED", "true")
    monkeypatch.setenv("AIX_MATERIALS_ENABLED", "true")
    monkeypatch.setenv("AIX_PRIVATE_RAG_ENABLED", "true")
    monkeypatch.setenv("AIX_STORAGE_PROVIDER", "local")
    monkeypatch.setenv("AIX_STORAGE_LOCAL_ROOT", str(tmp_path / "objects"))
    monkeypatch.setattr("aix.webui.lessons.uploads.UPLOAD_ROOT", tmp_path / "uploads")


@pytest.fixture
async def app(tmp_path, monkeypatch, flags):
    from aix.api.main import app as fastapi_app

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{(tmp_path / 'e2e.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    maker = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with maker() as session:
        session.add(
            User(
                id=ANNA,
                email="anna@scuola.it",
                hashed_password="x",
                is_active=True,
                is_superuser=False,
                is_verified=True,
            )
        )
        session.add(Lesson(id=LESSON, owner_id=ANNA, domain="neuro", status="draft"))
        await session.commit()

    monkeypatch.setattr("aix.webui.db.async_session_maker", maker)

    async def _session():
        async with maker() as session:
            yield session

    fastapi_app.dependency_overrides[get_async_session] = _session
    fastapi_app.dependency_overrides[optional_current_user] = lambda: User(
        id=ANNA,
        email="anna@scuola.it",
        hashed_password="x",
        is_active=True,
        is_superuser=False,
        is_verified=True,
    )
    try:
        yield fastapi_app, maker
    finally:
        fastapi_app.dependency_overrides.clear()
        await engine.dispose()


@pytest.fixture
def client(app):
    fastapi_app, _maker = app
    return TestClient(fastapi_app, follow_redirects=False)


@pytest.fixture
def maker(app):
    _fastapi_app, session_maker = app
    return session_maker


# ---------------------------------------------------------------------------
# Helpers that drive the real pipeline
# ---------------------------------------------------------------------------


def _pdf(text: str) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    for page in text.split("\n\n"):
        y = 780
        for line in page.splitlines():
            pdf.drawString(72, y, line)
            y -= 18
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def _upload(client, name: str, content: bytes):
    return client.post(
        f"/webui/lesson/{LESSON}/upload",
        files={"file": (name, content, "application/pdf")},
    )


async def _drain(maker, tmp_path):
    """Run the real worker until the queue is empty, exactly as the app does."""
    from aix.webui.materials.handlers import ExtractionHandler
    from aix.webui.materials.jobs import JobDispatcher
    from aix.webui.materials.storage import LocalDurableStorage
    from aix.webui.materials.worker import IngestionWorker

    dispatcher = JobDispatcher(maker)
    worker = IngestionWorker(
        dispatcher, ExtractionHandler(dispatcher, LocalDurableStorage(tmp_path / "objects"))
    )
    while await worker.run_once():
        pass


async def _rows(maker, model):
    async with maker() as session:
        return list((await session.execute(select(model))).scalars())


# ---------------------------------------------------------------------------
# The path
# ---------------------------------------------------------------------------


async def test_an_upload_becomes_searchable_material(client, maker, tmp_path, provider):
    """Route → intake → parser → chunker → embedder → material_chunk.

    The assertion that matters is the last one: the provider was really called
    through ``embedding._client()``. With the import broken, this test fails
    here instead of a teacher discovering it.
    """
    assert _upload(client, "storia.pdf", _pdf(STORIA)).status_code == 200

    await _drain(maker, tmp_path)

    chunks = await _rows(maker, MaterialChunk)
    assert chunks, "the document is searchable"
    assert all(len(c.embedding) == DIMENSION for c in chunks)
    assert all(c.owner_id == ANNA for c in chunks)
    assert all(c.locator_json.get("page") for c in chunks), "every chunk can be cited"
    assert provider.calls >= 1, "the real client path was exercised, not a stub of ours"

    versions = await _rows(maker, MaterialVersion)
    assert versions[0].status in ("ready", "ready_with_warnings")

    bindings = await _rows(maker, LessonMaterialBinding)
    assert [b.lesson_id for b in bindings] == [LESSON], "the lesson may search it"


async def test_the_lesson_is_written_from_the_passages_not_the_pasted_document(
    client, maker, tmp_path, provider
):
    """Retrieval → Writer prompt: the seam the whole feature exists for."""
    from aix.agent.agents.writer_agent import teacher_material_appendix
    from aix.agent.graph import material_source
    from aix.agent.graph.nodes import retrieve_private_node
    from aix.webui.materials.retrieval import search_for_lesson

    _upload(client, "storia.pdf", _pdf(STORIA))
    _upload(client, "scienze.pdf", _pdf(SCIENZE))
    await _drain(maker, tmp_path)

    # Registered the way ``run_agent_stream`` registers it for a real run.
    session_id = str(LESSON)
    material_source.register(session_id, search_for_lesson(LESSON, ANNA))
    try:
        updates = await retrieve_private_node(
            {
                "session_id": session_id,
                    "material_scope_id": session_id,
                "teacher_query": "crea una lezione basandoti sui documenti allegati",
                "educational_profile": {"specific_topic": "guerra fredda"},
                "subject_concepts": ["Conferenza di Yalta", "cortina di ferro"],
            }
        )
    finally:
        material_source.deregister(session_id)

    items = updates["material_context_items"]
    assert items, "the retriever found the teacher's own material"

    top = items[0]["text"].lower()
    assert "yalta" in top or "cortina di ferro" in top, "the subject decides the ranking"
    assert "fotosintesi" not in top

    appendix = teacher_material_appendix(
        items,
        "### storia.pdf\n" + STORIA * 40,  # the old behaviour: the whole document
        phase="write",
    )
    assert "[[material:" in appendix, "the model is given markers to copy back"
    assert "contenuto, non" in appendix, "and told the material is data, not instructions"
    assert STORIA * 3 not in appendix, "the whole document is not pasted alongside"


async def test_a_citation_reaches_the_teacher_readable(client, maker, tmp_path, provider):
    """§8.6: what the model writes is an address; what the teacher reads is a source."""
    from aix.agent.graph import material_source
    from aix.agent.graph.nodes import retrieve_private_node
    from aix.webui.lessons.routes import _markdown_to_html
    from aix.webui.materials.citations import resolve_for_lesson
    from aix.webui.materials.retrieval import search_for_lesson

    _upload(client, "storia.pdf", _pdf(STORIA))
    await _drain(maker, tmp_path)

    session_id = str(LESSON)
    material_source.register(session_id, search_for_lesson(LESSON, ANNA))
    try:
        items = (
            await retrieve_private_node(
                {
                    "session_id": session_id,
                    "material_scope_id": session_id,
                    "teacher_query": "la guerra fredda",
                    "subject_concepts": ["Yalta", "cortina di ferro"],
                }
            )
        )["material_context_items"]
    finally:
        material_source.deregister(session_id)

    # The lesson as the Writer would return it, citing what it was given.
    lesson_md = f"A Yalta l'Europa fu divisa {items[0]['citation']}."

    async with maker() as session:
        lesson = await session.get(Lesson, LESSON)
        resolved = await resolve_for_lesson(lesson_md, session, lesson)

    assert "storia.pdf" in resolved, "the document is named"
    assert re.search(r"p\.\s*\d+", resolved), "and the page with it"
    assert "[[material:" not in resolved
    # U3 — inline reference *and* the list of sources under the lesson.
    assert "Fonti dai tuoi materiali" in resolved
    assert "- storia.pdf — p." in resolved

    html = _markdown_to_html(lesson_md)
    assert "[[material:" not in html, "no internal address survives to the page"
    assert "Fonti dai tuoi materiali" in html


async def test_removing_an_attachment_removes_it_from_retrieval(client, maker, tmp_path, provider):
    """§12.3, and the reason it matters.

    Until 2026-09-18 «togli questo documento» soft-deleted one row and left the
    chunks active: the document went on answering questions and being quoted
    after the teacher had removed it.
    """
    from aix.agent.graph import material_source
    from aix.agent.graph.nodes import retrieve_private_node
    from aix.webui.materials.citations import TOMBSTONE_LABEL, resolve_for_lesson
    from aix.webui.materials.retrieval import search_for_lesson

    _upload(client, "storia.pdf", _pdf(STORIA))
    await _drain(maker, tmp_path)

    search = search_for_lesson(LESSON, ANNA)
    before = await search("guerra fredda Yalta")
    assert before, "the document answers while it is attached"
    citation = before[0]["citation"]

    async with maker() as session:
        lesson = await session.get(Lesson, LESSON)
        entry_id = lesson.uploaded_files_json[0]["id"]

    assert client.delete(f"/webui/lesson/{LESSON}/upload/{entry_id}").status_code == 200

    assert await _rows(maker, MaterialChunk) == [], "the searchable copy is gone"
    assert await search("guerra fredda Yalta") == [], "and nothing is retrievable"

    session_id = str(LESSON)
    material_source.register(session_id, search)
    try:
        updates = await retrieve_private_node(
            {"session_id": session_id,
                    "material_scope_id": session_id, "teacher_query": "guerra fredda"}
        )
    finally:
        material_source.deregister(session_id)
    assert not updates.get("material_context_items")

    async with maker() as session:
        lesson = await session.get(Lesson, LESSON)
        resolved = await resolve_for_lesson(f"A Yalta {citation}.", session, lesson)

    assert TOMBSTONE_LABEL in resolved, "the lesson says the source was removed"
    assert "storia.pdf" not in resolved


# ---------------------------------------------------------------------------
# What the teacher sees and can do while a document is being prepared (U1)
# ---------------------------------------------------------------------------


async def test_the_run_waits_for_a_document_that_is_still_being_prepared(
    client, maker, tmp_path, provider
):
    """The 2026-09-16 failure mode, as a test.

    The request was sent seconds after the upload and the lesson was written
    from the pasted text while the teacher believed the document had been read.
    """
    _upload(client, "storia.pdf", _pdf(STORIA))  # queued, deliberately not drained

    refused = client.post(
        f"/webui/lesson/{LESSON}/run", data={"query": "crea una lezione sulla guerra fredda"}
    )

    assert refused.status_code == 409
    assert "storia.pdf" in refused.text
    assert "pronto" in refused.text, "and it says the wait ends by itself"

    async with maker() as session:
        lesson = await session.get(Lesson, LESSON)
        assert lesson.status == "draft", "nothing was started"


async def test_once_ready_the_chip_says_so_and_the_send_button_returns(
    client, maker, tmp_path, provider
):
    _upload(client, "storia.pdf", _pdf(STORIA))

    preparing = client.get(f"/webui/lesson/{LESSON}/attachments-fragment")
    assert "in preparazione" in preparing.text
    assert "disabled" in preparing.text, "the send button waits with it"
    assert "attachments-fragment" in preparing.text, "and the tray asks again"
    # The button alone travels out of band, never the textarea: swapping the
    # whole input every two seconds wiped what the teacher was typing
    # (2026-09-29).
    assert 'id="chat-input-send"' in preparing.text
    assert "<textarea" not in preparing.text

    await _drain(maker, tmp_path)

    ready = client.get(f"/webui/lesson/{LESSON}/attachments-fragment")
    assert "pronto" in ready.text, "the upload is confirmed in as many words"
    assert "passaggi" in ready.text, "the teacher is told what is searchable"
    assert "in preparazione" not in ready.text
    assert "attachments-fragment" not in ready.text, "and the tray stops asking"


async def test_a_document_is_never_ready_before_its_passages_exist(
    client, maker, tmp_path, provider, monkeypatch
):
    """«ready» is a promise: the lesson may use it, the chip says so, Invia unlocks.

    The handler used to write that status before embedding, leaving a window of
    seconds where the version was ready with zero chunks. On 2026-09-21 a second
    attachment landed in it: the chip went blank and the tray stopped polling,
    and a run started in that window would have searched an empty document.

    The embedder here pauses long enough to inspect the database mid-flight.
    """
    from aix.webui.materials.models import MaterialVersion as MV

    seen: dict = {}

    async def _look_around(*args, **kwargs):
        async with maker() as session:
            version = (await session.execute(select(MV))).scalars().first()
            chunks = (await session.execute(select(MaterialChunk))).scalars().all()
            seen["status_while_embedding"] = version.status
            seen["chunks_while_embedding"] = len(chunks)
        return await _FakeEmbeddings.create(provider, *args, **kwargs)

    _upload(client, "storia.pdf", _pdf(STORIA))
    monkeypatch.setattr(provider, "create", _look_around)
    await _drain(maker, tmp_path)

    assert seen["chunks_while_embedding"] == 0, "the test looked at the right moment"
    assert seen["status_while_embedding"] != "ready", (
        "a document with no passages must not call itself ready"
    )

    async with maker() as session:
        version = (await session.execute(select(MV))).scalars().first()
        chunks = (await session.execute(select(MaterialChunk))).scalars().all()
    assert version.status == "ready" and chunks, "and afterwards it is ready, with them"


async def test_a_second_upload_does_not_freeze_the_first_one_s_tray(
    client, maker, tmp_path, provider
):
    """Two files, seconds apart — the case a teacher actually performs.

    The tray used to decide whether to keep polling from a snapshot of the
    *states*: with the first document ready and the second not yet registered,
    it saw nothing pending, stopped, and never asked again. The second chip sat
    with no badge at all while its document had finished indexing three seconds
    later (2026-09-21, observed in the browser).
    """
    _upload(client, "storia.pdf", _pdf(STORIA))
    await _drain(maker, tmp_path)

    first_done = client.get(f"/webui/lesson/{LESSON}/attachments-fragment")
    assert "attachments-fragment" not in first_done.text, "one file, all settled: no timer"

    # The second file arrives while the tray believes everything is finished.
    _upload(client, "scienze.pdf", _pdf(SCIENZE))

    waiting = client.get(f"/webui/lesson/{LESSON}/attachments-fragment")
    assert "attachments-fragment" in waiting.text, "the tray must start asking again"
    assert waiting.text.count("in preparazione") >= 1, "and the new chip says so"

    await _drain(maker, tmp_path)

    both = client.get(f"/webui/lesson/{LESSON}/attachments-fragment")
    assert both.text.count("aix-attach-chip--ready") == 2, "both documents confirmed"
    assert "in preparazione" not in both.text
    assert "attachments-fragment" not in both.text, "and only then does it stop"


async def test_the_chat_input_accepts_a_dropped_file(client, maker, tmp_path, provider):
    """Dropping a file posts through the same input the paperclip fills.

    The handlers must be attributes, not a <script>: htmx does not execute
    scripts it swaps in, and this partial is re-rendered out of band on every
    terminal SSE event. A script would work until the first generation and then
    quietly stop.
    """
    page = client.get(f"/webui/lesson/{LESSON}/chat-input-fragment").text

    assert "ondrop=" in page and "aixDrop" in page, "the drop target is an attribute"
    assert "<script" not in page, "a script here would not survive an out-of-band swap"
    assert "aixUploadNext" in page, "and several files at once go up one after another"
