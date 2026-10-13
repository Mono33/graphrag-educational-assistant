"""The upload, its parsing, and its shadow half (Fase 1 · PR 4, Fase 2 · PR 5).

``POST /webui/lesson/{id}/upload`` parses the file with
``aix.webui.materials.parsing`` and — with ``AIX_MATERIALS_ENABLED`` on — also
puts the same bytes through the durable pipeline. The guide's acceptance
conditions each have a section below.

The one that governs the design:

> «failure del worker non rompe la chat»

A teacher attaching a file has no idea a second pipeline exists. If it is
broken, misconfigured, or its disk is full, the upload must still succeed and
return the same fragment. So the test for it does not mock a tidy failure — it
makes storage raise, and then checks that the teacher got their file anyway.
"""

from __future__ import annotations

import io
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from aix.webui.auth.dependencies import optional_current_user
from aix.webui.auth.models import User
from aix.webui.db import Base, get_async_session
from aix.webui.lessons.models import Lesson
from aix.webui.materials.chunking import CHUNKER_VERSION
from aix.webui.materials.handlers import PARSER_PROFILE, measure_extraction, status_for
from aix.webui.materials.models import (
    AssetBlob,
    DocumentIR,
    IngestionJob,
    Material,
    MaterialChunk,
    MaterialVersion,
)

pytestmark = pytest.mark.unit

ANNA = uuid.UUID("11111111-1111-1111-1111-111111111111")
LESSON = uuid.UUID("33333333-3333-3333-3333-333333333333")

SYLLABUS = b"Programma di classe: si affronta la memoria di lavoro."
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _blank_pdf() -> bytes:
    """A valid PDF with a page and no text — a stand-in for a scanned document."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _docx() -> bytes:
    import docx

    document = docx.Document()
    document.add_heading("Programma di scienze", level=1)
    document.add_paragraph("La fotosintesi clorofilliana in tre fasi.")
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------


def test_a_readable_document_is_parsed_and_not_empty():
    ir, quality = measure_extraction(SYLLABUS, "text/plain")

    assert ir is not None
    assert quality["parser"] == PARSER_PROFILE == "ir@1"
    assert quality["empty"] is False
    assert quality["chars"] == len(SYLLABUS.decode())
    assert status_for(quality) == "ready"


def test_a_scanned_pdf_counts_its_unread_pages():
    """Wherever OCR is unavailable or finds nothing, the page is declared unread."""
    ir, quality = measure_extraction(_blank_pdf(), "application/pdf")

    assert quality["empty"] is True
    assert quality["pages"] == 1
    assert quality["pages_with_text_layer"] == 0
    assert quality["unread_pages"] == 1
    assert status_for(quality) == "ready_with_warnings"


def test_an_unreadable_pdf_is_recorded_rather_than_raised():
    """The bytes will not become readable on the third attempt."""
    ir, quality = measure_extraction(b"%PDF-1.4 questo non e' un PDF", "application/pdf")

    assert ir is None
    assert quality["unreadable"] is True
    assert quality["empty"] is True
    assert status_for(quality) == "failed", "«ready» would invite a lesson written from nothing"


# ---------------------------------------------------------------------------
# The route, against the real application
# ---------------------------------------------------------------------------


@pytest.fixture
async def _app(tmp_path, monkeypatch):
    """The real app on a temp database, with a temp object store."""
    monkeypatch.setenv("WEBUI_CSRF_ENABLED", "false")
    monkeypatch.setenv("AIX_STORAGE_PROVIDER", "local")
    monkeypatch.setenv("AIX_STORAGE_LOCAL_ROOT", str(tmp_path / "objects"))

    # The legacy path writes through a module-level constant, not a setting, so
    # without this the suite drops its fixtures into the repository's real
    # ``data/webui/uploads`` — which it did, until someone went looking for
    # sample documents and found thirty-three copies of a test string.
    monkeypatch.setattr("aix.webui.lessons.uploads.UPLOAD_ROOT", tmp_path / "legacy-uploads")

    from aix.api.main import app

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{(tmp_path / 'shadow.db').as_posix()}",
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

    # ``shadow.py`` imports the maker from ``aix.webui.db`` inside its
    # functions, deliberately, so it never holds the request's session. That is
    # also the seam a test needs: point the module attribute at the temp
    # database and the shadow writes land there.
    monkeypatch.setattr("aix.webui.db.async_session_maker", maker)

    async def _session():
        async with maker() as session:
            yield session

    app.dependency_overrides[get_async_session] = _session
    app.dependency_overrides[optional_current_user] = lambda: User(
        id=ANNA,
        email="anna@scuola.it",
        hashed_password="x",
        is_active=True,
        is_superuser=False,
        is_verified=True,
    )
    try:
        yield app, maker
    finally:
        app.dependency_overrides.clear()
        await engine.dispose()


@pytest.fixture
def client(_app):
    app, _maker = _app
    return TestClient(app, follow_redirects=False)


@pytest.fixture
def maker(_app):
    _app_obj, session_maker = _app
    return session_maker


def _upload(client, name="programma.txt", content=SYLLABUS, mime="text/plain"):
    return client.post(f"/webui/lesson/{LESSON}/upload", files={"file": (name, content, mime)})


async def _rows(maker, model):
    async with maker() as session:
        return list((await session.execute(select(model))).scalars())


async def _manifest(maker):
    async with maker() as session:
        lesson = await session.get(Lesson, LESSON)
        return lesson.uploaded_files_json or []


# ---------------------------------------------------------------------------
# Flags off — the paperclip the project had before this work (§5.2)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _parser_on(monkeypatch):
    """This module tests the parser, so it is on here — never in production.

    The default is off (§5.2). The three tests below switch it back off to pin
    what a deployment that has not opted in actually gets.
    """
    monkeypatch.setenv("AIX_UPLOAD_PARSER_ENABLED", "true")


@pytest.fixture
def _parser_off(monkeypatch):
    monkeypatch.setenv("AIX_UPLOAD_PARSER_ENABLED", "false")


async def test_with_the_parser_off_a_pdf_is_read_the_way_it_always_was(client, maker, _parser_off):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    pdf.drawString(72, 780, "Programma di storia")
    pdf.save()

    assert _upload(client, "programma.pdf", buffer.getvalue(), "application/pdf").status_code == 200

    entry = (await _manifest(maker))[0]
    assert "Programma di storia" in entry["text_excerpt"]
    assert "parse" not in entry, "the legacy manifest entry keeps exactly its old keys"


# ---------------------------------------------------------------------------
# Parsing through the upload (PR 5), with the flag on
# ---------------------------------------------------------------------------


async def test_a_docx_upload_reaches_the_writer_as_text(client, maker, monkeypatch):
    monkeypatch.setenv("AIX_MATERIALS_ENABLED", "false")

    response = _upload(client, "programma.docx", _docx(), DOCX_MIME)

    assert response.status_code == 200
    entry = (await _manifest(maker))[0]
    assert entry["mime"] == DOCX_MIME
    assert "fotosintesi clorofilliana" in entry["text_excerpt"]
    assert entry["parse"]["parser"] == "ir@1"


async def test_a_disguised_executable_is_refused_with_a_readable_message(client, maker):
    response = _upload(client, "programma.pdf.exe", b"MZ\x90\x00", "application/pdf")

    assert response.status_code == 422
    assert "non consentito" in response.text
    assert await _manifest(maker) == []


async def test_a_corrupt_pdf_is_a_422_not_a_500(client, maker):
    """The legacy path answered this with a server error."""
    response = _upload(client, "rotto.pdf", b"%PDF-1.4\n non un pdf", "application/pdf")

    assert response.status_code == 422
    assert "danneggiato" in response.text
    assert await _manifest(maker) == []


async def test_a_pdf_renamed_docx_is_refused(client, maker):
    response = _upload(client, "programma.docx", _blank_pdf(), DOCX_MIME)

    assert response.status_code == 422
    assert await _manifest(maker) == []


async def test_a_scan_without_text_is_accepted_and_says_so_on_the_chip(client, maker):
    response = _upload(client, "scansione.pdf", _blank_pdf(), "application/pdf")

    assert response.status_code == 200
    assert "nessun testo leggibile" in response.text
    entry = (await _manifest(maker))[0]
    assert entry["parse"]["empty"] is True
    assert entry["parse"]["unread_pages"] == 1


def test_a_three_page_handout_reaches_the_writer_whole():
    """The first real test lost the last fifth of a 9985-character handout at 8000."""
    from aix.webui.lessons.uploads import truncate_excerpt

    handout = "Due mondi contrapposti. " * 434  # ~10k characters, like guerra-fredda-def.pdf
    handout += "Nel 1989 fu abbattuto il muro di Berlino."

    assert truncate_excerpt(handout).endswith("muro di Berlino.")


async def test_ocr_text_is_flagged_to_the_writer():
    from aix.webui.agent.service import _teacher_upload_context

    class _Lesson:
        uploaded_files_json = [
            {"filename": "scheda.png", "text_excerpt": "Testo", "parse": {"ocr_pages": 1}},
            {"filename": "programma.docx", "text_excerpt": "Altro", "parse": {"ocr_pages": 0}},
        ]

    context = _teacher_upload_context(_Lesson())

    assert "### scheda.png (testo letto con OCR" in context
    assert "### programma.docx\n" in context


# ---------------------------------------------------------------------------
# Flag off — no shadow records
# ---------------------------------------------------------------------------


async def test_with_the_flag_off_nothing_shadow_happens(client, maker, monkeypatch):
    monkeypatch.setenv("AIX_MATERIALS_ENABLED", "false")

    assert _upload(client).status_code == 200

    assert await _rows(maker, Material) == []
    assert await _rows(maker, IngestionJob) == []


# ---------------------------------------------------------------------------
# Flag on — the shadow records appear
# ---------------------------------------------------------------------------


@pytest.fixture
def _on(monkeypatch):
    monkeypatch.setenv("AIX_MATERIALS_ENABLED", "true")


async def test_an_upload_creates_the_shadow_records(client, maker, _on):
    assert _upload(client).status_code == 200

    materials = await _rows(maker, Material)
    versions = await _rows(maker, MaterialVersion)
    blobs = await _rows(maker, AssetBlob)
    jobs = await _rows(maker, IngestionJob)

    assert len(materials) == len(versions) == len(blobs) == len(jobs) == 1
    assert materials[0].owner_id == ANNA
    assert materials[0].display_name == "programma.txt"
    assert jobs[0].parser_profile_version == PARSER_PROFILE
    assert jobs[0].status == "pending"


async def test_the_manifest_records_where_the_shadow_copy_went(client, maker, _on):
    _upload(client)

    entry = (await _manifest(maker))[0]
    assert entry["processing_status"] == "processing"
    assert uuid.UUID(entry["material_id"])
    assert uuid.UUID(entry["material_version_id"])
    assert entry["text_excerpt"], "and the legacy fields are still there"


async def test_the_chat_fragment_is_unchanged(client, maker, _on):
    """Shadow mode that the teacher can see is not shadow mode."""
    response = _upload(client)

    assert response.status_code == 200
    assert "programma.txt" in response.text
    for leaked in ("material_id", "processing_status", PARSER_PROFILE):
        assert leaked not in response.text


# ---------------------------------------------------------------------------
# A broken pipeline does not break the chat
# ---------------------------------------------------------------------------


async def test_the_upload_succeeds_when_storage_is_broken(client, maker, _on, monkeypatch):
    from aix.webui.materials import shadow as shadow_module
    from aix.webui.materials import storage as storage_module

    def _explode(*_args, **_kwargs):
        raise storage_module.StorageError("il volume non è montato")

    # Patched where it is used, not where it is defined: ``shadow`` imported
    # the name at module load, so replacing it on ``storage`` would leave the
    # original bound.
    monkeypatch.setattr(shadow_module, "get_storage_backend", _explode)

    response = _upload(client)

    assert response.status_code == 200, "the teacher's file is what matters"
    assert "programma.txt" in response.text
    assert len(await _manifest(maker)) == 1, "and the legacy manifest is intact"
    assert await _rows(maker, Material) == []


async def test_the_upload_succeeds_when_the_queue_is_broken(client, maker, _on, monkeypatch):
    from aix.webui.materials import shadow as shadow_module

    class _BrokenDispatcher:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("la coda non risponde")

    monkeypatch.setattr(shadow_module, "JobDispatcher", _BrokenDispatcher)

    assert _upload(client).status_code == 200
    assert len(await _manifest(maker)) == 1


# ---------------------------------------------------------------------------
# Removing the attachment retires the material
# ---------------------------------------------------------------------------


async def test_deleting_the_attachment_retires_the_material(client, maker, _on):
    _upload(client)
    entry = (await _manifest(maker))[0]

    assert client.delete(f"/webui/lesson/{LESSON}/upload/{entry['id']}").status_code == 200

    async with maker() as session:
        material = await session.get(Material, uuid.UUID(entry["material_id"]))
        assert material.deleted_at is not None

    versions = await _rows(maker, MaterialVersion)
    assert len(versions) == 1, "the version survives so the lesson stays explainable"


async def test_deleting_still_works_when_the_material_is_already_gone(client, maker, _on):
    """Best effort both ways: a missing shadow row must not block the removal."""
    _upload(client)
    entry = (await _manifest(maker))[0]

    async with maker() as session:
        material = await session.get(Material, uuid.UUID(entry["material_id"]))
        await session.delete(material)
        await session.commit()

    assert client.delete(f"/webui/lesson/{LESSON}/upload/{entry['id']}").status_code == 200
    assert await _manifest(maker) == []


# ---------------------------------------------------------------------------
# The worker actually processing what the upload queued
# ---------------------------------------------------------------------------


async def _drain_one(maker, tmp_path):
    """Run the real worker once over whatever the upload left in the queue."""
    from aix.webui.materials.handlers import ExtractionHandler
    from aix.webui.materials.jobs import JobDispatcher
    from aix.webui.materials.storage import LocalDurableStorage
    from aix.webui.materials.worker import IngestionWorker

    dispatcher = JobDispatcher(maker)
    storage = LocalDurableStorage(tmp_path / "objects")
    worker = IngestionWorker(dispatcher, ExtractionHandler(dispatcher, storage))
    return await worker.run_once()


async def test_the_worker_parses_what_the_upload_stored(client, maker, tmp_path, _on):
    """The whole chain: upload, storage, queue, worker, IR and summary."""
    _upload(client)

    assert await _drain_one(maker, tmp_path) is True

    version = (await _rows(maker, MaterialVersion))[0]
    assert version.status == "ready"
    assert version.parse_quality["parser"] == PARSER_PROFILE
    assert version.parse_quality["chars"] == len(SYLLABUS.decode())

    irs = await _rows(maker, DocumentIR)
    assert len(irs) == 1
    assert irs[0].material_version_id == version.id
    assert irs[0].owner_id == ANNA
    assert irs[0].ir_json["blocks"][0]["text"] == SYLLABUS.decode()

    job = (await _rows(maker, IngestionJob))[0]
    assert job.status == "ready"
    assert job.finished_at is not None
    assert job.lease_owner is None, "the lease is released when the job finishes"


async def test_a_docx_upload_is_stored_as_structured_blocks(client, maker, tmp_path, _on):
    _upload(client, "programma.docx", _docx(), DOCX_MIME)
    await _drain_one(maker, tmp_path)

    blocks = (await _rows(maker, DocumentIR))[0].ir_json["blocks"]
    assert [b["kind"] for b in blocks] == ["heading", "paragraph"]
    assert blocks[0]["source_locator"]["format"] == "docx"


async def test_a_scanned_upload_finishes_with_a_warning(client, maker, tmp_path, _on):
    _upload(client, "scansione.pdf", _blank_pdf(), "application/pdf")

    await _drain_one(maker, tmp_path)

    version = (await _rows(maker, MaterialVersion))[0]
    assert version.status == "ready_with_warnings"
    assert version.parse_quality["empty"] is True
    assert version.parse_quality["unread_pages"] == 1


# ---------------------------------------------------------------------------
# The private index (Fase 2 · PR 6): only when asked, and never duplicated
# ---------------------------------------------------------------------------


@pytest.fixture
def _rag_on(monkeypatch):
    """The private RAG flag, with a fake embedder in place of the provider."""
    monkeypatch.setenv("AIX_PRIVATE_RAG_ENABLED", "true")

    async def _fake_embed(texts, **_kwargs):
        return [[0.5] * 1024 for _ in texts]

    monkeypatch.setattr("aix.webui.materials.embedding.embed_texts", _fake_embed)


async def test_with_the_flag_off_nothing_is_chunked_or_embedded(client, maker, tmp_path, _on):
    """No flag, no chunks — and no text leaves the machine."""
    _upload(client)
    await _drain_one(maker, tmp_path)

    assert await _rows(maker, MaterialChunk) == []


async def test_the_worker_indexes_the_material_for_retrieval(client, maker, tmp_path, _on, _rag_on):
    _upload(client, "programma.docx", _docx(), DOCX_MIME)

    await _drain_one(maker, tmp_path)

    chunks = await _rows(maker, MaterialChunk)
    assert chunks, "the material is retrievable"
    assert all(c.owner_id == ANNA for c in chunks)
    assert all(len(c.embedding) == 1024 for c in chunks)
    assert all(c.chunker_version == CHUNKER_VERSION for c in chunks)
    assert all(c.embedding_model for c in chunks), "which model produced it is on the row"
    assert "fotosintesi clorofilliana" in " ".join(c.text for c in chunks)
    assert all("block_ids" in c.locator_json for c in chunks), "a citation can be resolved"


async def test_re_running_the_job_replaces_the_chunks_rather_than_duplicating_them(
    client, maker, tmp_path, _on, _rag_on
):
    """A second parse of the same version must leave the index as if it ran once."""
    _upload(client, "programma.docx", _docx(), DOCX_MIME)
    await _drain_one(maker, tmp_path)
    first = [c.text_hash for c in await _rows(maker, MaterialChunk)]

    from aix.webui.materials.handlers import ExtractionHandler
    from aix.webui.materials.jobs import JobDispatcher
    from aix.webui.materials.storage import LocalDurableStorage

    job = (await _rows(maker, IngestionJob))[0]
    dispatcher = JobDispatcher(maker)
    handler = ExtractionHandler(dispatcher, LocalDurableStorage(tmp_path / "objects"))
    content, media_type = await handler._load(job)
    from aix.webui.materials.parsing import parse_document

    await handler._index_privately(job, parse_document(content, media_type))

    assert [c.text_hash for c in await _rows(maker, MaterialChunk)] == first


async def test_the_queue_is_empty_afterwards(client, maker, tmp_path, _on):
    _upload(client)
    await _drain_one(maker, tmp_path)

    assert await _drain_one(maker, tmp_path) is False


async def test_the_lesson_binds_the_material_when_the_private_rag_is_on(
    client, maker, _on, _rag_on
):
    """The binding is what retrieval is authorized against (§8.5).

    Through shadow mode it was deliberately not written: it would have claimed
    the lesson used a material while the Writer was still reading the manifest
    excerpt.
    """
    from aix.webui.materials.models import LessonMaterialBinding

    _upload(client, "programma.docx", _docx(), DOCX_MIME)

    bindings = await _rows(maker, LessonMaterialBinding)
    assert len(bindings) == 1
    assert bindings[0].lesson_id == LESSON
    assert bindings[0].owner_id == ANNA

    versions = await _rows(maker, MaterialVersion)
    assert bindings[0].selected_version_id == versions[0].id


async def test_an_embedding_failure_leaves_the_job_retryable_not_ready(
    client, maker, tmp_path, _on, monkeypatch
):
    """The first real upload's failure, pinned.

    The job used to be completed *before* indexing. When embedding failed the job
    was already "ready": no retry, zero chunks, and a status saying all was well.
    Now it is completed last, so the failure is recorded and retried — and once
    the provider answers, the material becomes searchable.
    """
    monkeypatch.setenv("AIX_PRIVATE_RAG_ENABLED", "true")

    async def _provider_down(texts, **_kwargs):
        raise RuntimeError("OpenRouter non risponde")

    monkeypatch.setattr("aix.webui.materials.embedding.embed_texts", _provider_down)
    _upload(client, "programma.docx", _docx(), DOCX_MIME)
    await _drain_one(maker, tmp_path)

    job = (await _rows(maker, IngestionJob))[0]
    assert job.status != "ready", "a job whose material is not searchable is not ready"
    assert job.safe_error_code, "and the failure is recorded"
    assert await _rows(maker, MaterialChunk) == []

    async def _provider_back(texts, **_kwargs):
        return [[0.5] * 1024 for _ in texts]

    monkeypatch.setattr("aix.webui.materials.embedding.embed_texts", _provider_back)
    async with maker() as session:
        from sqlalchemy import update

        await session.execute(update(IngestionJob).values(next_attempt_at=IngestionJob.created_at))
        await session.commit()
    await _drain_one(maker, tmp_path)

    assert (await _rows(maker, IngestionJob))[0].status == "ready"
    assert await _rows(maker, MaterialChunk), "the retry made it searchable"
