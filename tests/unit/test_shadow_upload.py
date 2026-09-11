"""The upload's shadow half (Fase 1 · PR 4).

``POST /webui/lesson/{id}/upload`` keeps doing what it has always done, and —
with ``AIX_MATERIALS_ENABLED`` on — also puts the same bytes through the new
pipeline. The guide states four acceptance conditions, and each has a section
below.

The third of them is the one that governs the design:

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
from aix.webui.materials.handlers import PARSER_PROFILE, measure_extraction
from aix.webui.materials.models import AssetBlob, IngestionJob, Material, MaterialVersion

pytestmark = pytest.mark.unit

ANNA = uuid.UUID("11111111-1111-1111-1111-111111111111")
LESSON = uuid.UUID("33333333-3333-3333-3333-333333333333")

SYLLABUS = b"Programma di classe: si affronta la memoria di lavoro."


def _blank_pdf() -> bytes:
    """A valid PDF with a page and no text — a stand-in for a scanned document."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# The measurement (T2)
# ---------------------------------------------------------------------------


def test_a_readable_document_is_not_empty():
    quality = measure_extraction(SYLLABUS, "programma.txt")

    assert quality["extractor"] == PARSER_PROFILE
    assert quality["empty"] is False
    assert quality["chars"] == len(SYLLABUS.decode())


@pytest.mark.parametrize("content", [b"", b"   \n\t  "], ids=["nothing", "whitespace"])
def test_a_document_with_no_text_is_empty(content):
    assert measure_extraction(content, "vuoto.txt")["empty"] is True


def test_a_scanned_pdf_reads_as_empty_with_its_pages_counted():
    """The measurement this handler exists for — and the Docling decision."""
    quality = measure_extraction(_blank_pdf(), "scansione.pdf")

    assert quality["empty"] is True
    assert quality["pages"] == 1
    assert quality["pages_with_text"] == 0, "a page that exists and yields nothing"


def test_page_counts_are_only_for_pdfs():
    assert "pages" not in measure_extraction(SYLLABUS, "programma.txt")


def test_an_unreadable_pdf_is_recorded_rather_than_raised():
    """The bytes will not become readable on the third attempt.

    Retrying five times and then failing would spend a worker slot on a
    conclusion that is already known.
    """
    quality = measure_extraction(b"%PDF-1.4 questo non e' un PDF", "rotto.pdf")

    assert quality["unreadable"] is True
    assert quality["empty"] is True
    assert "pages" not in quality


# ---------------------------------------------------------------------------
# The route, against the real application
# ---------------------------------------------------------------------------


@pytest.fixture
async def _app(tmp_path, monkeypatch):
    """The real app on a temp database, with a temp object store."""
    monkeypatch.setenv("WEBUI_CSRF_ENABLED", "false")
    monkeypatch.setenv("AIX_STORAGE_PROVIDER", "local")
    monkeypatch.setenv("AIX_STORAGE_LOCAL_ROOT", str(tmp_path / "objects"))

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


def _upload(client, name="programma.txt", content=SYLLABUS):
    return client.post(
        f"/webui/lesson/{LESSON}/upload", files={"file": (name, content, "text/plain")}
    )


async def _rows(maker, model):
    async with maker() as session:
        return list((await session.execute(select(model))).scalars())


async def _manifest(maker):
    async with maker() as session:
        lesson = await session.get(Lesson, LESSON)
        return lesson.uploaded_files_json or []


# ---------------------------------------------------------------------------
# Flag off — the behaviour is the legacy behaviour
# ---------------------------------------------------------------------------


async def test_with_the_flag_off_nothing_shadow_happens(client, maker, monkeypatch):
    monkeypatch.delenv("AIX_MATERIALS_ENABLED", raising=False)

    assert _upload(client).status_code == 200

    assert await _rows(maker, Material) == []
    assert await _rows(maker, IngestionJob) == []


async def test_with_the_flag_off_the_manifest_is_the_legacy_one(client, maker, monkeypatch):
    monkeypatch.delenv("AIX_MATERIALS_ENABLED", raising=False)
    _upload(client)

    entry = (await _manifest(maker))[0]
    assert set(entry) == {"id", "filename", "mime", "size", "text_excerpt", "stored_name"}


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


async def test_the_worker_measures_what_the_upload_stored(client, maker, tmp_path, _on):
    """The whole chain: upload, storage, queue, worker, measurement."""
    _upload(client)

    assert await _drain_one(maker, tmp_path) is True

    version = (await _rows(maker, MaterialVersion))[0]
    assert version.status == "ready"
    assert version.parse_quality["extractor"] == PARSER_PROFILE
    assert version.parse_quality["empty"] is False
    assert version.parse_quality["chars"] == len(SYLLABUS.decode())

    job = (await _rows(maker, IngestionJob))[0]
    assert job.status == "ready"
    assert job.finished_at is not None
    assert job.lease_owner is None, "the lease is released when the job finishes"


async def test_a_scanned_upload_finishes_with_a_warning(client, maker, tmp_path, _on):
    """The number that decides whether Docling is worth its weight."""
    client.post(
        f"/webui/lesson/{LESSON}/upload",
        files={"file": ("scansione.pdf", _blank_pdf(), "application/pdf")},
    )

    await _drain_one(maker, tmp_path)

    version = (await _rows(maker, MaterialVersion))[0]
    assert version.status == "ready_with_warnings"
    assert version.parse_quality["empty"] is True
    assert version.parse_quality["pages"] == 1
    assert version.parse_quality["pages_with_text"] == 0


async def test_the_queue_is_empty_afterwards(client, maker, tmp_path, _on):
    _upload(client)
    await _drain_one(maker, tmp_path)

    assert await _drain_one(maker, tmp_path) is False
