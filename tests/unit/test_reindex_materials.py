"""Putting uploaded documents back through ingestion (``scripts/ops/reindex_materials.py``).

The private search reads only chunks of the current chunker version and
embedding model. Without a way to re-process what is already uploaded, any
change to either would make every document vanish from the search in silence,
and a parser improvement -- PowerPoint speaker notes, 2026-09-25 -- would never
reach the decks already there.

Its first dry run on the development database found a version marked ready with
no chunks at all: a document the chip called ready and the search could not
find. These tests pin the selection and the requeue, on the same SQLite setup
the retrieval tests use.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from aix.webui.auth.models import User
from aix.webui.db import Base
from aix.webui.materials.chunking import CHUNKER_VERSION
from aix.webui.materials.models import (
    AssetBlob,
    IngestionJob,
    Material,
    MaterialChunk,
    MaterialVersion,
)
from aix.webui.materials.parsing.parsers import PARSER_PROFILE
from scripts.ops.reindex_materials import MEDIA_TYPES, requeue, select_versions

pytestmark = pytest.mark.unit

ANNA = uuid.UUID("11111111-1111-1111-1111-111111111111")


@pytest.fixture(autouse=True)
def _current_model(monkeypatch):
    monkeypatch.setattr("aix.webui.materials.embedding.embedding_model", lambda: "model-now")
    monkeypatch.setenv("AIX_PRIVATE_EMBEDDING_DIMENSION", "3")


@pytest.fixture
async def session(tmp_path):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{(tmp_path / 'reindex.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        session.add(
            User(id=ANNA, email="anna@scuola.it", hashed_password="x", is_active=True,
                 is_superuser=False, is_verified=True)
        )
        await session.commit()
        yield session
    await engine.dispose()


async def _version(session, name="Dispensa.pdf", media_type="application/pdf", status="ready"):
    blob = AssetBlob(owner_id=ANNA, storage_provider="local", object_key=f"k-{uuid.uuid4()}",
                     sha256=uuid.uuid4().hex * 2, size_bytes=1, verified_media_type=media_type)
    material = Material(owner_id=ANNA, display_name=name)
    session.add_all([blob, material])
    await session.flush()
    version = MaterialVersion(material_id=material.id, version_number=1, blob_id=blob.id,
                              source_sha256=blob.sha256, media_type=media_type,
                              created_by=ANNA, status=status)
    session.add(version)
    await session.flush()
    return version, material


async def _chunk(session, version, *, chunker=CHUNKER_VERSION, model="model-now"):
    session.add(MaterialChunk(owner_id=ANNA, material_version_id=version.id, ordinal=0,
                              text="testo", text_hash="0" * 64, locator_json={"page": 1},
                              embedding=[1.0, 0.0, 0.0], embedding_model=model,
                              chunker_version=chunker))
    await session.flush()


async def _chosen(session, **kwargs):
    kwargs.setdefault("media_type", None)
    kwargs.setdefault("everything", False)
    return {name: reason for _version, name, reason in await select_versions(session, **kwargs)}


# ---------------------------------------------------------------------------
# Which versions
# ---------------------------------------------------------------------------


async def test_an_up_to_date_version_is_left_alone(session):
    version, _ = await _version(session)
    await _chunk(session, version)

    assert await _chosen(session) == {}


async def test_chunks_of_another_chunker_version_are_stale(session):
    version, _ = await _version(session, "Vecchia.pdf")
    await _chunk(session, version, chunker="chunk@1")

    assert "chunker" in (await _chosen(session))["Vecchia.pdf"]


async def test_chunks_of_another_embedding_model_are_stale(session):
    version, _ = await _version(session, "Altro-modello.pdf")
    await _chunk(session, version, model="model-before")

    assert "Altro-modello.pdf" in await _chosen(session)


async def test_with_the_map_on_an_indexed_version_without_a_map_is_chosen(session, monkeypatch):
    """Every document uploaded before the maps existed (2026-09-29)."""
    from aix.webui.materials.document_map import MAP_VERSION
    from aix.webui.materials.models import DocumentMap

    monkeypatch.setenv("AIX_PRIVATE_RAG_ENABLED", "true")
    unmapped, _ = await _version(session, "Libro.pdf")
    await _chunk(session, unmapped)
    mapped, _ = await _version(session, "Con-mappa.pdf")
    await _chunk(session, mapped)
    session.add(DocumentMap(owner_id=ANNA, material_version_id=mapped.id,
                            map_version=MAP_VERSION, model="m", map_json={}))
    await session.flush()

    assert await _chosen(session) == {"Libro.pdf": "indicizzato ma senza mappa del documento"}

    monkeypatch.setenv("AIX_DOCUMENT_MAP_ENABLED", "false")
    assert await _chosen(session) == {}


async def test_ready_with_no_chunks_is_the_case_the_first_run_found(session):
    await _version(session, "Pronto-vuoto.pdf", status="ready")

    assert (await _chosen(session))["Pronto-vuoto.pdf"] == "pronto ma senza passaggi"


async def test_deleted_material_is_never_brought_back(session):
    version, material = await _version(session, "Cancellato.pdf")
    await _chunk(session, version, chunker="chunk@1")
    material.deleted_at = datetime.now(timezone.utc)
    await session.flush()

    assert await _chosen(session, everything=True) == {}


async def test_a_media_type_selects_only_that_type(session):
    deck, _ = await _version(session, "Slide.pptx", media_type=MEDIA_TYPES["pptx"])
    pdf, _ = await _version(session, "Dispensa.pdf")
    await _chunk(session, deck)
    await _chunk(session, pdf)

    assert set(await _chosen(session, media_type=MEDIA_TYPES["pptx"])) == {"Slide.pptx"}


# ---------------------------------------------------------------------------
# The requeue
# ---------------------------------------------------------------------------


async def test_a_finished_job_goes_back_to_pending(session):
    version, _ = await _version(session)
    job = IngestionJob(owner_id=ANNA, material_version_id=version.id,
                       parser_profile_version=PARSER_PROFILE, status="failed",
                       current_step="failed", attempt_count=5,
                       next_attempt_at=datetime.now(timezone.utc) + timedelta(days=1),
                       safe_error_code="embedding_failed")
    session.add(job)
    await session.flush()

    assert await requeue(session, version) == "rimesso in coda"
    assert (job.status, job.attempt_count, job.safe_error_code) == ("pending", 0, None)
    assert job.next_attempt_at <= datetime.now(timezone.utc)


async def test_a_version_without_a_job_gets_one(session):
    version, _ = await _version(session)

    assert await requeue(session, version) == "creato"
    await session.flush()


async def test_a_job_being_worked_on_right_now_is_left_alone(session):
    version, _ = await _version(session)
    job = IngestionJob(owner_id=ANNA, material_version_id=version.id,
                       parser_profile_version=PARSER_PROFILE, status="parsing",
                       current_step="parsing", attempt_count=1,
                       next_attempt_at=datetime.now(timezone.utc),
                       lease_owner="worker-1",
                       lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5))
    session.add(job)
    await session.flush()

    assert "lasciato stare" in await requeue(session, version)
    assert job.status == "parsing"
