"""Owner-scoped access to material (Fase 0 · PR 2).

Two things are pinned here, and the first is the reason the module exists.

**No method reads a row without being told whose it is.** The failure this
prevents is not a crash — it is one teacher reading another's uploaded
material, which looks like nothing at all in a diff and produces no error in a
log. So the tests below ask for every row as the wrong owner and require the
answer to be the same as for a row that does not exist.

**A version is appended, never overwritten.** A lesson generated in March has
to keep resolving to the file it was actually written from, even after the
teacher uploads a corrected one in May. Otherwise an export cites a source that
no longer says what the lesson claims.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from aix.webui.auth.models import User
from aix.webui.db import Base
from aix.webui.lessons.models import Lesson
from aix.webui.materials.models import AssetBlob, LessonMaterialBinding, MaterialVersion
from aix.webui.materials.repository import (
    AssetBlobRepository,
    BindingRepository,
    MaterialRepository,
    MaterialService,
    MaterialVersionRepository,
)
from aix.webui.materials.storage import LocalDurableStorage, sha256_of

pytestmark = pytest.mark.unit

ANNA = uuid.UUID("11111111-1111-1111-1111-111111111111")
BRUNO = uuid.UUID("22222222-2222-2222-2222-222222222222")
LESSON = uuid.UUID("33333333-3333-3333-3333-333333333333")

SYLLABUS = b"Programma: si affronta solo la memoria di lavoro."
CORRECTED = b"Programma rivisto: memoria di lavoro e attenzione selettiva."


@pytest.fixture
async def session(tmp_path):
    """A real file-backed SQLite database with the whole webui schema.

    The full metadata rather than only the four new tables: the material
    tables carry foreign keys onto ``user`` and ``lesson``, and a test that
    skipped those would not be exercising the constraints that matter.
    """
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{(tmp_path / 'materials.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.exec_driver_sql("PRAGMA foreign_keys=ON")
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
        s.add(Lesson(id=LESSON, owner_id=ANNA, domain="neuro", status="draft"))
        await s.commit()

    async with maker() as s:
        try:
            yield s
        finally:
            await engine.dispose()


@pytest.fixture
def service(session, tmp_path):
    return MaterialService(session, LocalDurableStorage(tmp_path / "objects"))


# ---------------------------------------------------------------------------
# Storing a material
# ---------------------------------------------------------------------------


async def test_storing_an_upload_creates_the_three_rows(service, session):
    material, version = await service.store_new_material(
        SYLLABUS, owner_id=ANNA, display_name="Programma di classe", media_type="text/plain"
    )

    assert material.owner_id == ANNA
    assert version.version_number == 1
    assert version.material_id == material.id
    assert version.source_sha256 == sha256_of(SYLLABUS)

    blob = await AssetBlobRepository(session).get(version.blob_id, owner_id=ANNA)
    assert blob is not None
    assert blob.size_bytes == len(SYLLABUS)
    assert blob.object_key.startswith(f"{ANNA}/")


async def test_a_new_version_starts_quarantined(service):
    """Nothing reaches the agent before the pipeline says it is safe to."""
    _material, version = await service.store_new_material(
        SYLLABUS, owner_id=ANNA, display_name="Programma", media_type="text/plain"
    )

    assert version.status == "quarantined"


async def test_the_malware_status_records_the_decision_that_was_taken(service, session):
    """``not_scanned`` is a recorded choice, not a scan that never arrives."""
    _material, version = await service.store_new_material(
        SYLLABUS, owner_id=ANNA, display_name="Programma", media_type="text/plain"
    )
    blob = await AssetBlobRepository(session).get(version.blob_id, owner_id=ANNA)

    assert blob.malware_status == "not_scanned"


async def test_the_bytes_come_back(service):
    _material, version = await service.store_new_material(
        SYLLABUS, owner_id=ANNA, display_name="Programma", media_type="text/plain"
    )

    assert await service.read(version, owner_id=ANNA) == SYLLABUS


async def test_the_same_file_uploaded_twice_reuses_one_blob(service, session):
    """Keys are content-addressed, so the row must be reused as well as the object."""
    first, _ = await service.store_new_material(
        SYLLABUS, owner_id=ANNA, display_name="Programma", media_type="text/plain"
    )
    second, _ = await service.store_new_material(
        SYLLABUS, owner_id=ANNA, display_name="Programma (copia)", media_type="text/plain"
    )
    await session.commit()

    blobs = (await session.execute(AssetBlob.__table__.select())).all()
    assert len(blobs) == 1, "one object, one row"
    assert first.id != second.id, "but two distinct materials"


# ---------------------------------------------------------------------------
# Versions are appended, never overwritten
# ---------------------------------------------------------------------------


async def test_a_corrected_file_adds_a_version(service):
    material, first = await service.store_new_material(
        SYLLABUS, owner_id=ANNA, display_name="Programma", media_type="text/plain"
    )
    second = await service.add_version(
        CORRECTED, material_id=material.id, owner_id=ANNA, media_type="text/plain"
    )

    assert second.version_number == 2
    assert second.supersedes_version_id == first.id


async def test_the_earlier_version_still_resolves_to_what_it_was(service):
    """A lesson written from version 1 must keep reading version 1."""
    material, first = await service.store_new_material(
        SYLLABUS, owner_id=ANNA, display_name="Programma", media_type="text/plain"
    )
    await service.add_version(
        CORRECTED, material_id=material.id, owner_id=ANNA, media_type="text/plain"
    )

    assert await service.read(first, owner_id=ANNA) == SYLLABUS


async def test_versions_are_listed_newest_first(service):
    material, _ = await service.store_new_material(
        SYLLABUS, owner_id=ANNA, display_name="Programma", media_type="text/plain"
    )
    await service.add_version(
        CORRECTED, material_id=material.id, owner_id=ANNA, media_type="text/plain"
    )

    numbers = [
        v.version_number
        for v in await service.versions.list_for_material(material.id, owner_id=ANNA)
    ]
    assert numbers == [2, 1]


# ---------------------------------------------------------------------------
# Nobody reads anybody else's material
# ---------------------------------------------------------------------------


async def test_bruno_cannot_read_annas_material(service):
    material, _ = await service.store_new_material(
        SYLLABUS, owner_id=ANNA, display_name="Programma", media_type="text/plain"
    )

    assert await service.materials.get(material.id, owner_id=BRUNO) is None


async def test_bruno_cannot_read_annas_version(service):
    _material, version = await service.store_new_material(
        SYLLABUS, owner_id=ANNA, display_name="Programma", media_type="text/plain"
    )

    assert await service.versions.get(version.id, owner_id=BRUNO) is None


async def test_bruno_cannot_read_annas_blob(service):
    _material, version = await service.store_new_material(
        SYLLABUS, owner_id=ANNA, display_name="Programma", media_type="text/plain"
    )

    assert await service.blobs.get(version.blob_id, owner_id=BRUNO) is None


async def test_bruno_cannot_list_annas_materials(service):
    await service.store_new_material(
        SYLLABUS, owner_id=ANNA, display_name="Programma", media_type="text/plain"
    )

    assert await service.materials.list_for_owner(BRUNO) == []


async def test_bruno_cannot_append_to_annas_material(service):
    material, _ = await service.store_new_material(
        SYLLABUS, owner_id=ANNA, display_name="Programma", media_type="text/plain"
    )

    assert (
        await service.add_version(
            CORRECTED, material_id=material.id, owner_id=BRUNO, media_type="text/plain"
        )
        is None
    )


async def test_bruno_cannot_delete_annas_material(service):
    material, _ = await service.store_new_material(
        SYLLABUS, owner_id=ANNA, display_name="Programma", media_type="text/plain"
    )

    assert await service.materials.soft_delete(material.id, owner_id=BRUNO) is False
    assert await service.materials.get(material.id, owner_id=ANNA) is not None


async def test_every_read_takes_the_owner_as_a_keyword():
    """Not a style point: ``get(a, b)`` invites a transposition, ``owner_id=`` does not."""
    import inspect

    for repo, methods in (
        (AssetBlobRepository, ["get", "find_by_digest"]),
        (MaterialRepository, ["get", "soft_delete"]),
        (MaterialVersionRepository, ["get", "list_for_material", "latest_for_material"]),
        (BindingRepository, ["list_for_lesson"]),
    ):
        for name in methods:
            parameter = inspect.signature(getattr(repo, name)).parameters["owner_id"]
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, f"{repo.__name__}.{name}"
            assert parameter.default is inspect.Parameter.empty, f"{repo.__name__}.{name}"


# ---------------------------------------------------------------------------
# Soft delete
# ---------------------------------------------------------------------------


async def test_a_deleted_material_disappears_from_the_library(service):
    material, _ = await service.store_new_material(
        SYLLABUS, owner_id=ANNA, display_name="Programma", media_type="text/plain"
    )

    assert await service.materials.soft_delete(material.id, owner_id=ANNA) is True
    assert await service.materials.get(material.id, owner_id=ANNA) is None
    assert await service.materials.list_for_owner(ANNA) == []


async def test_the_row_survives_so_a_lesson_can_still_be_explained(service, session):
    material, _ = await service.store_new_material(
        SYLLABUS, owner_id=ANNA, display_name="Programma", media_type="text/plain"
    )
    await service.materials.soft_delete(material.id, owner_id=ANNA)
    await session.commit()

    rows = (await session.execute(MaterialVersion.__table__.select())).all()
    assert len(rows) == 1, "the version row is not removed by a soft delete"


# ---------------------------------------------------------------------------
# Binding a material to a lesson
# ---------------------------------------------------------------------------


async def test_a_lesson_binds_to_one_specific_version(service):
    material, first = await service.store_new_material(
        SYLLABUS, owner_id=ANNA, display_name="Programma", media_type="text/plain"
    )
    await service.add_version(
        CORRECTED, material_id=material.id, owner_id=ANNA, media_type="text/plain"
    )

    binding = await service.bindings.bind(
        lesson_id=LESSON,
        material_id=material.id,
        owner_id=ANNA,
        selected_version_id=first.id,
        created_by=ANNA,
        mode="must_follow",
    )

    assert binding.selected_version_id == first.id, "not silently upgraded to the newest"
    assert binding.mode == "must_follow"


async def test_a_binding_cannot_point_at_another_materials_version(service):
    """The composite key exists so this cannot happen; the repository says so too."""
    first_material, _ = await service.store_new_material(
        SYLLABUS, owner_id=ANNA, display_name="Programma", media_type="text/plain"
    )
    _other, other_version = await service.store_new_material(
        CORRECTED, owner_id=ANNA, display_name="Altro documento", media_type="text/plain"
    )

    assert (
        await service.bindings.bind(
            lesson_id=LESSON,
            material_id=first_material.id,
            owner_id=ANNA,
            selected_version_id=other_version.id,
            created_by=ANNA,
        )
        is None
    )


async def test_bruno_cannot_bind_annas_material_to_a_lesson(service):
    material, version = await service.store_new_material(
        SYLLABUS, owner_id=ANNA, display_name="Programma", media_type="text/plain"
    )

    assert (
        await service.bindings.bind(
            lesson_id=LESSON,
            material_id=material.id,
            owner_id=BRUNO,
            selected_version_id=version.id,
            created_by=BRUNO,
        )
        is None
    )


async def test_a_material_can_be_switched_off_without_being_forgotten(service):
    material, version = await service.store_new_material(
        SYLLABUS, owner_id=ANNA, display_name="Programma", media_type="text/plain"
    )
    await service.bindings.bind(
        lesson_id=LESSON,
        material_id=material.id,
        owner_id=ANNA,
        selected_version_id=version.id,
        created_by=ANNA,
    )

    assert await service.bindings.set_enabled(
        lesson_id=LESSON, material_id=material.id, owner_id=ANNA, enabled=False
    )
    assert await service.bindings.list_for_lesson(LESSON, owner_id=ANNA) == []

    still_there = await service.bindings.list_for_lesson(LESSON, owner_id=ANNA, enabled_only=False)
    assert len(still_there) == 1
    assert isinstance(still_there[0], LessonMaterialBinding)


async def test_bruno_sees_no_bindings_on_annas_lesson(service):
    material, version = await service.store_new_material(
        SYLLABUS, owner_id=ANNA, display_name="Programma", media_type="text/plain"
    )
    await service.bindings.bind(
        lesson_id=LESSON,
        material_id=material.id,
        owner_id=ANNA,
        selected_version_id=version.id,
        created_by=ANNA,
    )

    assert await service.bindings.list_for_lesson(LESSON, owner_id=BRUNO) == []


async def test_a_binding_refuses_a_lesson_that_belongs_to_someone_else(service, session):
    """Checking only the material let a teacher attach a document to another
    teacher's lesson: the binding's own ``owner_id`` records what the caller
    claimed, so every owner-filtered read afterwards looks correct
    (Louis, 2026-09-15)."""
    material, version = await service.store_new_material(
        b"Programma di classe",
        owner_id=ANNA,
        display_name="Programma",
        media_type="text/plain",
    )
    await session.flush()

    if await session.get(User, BRUNO) is None:
        session.add(
            User(
                id=BRUNO,
                email="bruno@scuola.it",
                hashed_password="x",
                is_active=True,
                is_superuser=False,
                is_verified=True,
            )
        )
    bruno_lesson = uuid.uuid4()
    session.add(Lesson(id=bruno_lesson, owner_id=BRUNO, domain="neuro", status="draft"))
    await session.flush()

    binding = await BindingRepository(session).bind(
        lesson_id=bruno_lesson,
        material_id=material.id,
        owner_id=ANNA,
        selected_version_id=version.id,
        created_by=ANNA,
    )

    assert binding is None
