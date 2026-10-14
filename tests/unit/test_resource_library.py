from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from aix.api.main import app
from aix.webui.auth.dependencies import current_active_user, optional_current_user
from aix.webui.auth.models import User
from aix.webui.db import Base, get_async_session
from aix.webui.lessons.models import Lesson
from aix.webui.materials.models import IngestionJob, LessonMaterialBinding, Material
from aix.webui.materials.repository import BindingRepository, MaterialRepository, MaterialService
from aix.webui.materials.storage import LocalDurableStorage

pytestmark = pytest.mark.unit

ANNA = uuid.UUID("11111111-1111-1111-1111-111111111111")
BRUNO = uuid.UUID("22222222-2222-2222-2222-222222222222")
LESSON = uuid.UUID("33333333-3333-3333-3333-333333333333")
SECOND_LESSON = uuid.UUID("44444444-4444-4444-4444-444444444444")
CONTENT = b"Programma privato di storia"


@pytest.fixture
async def library(tmp_path, monkeypatch):
    monkeypatch.setenv("WEBUI_CSRF_ENABLED", "false")
    monkeypatch.setenv("AIX_STORAGE_PROVIDER", "local")
    monkeypatch.setenv("AIX_STORAGE_LOCAL_ROOT", str(tmp_path / "objects"))
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{(tmp_path / 'library.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)

    async with maker() as session:
        anna = User(
            id=ANNA,
            email="anna@scuola.it",
            hashed_password="x",
            is_active=True,
            is_superuser=False,
            is_verified=True,
        )
        bruno = User(
            id=BRUNO,
            email="bruno@scuola.it",
            hashed_password="x",
            is_active=True,
            is_superuser=False,
            is_verified=True,
        )
        lesson = Lesson(
            id=LESSON,
            owner_id=ANNA,
            domain="neuro",
            status="complete",
            title="Storia",
            educational_profile_json={},
        )
        second = Lesson(
            id=SECOND_LESSON,
            owner_id=ANNA,
            domain="neuro",
            status="draft",
            title="Seconda lezione",
            educational_profile_json={},
        )
        session.add_all([anna, bruno, lesson, second])
        await session.flush()

        material, version = await MaterialService(
            session,
            LocalDurableStorage(tmp_path / "objects"),
        ).store_new_material(
            CONTENT,
            owner_id=ANNA,
            display_name="programma-storia.txt",
            media_type="text/plain",
        )
        version.status = "ready"
        await BindingRepository(session).bind(
            lesson_id=lesson.id,
            material_id=material.id,
            owner_id=ANNA,
            selected_version_id=version.id,
            created_by=ANNA,
        )
        lesson.uploaded_files_json = [
            {
                "id": "file-1",
                "filename": material.display_name,
                "mime": "text/plain",
                "size": len(CONTENT),
                "text_excerpt": "",
                "material_id": str(material.id),
                "material_version_id": str(version.id),
            }
        ]
        await session.commit()
        material_id = material.id
        version_id = version.id

    state = {"user_id": ANNA}

    async def session_override():
        async with maker() as session:
            yield session

    def user_override():
        user_id = state["user_id"]
        return User(
            id=user_id,
            email="anna@scuola.it" if user_id == ANNA else "bruno@scuola.it",
            hashed_password="x",
            is_active=True,
            is_superuser=False,
            is_verified=True,
        )

    app.dependency_overrides[get_async_session] = session_override
    app.dependency_overrides[optional_current_user] = user_override
    app.dependency_overrides[current_active_user] = user_override
    monkeypatch.setattr("aix.webui.db.async_session_maker", maker)
    client = TestClient(app, follow_redirects=False)
    client.act_as = lambda user_id: state.__setitem__("user_id", user_id)
    try:
        yield client, maker, material_id, version_id
    finally:
        app.dependency_overrides.clear()
        await engine.dispose()


def test_resources_is_a_real_owner_scoped_library_and_docs_stay_linked(library):
    client, _maker, _material_id, _version_id = library

    page = client.get("/webui/resources")
    assert page.status_code == 200
    assert "programma-storia.txt" in page.text
    assert 'href="/docs"' in page.text

    client.act_as(BRUNO)
    other = client.get("/webui/resources")
    assert other.status_code == 200
    assert "programma-storia.txt" not in other.text


def test_owner_can_download_but_another_teacher_gets_404(library):
    client, _maker, material_id, version_id = library
    path = f"/webui/materials/{material_id}/versions/{version_id}/source"

    own = client.get(path)
    assert own.status_code == 200
    assert own.content == CONTENT

    client.act_as(BRUNO)
    assert client.get(path).status_code == 404


async def test_library_mode_detaches_without_retiring(library, monkeypatch):
    client, maker, material_id, _version_id = library
    monkeypatch.setenv("AIX_MATERIAL_LIBRARY_ENABLED", "true")

    response = client.delete(f"/webui/lesson/{LESSON}/upload/file-1")
    assert response.status_code == 200

    async with maker() as session:
        material = await MaterialRepository(session).get(material_id, owner_id=ANNA)
        bindings = await BindingRepository(session).list_for_lesson(
            LESSON,
            owner_id=ANNA,
            enabled_only=False,
        )
        lesson = await session.get(Lesson, LESSON)
        assert material is not None
        assert bindings and bindings[0].enabled is False
        assert lesson.uploaded_files_json == []


async def test_material_api_can_detach_without_retiring(library, monkeypatch):
    client, maker, material_id, _version_id = library
    monkeypatch.setenv("AIX_MATERIAL_LIBRARY_ENABLED", "true")

    response = client.delete(f"/api/v1/materials/{material_id}/lessons/{LESSON}")
    assert response.status_code == 204

    async with maker() as session:
        material = await MaterialRepository(session).get(material_id, owner_id=ANNA)
        bindings = await BindingRepository(session).list_for_lesson(
            LESSON,
            owner_id=ANNA,
            enabled_only=False,
        )
        lesson = await session.get(Lesson, LESSON)
        assert material is not None
        assert bindings[0].enabled is False
        assert lesson.uploaded_files_json == []


async def test_existing_material_can_be_attached_to_another_lesson(library, monkeypatch):
    client, maker, material_id, _version_id = library
    monkeypatch.setenv("AIX_MATERIAL_LIBRARY_ENABLED", "true")

    response = client.post(f"/webui/lesson/{SECOND_LESSON}/materials/{material_id}/attach")
    assert response.status_code == 303

    async with maker() as session:
        bindings = await session.execute(
            LessonMaterialBinding.__table__.select().where(
                LessonMaterialBinding.lesson_id == SECOND_LESSON
            )
        )
        lesson = await session.get(Lesson, SECOND_LESSON)
        assert len(bindings.all()) == 1
        assert lesson.uploaded_files_json[0]["material_id"] == str(material_id)


async def test_retiring_a_library_material_removes_all_lesson_chips(library, monkeypatch):
    client, maker, material_id, _version_id = library
    monkeypatch.setenv("AIX_MATERIAL_LIBRARY_ENABLED", "true")
    assert (
        client.post(f"/webui/lesson/{SECOND_LESSON}/materials/{material_id}/attach").status_code
        == 303
    )
    assert client.post(f"/webui/materials/{material_id}/delete").status_code == 303

    async with maker() as session:
        material = await session.get(Material, material_id)
        first = await session.get(Lesson, LESSON)
        second = await session.get(Lesson, SECOND_LESSON)
        assert material.deleted_at is not None
        assert first.uploaded_files_json == []
        assert second.uploaded_files_json == []


def test_material_json_api_is_bearer_user_scoped(library):
    client, _maker, material_id, _version_id = library

    own = client.get("/api/v1/materials")
    assert own.status_code == 200
    assert own.json()["items"][0]["id"] == str(material_id)

    client.act_as(BRUNO)
    assert client.get(f"/api/v1/materials/{material_id}").status_code == 404


async def test_material_api_upload_is_flagged_owner_scoped_and_visible_in_lesson(
    library, monkeypatch
):
    client, maker, _material_id, _version_id = library
    assert (
        client.post(
            "/api/v1/materials",
            data={"lesson_id": str(SECOND_LESSON)},
            files={"file": ("nuovo.txt", b"Nuovo contesto privato", "text/plain")},
        ).status_code
        == 404
    )

    monkeypatch.setenv("AIX_MATERIAL_LIBRARY_ENABLED", "true")
    created = client.post(
        "/api/v1/materials",
        data={"lesson_id": str(SECOND_LESSON), "purpose": "reference"},
        files={"file": ("nuovo.txt", b"Nuovo contesto privato", "text/plain")},
    )
    assert created.status_code == 201
    material_id = created.json()["id"]

    async with maker() as session:
        lesson = await session.get(Lesson, SECOND_LESSON)
        assert lesson.uploaded_files_json[0]["material_id"] == material_id
        bindings = await BindingRepository(session).list_for_lesson(
            SECOND_LESSON,
            owner_id=ANNA,
        )
        assert str(bindings[0].material_id) == material_id
        jobs = await session.execute(IngestionJob.__table__.select())
        assert len(jobs.all()) == 1

    client.act_as(BRUNO)
    assert client.get(f"/api/v1/materials/{material_id}").status_code == 404


async def test_material_api_appends_an_immutable_version_and_job(library, monkeypatch):
    client, maker, material_id, _version_id = library
    monkeypatch.setenv("AIX_MATERIAL_LIBRARY_ENABLED", "true")

    response = client.post(
        f"/api/v1/materials/{material_id}/versions",
        files={"file": ("programma-v2.txt", b"Seconda versione", "text/plain")},
    )
    assert response.status_code == 201
    assert len(response.json()["versions"]) == 2

    async with maker() as session:
        jobs = await session.execute(IngestionJob.__table__.select())
        assert len(jobs.all()) == 1


def test_saved_external_resources_are_separate_and_owner_scoped(library):
    client, _maker, _material_id, _version_id = library
    response = client.post(
        "/webui/resources/saved",
        data={
            "kind": "video",
            "source": "live",
            "title": "Memoria di lavoro",
            "url": "https://example.test/watch?v=1&utm_source=aix",
            "video_id": "video-1",
            "next": "/webui/resources",
        },
    )
    assert response.status_code == 303
    assert "Memoria di lavoro" in client.get("/webui/resources").text

    client.act_as(BRUNO)
    assert "Memoria di lavoro" not in client.get("/webui/resources").text


def test_saved_resource_api_is_owner_scoped(library):
    client, _maker, _material_id, _version_id = library
    created = client.post(
        "/api/v1/materials/saved/resources",
        json={
            "kind": "article",
            "source": "curated",
            "title": "Working memory",
            "url": "https://example.test/article?utm_source=aix",
        },
    )
    assert created.status_code == 201
    resource_id = created.json()["id"]
    assert created.json()["url"] == "https://example.test/article"

    client.act_as(BRUNO)
    assert client.delete(f"/api/v1/materials/saved/resources/{resource_id}").status_code == 404
    assert client.get("/api/v1/materials/saved/resources").json() == []
