"""The library tied to the lessons, the bookmark that shows it took, the downloads after a reload.

Angelo's test of 2026-10-14: a bookmark reloaded the page and looked the same;
the reloaded lesson had no download buttons; the library listed the same file
twice and did not say which lesson anything came from.
"""

from __future__ import annotations

import uuid

import pytest
from tests.unit.test_resource_library import (  # noqa: F401 — the fixture and its ids
    ANNA,
    CONTENT,
    LESSON,
    SECOND_LESSON,
    library,
)

from aix.webui.lessons.models import Lesson, LessonMessage
from aix.webui.materials.repository import BindingRepository, MaterialService
from aix.webui.materials.storage import LocalDurableStorage

pytestmark = pytest.mark.unit

VIDEO = {
    "kind": "video",
    "source": "live",
    "title": "La Guerra Fredda in 10 minuti",
    "url": "https://www.youtube.com/watch?v=abc123",
    "video_id": "abc123",
}


def test_a_bookmark_saves_without_a_reload_and_shows_it_took(library):
    client, *_ = library

    response = client.post(
        "/webui/resources/saved",
        data={**VIDEO, "lesson_id": str(LESSON), "next": f"/webui/lesson/{LESSON}"},
        headers={"HX-Request": "true"},
    )

    assert response.status_code == 200, "a fragment for htmx, not a redirect"
    assert "is-saved" in response.text
    page = client.get("/webui/resources").text
    assert "La Guerra Fredda in 10 minuti" in page
    assert "Dalla lezione:" in page and f"/webui/lesson/{LESSON}" in page


def test_a_lesson_the_teacher_does_not_own_is_not_recorded(library):
    client, *_ = library

    client.post(
        "/webui/resources/saved",
        data={**VIDEO, "lesson_id": str(uuid.uuid4())},
        headers={"HX-Request": "true"},
    )

    assert "Dalla lezione:" not in client.get("/webui/resources").text


async def test_the_same_file_twice_is_one_card_with_both_lessons(library, tmp_path):
    client, maker, *_ = library
    async with maker() as session:
        material, version = await MaterialService(
            session, LocalDurableStorage(tmp_path / "objects")
        ).store_new_material(
            CONTENT, owner_id=ANNA, display_name="programma-storia.txt", media_type="text/plain"
        )
        version.status = "ready"
        await BindingRepository(session).bind(
            lesson_id=SECOND_LESSON,
            material_id=material.id,
            owner_id=ANNA,
            selected_version_id=version.id,
            created_by=ANNA,
        )
        await session.commit()

    page = client.get("/webui/resources").text

    assert page.count("programma-storia.txt</h2>") == 1
    assert "Caricato 2 volte" in page
    assert f"/webui/lesson/{LESSON}" in page and f"/webui/lesson/{SECOND_LESSON}" in page


async def test_a_reloaded_lesson_still_has_its_download_buttons(library):
    client, maker, *_ = library
    async with maker() as session:
        lesson = await session.get(Lesson, LESSON)
        lesson.lesson_plan_md = "# Storia\n\n## Introduzione *(10 minuti)*\n\nTesto."
        session.add(LessonMessage(lesson_id=LESSON, role="user", content_md="Crea", turn_index=1))
        session.add(
            LessonMessage(
                lesson_id=LESSON,
                role="assistant",
                content_md=lesson.lesson_plan_md,
                turn_index=1,
                agent_kind="writer",
                meta_json={"approved": True},
            )
        )
        await session.commit()

    page = client.get(f"/webui/lesson/{LESSON}").text

    assert f"/webui/lesson/{LESSON}/export?format=pdf" in page
    assert f"/webui/lesson/{LESSON}/export?format=docx" in page
