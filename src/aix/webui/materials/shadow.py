"""The shadow half of an upload (Fase 1 · PR 4).

``POST /webui/lesson/{id}/upload`` keeps doing exactly what it does today. These
two functions are what it does *in addition*, once ``AIX_MATERIALS_ENABLED`` is
on: put the same bytes through the durable storage, record the material and its
first version, and queue it for processing.

Everything here is best effort, and that is the contract rather than a
concession. The guide's acceptance condition for this PR is «failure del worker
non rompe la chat», and a teacher attaching a file to a lesson has no idea a
second pipeline exists — so if that pipeline is broken, misconfigured, or its
disk is full, the only correct behaviour is for the upload to succeed anyway and
for the problem to end up in the log. Neither function raises. Both return
something the caller can ignore.

They also take their own sessions rather than the request's. A failed flush
poisons the session it happens in, and the request's session is still needed
after this returns; borrowing it would turn a shadow failure into a broken page,
which is the exact thing this module exists to prevent.

**The binding is created only once something honours it.** ``lesson_material_binding``
says a lesson *uses* a material. Through shadow mode it was not written at all,
because the Writer was still being fed the manifest excerpt and a table that
claims a relationship nobody acts on is a table that lies. With
``AIX_PRIVATE_RAG_ENABLED`` on it becomes true and load-bearing: it is exactly
the set of versions the private retrieval is authorized against (§8.5).
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Optional

from aix.webui.errors import report
from aix.webui.materials.embedding import private_rag_enabled
from aix.webui.materials.handlers import PARSER_PROFILE
from aix.webui.materials.jobs import JobDispatcher
from aix.webui.materials.repository import (
    BindingRepository,
    ChunkRepository,
    MaterialRepository,
    MaterialService,
    MaterialVersionRepository,
)
from aix.webui.materials.storage import get_storage_backend, materials_enabled

logger = logging.getLogger(__name__)


def _media_type_for(filename: str) -> str:
    """The same mapping the legacy manifest records, not a second one."""
    from aix.webui.lessons.uploads import MIME_FOR_EXT

    return MIME_FOR_EXT.get(Path(filename).suffix.lower(), "application/octet-stream")


async def store_and_queue(
    *,
    owner_id: uuid.UUID,
    filename: str,
    content: bytes,
    lesson_id: Optional[uuid.UUID] = None,
) -> dict[str, Any]:
    """Store the bytes as a material, bind it to the lesson if any, queue the parse.

    The core of :func:`record_upload`, without its safety net: it raises. The
    WebUI calls it through ``record_upload``, after the legacy path has already
    kept the teacher's file; the API (``materials.library``) calls it directly,
    where a failure must be an answer, not a silence.
    """
    from aix.webui.db import async_session_maker

    storage = get_storage_backend()
    async with async_session_maker() as session:
        service = MaterialService(session, storage)
        material, version = await service.store_new_material(
            content,
            owner_id=owner_id,
            display_name=Path(filename).name[:240] or "materiale",
            media_type=_media_type_for(filename),
        )
        await session.commit()

    # Fase 2 · PR 6 — now the binding is honest. In shadow mode it would
    # have said a lesson *uses* a material while the Writer was still being
    # fed the manifest excerpt; with the private retriever on, the binding is
    # exactly what the retrieval is authorized against (§8.5).
    if lesson_id is not None and private_rag_enabled():
        async with async_session_maker() as session:
            bound = await BindingRepository(session).bind(
                lesson_id=lesson_id,
                material_id=material.id,
                owner_id=owner_id,
                selected_version_id=version.id,
                created_by=owner_id,
            )
            await session.commit()
        if bound is None:
            logger.warning(
                "[private-rag] lesson=%s refused the binding for material=%s",
                lesson_id,
                material.id,
            )

    job = await JobDispatcher(async_session_maker).enqueue(
        owner_id=owner_id,
        material_version_id=version.id,
        parser_profile_version=PARSER_PROFILE,
    )

    logger.info(
        "[shadow] material=%s version=%s job=%s queued for %s",
        material.id,
        version.id,
        job.id if job else "already-queued",
        PARSER_PROFILE,
    )
    return {
        "material_id": str(material.id),
        "material_version_id": str(version.id),
        "processing_status": "processing",
    }


async def record_upload(
    *,
    owner_id: uuid.UUID,
    filename: str,
    content: bytes,
    lesson_id: Optional[uuid.UUID] = None,
) -> Optional[dict[str, Any]]:
    """Store the bytes and queue them, returning the fields for the manifest.

    ``None`` when the pipeline is off, or when anything at all went wrong. The
    caller writes the returned dict onto the manifest entry the legacy path just
    created, which is what makes a shadow run inspectable afterwards without
    changing anything the teacher sees.
    """
    if not materials_enabled():
        return None

    try:
        return await store_and_queue(
            owner_id=owner_id, filename=filename, content=content, lesson_id=lesson_id
        )
    except Exception as exc:  # noqa: BLE001 — the legacy upload must still succeed
        report(exc, context=f"shadow upload owner={owner_id}")
        return None


async def retire_material(
    *,
    material_id: str,
    owner_id: uuid.UUID,
    force: bool = False,
) -> None:
    """Remove a material from retrieval when its attachment is removed (§12.3).

    The teacher's action is «togli questo documento dalla lezione», and until
    2026-09-18 it only soft-deleted the ``material`` row: the bindings stayed
    enabled and the chunks stayed ``active``, so the document went on answering
    questions and being quoted. §12.3's order is followed here instead:

    1. disable the bindings, so a run starting mid-deletion cannot use it;
    2. delete the chunks — the copy that retrieval actually reads;
    3. tombstone the versions and the material.

    Soft at the end, deliberately: a lesson already written from the material
    stays explainable, and the blob it points at is never orphaned
    (``material_version.blob_id`` is ``ON DELETE RESTRICT``).

    ``force`` is reserved for the explicit library/API retirement action. The
    legacy shadow hook keeps its original feature-gated no-op behavior.
    """
    if not force and not materials_enabled():
        return

    try:
        from aix.webui.db import async_session_maker

        async with async_session_maker() as session:
            material_uuid = uuid.UUID(str(material_id))

            # Before the soft delete: ``list_for_material`` filters deleted
            # materials out, so afterwards there would be no versions to clean.
            versions = await MaterialVersionRepository(session).list_for_material(
                material_uuid, owner_id=owner_id
            )
            lessons = await BindingRepository(session).disable_for_material(
                material_uuid, owner_id=owner_id
            )

            # A permanent library deletion must also remove the attachment
            # chips from every linked lesson.  The legacy remove route already
            # did this before calling ``retire_material``; keeping the cleanup
            # here makes the new explicit library action equally complete.
            if lessons:
                from sqlalchemy import select

                from aix.webui.lessons.models import Lesson

                linked = (
                    await session.execute(
                        select(Lesson).where(
                            Lesson.id.in_(lessons),
                            Lesson.owner_id == owner_id,
                        )
                    )
                ).scalars()
                for lesson in linked:
                    manifest = list(lesson.uploaded_files_json or [])
                    lesson.uploaded_files_json = [
                        entry
                        for entry in manifest
                        if not (
                            isinstance(entry, dict)
                            and str(entry.get("material_id")) == str(material_uuid)
                        )
                    ]

            chunks_removed = 0
            for version in versions:
                chunks_removed += await ChunkRepository(session).delete_for_version(
                    version.id, owner_id=owner_id
                )
            await MaterialVersionRepository(session).mark_deleted(
                [version.id for version in versions], owner_id=owner_id
            )

            removed = await MaterialRepository(session).soft_delete(
                material_uuid, owner_id=owner_id
            )
            await session.commit()

        if removed:
            logger.info(
                "[shadow] material=%s retired: %d chunk(s) removed, %d binding(s) disabled",
                material_id,
                chunks_removed,
                len(lessons),
            )
    except Exception as exc:  # noqa: BLE001 — the legacy delete must still succeed
        report(exc, context=f"shadow retire material={material_id}")
