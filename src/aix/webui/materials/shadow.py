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

**No binding is created.** ``lesson_material_binding`` says a lesson *uses* a
material, and in shadow mode it does not — the Writer is still fed the legacy
manifest excerpt. Recording a relationship the agent does not honour would make
the table lie for as long as shadow mode lasts. It belongs with the retriever
that resolves it, in PR 6.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Optional

from aix.webui.errors import report
from aix.webui.materials.handlers import PARSER_PROFILE
from aix.webui.materials.jobs import JobDispatcher
from aix.webui.materials.repository import MaterialRepository, MaterialService
from aix.webui.materials.storage import get_storage_backend, materials_enabled

logger = logging.getLogger(__name__)


def _media_type_for(filename: str) -> str:
    """The same mapping the legacy manifest records, not a second one."""
    from aix.webui.lessons.uploads import MIME_FOR_EXT

    return MIME_FOR_EXT.get(Path(filename).suffix.lower(), "application/octet-stream")


async def record_upload(
    *,
    owner_id: uuid.UUID,
    filename: str,
    content: bytes,
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
    except Exception as exc:  # noqa: BLE001 — the legacy upload must still succeed
        report(exc, context=f"shadow upload owner={owner_id}")
        return None


async def retire_material(*, material_id: str, owner_id: uuid.UUID) -> None:
    """Soft-delete the shadow material behind a removed attachment.

    Soft, so a lesson already written from it stays explainable, and so the
    blob it points at is never orphaned — ``material_version.blob_id`` is
    ``ON DELETE RESTRICT`` precisely to make that impossible.
    """
    if not materials_enabled():
        return

    try:
        from aix.webui.db import async_session_maker

        async with async_session_maker() as session:
            removed = await MaterialRepository(session).soft_delete(
                uuid.UUID(str(material_id)), owner_id=owner_id
            )
            await session.commit()

        if removed:
            logger.info("[shadow] material=%s retired with its attachment", material_id)
    except Exception as exc:  # noqa: BLE001 — the legacy delete must still succeed
        report(exc, context=f"shadow retire material={material_id}")
