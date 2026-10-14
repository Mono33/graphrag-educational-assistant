"""A user's documents, independent of any lesson (2026-10-14).

The WebUI attaches a document to one lesson: the upload stores it, binds it and
queues its parse in one gesture. A chatbot platform uploads a document once and
uses it across many conversations, so the API needs the two halves apart:
*add* a document to the owner's library, then *bind* it to a session when a
conversation wants it. Both halves were already there -- ``store_and_queue``
and ``BindingRepository.bind`` -- and the authorization is the same
``owner_id`` filter every material table carries.

What a document's state means for a caller:

* ``processing`` -- uploaded, being parsed, chunked and indexed;
* ``ready`` -- searchable (``ready_with_warnings`` counts: some pages unread,
  reported in ``quality``);
* ``failed`` -- it will never be searchable; ``error_code`` says why.

A turn on a session waits for its documents exactly as in the WebUI
(``still_preparing``): a lesson is never written from a document still being read.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from aix.webui.materials.embedding import private_rag_enabled
from aix.webui.materials.models import (
    IngestionJob,
    LessonMaterialBinding,
    Material,
    MaterialChunk,
)
from aix.webui.materials.parsing.intake import IntakeRejected, check_upload
from aix.webui.materials.readiness import PREPARING_STATUSES
from aix.webui.materials.repository import (
    READY_STATUSES,
    BindingRepository,
    MaterialRepository,
    MaterialVersionRepository,
)
from aix.webui.materials.storage import materials_enabled


class LibraryUnavailable(RuntimeError):
    """The document pipeline or the private retrieval is switched off on this server."""

    code = "materials_disabled"


class UnknownMaterial(LookupError):
    """An id that is not one of this owner's documents. Never says whose it is."""

    code = "unknown_material"


@dataclass(frozen=True)
class MaterialInfo:
    """One document as a caller sees it."""

    material_id: str
    version_id: Optional[str]
    name: str
    status: str
    raw_status: Optional[str]
    passages: int = 0
    quality: Optional[dict[str, Any]] = None
    error_code: Optional[str] = None
    created_at: Optional[datetime] = None
    sessions: list[str] = field(default_factory=list)


def available() -> bool:
    """Both halves on: the materials pipeline and the private retrieval that reads it."""
    return materials_enabled() and private_rag_enabled()


def public_status(raw: Optional[str]) -> str:
    if raw in READY_STATUSES:
        return "ready"
    if raw is None or raw in PREPARING_STATUSES:
        return "processing"
    return "failed"


async def add_material(*, owner_id: uuid.UUID, filename: str, content: bytes) -> MaterialInfo:
    """Check the file the way the WebUI does, store it and queue its parse.

    Raises ``IntakeRejected`` (with its ``code``) for a file the WebUI would
    refuse too, and ``LibraryUnavailable`` when the pipeline is off.
    """
    if not available():
        raise LibraryUnavailable("I documenti non sono attivi su questo server.")
    check_upload(filename, content)

    from aix.webui.materials.shadow import store_and_queue

    stored = await store_and_queue(owner_id=owner_id, filename=filename, content=content)
    return MaterialInfo(
        material_id=stored["material_id"],
        version_id=stored["material_version_id"],
        name=filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1][:240] or "materiale",
        status="processing",
        raw_status=None,
    )


async def list_materials(
    session: AsyncSession,
    *,
    owner_id: uuid.UUID,
    material_ids: Optional[list[uuid.UUID]] = None,
) -> list[MaterialInfo]:
    """This owner's documents with their latest version's state, newest first."""
    query = select(Material).where(Material.owner_id == owner_id, Material.deleted_at.is_(None))
    if material_ids is not None:
        query = query.where(Material.id.in_(material_ids))
    materials = (await session.execute(query.order_by(Material.created_at.desc()))).scalars().all()

    versions_repo = MaterialVersionRepository(session)
    out: list[MaterialInfo] = []
    for material in materials:
        version = await versions_repo.latest_for_material(material.id, owner_id=owner_id)
        passages = 0
        error_code = None
        if version is not None:
            passages = int(
                await session.scalar(
                    select(func.count(MaterialChunk.id)).where(
                        MaterialChunk.owner_id == owner_id,
                        MaterialChunk.material_version_id == version.id,
                        MaterialChunk.status == "active",
                    )
                )
                or 0
            )
            if public_status(version.status) == "failed":
                error_code = await session.scalar(
                    select(IngestionJob.safe_error_code)
                    .where(
                        IngestionJob.owner_id == owner_id,
                        IngestionJob.material_version_id == version.id,
                    )
                    .order_by(IngestionJob.created_at.desc())
                    .limit(1)
                )
        sessions = (
            (
                await session.execute(
                    select(LessonMaterialBinding.lesson_id).where(
                        LessonMaterialBinding.owner_id == owner_id,
                        LessonMaterialBinding.material_id == material.id,
                        LessonMaterialBinding.enabled.is_(True),
                    )
                )
            )
            .scalars()
            .all()
        )
        out.append(
            MaterialInfo(
                material_id=str(material.id),
                version_id=str(version.id) if version else None,
                name=material.display_name,
                status=public_status(version.status if version else None),
                raw_status=version.status if version else None,
                passages=passages,
                quality=version.parse_quality
                if version and isinstance(version.parse_quality, dict)
                else None,
                error_code=error_code,
                created_at=material.created_at,
                sessions=[str(lesson_id) for lesson_id in sessions],
            )
        )
    return out


async def get_material(
    session: AsyncSession, *, owner_id: uuid.UUID, material_id: uuid.UUID
) -> MaterialInfo:
    found = await list_materials(session, owner_id=owner_id, material_ids=[material_id])
    if not found:
        raise UnknownMaterial(str(material_id))
    return found[0]


async def bind_materials(
    session: AsyncSession,
    *,
    lesson_id: uuid.UUID,
    owner_id: uuid.UUID,
    material_ids: list[uuid.UUID],
) -> list[str]:
    """Let a session's turns search these documents. Already bound ones are left as they are.

    Every id must be this owner's: one that is not raises ``UnknownMaterial``
    before anything is bound, so a request is all or nothing.
    """
    if not material_ids:
        return []
    if not available():
        raise LibraryUnavailable("I documenti non sono attivi su questo server.")

    materials_repo = MaterialRepository(session)
    versions_repo = MaterialVersionRepository(session)
    plan: list[tuple[uuid.UUID, uuid.UUID]] = []
    for material_id in dict.fromkeys(material_ids):
        material = await materials_repo.get(material_id, owner_id=owner_id)
        version = (
            await versions_repo.latest_for_material(material_id, owner_id=owner_id)
            if material is not None
            else None
        )
        if material is None or version is None:
            raise UnknownMaterial(str(material_id))
        plan.append((material_id, version.id))

    already = set(
        (
            await session.execute(
                select(LessonMaterialBinding.material_id).where(
                    LessonMaterialBinding.lesson_id == lesson_id,
                    LessonMaterialBinding.owner_id == owner_id,
                )
            )
        )
        .scalars()
        .all()
    )
    bindings = BindingRepository(session)
    bound: list[str] = []
    for material_id, version_id in plan:
        if material_id in already:
            await bindings.set_enabled(
                lesson_id=lesson_id, material_id=material_id, owner_id=owner_id, enabled=True
            )
        elif (
            await bindings.bind(
                lesson_id=lesson_id,
                material_id=material_id,
                owner_id=owner_id,
                selected_version_id=version_id,
                created_by=owner_id,
            )
            is None
        ):
            raise UnknownMaterial(str(material_id))
        bound.append(str(material_id))
    await session.commit()
    return bound


async def remove_material(
    session: AsyncSession, *, owner_id: uuid.UUID, material_id: uuid.UUID
) -> None:
    """Out of retrieval and out of the library, in the order §12.3 sets (``retire_material``)."""
    if await MaterialRepository(session).get(material_id, owner_id=owner_id) is None:
        raise UnknownMaterial(str(material_id))
    from aix.webui.materials.shadow import retire_material

    await retire_material(material_id=str(material_id), owner_id=owner_id)


__all__ = [
    "IntakeRejected",
    "LibraryUnavailable",
    "MaterialInfo",
    "UnknownMaterial",
    "add_material",
    "available",
    "bind_materials",
    "get_material",
    "list_materials",
    "public_status",
    "remove_material",
]
