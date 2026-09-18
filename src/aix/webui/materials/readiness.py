"""Whether a lesson's attachments are ready to be used (§6.1).

The paperclip contract says the chip shows a state and that generation uses only
material that is ``ready``. Until now neither happened: the chip showed a size
and the run started whenever the teacher pressed Invia. On 2026-09-16 that
produced the worst possible outcome — a lesson generated seconds after the
upload, before the document had been indexed, written from the pasted text while
the teacher believed it had been read. Nothing on the screen said so.

One function answers both questions, because two implementations would drift:
the chip and the run guard must never disagree about what «ready» means.

States, as §6.1 names them:

``preparing``
    Uploaded, not finished: quarantined, scanning or processing. The send button
    waits for it (decision U1, 2026-09-18).
``ready`` / ``ready_with_warnings``
    Usable. Warnings travel with the passages — a scan read by OCR is still the
    teacher's material.
``failed``
    Nothing could be read. The lesson proceeds without it rather than waiting
    forever, and the chip says which document was left out.
``deleted``
    Removed by the teacher; kept here only so a stale chip resolves to something
    honest.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from aix.webui.materials.models import (
    LessonMaterialBinding,
    Material,
    MaterialChunk,
    MaterialVersion,
)
from aix.webui.materials.repository import READY_STATUSES

#: Statuses that mean "still working on it" — the run waits for these.
PREPARING_STATUSES = ("quarantined", "scanning", "processing")


@dataclass(frozen=True)
class MaterialState:
    """What the teacher is told about one attached document."""

    version_id: str
    name: str
    status: str
    #: Retrievable passages. Zero while the private RAG is off, which is why it
    #: is shown only when it is greater than zero.
    passages: int = 0

    @property
    def preparing(self) -> bool:
        return self.status in PREPARING_STATUSES

    @property
    def usable(self) -> bool:
        return self.status in READY_STATUSES


async def states_for_lesson(
    session: AsyncSession, *, lesson_id: uuid.UUID, owner_id: uuid.UUID
) -> dict[str, MaterialState]:
    """State of every material bound to this lesson, keyed by version id.

    Empty when the material pipeline is off: there are no bindings, so there is
    nothing to wait for and the chip renders as it always did.
    """
    chunk_counts = (
        select(
            MaterialChunk.material_version_id.label("version_id"),
            func.count(MaterialChunk.id).label("passages"),
        )
        .where(MaterialChunk.owner_id == owner_id, MaterialChunk.status == "active")
        .group_by(MaterialChunk.material_version_id)
        .subquery()
    )

    rows = await session.execute(
        select(
            MaterialVersion.id,
            Material.display_name,
            MaterialVersion.status,
            func.coalesce(chunk_counts.c.passages, 0),
        )
        .join(Material, Material.id == MaterialVersion.material_id)
        .join(
            LessonMaterialBinding,
            LessonMaterialBinding.selected_version_id == MaterialVersion.id,
        )
        .outerjoin(chunk_counts, chunk_counts.c.version_id == MaterialVersion.id)
        .where(
            LessonMaterialBinding.lesson_id == lesson_id,
            LessonMaterialBinding.owner_id == owner_id,
            LessonMaterialBinding.enabled.is_(True),
        )
    )

    return {
        str(version_id): MaterialState(
            version_id=str(version_id),
            name=name,
            status=status,
            passages=int(passages or 0),
        )
        for version_id, name, status, passages in rows
    }


async def still_preparing(
    session: AsyncSession, *, lesson_id: uuid.UUID, owner_id: uuid.UUID
) -> list[str]:
    """Names of the documents the lesson should wait for. Empty means go ahead."""
    states = await states_for_lesson(session, lesson_id=lesson_id, owner_id=owner_id)
    return sorted(state.name for state in states.values() if state.preparing)


async def states_for_lesson_safe(
    session: AsyncSession, *, lesson_id: uuid.UUID, owner_id: Optional[uuid.UUID]
) -> dict[str, MaterialState]:
    """``states_for_lesson`` that never raises, for rendering a template.

    A chip that cannot be decorated is a cosmetic loss; a lesson page that fails
    to render because of it is not. The run guard uses the strict version.
    """
    if owner_id is None:
        return {}
    try:
        return await states_for_lesson(session, lesson_id=lesson_id, owner_id=owner_id)
    except Exception:  # noqa: BLE001 — the page matters more than the badge
        return {}
