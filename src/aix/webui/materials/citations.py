"""Turning the model's citation markers into something a teacher can read (§8.6).

The Writer is given markers and copies them back:

    [[material:9f1c…#page=7&block=1a2b]]

It never writes a URL, and that is the point: a citation the model composed
could point anywhere, including at another teacher's document. Here the marker
is resolved against what this lesson is actually bound to, and anything else is
removed rather than shown.

So there are exactly three outcomes for a marker:

* it names a version bound to this lesson → the teacher sees the document's name
  and the page, e.g. «(guerra-fredda.pdf, p. 7)»;
* it names something else, or nothing at all → the marker disappears. A model
  that invents a UUID produces no citation, not a broken one;
* labels are not available (a preview rendered without a database session) → a
  neutral «(materiale allegato, p. 7)», never the raw marker.

A raw ``[[material:…]]`` must never reach a teacher: it is an internal address,
it is noise in a printed lesson, and it invites someone to paste it somewhere.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

#: ``[[material:<uuid>]]`` or ``[[material:<uuid>#page=7&block=…]]``
CITATION_RE = re.compile(
    r"\[\[material:([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
    r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})(?:#([^\]\s]*))?\]\]"
)

GENERIC_LABEL = "materiale allegato"


def _page_of(anchor: Optional[str]) -> Optional[str]:
    for part in (anchor or "").split("&"):
        key, _, value = part.partition("=")
        if key == "page" and value.strip():
            return value.strip()
    return None


def render_citations(text: str, labels: Optional[dict[str, str]] = None) -> str:
    """Replace every marker with a readable reference, or remove it."""
    if not text:
        return text

    def _replace(match: re.Match) -> str:
        version_id, anchor = match.group(1), match.group(2)
        if labels is not None:
            label = labels.get(version_id.lower())
            if label is None:
                return ""  # not this lesson's material: say nothing at all
        else:
            label = GENERIC_LABEL

        page = _page_of(anchor)
        return f"({label}, p. {page})" if page else f"({label})"

    # Tidy the space a removed citation leaves behind mid-sentence.
    return re.sub(r" +([.,;:])", r"\1", CITATION_RE.sub(_replace, text))


def has_citations(text: str) -> bool:
    return bool(CITATION_RE.search(text or ""))


async def labels_for_lesson(
    session: AsyncSession, *, lesson_id: uuid.UUID, owner_id: uuid.UUID
) -> dict[str, str]:
    """``version id -> document name``, for the material this lesson may cite.

    Built from the bindings, so the authorization that governs retrieval governs
    the citations too: a version this lesson is not bound to has no label, and a
    marker naming it is therefore removed.
    """
    from sqlalchemy import select

    from aix.webui.materials.models import LessonMaterialBinding, Material, MaterialVersion

    rows = await session.execute(
        select(MaterialVersion.id, Material.display_name)
        .join(Material, Material.id == MaterialVersion.material_id)
        .join(
            LessonMaterialBinding,
            LessonMaterialBinding.selected_version_id == MaterialVersion.id,
        )
        .where(
            LessonMaterialBinding.lesson_id == lesson_id,
            LessonMaterialBinding.owner_id == owner_id,
        )
    )
    return {str(version_id).lower(): name for version_id, name in rows}


async def resolve_for_lesson(text: str, session: AsyncSession, lesson: Any) -> str:
    """Render the citations of one lesson's text, best effort.

    Never raises: a lesson that cannot be shown because its citations could not
    be looked up would be a poor trade, so on failure the neutral label is used.
    """
    if not has_citations(text):
        return text

    try:
        labels = await labels_for_lesson(session, lesson_id=lesson.id, owner_id=lesson.owner_id)
    except Exception:  # noqa: BLE001 — the lesson matters more than the filename
        return render_citations(text)
    return render_citations(text, labels)
