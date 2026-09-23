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

import logging
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

#: §12.3 — a lesson keeps a citation tombstone, never the deleted content. The
#: teacher removed the document; the sentence that leaned on it stays, and says
#: so. Silently dropping the reference would leave a claim with no source at all.
TOMBSTONE_LABEL = "materiale rimosso"


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
        if label == TOMBSTONE_LABEL:
            return f"({label})"  # the page of a document nobody can open means nothing
        return f"({label}, p. {page})" if page else f"({label})"

    # Tidy the space a removed citation leaves behind mid-sentence.
    return re.sub(r" +([.,;:])", r"\1", CITATION_RE.sub(_replace, text))


def has_citations(text: str) -> bool:
    return bool(CITATION_RE.search(text or ""))


#: The heading of the block appended below a lesson that quoted the teacher's
#: own documents. Decision U3 (2026-09-18): the inline reference says *where* a
#: sentence comes from, this says *what was used* for the lesson as a whole.
SOURCES_HEADING = "**Fonti dai tuoi materiali**"


def sources_block(text: str, labels: Optional[dict[str, str]] = None) -> str:
    """The list of documents this text cites, or ``""`` if it cites none.

    Built by reading the markers, never by asking the model for a bibliography:
    a list the Writer composed would eventually name a document the lesson did
    not use, and nobody could tell by looking. Everything here comes from
    markers that survived :func:`render_citations`'s authorization check, so the
    block cannot name material this lesson is not bound to.

    Documents appear in the order they are first cited, pages in numeric order.
    """
    if not text:
        return ""

    pages: dict[str, list[str]] = {}
    for match in CITATION_RE.finditer(text):
        version_id, anchor = match.group(1).lower(), match.group(2)
        if labels is not None:
            label = labels.get(version_id)
            if label is None:
                continue  # not this lesson's material — it is not a source
        else:
            label = GENERIC_LABEL

        seen = pages.setdefault(label, [])
        page = _page_of(anchor)
        # A tombstone has no page worth printing: nobody can open it.
        if page and page not in seen and label != TOMBSTONE_LABEL:
            seen.append(page)

    if not pages:
        return ""

    lines = [SOURCES_HEADING, ""]
    for label, found in pages.items():
        numbers = sorted(found, key=lambda p: (not p.isdigit(), int(p) if p.isdigit() else 0, p))
        if not numbers:
            lines.append(f"- {label}")
        elif len(numbers) == 1:
            lines.append(f"- {label} — p. {numbers[0]}")
        else:
            lines.append(f"- {label} — pp. {', '.join(numbers)}")
    return "\n".join(lines)


logger = logging.getLogger(__name__)

# The Writer's output hygiene lives with the Writer; re-exported here for
# the callers that learnt this address first.
from aix.agent.validation.answer_shape import (  # noqa: E402
    once_only_disclaimer,
    tidy_answer,
)


def with_sources(text: str, labels: Optional[dict[str, str]] = None) -> str:
    """``text`` with its citations made readable and its sources listed below.

    Idempotent where it matters: the block is built from the markers, and
    rendering removes them, so text that has already been through this function
    gains nothing a second time.
    """
    text = tidy_answer(text)
    block = sources_block(text, labels)
    rendered = render_citations(text, labels)
    return f"{rendered.rstrip()}\n\n{block}\n" if block else rendered


async def labels_for_lesson(
    session: AsyncSession, *, lesson_id: uuid.UUID, owner_id: uuid.UUID
) -> dict[str, str]:
    """``version id -> what to print``, for the material this lesson ever cited.

    Built from the bindings, so the authorization that governs retrieval governs
    the citations too: a version this lesson was never bound to has no label, and
    a marker naming it is removed rather than shown.

    Bindings that have been switched off, and versions tombstoned by a deletion,
    are included with :data:`TOMBSTONE_LABEL`. That is the difference between «the
    teacher removed this document» — which the lesson should say — and «the model
    invented an identifier», which it should not repeat.
    """
    from sqlalchemy import select

    from aix.webui.materials.models import LessonMaterialBinding, Material, MaterialVersion

    rows = await session.execute(
        select(MaterialVersion.id, Material.display_name, MaterialVersion.status)
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
    return {
        str(version_id).lower(): (TOMBSTONE_LABEL if status == "deleted" else name)
        for version_id, name, status in rows
    }


async def resolve_for_lesson(text: str, session: AsyncSession, lesson: Any) -> str:
    """Render the citations of one lesson's text and list its sources, best effort.

    Never raises: a lesson that cannot be shown because its citations could not
    be looked up would be a poor trade, so on failure the neutral label is used.
    """
    # Tidied before the early return, not after it. The first version of the
    # disclaimer fix lived only in ``with_sources``, which this function skips
    # for a lesson without private citations -- so the PDF of such a lesson kept
    # both copies while the WebUI showed one (2026-09-23).
    text = tidy_answer(text)
    if not has_citations(text):
        return text

    try:
        labels = await labels_for_lesson(session, lesson_id=lesson.id, owner_id=lesson.owner_id)
    except Exception:  # noqa: BLE001 — the lesson matters more than the filename
        return with_sources(text)
    return with_sources(text, labels)


async def labels_for_lesson_safe(session: AsyncSession, lesson: Any) -> Optional[dict[str, str]]:
    """``labels_for_lesson`` for the render path: ``None`` rather than an error.

    ``None`` means "print the neutral label", which is what the chat page did
    before there was anything to look up — a page that fails to paint because a
    filename could not be read would be a bad trade.
    """
    try:
        return await labels_for_lesson(session, lesson_id=lesson.id, owner_id=lesson.owner_id)
    except Exception:  # noqa: BLE001 — see the docstring
        return None
