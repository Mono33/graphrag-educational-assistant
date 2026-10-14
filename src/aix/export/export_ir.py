"""The target-neutral snapshot every renderer reads (§10.3).

``ExportIRV1`` is the lesson grouped into sections, with the timed outline, the
disclosure and the provenance an export must carry. It is what the PDF renderer
reads today, what the outline renderer reads today, and what the Slides planner
will read in PR 9.

§10.3 lists what it must *not* contain: no HTML, no ReportLab objects, no Google
file id, no token, no local path. That list is the design. A snapshot free of
target details is one that can be rendered again later — to a second format, or
to the same one after a renderer fix — and still describe exactly the revision
the teacher exported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from aix.core.ai_marking import DISCLOSURE_LONG, DISCLOSURE_SHORT
from aix.export.lesson_document import Block, LessonDocumentV1

IR_VERSION = 1


@dataclass(frozen=True)
class ExportSection:
    """A heading and the blocks under it, up to the next heading."""

    title: str
    level: int
    minutes: Optional[int]
    heading_id: Optional[str]
    blocks: tuple[Block, ...]


@dataclass(frozen=True)
class ExportIRV1:
    lesson_id: Optional[str]
    content_hash: str
    title: str
    language: str
    duration_minutes: Optional[int]
    #: Blocks before the first heading, then one entry per heading.
    preamble: tuple[Block, ...]
    sections: tuple[ExportSection, ...]
    timed_outline: tuple[tuple[str, int], ...]
    disclosure_short: str
    disclosure_long: str
    warnings: tuple[str, ...]
    ir_version: int = IR_VERSION

    @property
    def outline_total_minutes(self) -> Optional[int]:
        if not self.timed_outline:
            return None
        return sum(minutes for _title, minutes in self.timed_outline)


def build_export_ir(document: LessonDocumentV1) -> ExportIRV1:
    """Group the document's blocks under their headings.

    The level-1 title heading is not a section of its own: it is the document's
    title, and the renderers draw it once, as a title. Everything else is kept
    in order, including headings with no content under them — a heading the
    Writer produced is part of the lesson even when it is empty.
    """
    preamble: list[Block] = []
    sections: list[ExportSection] = []
    current_heading: Optional[Block] = None
    current_blocks: list[Block] = []
    title_consumed = False

    def _flush() -> None:
        if current_heading is not None:
            sections.append(
                ExportSection(
                    title=current_heading.text.strip(),
                    level=current_heading.level,
                    minutes=current_heading.minutes,
                    heading_id=current_heading.block_id,
                    blocks=tuple(current_blocks),
                )
            )

    for block in document.blocks:
        if block.kind == "heading":
            if block.level == 1 and not title_consumed and block.text.strip() == document.title:
                title_consumed = True
                continue
            _flush()
            current_heading, current_blocks = block, []
            continue

        if current_heading is None:
            preamble.append(block)
        else:
            current_blocks.append(block)

    _flush()

    return ExportIRV1(
        lesson_id=document.lesson_id,
        content_hash=document.content_hash,
        title=document.title,
        language=document.language,
        duration_minutes=document.duration_minutes,
        preamble=tuple(preamble),
        sections=tuple(sections),
        timed_outline=document.timed_outline,
        disclosure_short=DISCLOSURE_SHORT,
        disclosure_long=DISCLOSURE_LONG,
        warnings=document.warnings,
    )
