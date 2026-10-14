"""One lesson, one download, whoever asks for it (2026-10-14).

The body of the WebUI's export route, moved here so the API's session export
produces the same files: citations resolved to the documents' names once (§8.6),
the AI notice in every format (#21), the same flags for the formats a pilot may
switch off. A route adds only its own authentication and the HTTP response.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Union

from sqlalchemy.ext.asyncio import AsyncSession

from aix.core.ai_marking import DISCLOSURE_LONG, DISCLOSURE_SHORT, strip_marking

#: Every format a lesson downloads as.
FORMATS = ("md", "txt", "pdf", "docx", "outline")

DOCX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


class ExportRefused(Exception):
    """A download that cannot be served: ``status`` and an Italian ``message``."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


@dataclass(frozen=True)
class ExportFile:
    content: Union[bytes, str]
    media_type: str
    filename: str


def _slug(lesson) -> str:
    slug = (lesson.title or f"lezione-{lesson.id}").lower()
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in slug).strip("-")[:60]


def _enabled(fmt: str) -> bool:
    from aix.webui.features import (
        export_docx_enabled,
        export_outline_enabled,
        export_pdf_enabled,
    )

    return {
        "pdf": export_pdf_enabled,
        "docx": export_docx_enabled,
        "outline": export_outline_enabled,
    }.get(fmt, lambda: True)()


async def render_lesson_export(session: AsyncSession, lesson, fmt: str) -> ExportFile:
    """The lesson as ``fmt``; raises ``ExportRefused`` for a format or a lesson that cannot be."""
    if not lesson.lesson_plan_md:
        raise ExportRefused(404, "no_lesson_yet", "Nessun contenuto disponibile")
    if fmt not in FORMATS:
        raise ExportRefused(
            400,
            "unsupported_format",
            "Formato non supportato. Usa ?format=md, txt, pdf, docx oppure outline",
        )
    # A hidden button is not a closed door: the flag is checked here too, or a
    # bookmarked URL would still render what the pilot switched off.
    if not _enabled(fmt):
        raise ExportRefused(404, "format_disabled", "Questo formato di export non è attivo.")

    from aix.webui.materials.citations import resolve_for_lesson

    slug = _slug(lesson)
    plan_md = await resolve_for_lesson(lesson.lesson_plan_md, session, lesson)

    if fmt == "md":
        # The machine-readable Art. 50 comment the agent stamped stays on top.
        return ExportFile(plan_md, "text/markdown; charset=utf-8", f"{slug}.md")

    if fmt == "txt":
        # The HTML comment means nothing in plain text: a readable notice instead.
        text = strip_marking(plan_md)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # links
        text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)  # images
        text = re.sub(r"#{1,6}\s*", "", text)  # headings
        text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)  # bold/italic
        text = re.sub(r"`{1,3}[^`]*`{1,3}", lambda m: m.group().strip("`"), text)  # code
        text = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", text)  # underscores
        text = re.sub(r"^\s*[-*+]\s+", "• ", text, flags=re.MULTILINE)  # bullets
        notice = f"[{DISCLOSURE_SHORT}] {DISCLOSURE_LONG}\n{'-' * 60}\n\n"
        return ExportFile(notice + text.lstrip(), "text/plain; charset=utf-8", f"{slug}.txt")

    # PDF, Word and the outline read the document model: Markdown into
    # LessonDocumentV1, grouped into ExportIRV1, then rendered (Fase 3 · PR 7).
    from aix.export import adapt_lesson_markdown, build_export_ir, render_outline

    ir = build_export_ir(
        adapt_lesson_markdown(
            plan_md, lesson_id=str(lesson.id), fallback_title=lesson.title or "Lezione"
        )
    )
    if fmt == "outline":
        return ExportFile(render_outline(ir), "text/plain; charset=utf-8", f"{slug}-scaletta.txt")
    if fmt == "docx":
        from aix.export.word import render_docx

        # Off the event loop: rendering is CPU work, and a request thread
        # blocked on it would stall every stream in the process.
        document = await asyncio.to_thread(render_docx, ir)
        return ExportFile(document.content, DOCX_MEDIA_TYPE, f"{slug}.docx")

    from aix.export.pdf import render_pdf

    result = await asyncio.to_thread(render_pdf, ir)
    return ExportFile(result.content, "application/pdf", f"{slug}.pdf")
