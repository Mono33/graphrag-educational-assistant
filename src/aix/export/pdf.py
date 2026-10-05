"""The lesson as a real PDF (§10.4).

Until now the "PDF" button opened ``/print`` and left the rest to the browser's
print dialog: the result depended on the browser, the margins on the printer
settings, and nothing about it could be tested. This renders the file on the
server, with ReportLab Platypus, from ``ExportIRV1``.

What §10.4 asks for, and where it happens:

* **Unicode** — Noto Sans is bundled (``fonts/``, SIL OFL 1.1) and registered
  with all four faces, because ReportLab's built-in Helvetica covers WinAnsi and
  little else;
* **headings, pagination, table splitting** — Platypus flowables, with table
  header rows repeated when a table crosses a page;
* **links and bookmarks** — clickable links for the schemes the adapter allows,
  and a PDF outline built from the headings;
* **disclosure and sources** — the Art. 50 notice on every page, and the lesson's
  own sources footer, which is part of the document.

**Characters the font cannot draw.** Noto Sans has no arrows, no ≤ ≥, no check
marks and no emoji — all of which the Writer uses. A missing glyph would print as
an empty box, so each character is checked against the font's own character map:
a short list is substituted (→ becomes "->"), everything else is dropped and
reported. A box in a printed lesson looks like a broken file; a missing emoji
looks like nothing.

Synchronous on purpose. A lesson is a few pages and renders in a fraction of a
second; §10.4 reserves ``export_job`` for what is too big for that, which no
lesson is yet. The route runs it off the event loop regardless.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.fonts import addMapping
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from aix.export.export_ir import ExportIRV1
from aix.export.lesson_document import Block, Run

logger = logging.getLogger(__name__)

RENDERER_VERSION = "pdf@1"

FONT = "NotoSans"
_FONT_DIR = Path(__file__).resolve().parent / "fonts"
_FACES = {
    "NotoSans": ("NotoSans-Regular.ttf", 0, 0),
    "NotoSans-Bold": ("NotoSans-Bold.ttf", 1, 0),
    "NotoSans-Italic": ("NotoSans-Italic.ttf", 0, 1),
    "NotoSans-BoldItalic": ("NotoSans-BoldItalic.ttf", 1, 1),
}

#: Glyphs Noto Sans lacks but lessons use, with a plain rendering that keeps the
#: meaning. Everything not here and not in the font is dropped and reported.
SUBSTITUTES = {
    "→": "->",
    "←": "<-",
    "↔": "<->",
    "⇒": "=>",
    "≤": "<=",
    "≥": ">=",
    "≠": "!=",
    "✓": "v",
    "✔": "v",
    "✗": "x",
    "✘": "x",
}

_INK = colors.HexColor("#1f2a37")
_MUTED = colors.HexColor("#6b7280")
_LINK = colors.HexColor("#1f5fa8")
_RULE = colors.HexColor("#d1d5db")
_HEADER_FILL = colors.HexColor("#eef2f6")

_PAGE_W, _PAGE_H = A4
_MARGIN_X, _MARGIN_TOP, _MARGIN_BOTTOM = 20 * mm, 22 * mm, 20 * mm


@dataclass(frozen=True)
class PdfResult:
    content: bytes
    #: Characters the font could not draw and that were removed from the output.
    dropped_characters: tuple[str, ...] = ()
    page_count: int = 0


# ---------------------------------------------------------------------------
# Fonts and characters
# ---------------------------------------------------------------------------


def _register_fonts() -> None:
    """Register the four faces once per process, and map them as one family."""
    registered = set(pdfmetrics.getRegisteredFontNames())
    for name, (filename, bold, italic) in _FACES.items():
        if name not in registered:
            pdfmetrics.registerFont(TTFont(name, str(_FONT_DIR / filename)))
        addMapping(FONT, bold, italic, name)


class _GlyphFilter:
    """Keeps what the font can draw, substitutes what it can approximate."""

    def __init__(self) -> None:
        self._cmap = pdfmetrics.getFont(FONT).face.charToGlyph
        self.dropped: set[str] = set()

    def clean(self, text: str) -> str:
        out = []
        for ch in text:
            if ch in "\n\t" or ord(ch) in self._cmap:
                out.append(ch)
            elif ch in SUBSTITUTES:
                out.append(SUBSTITUTES[ch])
            elif ch.isspace():
                out.append(" ")
            else:
                self.dropped.add(ch)
        return "".join(out)


# ---------------------------------------------------------------------------
# Inline markup
# ---------------------------------------------------------------------------


def _markup(runs: tuple[Run, ...], glyphs: _GlyphFilter) -> str:
    """Runs to ReportLab's paragraph mini-markup, escaping everything first.

    Order matters: clean the characters, *then* escape, *then* wrap in our own
    tags. Escaping first would let a substitute such as "<=" reach the parser
    unescaped; wrapping first would escape our own tags.
    """
    parts = []
    for run in runs:
        text = escape(glyphs.clean(run.text))
        if not text:
            continue
        if run.code:
            text = f'<font color="#4b5563">{text}</font>'
        if run.bold:
            text = f"<b>{text}</b>"
        if run.italic:
            text = f"<i>{text}</i>"
        if run.href:
            href = escape(run.href, {'"': "&quot;"})
            text = f'<a href="{href}" color="#1f5fa8"><u>{text}</u></a>'
        parts.append(text)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------


def _styles() -> dict[str, ParagraphStyle]:
    body = ParagraphStyle(
        "body", fontName=FONT, fontSize=10.5, leading=14.5, textColor=_INK, spaceAfter=4
    )
    return {
        "title": ParagraphStyle(
            "title", parent=body, fontName="NotoSans-Bold", fontSize=20, leading=25, spaceAfter=4
        ),
        "meta": ParagraphStyle("meta", parent=body, fontSize=9.5, textColor=_MUTED, spaceAfter=10),
        "h2": ParagraphStyle(
            "h2",
            parent=body,
            fontName="NotoSans-Bold",
            fontSize=14.5,
            leading=19,
            spaceBefore=12,
            spaceAfter=5,
        ),
        "h3": ParagraphStyle(
            "h3",
            parent=body,
            fontName="NotoSans-Bold",
            fontSize=12,
            leading=16,
            spaceBefore=9,
            spaceAfter=4,
        ),
        "h4": ParagraphStyle(
            "h4", parent=body, fontName="NotoSans-Bold", fontSize=11, spaceBefore=6, spaceAfter=3
        ),
        "body": body,
        "list": ParagraphStyle("list", parent=body, spaceAfter=2),
        "quote": ParagraphStyle(
            "quote", parent=body, leftIndent=12, textColor=_MUTED, fontName="NotoSans-Italic"
        ),
        "code": ParagraphStyle(
            "code", parent=body, fontSize=9.5, backColor=_HEADER_FILL, borderPadding=4
        ),
        "cell": ParagraphStyle("cell", parent=body, fontSize=9.5, leading=12.5, spaceAfter=0),
        "cell_head": ParagraphStyle(
            "cell_head", parent=body, fontName="NotoSans-Bold", fontSize=9.5, leading=12.5
        ),
    }


# ---------------------------------------------------------------------------
# Document template: disclosure on every page, bookmarks from headings
# ---------------------------------------------------------------------------


class _LessonTemplate(BaseDocTemplate):
    def __init__(self, buffer: io.BytesIO, ir: ExportIRV1, **kwargs: Any) -> None:
        super().__init__(buffer, pagesize=A4, **kwargs)
        self._ir = ir
        self._outline_level = -1
        frame = Frame(
            _MARGIN_X,
            _MARGIN_BOTTOM,
            _PAGE_W - 2 * _MARGIN_X,
            _PAGE_H - _MARGIN_TOP - _MARGIN_BOTTOM,
            id="body",
        )
        self.addPageTemplates([PageTemplate(id="lesson", frames=[frame], onPage=self._decorate)])

    def _decorate(self, canvas: Any, doc: Any) -> None:
        """The Art. 50 notice and the page number, on every page."""
        canvas.saveState()
        canvas.setFont(FONT, 8)
        canvas.setFillColor(_MUTED)
        canvas.drawRightString(_PAGE_W - _MARGIN_X, _PAGE_H - 12 * mm, self._ir.disclosure_short)
        canvas.setStrokeColor(_RULE)
        canvas.line(_MARGIN_X, 14 * mm, _PAGE_W - _MARGIN_X, 14 * mm)
        canvas.drawString(_MARGIN_X, 10 * mm, self._ir.disclosure_long)
        canvas.drawRightString(_PAGE_W - _MARGIN_X, 10 * mm, f"pagina {doc.page}")
        canvas.restoreState()

    def afterFlowable(self, flowable: Any) -> None:
        """Turn headings into PDF bookmarks.

        Outline levels must be contiguous, and a lesson can jump from ``##`` to
        ``####``. Clamping each level to at most one below the previous one
        keeps the outline valid instead of letting ReportLab raise mid-render.
        """
        bookmark = getattr(flowable, "_aix_bookmark", None)
        if not bookmark:
            return
        key, title, level = bookmark
        level = min(level, self._outline_level + 1)
        self._outline_level = level
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(title, key, level=level, closed=False)


# ---------------------------------------------------------------------------
# Blocks to flowables
# ---------------------------------------------------------------------------


def _heading(text: str, level: int, key: str, styles: dict, glyphs: _GlyphFilter) -> Paragraph:
    clean = glyphs.clean(text)
    style = styles["h2"] if level <= 2 else styles["h3"] if level == 3 else styles["h4"]
    paragraph = Paragraph(escape(clean), style)
    # The outline starts at the title (level 0), so a ``##`` heading is level 1.
    paragraph._aix_bookmark = (key, clean, max(1, level - 1))
    return paragraph


def _list(block: Block, styles: dict, glyphs: _GlyphFilter) -> list[Any]:
    flowables = []
    counter = 0
    # strict: the adapter appends an item and its depth together, so a length
    # mismatch is a bug in the adapter and should fail loudly, not misindent.
    for runs, depth in zip(block.items, block.depths, strict=True):
        if block.ordered and depth == 0:
            counter += 1
            bullet = f"{counter}."
        else:
            bullet = "•" if depth == 0 else "–"
        indent = 14 + 14 * depth
        style = ParagraphStyle(
            f"list{depth}",
            parent=styles["list"],
            leftIndent=indent,
            bulletIndent=indent - 11,
        )
        flowables.append(Paragraph(_markup(runs, glyphs), style, bulletText=bullet))
    flowables.append(Spacer(1, 4))
    return flowables


def _table(block: Block, styles: dict, glyphs: _GlyphFilter, width: float) -> Optional[Table]:
    if not block.rows:
        return None
    columns = max(len(row) for row in block.rows)
    data = []
    for index, row in enumerate(block.rows):
        style = styles["cell_head"] if index == 0 else styles["cell"]
        cells = [Paragraph(_markup(cell, glyphs), style) for cell in row]
        cells += [Paragraph("", style)] * (columns - len(cells))
        data.append(cells)

    table = Table(data, colWidths=[width / columns] * columns, repeatRows=1, splitByRow=1)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, _RULE),
                ("BACKGROUND", (0, 0), (-1, 0), _HEADER_FILL),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _block(block: Block, styles: dict, glyphs: _GlyphFilter, width: float) -> list[Any]:
    if block.kind == "paragraph":
        return [Paragraph(_markup(block.runs, glyphs), styles["body"])]
    if block.kind == "list":
        return _list(block, styles, glyphs)
    if block.kind == "table":
        table = _table(block, styles, glyphs, width)
        return [table, Spacer(1, 6)] if table else []
    if block.kind == "quote":
        return [Paragraph(_markup(block.runs, glyphs), styles["quote"])]
    if block.kind == "code":
        text = escape(glyphs.clean(block.text)).replace("\n", "<br/>")
        return [Paragraph(text, styles["code"])]
    if block.kind == "heading":  # a heading nested inside a section's blocks
        return [_heading(block.text, block.level, block.block_id, styles, glyphs)]
    return []


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def render_pdf(ir: ExportIRV1) -> PdfResult:
    """Render the snapshot to PDF bytes."""
    _register_fonts()
    glyphs = _GlyphFilter()
    styles = _styles()
    width = _PAGE_W - 2 * _MARGIN_X

    buffer = io.BytesIO()
    doc = _LessonTemplate(
        buffer,
        ir,
        title=glyphs.clean(ir.title),
        author="AixLearning — " + ir.disclosure_short,
        subject=f"lesson={ir.lesson_id or 'n/a'} content={ir.content_hash[:16]}",
        creator=f"aix-export {RENDERER_VERSION}",
    )

    title = Paragraph(escape(glyphs.clean(ir.title)), styles["title"])
    title._aix_bookmark = ("title", glyphs.clean(ir.title), 0)
    story: list[Any] = [title]

    meta = []
    if ir.duration_minutes:
        meta.append(f"Durata: {ir.duration_minutes} minuti")
    meta.append(ir.disclosure_long)
    story.append(Paragraph(escape(" · ".join(meta)), styles["meta"]))

    for block in ir.preamble:
        story += _block(block, styles, glyphs, width)

    for section in ir.sections:
        story.append(
            _heading(
                section.title, section.level, section.heading_id or section.title, styles, glyphs
            )
        )
        for block in section.blocks:
            story += _block(block, styles, glyphs, width)

    doc.build(story)

    if glyphs.dropped:
        logger.info(
            "[export] pdf for lesson=%s dropped %d character(s) the font cannot draw",
            ir.lesson_id,
            len(glyphs.dropped),
        )

    return PdfResult(
        content=buffer.getvalue(),
        dropped_characters=tuple(sorted(glyphs.dropped)),
        page_count=doc.page,
    )
