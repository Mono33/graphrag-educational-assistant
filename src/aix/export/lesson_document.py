"""The lesson as a document instead of a string (§10.2).

``LessonDocumentV1`` is what every export reads. The Writer does not produce it
yet — it produces Markdown — so ``adapt_lesson_markdown`` builds one from the
Markdown, following the four rules §10.2 sets for the legacy adapter:

* **it recognises** headings, paragraphs, lists, tables, quotes and links;
* **it keeps the order** the Writer wrote;
* **it invents nothing.** A lesson without an assessment section gets no empty
  "Valutazione" heading. What is missing is reported as a warning, never filled;
* **its block ids are stable.** The same content yields the same ids on every
  export, so a later export can be compared block by block with an earlier one.

It parses with ``markdown-it-py`` rather than regexes, because lesson Markdown
nests — lists inside sections, bold inside table cells, links inside list items
— and a regex adapter gets nesting wrong in exactly the places a teacher would
notice.

Images are the one thing it deliberately does not carry across as images. A
remote image in a lesson has unknown licensing and would need a network fetch
inside the renderer; §10.10 says an unknown licence is link-only, so it becomes
a link and a warning.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal, Optional

from markdown_it import MarkdownIt
from markdown_it.token import Token

from aix.agent.validation.rules import section_minutes
from aix.core.ai_marking import is_marked, strip_marking

SCHEMA_VERSION = 1

BlockKind = Literal["heading", "paragraph", "list", "table", "quote", "code"]

#: Link targets allowed into an export. The same policy the WebUI sanitiser
#: applies (``aix.webui.sanitize.ALLOWED_SCHEMES``): a ``javascript:`` or
#: ``data:`` URL is dropped to its text, never made clickable in a PDF.
ALLOWED_LINK_SCHEMES = ("http://", "https://", "mailto:")

_DURATION = re.compile(r"(?:durata|duration)\W{0,6}(\d{1,3})\s*(?:min|minuti|minutes)", re.I)
_MINUTES = re.compile(r"(\d{1,3})\s*(?:min\b|min\.|minut|minute)", re.I)


@dataclass(frozen=True)
class Run:
    """A span of inline text with the formatting it carries."""

    text: str
    bold: bool = False
    italic: bool = False
    code: bool = False
    href: Optional[str] = None


@dataclass(frozen=True)
class Block:
    """One semantic unit of the lesson, in the order the Writer produced it."""

    block_id: str
    kind: BlockKind
    #: heading / paragraph / quote / code
    runs: tuple[Run, ...] = ()
    #: heading only
    level: int = 0
    #: heading only — the minutes the heading declares, if any
    minutes: Optional[int] = None
    #: list only — each item is a run sequence, with its nesting depth
    items: tuple[tuple[Run, ...], ...] = ()
    depths: tuple[int, ...] = ()
    ordered: bool = False
    #: table only — the first row is the header
    rows: tuple[tuple[tuple[Run, ...], ...], ...] = ()

    @property
    def text(self) -> str:
        return "".join(run.text for run in self.runs)


@dataclass(frozen=True)
class LessonDocumentV1:
    """Everything an export needs to know about the lesson, and nothing about rendering."""

    content_hash: str
    title: str
    blocks: tuple[Block, ...]
    lesson_id: Optional[str] = None
    language: str = "it"
    duration_minutes: Optional[int] = None
    #: (section title, minutes), without double-counting nested timings.
    timed_outline: tuple[tuple[str, int], ...] = ()
    ai_generated: bool = True
    warnings: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION
    extra: dict = field(default_factory=dict, compare=False)


# ---------------------------------------------------------------------------
# Inline
# ---------------------------------------------------------------------------


def _safe_href(href: Optional[str]) -> Optional[str]:
    if not href:
        return None
    return href if href.strip().lower().startswith(ALLOWED_LINK_SCHEMES) else None


def _runs(inline: Optional[Token], warnings: list[str]) -> tuple[Run, ...]:
    """Flatten an inline token into runs, carrying bold/italic/link state."""
    if inline is None or not inline.children:
        return ()

    runs: list[Run] = []
    bold = italic = 0
    href: Optional[str] = None

    for child in inline.children:
        kind = child.type
        if kind == "strong_open":
            bold += 1
        elif kind == "strong_close":
            bold = max(0, bold - 1)
        elif kind == "em_open":
            italic += 1
        elif kind == "em_close":
            italic = max(0, italic - 1)
        elif kind == "link_open":
            href = _safe_href(child.attrGet("href"))
        elif kind == "link_close":
            href = None
        elif kind in ("softbreak", "hardbreak"):
            runs.append(Run(" "))
        elif kind == "code_inline":
            runs.append(Run(child.content, code=True, href=href))
        elif kind == "image":
            src = _safe_href(child.attrGet("src"))
            alt = child.content or "immagine"
            runs.append(Run(f"[immagine: {alt}]", italic=True, href=src))
            warnings.append(
                f"Immagine non incorporata («{alt}»): licenza non verificata, resa come link."
            )
        elif kind == "html_inline":
            # Inline HTML is dropped, not rendered: a PDF is not a browser, and
            # the sanitiser's allowlist does not exist here.
            continue
        elif child.content:
            runs.append(Run(child.content, bold=bold > 0, italic=italic > 0, href=href))

    return tuple(runs)


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


class _IdMaker:
    """Stable ids: same content, same place in the outline → same id.

    Hashes the block's kind, the heading it sits under and its text, plus an
    occurrence counter for genuine duplicates (two identical bullets under one
    heading). Position in the file is deliberately not part of it: inserting a
    paragraph above a table must not renumber the table.
    """

    def __init__(self) -> None:
        self._seen: dict[str, int] = {}
        self.section = ""

    def make(self, kind: str, text: str) -> str:
        base = f"{kind}\x1f{self.section}\x1f{text.strip()}"
        count = self._seen.get(base, 0)
        self._seen[base] = count + 1
        return hashlib.sha256(f"{base}\x1f{count}".encode()).hexdigest()[:12]


def _find_close(tokens: list[Token], start: int, open_type: str, close_type: str) -> int:
    depth = 0
    for index in range(start, len(tokens)):
        if tokens[index].type == open_type:
            depth += 1
        elif tokens[index].type == close_type:
            depth -= 1
            if depth == 0:
                return index
    return len(tokens) - 1


def _blocks(tokens: list[Token], warnings: list[str]) -> Iterator[Block]:
    ids = _IdMaker()
    index = 0

    while index < len(tokens):
        token = tokens[index]
        kind = token.type

        if kind == "heading_open":
            runs = _runs(tokens[index + 1], warnings)
            text = "".join(r.text for r in runs)
            ids.section = text
            minutes = _MINUTES.search(text)
            yield Block(
                block_id=ids.make("heading", text),
                kind="heading",
                runs=runs,
                level=int(token.tag[1]),
                minutes=int(minutes.group(1)) if minutes else None,
            )
            index += 3
            continue

        if kind == "paragraph_open":
            runs = _runs(tokens[index + 1], warnings)
            if runs:
                yield Block(ids.make("paragraph", "".join(r.text for r in runs)), "paragraph", runs)
            index += 3
            continue

        if kind in ("bullet_list_open", "ordered_list_open"):
            close = _find_close(tokens, index, kind, kind.replace("_open", "_close"))
            items, depths = [], []
            depth = -1
            for inner in tokens[index : close + 1]:
                if inner.type in ("bullet_list_open", "ordered_list_open"):
                    depth += 1
                elif inner.type in ("bullet_list_close", "ordered_list_close"):
                    depth -= 1
                elif inner.type == "inline":
                    items.append(_runs(inner, warnings))
                    depths.append(max(depth, 0))
            text = " | ".join("".join(r.text for r in item) for item in items)
            yield Block(
                block_id=ids.make("list", text),
                kind="list",
                items=tuple(items),
                depths=tuple(depths),
                ordered=kind == "ordered_list_open",
            )
            index = close + 1
            continue

        if kind == "table_open":
            close = _find_close(tokens, index, "table_open", "table_close")
            rows: list[tuple[tuple[Run, ...], ...]] = []
            current: list[tuple[Run, ...]] = []
            for inner in tokens[index : close + 1]:
                if inner.type == "tr_open":
                    current = []
                elif inner.type == "inline":
                    current.append(_runs(inner, warnings))
                elif inner.type == "tr_close":
                    rows.append(tuple(current))
            text = " | ".join("".join(r.text for r in cell) for row in rows for cell in row)
            yield Block(ids.make("table", text), "table", rows=tuple(rows))
            index = close + 1
            continue

        if kind == "blockquote_open":
            close = _find_close(tokens, index, "blockquote_open", "blockquote_close")
            runs: tuple[Run, ...] = ()
            for inner in tokens[index : close + 1]:
                if inner.type == "inline":
                    runs = runs + (
                        _runs(inner, warnings) if not runs else (Run(" "),) + _runs(inner, warnings)
                    )
            if runs:
                yield Block(ids.make("quote", "".join(r.text for r in runs)), "quote", runs)
            index = close + 1
            continue

        if kind in ("fence", "code_block"):
            content = token.content.rstrip("\n")
            if content:
                yield Block(ids.make("code", content), "code", (Run(content, code=True),))
            index += 1
            continue

        if kind == "html_block":
            # The Art. 50 provenance comment is expected and handled by the
            # caller; anything else is raw HTML a PDF cannot and should not run.
            if "ai-generated" not in token.content:
                warnings.append("Blocco HTML non riprodotto nell'export.")
            index += 1
            continue

        # hr and anything structural with no content of its own.
        index += 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parser() -> MarkdownIt:
    return MarkdownIt("commonmark").enable("table")


def adapt_lesson_markdown(
    markdown: str,
    *,
    lesson_id: Optional[str] = None,
    fallback_title: str = "Lezione",
    language: str = "it",
) -> LessonDocumentV1:
    """Read a lesson's Markdown into a ``LessonDocumentV1``. Never raises on content.

    ``content_hash`` is taken over the Markdown *as stored*, marker included, so
    it identifies the exact revision an export was made from.
    """
    source = markdown or ""
    content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    body = strip_marking(source)
    warnings: list[str] = []

    blocks = tuple(_blocks(_parser().parse(body), warnings))

    title_block = next((b for b in blocks if b.kind == "heading" and b.level == 1), None)
    if title_block is None:
        warnings.append("Nessun titolo di primo livello: usato il titolo della lezione.")
    title = title_block.text.strip() if title_block else fallback_title

    duration = _DURATION.search(body)
    if duration is None:
        warnings.append("Durata non indicata nel testo della lezione.")

    return LessonDocumentV1(
        content_hash=content_hash,
        title=title,
        blocks=blocks,
        lesson_id=lesson_id,
        language=language,
        duration_minutes=int(duration.group(1)) if duration else None,
        timed_outline=tuple(section_minutes(body)),
        ai_generated=is_marked(source) or True,
        warnings=tuple(dict.fromkeys(warnings)),
    )
