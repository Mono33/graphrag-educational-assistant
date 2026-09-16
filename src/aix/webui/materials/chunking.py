"""Cutting a ``DocumentIRV1`` into retrievable chunks (§8.3).

The chunker is what makes a long document usable at all. Today the whole
extracted text is pasted into the Writer's prompt, which is why two attachments
plus a thread exhausted the model's budget on 2026-09-16. A chunk is instead a
piece small enough to retrieve and large enough to mean something on its own.

The rules in §8.3, and why each one is there:

* **structure-aware, not fixed-size.** A cut every N characters splits a
  sentence between two chunks and neither half retrieves well;
* **a heading travels with the content under it.** "La crisi di Cuba" alone is
  not an answer to anything; the heading opens the chunk it introduces;
* **slides never mix.** Two slides in one chunk produce a citation that points
  at two places at once, so the teacher cannot check it;
* **a table is one unit.** Half a table is worse than no table: the rows that
  survive look complete and are not;
* **overlap between consecutive chunks of the same section**, so a passage that
  straddles a boundary is still whole in one of them;
* **the locator is kept on every chunk** — page, slide, block ids — because a
  citation that cannot be opened is a claim, not a source (§8.6).

Sizes come from §8.3: child chunks of 250–450 tokens, 10–15% overlap. Tokens are
estimated, not counted: the exact count depends on the tokenizer and would tie
the chunker to one model.

``CHUNKER_VERSION`` is stored on every row. Changing these rules means a new
version and a re-chunk, never a silent mix of two chunkings in one index.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from aix.webui.materials.parsing.ir import DocumentIRV1, IRBlock

CHUNKER_VERSION = "chunk@1"

#: §8.3 child size. The target is what a chunk aims for, the max what it may
#: never exceed, and the min the size below which a chunk is merged forward
#: rather than left as a fragment nobody can retrieve.
TARGET_TOKENS = 380
MAX_TOKENS = 450
MIN_TOKENS = 120

#: §8.3: 10–15%. Applied only between chunks of the same section, never across a
#: heading or a slide — carrying a slide's last line onto the next slide would
#: put text under a locator it does not belong to.
OVERLAP_RATIO = 0.12

#: Italian prose through this parser averages ~3.6 characters per token.
_CHARS_PER_TOKEN = 3.6

_HEADING_KINDS = ("heading", "slide_title")


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / _CHARS_PER_TOKEN))


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    text: str
    text_hash: str
    #: What a citation resolves against: page or slide, the blocks it came from,
    #: and the heading it sits under (§8.6).
    locator: dict[str, Any]
    token_estimate: int
    chunker_version: str = CHUNKER_VERSION


@dataclass
class _Draft:
    """A chunk being accumulated. Mutable on purpose; frozen when emitted."""

    heading: Optional[str] = None
    parts: list[str] = field(default_factory=list)
    block_ids: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    carried_tokens: int = 0

    @property
    def text(self) -> str:
        return "\n\n".join(p for p in self.parts if p.strip()).strip()

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text) if self.parts else 0

    def add(self, block: IRBlock, text: str) -> None:
        self.parts.append(text)
        self.block_ids.append(block.block_id)
        if block.page is not None and block.page not in self.pages:
            self.pages.append(block.page)


def _block_text(block: IRBlock) -> str:
    if block.kind == "table" and block.rows:
        return "\n".join(" | ".join(cell for cell in row) for row in block.rows)
    return block.text


def _tail_for_overlap(text: str) -> str:
    """The last ~12% of a chunk, cut at a sentence start where possible."""
    wanted = int(len(text) * OVERLAP_RATIO)
    if wanted < 80:
        return ""
    tail = text[-wanted:]
    match = re.search(r"(?<=[.!?…])\s+", tail)
    return tail[match.end() :].strip() if match else tail.strip()


def _split_long(text: str) -> list[str]:
    """Split one oversized block into pieces of at most ``MAX_TOKENS``.

    Sentence-first: a paragraph of 900 tokens is real (a dense page of a
    handout), and cutting it mid-sentence would leave both halves unusable.
    """
    limit = int(MAX_TOKENS * _CHARS_PER_TOKEN)
    sentences = re.split(r"(?<=[.!?…])\s+", text)

    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        while len(sentence) > limit:  # a "sentence" with no punctuation at all
            pieces.append(sentence[:limit])
            sentence = sentence[limit:]
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= limit:
            current = f"{current} {sentence}"
        else:
            pieces.append(current)
            current = sentence
    if current.strip():
        pieces.append(current)
    return [p.strip() for p in pieces if p.strip()]


class _Chunker:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.draft = _Draft()

    def emit(self) -> None:
        text = self.draft.text
        if not text:
            self.draft = _Draft(heading=self.draft.heading)
            return

        locator: dict[str, Any] = {"block_ids": list(self.draft.block_ids)}
        if self.draft.heading:
            locator["heading"] = self.draft.heading
        if self.draft.pages:
            locator["page"] = self.draft.pages[0]
            if len(self.draft.pages) > 1:
                locator["pages"] = list(self.draft.pages)

        self.chunks.append(
            Chunk(
                ordinal=len(self.chunks),
                text=text,
                text_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                locator=locator,
                token_estimate=estimate_tokens(text),
            )
        )
        self.draft = _Draft(heading=self.draft.heading)

    def start_section(self, heading: Optional[str]) -> None:
        """A heading or a new slide closes what came before it."""
        self.emit()
        self.draft = _Draft(heading=heading)

    def continue_with_overlap(self) -> None:
        """Close a full chunk and open the next carrying its tail (§8.3)."""
        tail = _tail_for_overlap(self.draft.text)
        heading = self.draft.heading
        self.emit()
        if tail:
            self.draft.parts.append(tail)
            self.draft.carried_tokens = estimate_tokens(tail)
        self.draft.heading = heading


def chunk_document(ir: DocumentIRV1) -> list[Chunk]:
    """Cut an IR into chunks, in reading order."""
    chunker = _Chunker()
    previous_page: Optional[int] = None

    for block in ir.blocks:
        text = _block_text(block)
        if not text.strip():
            continue

        # Never mix two slides, and never mix two pages of a presentation: the
        # locator would name one of them and the text would come from both.
        crosses_slide = (
            ir.media_type.endswith("presentationml.presentation")
            and block.page is not None
            and previous_page is not None
            and block.page != previous_page
        )
        if crosses_slide:
            chunker.start_section(chunker.draft.heading)
        previous_page = block.page if block.page is not None else previous_page

        if block.kind in _HEADING_KINDS:
            chunker.start_section(text)
            chunker.draft.add(block, text)
            continue

        if block.kind == "table":
            # Its own unit, with the heading that introduces it.
            heading = chunker.draft.heading
            chunker.emit()
            chunker.draft.heading = heading
            chunker.draft.add(block, text)
            chunker.emit()
            continue

        for piece in _split_long(text) if estimate_tokens(text) > MAX_TOKENS else [text]:
            if chunker.draft.tokens + estimate_tokens(piece) > MAX_TOKENS:
                if chunker.draft.tokens >= MIN_TOKENS:
                    chunker.continue_with_overlap()
                else:
                    chunker.emit()
            chunker.draft.add(block, piece)

            if chunker.draft.tokens >= TARGET_TOKENS:
                chunker.continue_with_overlap()

    chunker.emit()
    return chunker.chunks
