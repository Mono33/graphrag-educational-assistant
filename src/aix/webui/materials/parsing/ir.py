"""``DocumentIRV1`` — what a parser makes of a document (§7.7).

The IR is the only thing downstream of parsing ever reads: the private retriever
chunks it, the preview shows it, a citation points into it. So it records not
only the text but *where the text came from* and *how sure the parser is*:

* every block has a ``source_locator`` — page for a PDF, slide and shape for a
  PPTX, paragraph index for a DOCX — which is what lets a citation open the right
  page (§8.6);
* every block has a ``confidence``: 1.0 for text taken from a text layer, the
  mean word confidence for text OCR produced. A citation to a low-confidence
  block should say so, not present a guess as a quotation;
* ``ocr_pages`` lists the pages that were read by OCR, so a teacher can be told
  which parts of their document are a machine's reading of an image.

Plain dataclasses, serialised with ``to_json`` into the ``document_ir.ir_json``
column. §7.7: no base64 assets inside the IR or the database — assets are blobs.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

IR_SCHEMA_VERSION = 1

#: Below this mean confidence an OCR page is flagged ``ocr_low_confidence``.
LOW_CONFIDENCE = 0.60

BlockKind = Literal["heading", "paragraph", "list_item", "table", "slide_title"]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class IRBlock:
    block_id: str
    kind: BlockKind
    text: str
    reading_order: int
    source_locator: dict[str, Any]
    confidence: float = 1.0
    #: 1-based page for PDFs and images, slide number for PPTX, None for DOCX.
    page: Optional[int] = None
    #: heading level, when ``kind == "heading"``
    level: Optional[int] = None
    #: table rows, when ``kind == "table"``
    rows: tuple[tuple[str, ...], ...] = ()
    content_hash: str = ""


@dataclass(frozen=True)
class DocumentIRV1:
    parser_name: str
    parser_version: str
    parser_profile_version: str
    source_sha256: str
    media_type: str
    blocks: tuple[IRBlock, ...]
    page_count: int = 0
    #: Pages whose text came from OCR rather than a text layer.
    ocr_pages: tuple[int, ...] = ()
    #: Pages that had no usable text and could not be read either.
    unread_pages: tuple[int, ...] = ()
    #: Codes, not sentences: ``ocr_low_confidence:page=3``, ``ocr_unavailable``.
    warnings: tuple[str, ...] = ()
    #: Things present in the source that the parser does not carry across.
    unsupported: tuple[str, ...] = ()
    language: Optional[str] = None
    schema_version: int = IR_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(block.text for block in self.blocks if block.text)

    def quality(self) -> dict[str, Any]:
        """What ``material_version.parse_quality`` records about this parse.

        ``empty`` alone was the wrong signal: a 50-page scan with text on its
        cover page is not empty and is 98% unread. The page counts say that.
        """
        text_pages = {b.page for b in self.blocks if b.page and b.confidence >= 1.0}
        ocr_confidences = [b.confidence for b in self.blocks if b.confidence < 1.0]
        return {
            "parser": self.parser_profile_version,
            "chars": len(self.text),
            "blocks": len(self.blocks),
            "pages": self.page_count,
            "pages_with_text_layer": len(text_pages),
            "ocr_pages": len(self.ocr_pages),
            "unread_pages": len(self.unread_pages),
            "ocr_mean_confidence": (
                round(sum(ocr_confidences) / len(ocr_confidences), 3) if ocr_confidences else None
            ),
            "empty": not self.text.strip(),
            "warnings": list(self.warnings),
        }

    def to_json(self) -> dict[str, Any]:
        return asdict(self)
