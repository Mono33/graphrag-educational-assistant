"""OCR for pages without a usable text layer (§7.8), sized for a 4 GB VM.

``recognise`` is the entry point. With ``AIX_OCR_BACKEND=vision`` a page goes to
a vision model (``vision_ocr``) and Tesseract is the fallback; otherwise
Tesseract alone, as below.

Tesseract, called through ``pytesseract``, one page at a time:

* **only where needed** — a page whose text layer already yields text is never
  rasterised, so a digital PDF costs no OCR at all;
* **bounded** — every page has a timeout, images are downsampled before
  recognition, and nothing runs in parallel: the worker is single-threaded by
  design, and an OCR burst must slow the queue, not the chat;
* **honest about itself** — the result carries Tesseract's mean word confidence,
  which becomes the block confidence and, under a threshold, a warning. §7.8 also
  forbids correcting names or formulas; nothing here post-processes the text.

Tesseract is a system binary. It is installed by apt in the image; on a machine
without it ``tesseract_available()`` is False, the parser records
``ocr_unavailable`` for the pages it could not read, and nothing crashes.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

#: Rendering resolution for PDF pages. 200 dpi is the usual floor for reliable
#: Tesseract results on body text, and a third of the pixels of 300 dpi.
PDF_RENDER_DPI = 200

#: Longest side, in pixels, an image is reduced to before recognition. Larger
#: buys little accuracy and a lot of memory — Tesseract's rare 2 GB peaks come
#: from exactly these oversized scans.
MAX_SIDE_PX = 3_500

DEFAULT_PAGE_TIMEOUT_S = 45


@dataclass(frozen=True)
class OcrResult:
    text: str
    #: Mean word confidence, 0–1. 0 when nothing was recognised.
    confidence: float
    words: int
    #: ``tesseract`` or ``vision`` — which one read the page.
    engine: str = "tesseract"


def _ocr_switched_off() -> bool:
    return os.getenv("AIX_OCR_ENABLED", "true").strip().lower() in ("0", "false", "no", "off")


@lru_cache(maxsize=1)
def tesseract_available() -> bool:
    if _ocr_switched_off():
        return False
    return shutil.which(os.getenv("TESSERACT_CMD") or "tesseract") is not None


def ocr_available() -> bool:
    """Whether any engine can read a page: the vision model, Tesseract, or both."""
    from aix.webui.materials.parsing.vision_ocr import vision_available

    if _ocr_switched_off():
        return False
    return vision_available() or tesseract_available()


def recognise(image: Any, *, timeout_s: int = DEFAULT_PAGE_TIMEOUT_S) -> OcrResult:
    """Read one page with the configured engine, falling back to Tesseract.

    ``AIX_OCR_BACKEND=vision`` sends the page to the vision model; if that call
    fails — network, provider, a refused zero-retention route — the page is read
    locally instead, so an outage costs quality, not the document.
    """
    from aix.webui.materials.parsing.vision_ocr import vision_available, vision_ocr_image

    if vision_available():
        try:
            return vision_ocr_image(image, timeout_s=timeout_s)
        except Exception:  # noqa: BLE001 — any failure falls back
            if not tesseract_available():
                raise
            logger.warning("[parsing] vision OCR failed; reading the page with Tesseract", exc_info=True)
    return ocr_image(image, timeout_s=timeout_s)


def languages() -> str:
    """Italian first: the corpus is Italian school material. English as fallback."""
    return os.getenv("AIX_OCR_LANGUAGES") or "ita+eng"


def _prepare(image: Any) -> Any:
    """Greyscale and downsample. Colour carries nothing OCR can use."""
    image = image.convert("L")
    width, height = image.size
    longest = max(width, height)
    if longest > MAX_SIDE_PX:
        scale = MAX_SIDE_PX / longest
        image = image.resize((int(width * scale), int(height * scale)))
    return image


def ocr_image(image: Any, *, timeout_s: int = DEFAULT_PAGE_TIMEOUT_S) -> OcrResult:
    """Recognise one image. Raises ``RuntimeError`` on timeout."""
    import pytesseract

    if os.getenv("TESSERACT_CMD"):
        pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]

    data = pytesseract.image_to_data(
        _prepare(image),
        lang=languages(),
        output_type=pytesseract.Output.DICT,
        timeout=timeout_s,
    )

    words, confidences, lines = [], [], {}
    for index, word in enumerate(data.get("text", [])):
        word = (word or "").strip()
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            confidence = -1.0
        if not word or confidence < 0:
            continue
        words.append(word)
        confidences.append(confidence)
        key = (data["block_num"][index], data["par_num"][index], data["line_num"][index])
        lines.setdefault(key, []).append(word)

    text = "\n".join(" ".join(line) for _key, line in sorted(lines.items()))
    mean = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0
    return OcrResult(text=text, confidence=round(mean, 3), words=len(words))


def render_pdf_page(content: bytes, page_index: int) -> Any:
    """Rasterise one PDF page with pypdfium2 — no Ghostscript, no poppler."""
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(content)
    try:
        page = document[page_index]
        try:
            bitmap = page.render(scale=PDF_RENDER_DPI / 72)
            return bitmap.to_pil()
        finally:
            page.close()
    finally:
        document.close()
