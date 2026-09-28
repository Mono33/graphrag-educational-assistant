"""OCR by a vision model through OpenRouter, for pages Tesseract reads badly.

Measured on 2026-09-25 and 2026-09-28 on the same four pages (an INVALSI maths
sheet, a phone photo of a textbook page, an old scanned grammar, a poster):
Tesseract kept 2 of the sheet's 8 critical elements — «1/10» became «i», the
option «45» became «045» — while Gemini 2.5 Flash kept 8 of 8 on every run, read
91 % of the photo's words against 85 %, in 2–6 s and under half a cent a page.

2.5 Flash is being retired. On 2026-09-29 four models read the same sixteen
cases (``scripts/eval/check_ocr.py``): 3.5 Flash-Lite got all sixteen, twice,
in 1–7 s a page, and is the cheapest. 3.8 Flash wrote every ticked box as
empty; 3.7 Flash did so on one run of two. See Stato_Lavori.

What the model is told, and why:

* **transcribe, never interpret** — asked to describe drawings it turned a
  pinwheel of triangles into a swastika and every rectangle into a square. A
  figure is written as ``[figura]`` and nothing else;
* **mark what it cannot read** as ``[illeggibile]`` instead of guessing. Those
  markers are the only signal of doubt a vision model gives, so they become the
  page's confidence (``_confidence``), and the same floor that keeps a bad
  Tesseract page out of the index applies here.

The page leaves the server. Every request asks OpenRouter for zero-data-retention
endpoints only; the account enforces the same, and a request that cannot be
honoured fails instead of going elsewhere. A failure is the caller's to handle —
``ocr.recognise`` falls back to Tesseract.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "google/gemini-3.5-flash-lite"

#: Longest side sent to the model. The measurements used pages of 1190×1684 and
#: 1654×2339 pixels; beyond ~2000 the image costs more tokens and reads no better.
MAX_SIDE_PX = 2_000

#: A page of dense text is ~1 500 tokens; this leaves room for two.
MAX_OUTPUT_TOKENS = 6_000

PROMPT = (
    "Trascrivi fedelmente tutto il testo visibile nell'immagine, nell'ordine di lettura.\n"
    "Regole:\n"
    "- non correggere, non completare, non riassumere, non tradurre;\n"
    "- scrivi le frazioni come a/b e le caselle da spuntare come □;\n"
    "- se una parte non si legge scrivi [illeggibile], non indovinare;\n"
    "- per un disegno o una figura scrivi solo [figura], senza descriverla;\n"
    "- lascia una riga vuota fra un paragrafo e l'altro;\n"
    "- rispondi solo con la trascrizione."
)

ILLEGIBLE = "[illeggibile]"

#: How many words one ``[illeggibile]`` is assumed to hide. A marker usually
#: stands for a smudged phrase, not a single letter.
_WORDS_PER_MARKER = 3

#: The ceiling of the estimate. Nothing the model says makes its reading
#: certain, and the sheet and photo it was measured on were not perfect either.
_BEST_CONFIDENCE = 0.95


class IncompleteTranscription(RuntimeError):
    """The model stopped before the end of the page (length or content filter)."""


def backend() -> str:
    """``vision`` or ``tesseract`` (the default: it needs no network and no key)."""
    return (os.getenv("AIX_OCR_BACKEND") or "tesseract").strip().lower()


def model() -> str:
    return (os.getenv("AIX_OCR_MODEL") or DEFAULT_MODEL).strip()


def vision_available() -> bool:
    if backend() != "vision":
        return False
    from aix.core.config import config

    return bool(config.openai.api_key)


@lru_cache(maxsize=1)
def _client() -> Any:
    from aix.core.config import config

    return config.openai.get_client()


def _png(image: Any) -> str:
    image = image.convert("RGB")
    width, height = image.size
    longest = max(width, height)
    if longest > MAX_SIDE_PX:
        scale = MAX_SIDE_PX / longest
        image = image.resize((int(width * scale), int(height * scale)))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n\s*```\s*$", re.S)


def _clean(text: str) -> str:
    fenced = _FENCE.match(text or "")
    return (fenced.group(1) if fenced else (text or "")).strip()


def _confidence(text: str) -> tuple[float, int]:
    """An estimate from the share of the page the model marked unreadable."""
    markers = text.count(ILLEGIBLE)
    words = len(re.findall(r"\w+", text.replace(ILLEGIBLE, " ")))
    if words == 0:
        return 0.0, 0
    readable = words / (words + _WORDS_PER_MARKER * markers)
    return round(_BEST_CONFIDENCE * readable, 3), words


def vision_ocr_image(image: Any, *, timeout_s: float) -> "OcrResult":
    """Transcribe one image. Raises on any failure; the caller decides what next."""
    from aix.webui.materials.parsing.ocr import OcrResult

    response = _client().with_options(timeout=timeout_s).chat.completions.create(
        model=model(),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_png(image)}"}},
                ],
            }
        ],
        temperature=0,
        max_tokens=MAX_OUTPUT_TOKENS,
        extra_body={"provider": {"zdr": True, "data_collection": "deny"}},
    )
    choice = response.choices[0]
    finish = getattr(choice, "finish_reason", None)
    if finish not in (None, "stop"):
        # A transcription that stopped early looks exactly like a short page.
        # On 2026-09-29 two Gemini models stopped halfway down a dense page of
        # the national curriculum with no error at all — likely the provider's
        # filter against reciting well-known text, which a scanned textbook can
        # trip as well. Half a page must not be indexed as the whole page.
        raise IncompleteTranscription(f"vision OCR stopped early: finish_reason={finish}")
    text = _clean(choice.message.content or "")
    confidence, words = _confidence(text)
    return OcrResult(text=text, confidence=confidence, words=words, engine="vision")
