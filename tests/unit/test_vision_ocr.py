"""OCR by a vision model: when it is used, what it is told, and what happens when it fails.

Chosen on measurements (Stato_Lavori, 2026-09-25 and 2026-09-28): on an INVALSI
sheet Tesseract kept 2 of 8 critical elements, Gemini 2.5 Flash 8 of 8. The
model is replaced here by a fake client; nothing leaves the machine.
"""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from aix.webui.materials.parsing import ocr, parse_document, vision_ocr
from aix.webui.materials.parsing.intake import PDF
from aix.webui.materials.parsing.ocr import OcrResult

pytestmark = pytest.mark.unit


class _FakeClient:
    def __init__(self, reply="", *, error=None, finish_reason="stop"):
        self.reply, self.error, self.finish_reason = reply, error, finish_reason
        self.calls: list[dict] = []
        self.timeouts: list[float] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def with_options(self, *, timeout):
        self.timeouts.append(timeout)
        return self

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        message = SimpleNamespace(content=self.reply)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason=self.finish_reason)]
        )


@pytest.fixture
def vision(monkeypatch):
    """Vision backend on, a key present, no Tesseract; returns an installer."""

    def install(reply="", *, error=None, tesseract=False, finish_reason="stop"):
        client = _FakeClient(reply, error=error, finish_reason=finish_reason)
        monkeypatch.setenv("AIX_OCR_BACKEND", "vision")
        monkeypatch.setattr("aix.core.config.config.openai.api_key", "test-key")
        monkeypatch.setattr(vision_ocr, "_client", lambda: client)
        monkeypatch.setattr(ocr, "tesseract_available", lambda: tesseract)
        return client

    return install


def _image(size=(400, 300)):
    from PIL import Image

    return Image.new("RGB", size, "white")


def _scan(pages: int = 1) -> bytes:
    images = [_image((200, 200)) for _ in range(pages)]
    buffer = io.BytesIO()
    images[0].save(buffer, format="PDF", save_all=True, append_images=images[1:])
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# When it is used
# ---------------------------------------------------------------------------


def test_the_vision_model_is_the_default(monkeypatch):
    """Since 2026-10-05: the pilot's setting. Tesseract is the fallback."""
    monkeypatch.delenv("AIX_OCR_BACKEND", raising=False)
    monkeypatch.setattr("aix.core.config.config.openai.api_key", "test-key")

    assert vision_ocr.backend() == "vision"
    assert vision_ocr.vision_available()


def test_tesseract_when_asked_for(monkeypatch):
    """No network and no key needed."""
    monkeypatch.setenv("AIX_OCR_BACKEND", "tesseract")
    monkeypatch.setattr("aix.core.config.config.openai.api_key", "test-key")

    assert vision_ocr.backend() == "tesseract"
    assert not vision_ocr.vision_available()


def test_without_a_key_the_vision_backend_is_not_available(monkeypatch):
    monkeypatch.setenv("AIX_OCR_BACKEND", "vision")
    monkeypatch.setattr("aix.core.config.config.openai.api_key", "")

    assert not vision_ocr.vision_available()


def test_the_vision_model_alone_makes_ocr_available(vision):
    vision("testo")

    assert ocr.ocr_available(), "a VM without Tesseract can still read scans"


def test_the_general_switch_turns_every_engine_off(vision, monkeypatch):
    vision("testo")
    monkeypatch.setenv("AIX_OCR_ENABLED", "false")

    assert not ocr.ocr_available()


# ---------------------------------------------------------------------------
# What it is told
# ---------------------------------------------------------------------------


def test_the_page_goes_to_zero_retention_endpoints_only(vision):
    client = vision("Questa figura è 1/10 di una figura intera.")

    ocr.recognise(_image(), timeout_s=45)

    call = client.calls[0]
    assert call["extra_body"]["provider"] == {"zdr": True, "data_collection": "deny"}
    assert call["temperature"] == 0
    assert call["model"] == vision_ocr.DEFAULT_MODEL
    assert client.timeouts == [45]


def test_the_model_can_be_changed_without_code(vision, monkeypatch):
    client = vision("testo")
    monkeypatch.setenv("AIX_OCR_MODEL", "google/gemini-3.1-flash-lite")

    ocr.recognise(_image(), timeout_s=45)

    assert client.calls[0]["model"] == "google/gemini-3.1-flash-lite"


def test_it_is_told_to_transcribe_and_not_to_describe_figures(vision):
    """Asked to describe, it turned a pinwheel of triangles into a swastika."""
    client = vision("testo")

    ocr.recognise(_image(), timeout_s=45)

    prompt = client.calls[0]["messages"][0]["content"][0]["text"]
    assert "[figura], senza descriverla" in prompt
    assert "[illeggibile], non indovinare" in prompt
    assert "non correggere" in prompt


def test_a_large_scan_is_reduced_before_it_is_sent(vision):
    import base64

    from PIL import Image

    client = vision("testo")

    ocr.recognise(_image((5000, 3000)), timeout_s=45)

    url = client.calls[0]["messages"][0]["content"][1]["image_url"]["url"]
    sent = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert max(sent.size) == vision_ocr.MAX_SIDE_PX


# ---------------------------------------------------------------------------
# What comes back
# ---------------------------------------------------------------------------


def test_a_clean_transcription_is_trusted_and_says_who_read_it(vision):
    vision("```\nQuale scrittura non rappresenta 45 decimi?\n\nA. □ 4,5\n```")

    result = ocr.recognise(_image(), timeout_s=45)

    assert result.engine == "vision"
    assert result.text.startswith("Quale scrittura"), "a wrapping code fence is removed"
    assert result.confidence == pytest.approx(0.95)


def test_a_page_mostly_marked_unreadable_falls_under_the_floor(vision):
    vision("[illeggibile] del [illeggibile] [illeggibile] parte")

    ir = parse_document(_scan(1), PDF)

    assert "ocr_discarded:page=1" in ir.warnings
    assert "parte" not in ir.text


def test_the_index_records_which_engine_read_the_page(vision):
    vision("Disegna a fianco del quadrato un rettangolo con lo stesso perimetro.")

    ir = parse_document(_scan(1), PDF)

    assert ir.ocr_pages == (1,)
    block = ir.blocks[0]
    assert block.source_locator["ocr_engine"] == "vision"
    assert block.source_locator["ocr"] is True


def test_an_empty_page_stays_unread(vision):
    vision("")

    ir = parse_document(_scan(1), PDF)

    assert ir.unread_pages == (1,)


# ---------------------------------------------------------------------------
# When it fails
# ---------------------------------------------------------------------------


def test_a_failed_call_falls_back_to_tesseract(vision, monkeypatch):
    vision(error=RuntimeError("no zero-retention endpoint"), tesseract=True)
    monkeypatch.setattr(
        ocr, "ocr_image", lambda image, timeout_s: OcrResult("letto in locale", 0.9, 3)
    )

    result = ocr.recognise(_image(), timeout_s=45)

    assert result.engine == "tesseract"
    assert result.text == "letto in locale"


@pytest.mark.parametrize("finish", ["length", "content_filter"])
def test_a_transcription_that_stopped_early_is_not_taken_as_the_page(vision, monkeypatch, finish):
    """Two models stopped halfway down a dense page with no error (2026-09-29)."""
    vision("La storia generale a scuola. Occorre, dunque,", tesseract=True, finish_reason=finish)
    monkeypatch.setattr(
        ocr, "ocr_image", lambda image, timeout_s: OcrResult("pagina intera", 0.9, 2)
    )

    result = ocr.recognise(_image(), timeout_s=45)

    assert result.engine == "tesseract", "half a page must not be indexed as the page"
    assert result.text == "pagina intera"


def test_a_failed_call_without_tesseract_loses_only_that_page(vision):
    vision(error=RuntimeError("timeout"))

    ir = parse_document(_scan(2), PDF)

    assert "ocr_failed:page=1" in ir.warnings and "ocr_failed:page=2" in ir.warnings
    assert ir.unread_pages == (1, 2)
