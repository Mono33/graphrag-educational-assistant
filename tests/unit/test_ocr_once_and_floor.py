"""OCR runs once, keeps out what it cannot trust, and the chip says it in proportion.

A 114-page magazine in the production image, 2026-09-25:

* **twice** -- the upload request ran OCR for a fallback excerpt nobody sends
  with the private RAG on, and the worker ran it again for the index; the
  teacher waited through the first, and the VM's two CPUs ran both at once;
* **noise indexed** -- a poster read at 53% as «$$ eh = <3he \\ AS a n»
  became a passage a lesson could cite;
* **one page, one triangle** -- that single poster put a warning triangle on
  the whole document.

Tesseract is not needed here: ``recognise`` is replaced, as in the parser tests.
"""

from __future__ import annotations

import io

import pytest
from jinja2 import Environment, FileSystemLoader

from aix.webui.materials.parsing import ParseOptions, parse_document, parsers
from aix.webui.materials.parsing.intake import PDF
from aix.webui.materials.parsing.ocr import OcrResult

pytestmark = pytest.mark.unit


def _scan(pages: int) -> bytes:
    """A PDF of ``pages`` pages with no text layer, so every page goes to OCR."""
    from PIL import Image

    images = [Image.new("RGB", (200, 200), "white") for _ in range(pages)]
    buffer = io.BytesIO()
    images[0].save(buffer, format="PDF", save_all=True, append_images=images[1:])
    return buffer.getvalue()


@pytest.fixture
def ocr_reads(monkeypatch):
    """Page ``i`` (0-based) is read with confidence ``confidences[i]``."""

    def install(confidences):
        monkeypatch.setattr(parsers, "ocr_available", lambda: True)
        monkeypatch.setattr(parsers, "render_pdf_page", lambda content, index: index)
        monkeypatch.setattr(
            parsers,
            "recognise",
            lambda image, timeout_s: OcrResult(f"testo della pagina {image + 1}", confidences[image], 4),
        )

    return install


# ---------------------------------------------------------------------------
# The floor
# ---------------------------------------------------------------------------


def test_a_page_the_ocr_does_not_trust_is_not_indexed(ocr_reads):
    ocr_reads([0.93, 0.53, 0.88])

    ir = parse_document(_scan(3), PDF)

    assert ir.ocr_pages == (1, 3)
    assert "testo della pagina 2" not in ir.text, "noise must not become a passage"
    assert "ocr_discarded:page=2" in ir.warnings
    assert 2 not in ir.unread_pages, "it had text; it was not trusted -- a different thing"


def test_the_floor_sits_in_the_measured_gap(ocr_reads):
    """Noise was at 46-67%, real text at 80-95%."""
    ocr_reads([0.67, 0.80])

    ir = parse_document(_scan(2), PDF)

    assert ir.ocr_pages == (2,)
    assert "ocr_discarded:page=1" in ir.warnings


def test_the_floor_can_be_moved(ocr_reads, monkeypatch):
    monkeypatch.setenv("AIX_OCR_MIN_CONFIDENCE", "0.5")
    ocr_reads([0.53])

    assert parse_document(_scan(1), PDF).ocr_pages == (1,)


def test_an_image_under_the_floor_is_not_indexed(monkeypatch):
    from PIL import Image

    monkeypatch.setattr(parsers, "ocr_available", lambda: True)
    monkeypatch.setattr(parsers, "recognise", lambda image, timeout_s: OcrResult("$$ eh", 0.4, 2))
    buffer = io.BytesIO()
    Image.new("RGB", (50, 50), "white").save(buffer, format="PNG")

    ir = parse_document(buffer.getvalue(), "image/png")

    assert ir.text.strip() == ""
    assert "ocr_discarded:page=1" in ir.warnings


# ---------------------------------------------------------------------------
# Once
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rag_on, expected", [(True, False), (False, True)])
def test_the_upload_request_runs_ocr_only_when_nothing_else_will(monkeypatch, rag_on, expected):
    """With the private RAG on, the worker reads the file for the index."""
    from aix.webui.lessons import uploads

    seen: dict = {}

    def _parse(content, media_type, options=None):
        seen["ocr"] = (options or ParseOptions()).ocr_enabled
        raise RuntimeError("stop here")

    monkeypatch.setenv("AIX_UPLOAD_PARSER_ENABLED", "true")
    monkeypatch.setattr("aix.webui.materials.embedding.private_rag_enabled", lambda: rag_on)
    monkeypatch.setattr(uploads, "parse_document", _parse)

    with pytest.raises(ValueError):
        uploads.read_upload(_scan(1), "scansione.pdf")

    assert seen["ocr"] is expected


# ---------------------------------------------------------------------------
# The chip
# ---------------------------------------------------------------------------

TEMPLATES = "src/aix/webui/templates"


class _Lesson:
    id = "11111111-1111-1111-1111-111111111111"
    status = "draft"

    def __init__(self, entry):
        self.uploaded_files_json = [entry]


class _State:
    def __init__(self, status, quality=None, passages=282):
        self.status, self.quality, self.passages = status, quality, passages
        self.usable = status in ("ready", "ready_with_warnings")


def _chip(entry, states):
    env = Environment(loader=FileSystemLoader(TEMPLATES))
    env.globals.setdefault("upload_accept", lambda: ".pdf")
    return env.get_template("partials/chat_attachments.html").render(
        lesson=_Lesson(entry), material_states=states
    )


MAGAZINE = {
    "pages": 114, "ocr_pages": 9, "unread_pages": 4, "empty": False,
    "ocr_mean_confidence": 0.9,
    "warnings": ["ocr_discarded:page=15", "ocr_discarded:page=27"],
}
ENTRY = {
    "id": "f1", "filename": "Internazionale1682.pdf", "size": 14_130_313,
    "material_version_id": "v1",
    # What the request saw without OCR: thirteen image pages, no text.
    "parse": {"pages": 114, "ocr_pages": 0, "unread_pages": 13, "empty": False,
              "warnings": []},
}


def test_the_chip_reads_the_workers_summary_not_the_requests():
    html = _chip(ENTRY, {"v1": _State("ready_with_warnings", MAGAZINE)})

    assert "OCR 9 pag." in html
    assert "2 pag. illeggibili" in html
    assert "13 pag. senza testo" not in html, "that was the request's OCR-less view"


def test_two_bad_pages_in_114_are_not_a_warning():
    html = _chip(ENTRY, {"v1": _State("ready_with_warnings", MAGAZINE)})

    assert "aix-attach-chip--warn" not in html


def test_while_the_worker_is_still_reading_the_chip_makes_no_claim_about_pages():
    html = _chip(ENTRY, {"v1": _State("processing", None)})

    assert "in preparazione" in html
    assert "senza testo" not in html and "OCR" not in html


def test_outside_the_pipeline_the_requests_summary_is_still_used():
    legacy = {"id": "f2", "filename": "vecchio.pdf", "size": 1000,
              "parse": {"pages": 10, "ocr_pages": 2, "unread_pages": 0, "empty": False,
                        "warnings": []}}

    assert "OCR 2 pag." in _chip(legacy, {})
