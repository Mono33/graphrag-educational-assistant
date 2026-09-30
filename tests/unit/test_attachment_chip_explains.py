"""The chip says why pages went unread, and warns only when something is wrong.

A magazine issue reported «13 pag. non lette» and nothing else. The pages were
its full-page adverts and photographs -- normal for that document -- and OCR
had not run because tesseract is not installed on that machine, which the
parser records as ``ocr_unavailable`` and the chip was discarding. The teacher
was left with a warning triangle, a number, and no way to tell an ordinary
magazine from a broken upload.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

pytestmark = pytest.mark.unit

TEMPLATES = Path(__file__).resolve().parents[2] / "src" / "aix" / "webui" / "templates"


class _Lesson:
    id = "11111111-1111-1111-1111-111111111111"
    status = "draft"

    def __init__(self, parse: dict):
        self.uploaded_files_json = [
            {"id": "f1", "filename": "rivista.pdf", "size": 14_130_313, "parse": parse}
        ]


def _render(parse: dict) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)))
    env.globals.setdefault("upload_accept", lambda: ".pdf")
    template = env.get_template("partials/chat_attachments.html")
    return template.render(lesson=_Lesson(parse), material_states={})


MAGAZINE = {
    "pages": 114,
    "unread_pages": 13,
    "ocr_pages": 0,
    "empty": False,
    "warnings": ["ocr_unavailable"],
}


def test_the_chip_says_what_those_pages_are():
    html = _render(MAGAZINE)

    assert "13 pag. senza testo" in html, "«non lette» reads like a failure"
    assert "pubblicit" in html, "photographs and adverts are the usual reason"


def test_it_says_when_ocr_was_never_tried():
    """``ocr_unavailable`` is the difference between "could not" and "did not"."""
    assert "non è disponibile" in _render(MAGAZINE), "OCR was never attempted here"

    scanned = dict(MAGAZINE, warnings=[])
    assert "non è disponibile" not in _render(scanned), "there it ran and failed"
    assert "non è riuscito" in _render(scanned)


def test_a_few_image_pages_in_a_long_document_are_not_a_warning():
    """13 of 114 is a magazine. The triangle is for documents read badly."""
    assert "aix-attach-chip--warn" not in _render(MAGAZINE)


def test_a_document_mostly_unread_still_warns():
    """Written from a fraction of what was attached: the teacher must know."""
    mostly = dict(MAGAZINE, pages=20, unread_pages=15)

    assert "aix-attach-chip--warn" in _render(mostly)
