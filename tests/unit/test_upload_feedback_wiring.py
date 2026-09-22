"""The page has somewhere to say what an upload is doing, and why it failed.

On 2026-09-22 a teacher attached the same document three times and the page did
nothing at all, three times. Two defects, neither visible from the server:

* **htmx swaps 2xx only.** The route answers a file it cannot take with 422 and
  an Italian sentence, and htmx was discarding the whole response. The reason
  existed solely in the log;
* the file input targets ``#chat-attachments`` with ``outerHTML``, so an error
  allowed to swap there would replace the tray with a bare paragraph --
  destroying the swap target, after which nothing could render again.

What is pinned here is the structure the JavaScript in ``_base.html`` depends
on. The listeners themselves are not run by these tests; what would silently
break them is someone moving or renaming these elements, and that is caught.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

TEMPLATES = Path(__file__).resolve().parents[2] / "src" / "aix" / "webui" / "templates"
CHAT_INPUT = (TEMPLATES / "partials" / "chat_input.html").read_text(encoding="utf-8")
BASE = (TEMPLATES / "_base.html").read_text(encoding="utf-8")
TRAY = (TEMPLATES / "partials" / "chat_attachments.html").read_text(encoding="utf-8")


def test_both_shells_have_somewhere_to_report_an_upload():
    """Draft and non-draft render different inputs; both need the slot."""
    assert CHAT_INPUT.count('id="chat-upload-error"') == 2
    assert CHAT_INPUT.count('id="chat-upload-busy"') == 2


def test_the_error_slot_is_not_inside_the_tray():
    """The tray is replaced wholesale by every upload and delete response.

    A message rendered inside it would be swept away by the next swap, and an
    error swapped over it would take ``#chat-attachments`` with it.
    """
    assert "chat-upload-error" not in TRAY

    for shell in CHAT_INPUT.split('id="chat-upload-error"')[1:]:
        include = shell.find('{% include "partials/chat_attachments.html" %}')
        assert include != -1, "the slot is rendered before the tray, not inside it"


def test_the_error_slot_comes_before_the_tray_in_every_shell():
    slots = [m.start() for m in re.finditer('id="chat-upload-error"', CHAT_INPUT)]
    trays = [m.start() for m in re.finditer(r'include "partials/chat_attachments\.html"', CHAT_INPUT)]

    assert len(slots) == len(trays) == 2
    assert all(slot < tray for slot, tray in zip(slots, trays, strict=True))


def test_the_page_allows_an_error_response_to_be_shown():
    """Without ``shouldSwap`` htmx drops a 422 body and the teacher sees nothing."""
    assert "htmx:beforeSwap" in BASE
    assert "shouldSwap" in BASE
    assert "swapOverride" in BASE, "the input's outerHTML is wrong for the error slot"


def test_a_request_that_never_answers_is_reported_too():
    """Silence is the one outcome a page must not present as nothing happening.

    A database lock left an upload hanging with no response at all: no chip, no
    error, and the document indexed all the same.
    """
    assert "htmx:sendError" in BASE
    assert "htmx:timeout" in BASE
