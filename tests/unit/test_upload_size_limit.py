"""The size limit, and the two ways it used to be wrong.

``MAX_FILE_BYTES`` was 10 MB, sized when the whole of an attachment was pasted
into the Writer's prompt. On 2026-09-22 it refused an issue of a magazine --
13.5 MB, 114 pages -- which pypdf reads in four seconds, no OCR, 375 000
characters of clean text. Bytes are a poor proxy for the work: the real guards
are ``intake.MAX_PAGES`` and OCR concurrency. What the byte limit still guards
is memory, because the upload is held whole in RAM while it is parsed.

Two things are pinned here, and neither is the number itself:

* the refusal quotes the limit that is actually in force. It used to say
  "massimo 10 MB" in a string beside the constant, so raising one would have
  left teachers reading the other;
* an oversized file is refused *before* it is read. The check inside
  ``save_upload`` runs after the bytes are already in memory, which at 100 MB
  is the difference between a rejection and a bad afternoon on a 4 GB VM.
"""

from __future__ import annotations

import uuid

import pytest

from aix.webui.lessons.uploads import MAX_FILE_BYTES, save_upload, too_large_message
from aix.webui.materials.parsing import intake

pytestmark = pytest.mark.unit

LESSON = uuid.UUID("88888888-8888-8888-8888-888888888888")


@pytest.fixture(autouse=True)
def _uploads_in_a_temporary_place(tmp_path, monkeypatch):
    monkeypatch.setattr("aix.webui.lessons.uploads.UPLOAD_ROOT", tmp_path / "uploads")


def test_a_file_over_the_limit_is_refused():
    with pytest.raises(ValueError) as refusal:
        save_upload(LESSON, "grande.txt", b"x" * (MAX_FILE_BYTES + 1), [])

    assert str(MAX_FILE_BYTES // (1024 * 1024)) in str(refusal.value)


def test_a_magazine_sized_file_is_no_longer_refused_for_its_size():
    """13.5 MB of photographs around clean text: the case that prompted this.

    Only the size gate is under test -- the bytes here are text, so what is
    proved is that a file of that *weight* gets through to the parser at all.
    """
    magazine = 14_130_313  # the real file, to the byte
    assert magazine < MAX_FILE_BYTES

    manifest, entry = save_upload(
        LESSON, "rivista.txt", b"Un articolo di attualita. " * (magazine // 26), []
    )

    assert len(manifest) == 1
    assert entry["filename"] == "rivista.txt"
    assert entry["size"] > 10 * 1024 * 1024, "the weight the old limit refused"


def test_the_two_gates_hold_the_same_number():
    """A file passes two size checks, and they used to disagree.

    ``lessons.uploads`` is the chat attachment path; ``parsing.intake`` is the
    gate of the RAG pipeline, and ``IntakeRejected`` is a ``ValueError``, so
    both surface as the same 422. On 2026-09-22 the limit was raised in the
    first and the teacher went on being refused by the second — with a message
    quoting the old number, which is what made it look like a broken upload
    rather than a rejected file.
    """
    assert MAX_FILE_BYTES == intake.MAX_BYTES


def test_raising_the_limit_moves_every_sentence_that_quotes_it(monkeypatch):
    """The number was written in three places and the wording in three more."""
    from aix.webui import features

    monkeypatch.setattr(intake, "MAX_BYTES", 250 * 1024 * 1024)

    assert "250 MB" in intake.too_large_message()
    assert "250 MB" in too_large_message(), "the attachment path forwards"
    assert "250 MB" in features.upload_hint(), "and so does the paperclip's tooltip"


def test_the_magazine_is_under_both_gates_now():
    """13.5 MB, the real file to the byte. Refused by ``check_upload`` until today.

    Asserted on the number rather than on a fixture: a 14 MB PDF in the repo to
    prove a size check is a poor trade, and a *fake* one proves nothing — the
    first version of this test padded a PDF header with zeroes and intake
    rejected it as corrupt, correctly, before ever reaching the size.
    """
    magazine = 14_130_313

    assert magazine > 10 * 1024 * 1024, "over the limit that refused it"
    assert magazine < intake.MAX_BYTES
    assert magazine < MAX_FILE_BYTES


def test_intake_refuses_by_size_before_it_reads_anything(monkeypatch):
    """The size gate runs first, so the bytes need not be a real document."""
    monkeypatch.setattr(intake, "MAX_BYTES", 1024)

    with pytest.raises(intake.IntakeRejected) as refused:
        intake.check_upload("enorme.pdf", b"x" * 2048)

    assert refused.value.code == "too_large"
    assert "1 MB" not in str(refused.value), "the sentence quotes the limit in force"
