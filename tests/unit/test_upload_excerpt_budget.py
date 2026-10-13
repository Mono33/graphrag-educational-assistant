"""The excerpt budget stops refusing documents it has no business refusing.

``MAX_TOTAL_EXCERPT`` caps the plain text pasted into the Writer's prompt when
there is no private index. It dates from when that paste was the only way a
document reached the agent at all.

With the private RAG on it is a fallback: the lesson is written from retrieved
passages and the excerpt is never sent. Refusing an upload because the *fallback*
does not fit refuses a document for the weight of something nobody reads — and on
2026-09-22 it did: the Indicazioni nazionali filled the budget alone, the third
attachment came back 422, and the teacher read it as "this cannot take more than
two documents".

With the flag off the refusal is honest, because then there is nowhere else for
the document to go. Both halves are pinned here.
"""

from __future__ import annotations

import uuid

import pytest

from aix.webui.lessons.uploads import MAX_TOTAL_EXCERPT, save_upload

pytestmark = pytest.mark.unit

LESSON = uuid.UUID("77777777-7777-7777-7777-777777777777")


@pytest.fixture(autouse=True)
def _uploads_in_a_temporary_place(tmp_path, monkeypatch):
    monkeypatch.setattr("aix.webui.lessons.uploads.UPLOAD_ROOT", tmp_path / "uploads")


def _full_manifest() -> list[dict]:
    """A manifest whose excerpts already fill the budget."""
    return [
        {
            "id": str(uuid.uuid4()),
            "filename": "indicazioni_nazionali.pdf",
            "stored_name": "x.txt",
            "size": 10,
            "mime": "text/plain",
            "text_excerpt": "x" * MAX_TOTAL_EXCERPT,
        }
    ]


def test_with_the_private_rag_on_the_document_is_kept_without_its_excerpt(monkeypatch):
    monkeypatch.setenv("AIX_PRIVATE_RAG_ENABLED", "true")

    manifest, entry = save_upload(
        LESSON, "modello_pei.txt", b"Il PEI e le misure di sostegno.", _full_manifest()
    )

    assert len(manifest) == 2, "the document is accepted"
    assert entry["filename"] == "modello_pei.txt"
    assert entry["text_excerpt"] == "", (
        "only the fallback text is dropped — the document is still indexed and searchable"
    )


def test_a_document_that_fits_keeps_its_excerpt(monkeypatch):
    monkeypatch.setenv("AIX_PRIVATE_RAG_ENABLED", "true")

    _manifest, entry = save_upload(LESSON, "breve.txt", b"La fotosintesi in tre fasi.", [])

    assert "fotosintesi" in entry["text_excerpt"]
