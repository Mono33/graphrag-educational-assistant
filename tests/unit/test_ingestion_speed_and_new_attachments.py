"""A 108-page magazine, then four screenshots of one of its articles (2026-10-02).

Two problems from one afternoon:

- **the magazine took five minutes.** Its 19 photo pages were read one at a time
  (80 s), a busy embedding provider failed the job, the failure could not be
  recorded on Postgres, and the retry waited out the lease and parsed the
  whole PDF again;
- **the screenshots were never cited.** The magazine's 17-part map used the
  whole maps budget, so the Writer never saw them named and wrote that no
  screenshots had arrived; its passages took 9 places of 12.
"""

from __future__ import annotations

import io
import threading
import time
import uuid
from types import SimpleNamespace

import pytest
from tests.unit.test_document_parsing import PDF, _digital_pdf, _scanned_pdf, _text_image

from aix.agent.agents.writer_agent import render_document_maps, render_material_passages
from aix.agent.graph.nodes import _private_search_query
from aix.webui.materials import embedding
from aix.webui.materials.parsing import parse_document, parsers
from aix.webui.materials.parsing.ocr import OcrResult, ocr_concurrency
from aix.webui.materials.retrieval import MaterialContextItem, _passages_from_new_attachments

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# The pages without text are read together, and kept in order
# ---------------------------------------------------------------------------


def _magazine() -> bytes:
    """A text page, three photo pages, a text page, two photo pages."""
    from pypdf import PdfReader, PdfWriter

    text = PdfReader(
        io.BytesIO(
            _digital_pdf(["Editoriale del numero sulla guerra", "Lettere dei lettori alla rivista"])
        )
    ).pages
    scan = PdfReader(io.BytesIO(_scanned_pdf(_text_image("x")))).pages[0]
    writer = PdfWriter()
    for page in (text[0], scan, scan, scan, text[1], scan, scan):
        writer.add_page(page)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_photo_pages_are_read_together_and_land_in_page_order(monkeypatch):
    monkeypatch.setenv("AIX_OCR_CONCURRENCY", "3")
    monkeypatch.setattr(parsers, "ocr_available", lambda: True)
    monkeypatch.setattr(parsers, "render_pdf_page", lambda content, index: index)
    busy, peak = [0], [0]
    lock = threading.Lock()

    def read(image, timeout_s):
        with lock:
            busy[0] += 1
            peak[0] = max(peak[0], busy[0])
        time.sleep(0.15)
        with lock:
            busy[0] -= 1
        return OcrResult(f"Didascalia della pagina {image + 1}", 0.95, 4, engine="vision")

    monkeypatch.setattr(parsers, "recognise", read)

    ir = parse_document(_magazine(), PDF)

    assert peak[0] == 3
    assert ir.ocr_pages == (2, 3, 4, 6, 7)
    pages = [block.page for block in ir.blocks]
    assert pages == sorted(pages), "reading order is page order"
    assert "Didascalia della pagina 6" in ir.text


def test_one_page_that_fails_loses_only_that_page(monkeypatch):
    monkeypatch.setattr(parsers, "ocr_available", lambda: True)
    monkeypatch.setattr(parsers, "render_pdf_page", lambda content, index: index)

    def read(image, timeout_s):
        if image == 2:
            raise TimeoutError("page too slow")
        return OcrResult(f"pagina {image + 1}", 0.95, 2)

    monkeypatch.setattr(parsers, "recognise", read)

    ir = parse_document(_magazine(), PDF)

    assert ir.unread_pages == (3,)
    assert "ocr_failed:page=3" in ir.warnings
    assert ir.ocr_pages == (2, 4, 6, 7)


def test_tesseract_reads_one_page_at_a_time(monkeypatch):
    """It is CPU and memory on the 4 GB VM; the vision model is network time."""
    monkeypatch.delenv("AIX_OCR_CONCURRENCY", raising=False)
    monkeypatch.setenv("AIX_OCR_BACKEND", "tesseract")
    assert ocr_concurrency() == 1
    monkeypatch.setenv("AIX_OCR_CONCURRENCY", "6")
    assert ocr_concurrency() == 6


# ---------------------------------------------------------------------------
# A busy embedding provider is waited for, not failed on
# ---------------------------------------------------------------------------


class _Busy(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class _Embeddings:
    def __init__(self, failures):
        self.failures, self.calls = list(failures), 0
        self.embeddings = SimpleNamespace(create=self.create)

    async def create(self, *, model, input):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        dim = embedding.embedding_dimension()
        return SimpleNamespace(
            data=[SimpleNamespace(index=i, embedding=[0.1] * dim) for i in range(len(input))]
        )


async def test_an_overloaded_provider_is_tried_again(monkeypatch):
    monkeypatch.setattr(embedding, "RETRY_DELAYS_S", (0, 0, 0))
    client = _Embeddings([_Busy(429), _Busy(503)])

    vectors = await embedding.embed_texts(["uno", "due"], client=client)

    assert len(vectors) == 2 and client.calls == 3


async def test_a_request_the_provider_refuses_is_not_retried(monkeypatch):
    monkeypatch.setattr(embedding, "RETRY_DELAYS_S", (0, 0, 0))
    client = _Embeddings([_Busy(400)])

    with pytest.raises(embedding.EmbeddingError):
        await embedding.embed_texts(["uno"], client=client)
    assert client.calls == 1


async def test_it_gives_up_after_its_last_wait(monkeypatch):
    monkeypatch.setattr(embedding, "RETRY_DELAYS_S", (0, 0))
    client = _Embeddings([_Busy(429)] * 5)

    with pytest.raises(embedding.EmbeddingError):
        await embedding.embed_texts(["uno"], client=client)
    assert client.calls == 3


# ---------------------------------------------------------------------------
# The documents attached for this turn are seen, named and searched
# ---------------------------------------------------------------------------


def _map(name, position, *, parts=1, turn=1, new=False):
    return {
        "source_label": name,
        "position": position,
        "turn": turn,
        "new": new,
        "summary": f"Sintesi di {name}",
        "sections": [
            {"title": f"Parte {i}", "pages": [i, i], "summary": "x" * 400}
            for i in range(1, parts + 1)
        ],
    }


def test_every_attachment_is_named_even_after_a_long_magazine():
    maps = [_map("Internazionale.pdf", 1, parts=17)] + [
        _map(f"Screenshot {i}.png", i, turn=5, new=True) for i in range(2, 6)
    ]

    text = render_document_maps(maps)

    for i in range(2, 6):
        assert f"Allegato {i} · Screenshot {i}.png** — aggiunto al turno 5, in questo turno" in text
    assert len(text) <= 6500
    assert "- …" in text, "the magazine's parts are cut, and the cut is visible"


def test_the_passages_of_this_turn_s_attachments_come_first():
    old = {"source_id": "a", "source_label": "Rivista.pdf", "attachment": 1}
    new = {"source_id": "b", "source_label": "Screenshot.png", "attachment": 2, "turn": 5}
    items = [
        {**old, "text": "y" * 900, "citation": f"[[material:a#page={i}]]"} for i in range(1, 6)
    ] + [{**new, "new": True, "text": "articolo", "citation": "[[material:b#page=1]]"}]

    text = render_material_passages(items, budget_chars=2000)

    assert text.index("Screenshot.png") < text.index("Rivista.pdf")
    assert "in questo turno" in text
    assert "[[material:b#page=1]]" in text


class _Retriever:
    def __init__(self):
        self.asked = []

    async def search(self, query, *, owner_id, version_ids, limit=None, **_):
        self.asked.append(version_ids[0])
        source = str(version_ids[0])
        return [
            MaterialContextItem(
                source_id=source, text=f"{source} {i}", score=0.5, locator={"page": i}
            )
            for i in range(1, (limit or 8) + 1)
        ]


async def test_each_new_attachment_brings_its_best_passages():
    magazine, shot_a, shot_b, older = (uuid.uuid4() for _ in range(4))
    attachments = {
        str(magazine): {"new": False},
        str(older): {"new": False},
        str(shot_a): {"new": True},
        str(shot_b): {"new": True},
    }
    found = [
        MaterialContextItem(source_id=str(magazine), text="m", score=0.6, locator={"page": p})
        for p in range(9)
    ] + [MaterialContextItem(source_id=str(shot_a), text="a", score=0.55, locator={"page": 1})]
    retriever = _Retriever()

    added = await _passages_from_new_attachments(retriever, "ia", attachments, found, uuid.uuid4())

    assert retriever.asked == [shot_a, shot_b], "only the new ones, and only when short"
    assert [(i.source_id, i.locator["page"]) for i in added] == [
        (str(shot_a), 2),
        (str(shot_b), 1),
        (str(shot_b), 2),
    ]


# ---------------------------------------------------------------------------
# A new lesson in the same thread is not searched for under the first one's topic
# ---------------------------------------------------------------------------


def test_a_later_turn_searches_for_its_own_subject():
    state = {
        "educational_profile": {"specific_topic": "crisi israelo-palestinese"},
        "subject_concepts": ["intelligenza artificiale", "economia dell'IA"],
        "conversation_history": [{"role": "user", "content": "lezione su Gaza"}],
    }

    assert _private_search_query(state) == "intelligenza artificiale · economia dell'IA"


def test_without_new_concepts_the_topic_still_leads():
    state = {
        "educational_profile": {"specific_topic": "guerra fredda"},
        "conversation_history": [{"role": "user", "content": "x"}],
        "teacher_query": "accorcia",
    }

    assert _private_search_query(state) == "guerra fredda"
