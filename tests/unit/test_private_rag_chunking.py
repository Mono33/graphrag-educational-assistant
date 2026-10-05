"""Chunking and embedding the teacher's material (Fase 2 · PR 6, §8.3–8.4).

The chunker is the part of the private RAG a reader can check by eye, so these
tests state its rules as sentences: a heading opens the chunk it introduces, two
slides never share one, a table is never cut in half, and every chunk keeps the
locator a citation will resolve (§8.6).

The embedder is checked for the two failures that would be silent: a provider
returning vectors of the wrong width, which the database would report much later
as an error about arrays, and vectors coming back in a different order from the
texts, which would attach every chunk to its neighbour's meaning.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aix.webui.materials.chunking import (
    CHUNKER_VERSION,
    MAX_TOKENS,
    chunk_document,
    estimate_tokens,
)
from aix.webui.materials.parsing.intake import PDF, PPTX
from aix.webui.materials.parsing.ir import DocumentIRV1, IRBlock

pytestmark = pytest.mark.unit


def _block(kind, text, *, page=None, rows=(), block_id=None) -> IRBlock:
    return IRBlock(
        block_id=block_id or f"b{abs(hash((kind, text, page))) % 10**8}",
        kind=kind,
        text=text,
        reading_order=0,
        source_locator={"format": "pdf", "page": page} if page else {"format": "docx"},
        page=page,
        rows=rows,
    )


def _ir(blocks, media_type=PDF) -> DocumentIRV1:
    return DocumentIRV1(
        parser_name="test",
        parser_version="1",
        parser_profile_version="ir@1",
        source_sha256="0" * 64,
        media_type=media_type,
        blocks=tuple(blocks),
    )


PARAGRAPH = (
    "La Guerra Fredda oppose USA e URSS senza mai sfociare in un conflitto diretto. "
    "L'equilibrio del terrore rese impossibile una guerra aperta fra le due potenze. "
)


# ---------------------------------------------------------------------------
# The rules of §8.3
# ---------------------------------------------------------------------------


def test_a_heading_opens_the_chunk_it_introduces():
    """«La crisi di Cuba» on its own answers nothing; it belongs with its text."""
    chunks = chunk_document(
        _ir(
            [
                _block("heading", "Le origini"),
                _block("paragraph", PARAGRAPH),
                _block("heading", "La crisi di Cuba"),
                _block("paragraph", "Nel 1962 l'URSS installò missili a Cuba."),
            ]
        )
    )

    assert len(chunks) == 2
    assert chunks[0].text.startswith("Le origini")
    assert chunks[1].text.startswith("La crisi di Cuba")
    assert "1962" in chunks[1].text
    assert chunks[1].locator["heading"] == "La crisi di Cuba"


def test_two_slides_never_share_a_chunk():
    """A citation must point at one slide, not at the seam between two."""
    chunks = chunk_document(
        _ir(
            [
                _block("slide_title", "Yalta", page=1),
                _block("paragraph", "Europa divisa in due sfere.", page=1),
                _block("slide_title", "Il muro", page=2),
                _block("paragraph", "Costruito nel 1961.", page=2),
            ],
            media_type=PPTX,
        )
    )

    for chunk in chunks:
        assert len(chunk.locator.get("pages", [chunk.locator["page"]])) == 1
    assert {c.locator["page"] for c in chunks} == {1, 2}


def test_a_table_is_one_unit():
    """Half a table looks complete and is not."""
    rows = (("Anno", "Evento"), ("1947", "Piano Marshall"), ("1961", "Muro di Berlino"))
    chunks = chunk_document(
        _ir(
            [
                _block("heading", "Cronologia"),
                _block("paragraph", PARAGRAPH),
                _block("table", "", rows=rows),
                _block("paragraph", "Dopo la tabella il discorso continua."),
            ]
        )
    )

    table_chunks = [c for c in chunks if "Piano Marshall" in c.text]
    assert len(table_chunks) == 1
    assert "1961 | Muro di Berlino" in table_chunks[0].text
    assert table_chunks[0].locator["heading"] == "Cronologia"


def test_no_chunk_exceeds_the_maximum_size():
    chunks = chunk_document(_ir([_block("paragraph", PARAGRAPH * 40, page=3)]))

    assert len(chunks) > 1
    assert all(estimate_tokens(c.text) <= MAX_TOKENS * 1.1 for c in chunks)


def test_consecutive_chunks_of_a_section_overlap():
    """A passage on the boundary has to survive whole in one of the two."""
    chunks = chunk_document(_ir([_block("paragraph", PARAGRAPH * 40, page=3)]))

    tail_of_first = chunks[0].text[-60:].strip()
    assert tail_of_first and tail_of_first in chunks[1].text


def test_every_chunk_keeps_what_a_citation_needs():
    chunks = chunk_document(
        _ir(
            [
                _block("heading", "Le origini", page=1),
                _block("paragraph", PARAGRAPH, page=1),
                _block("paragraph", "Seconda pagina.", page=2),
            ]
        )
    )

    for chunk in chunks:
        assert chunk.locator["block_ids"], "a citation resolves to the blocks it came from"
        assert chunk.chunker_version == CHUNKER_VERSION
        assert chunk.text_hash and len(chunk.text_hash) == 64
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_an_empty_document_produces_no_chunks():
    assert chunk_document(_ir([])) == []
    assert chunk_document(_ir([_block("paragraph", "   ")])) == []


def test_the_same_document_chunks_identically_twice():
    blocks = [_block("heading", "Le origini"), _block("paragraph", PARAGRAPH * 3)]

    assert [c.text_hash for c in chunk_document(_ir(blocks))] == [
        c.text_hash for c in chunk_document(_ir(blocks))
    ]


# ---------------------------------------------------------------------------
# Embedding (§8.4)
# ---------------------------------------------------------------------------


class _FakeEmbeddings:
    def __init__(self, dimension=1024, shuffle=False):
        self.dimension = dimension
        self.shuffle = shuffle
        self.batches: list[int] = []

    async def create(self, *, model, input):  # noqa: A002 — the provider's name
        self.batches.append(len(input))
        data = [
            SimpleNamespace(index=i, embedding=[float(i)] * self.dimension)
            for i in range(len(input))
        ]
        return SimpleNamespace(data=list(reversed(data)) if self.shuffle else data)


def _client(**kwargs):
    return SimpleNamespace(embeddings=_FakeEmbeddings(**kwargs))


async def test_texts_are_embedded_in_batches():
    from aix.webui.materials.embedding import BATCH_SIZE, embed_texts

    client = _client()
    vectors = await embed_texts([f"chunk {i}" for i in range(BATCH_SIZE + 5)], client=client)

    assert len(vectors) == BATCH_SIZE + 5
    assert client.embeddings.batches == [BATCH_SIZE, 5], "not one request per chunk"


async def test_vectors_follow_the_texts_not_the_response_order():
    from aix.webui.materials.embedding import embed_texts

    vectors = await embed_texts(["primo", "secondo", "terzo"], client=_client(shuffle=True))

    assert [v[0] for v in vectors] == [0.0, 1.0, 2.0]


async def test_the_wrong_dimension_is_refused_here_not_by_the_database():
    from aix.webui.materials.embedding import EmbeddingError, embed_texts

    with pytest.raises(EmbeddingError, match="1536"):
        await embed_texts(["prova"], client=_client(dimension=1536))


async def test_nothing_to_embed_is_not_a_request():
    from aix.webui.materials.embedding import embed_texts

    client = _client()
    assert await embed_texts([], client=client) == []
    assert client.embeddings.batches == []


def test_the_private_rag_is_on_unless_switched_off(monkeypatch):
    from aix.webui.materials.embedding import private_rag_enabled

    monkeypatch.delenv("AIX_PRIVATE_RAG_ENABLED", raising=False)
    assert private_rag_enabled() is True
    monkeypatch.setenv("AIX_PRIVATE_RAG_ENABLED", "false")
    assert private_rag_enabled() is False


def test_the_real_client_is_reachable():
    """The path every other test here skips by injecting a fake client.

    The first version imported ``aix.config.settings``, which does not exist.
    Nothing caught it, because nothing called ``_client()`` without a fake — the
    first real upload did, and both ingestion jobs and the search failed on it.
    No network here: building the client makes no request.
    """
    from aix.webui.materials.embedding import _client

    assert hasattr(_client(), "embeddings")


# ---------------------------------------------------------------------------
# What the first real documents taught (2026-09-16) — chunk@2
# ---------------------------------------------------------------------------


def test_no_chunk_is_made_only_of_overlap():
    """17 of 55 chunks were the carried overlap alone: no block, no page, no citation."""
    blocks = [
        _block("paragraph", PARAGRAPH * 6, page=1),
        _block("paragraph", PARAGRAPH * 9, page=2),
        _block("paragraph", PARAGRAPH * 7, page=3),
    ]

    chunks = chunk_document(_ir(blocks))

    assert chunks
    assert all(chunk.locator["block_ids"] for chunk in chunks)
    assert all("page" in chunk.locator for chunk in chunks), "every chunk can be cited"


def test_an_overlap_never_starts_in_the_middle_of_a_word():
    """«icoltura erano in mano…» — the tail was cut at a character count."""
    unpunctuated = " ".join(f"parola{i}" for i in range(900))

    chunks = chunk_document(_ir([_block("paragraph", unpunctuated, page=1)]))

    for chunk in chunks[1:]:
        assert chunk.text.split()[0].startswith("parola"), chunk.text[:30]


def test_pdf_line_breaks_and_hyphenation_are_undone():
    """«pol-/icy» across a line break embeds as two unknown tokens, not one word."""
    raw = "He abandoned his pol-\nicy of favoring coalition\ngovernments in Europe."

    text = chunk_document(_ir([_block("paragraph", raw, page=1)]))[0].text

    assert "policy" in text
    assert "coalition governments" in text
    assert "\n" not in text


def test_paragraph_breaks_survive_the_normalisation():
    raw = "Primo paragrafo sulla guerra.\n\nSecondo paragrafo su Yalta."

    text = chunk_document(_ir([_block("paragraph", raw, page=1)]))[0].text

    assert "Primo paragrafo sulla guerra.\n\nSecondo paragrafo" in text
