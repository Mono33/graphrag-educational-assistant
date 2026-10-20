"""The document map: parts, summaries, and the requests that point into them.

On 2026-09-29 a teacher attached a 222-page book and asked for a lesson on
personal finance. The private search returned the 8 passages closest to that
sentence out of 297; the Writer, knowing nothing else of the book, wrote its
usual lesson from the Knowledge Graph and cited the book zero times, and the
Critic approved it at 3.8. The map gives the Writer the book's chapters, lets a
search reach the chapter a request names, and a draft that ignores relevant
passages is sent back.

The model is a fake client throughout: nothing leaves the machine.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from aix.webui.materials.chunking import chunk_document
from aix.webui.materials.document_map import (
    MAX_SECTIONS,
    build_map,
    named_parts,
    plan_parts,
)
from aix.webui.materials.parsing.ir import DocumentIRV1, IRBlock

pytestmark = pytest.mark.unit

PDF = "application/pdf"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

PAGE_TEXT = (
    "Il capitalismo è un sistema economico in cui individui e imprese private producono "
    "e scambiano beni e servizi su mercati liberi, guidati dalla concorrenza e dal profitto. "
) * 12


def _block(i, kind, text, page=None, level=None):
    return IRBlock(
        block_id=f"b{i}",
        kind=kind,
        text=text,
        reading_order=i,
        source_locator={"format": "pdf", "page": page} if page else {"format": "docx"},
        page=page,
        level=level,
    )


def _ir(blocks, media_type=PDF, pages=None) -> DocumentIRV1:
    return DocumentIRV1(
        parser_name="test",
        parser_version="1",
        parser_profile_version="ir@1",
        source_sha256="0" * 64,
        media_type=media_type,
        blocks=tuple(blocks),
        page_count=pages,
    )


def _book(chapters=6, pages_each=10) -> DocumentIRV1:
    """A book whose chapters open with «Chapter N» and a title on the next line."""
    blocks, n, page = [], 0, 1
    for chapter in range(1, chapters + 1):
        for p in range(pages_each):
            text = PAGE_TEXT
            if p == 0:
                text = f"Chapter {chapter}\nTitle of chapter {chapter}\n" + PAGE_TEXT
            blocks.append(_block(n, "paragraph", text, page=page))
            n += 1
            page += 1
    return _ir(blocks, pages=page - 1)


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------


def test_a_book_split_without_bookmarks_is_cut_at_its_chapters():
    """The 222-page book, split in two, had lost its outline but not «Chapter 1..6»."""
    parts = plan_parts(_book())

    assert [p.source for p in parts] == ["chapters"] * 6
    assert parts[0].title == "Chapter 1 — Title of chapter 1"
    assert parts[3].title.startswith("Chapter 4")


def test_a_handout_is_one_part():
    parts = plan_parts(_ir([_block(0, "paragraph", "Una scheda breve sulla fotosintesi.")]))

    assert len(parts) == 1 and parts[0].source == "whole"


def test_a_word_file_is_cut_at_its_headings():
    blocks = []
    for n, title in enumerate(["Storia", "Geografia", "Musica"]):
        blocks.append(_block(2 * n, "heading", title, level=1))
        blocks.append(_block(2 * n + 1, "paragraph", PAGE_TEXT * 2))

    parts = plan_parts(_ir(blocks, media_type=DOCX))

    assert [p.title for p in parts] == ["Storia", "Geografia", "Musica"]


def test_without_any_structure_the_pages_are_cut_in_windows():
    blocks = [_block(n, "paragraph", PAGE_TEXT, page=n + 1) for n in range(60)]

    parts = plan_parts(_ir(blocks, pages=60))

    assert all(p.source == "window" for p in parts)
    assert 2 <= len(parts) <= MAX_SECTIONS
    assert parts[0].title == "pp. 1–6"


def test_a_copyright_page_does_not_get_a_part_of_its_own():
    blocks = [_block(0, "paragraph", "Chapter 1\nTitle\n" + PAGE_TEXT * 3, page=1)]
    blocks.append(_block(1, "paragraph", "Chapter 2\nCopyright\n© 2001", page=2))
    blocks.append(_block(2, "paragraph", "Chapter 3\nTitle\n" + PAGE_TEXT * 3, page=3))

    parts = plan_parts(_ir(blocks, pages=3))

    assert len(parts) == 2, "the two-line page is folded into its neighbour"


# ---------------------------------------------------------------------------
# Requests that name a part
# ---------------------------------------------------------------------------

SECTIONS = [
    {"title": "Inizio"},
    {"title": "Chapter 1 — The Development and Growth of Capitalism"},
    {"title": "Chapter 2 — Modern Capitalism"},
    {"title": "Chapter 3 — The Future of Capitalism"},
    {"title": "Chapter 4 — Entrepreneurs in the 21st Century"},
]


@pytest.mark.parametrize(
    "request_text, expected",
    [
        ("approfondisci il capitolo 4", ["Chapter 4"]),
        ("partiamo dal cap. IV", ["Chapter 4"]),
        ("una lezione su modern capitalism", ["Chapter 2"]),
        ("chapter 2 e capitolo 3", ["Chapter 2", "Chapter 3"]),
        ("crea una lezione sulla finanza personale", []),
    ],
)
def test_a_request_can_name_a_part_by_number_or_title(request_text, expected):
    found = [s["title"].split(" —")[0] for s in named_parts(request_text, SECTIONS)]

    assert found == expected


def test_a_number_is_the_chapter_s_own_number_not_its_position():
    """The front matter makes the fourth part rarely chapter 4."""
    found = named_parts("capitolo 1", SECTIONS)

    assert found[0]["title"].startswith("Chapter 1")


# ---------------------------------------------------------------------------
# Building the map
# ---------------------------------------------------------------------------


class _FakeModel:
    def __init__(self, *, fail_on: str = ""):
        self.calls: list[dict] = []
        self.fail_on = fail_on
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][0]["content"]
        if self.fail_on and self.fail_on in prompt:
            raise RuntimeError("provider error")
        if "sintesi delle parti" in prompt:
            reply = {"sintesi": "Un libro sul capitalismo.", "temi": ["capitalismo", "impresa"]}
        else:
            reply = {"titolo": "x", "sintesi": "Parla di mercati.", "concetti": ["mercato"]}
        message = SimpleNamespace(content="```json\n" + json.dumps(reply) + "\n```")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


async def test_the_map_has_every_part_its_pages_and_its_passages():
    ir = _book()
    model = _FakeModel()

    document_map = await build_map(ir, chunks=chunk_document(ir), client=model)

    assert document_map["summary"] == "Un libro sul capitalismo."
    assert document_map["structure"] == "chapters"
    sections = document_map["sections"]
    assert len(sections) == 6
    assert sections[1]["title"].startswith("Chapter 2")
    assert sections[1]["pages"] == [11, 20]
    first, last = sections[1]["ordinals"]
    assert sections[0]["ordinals"][1] < first <= last < sections[2]["ordinals"][0]
    assert all(c["extra_body"]["provider"]["zdr"] is True for c in model.calls)


async def test_a_part_the_model_could_not_describe_does_not_lose_the_map():
    ir = _book()

    document_map = await build_map(
        ir, chunks=chunk_document(ir), client=_FakeModel(fail_on="Chapter 3")
    )

    assert len(document_map["sections"]) == 6
    assert document_map["sections"][2]["summary"] == ""
    assert document_map["sections"][3]["summary"] == "Parla di mercati."


# ---------------------------------------------------------------------------
# What the Writer reads
# ---------------------------------------------------------------------------

MAP_FOR_PROMPT = {
    "source_id": "v1",
    "source_label": "Understanding Global Capitalism.pdf",
    "summary": "Un libro sul capitalismo e sull'imprenditoria.",
    "themes": ["capitalismo", "imprenditoria"],
    "sections": [
        {"title": "Chapter 1 — Growth", "pages": [15, 28], "summary": "Storia del capitalismo."},
        {"title": "Chapter 2 — Modern Capitalism", "pages": [29, 46], "summary": "Concorrenza."},
    ],
}

PASSAGE = {
    "source_id": "v1",
    "source_label": "Understanding Global Capitalism.pdf",
    "text": "Competition is the engine of the market.",
    "citation": "[[material:v1#page=30]]",
    "score": 0.62,
    "part": "Chapter 2 — Modern Capitalism",
}


def test_the_documents_are_the_primary_source_and_the_map_is_not_cited():
    from aix.agent.agents.writer_agent import teacher_material_appendix

    appendix = teacher_material_appendix(
        [PASSAGE], None, phase="write", material_maps=[MAP_FOR_PROMPT]
    )

    assert "Sono la fonte\nprincipale dei contenuti" in appendix
    assert "Chapter 2 — Modern Capitalism (pp. 29–46): Concorrenza." in appendix
    assert "**Non è una\nfonte**" in appendix
    assert "[[material:v1#page=30]] · parte: Chapter 2 — Modern Capitalism" in appendix


def test_with_a_map_the_passages_get_more_room(monkeypatch):
    from aix.agent.agents import writer_agent

    long = [dict(PASSAGE, text="x" * 1400, citation=f"[[material:v1#page={n}]]") for n in range(20)]

    without = writer_agent.teacher_material_appendix(long, None, phase="write")
    with_map = writer_agent.teacher_material_appendix(
        long, None, phase="write", material_maps=[MAP_FOR_PROMPT]
    )

    assert with_map.count("[[material:") > without.count("[[material:") + 4


def test_without_maps_or_passages_nothing_is_added():
    from aix.agent.agents.writer_agent import teacher_material_appendix

    assert teacher_material_appendix(None, None, phase="write", material_maps=[]) == ""


# ---------------------------------------------------------------------------
# A draft that ignores the documents
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "draft, score, expected",
    [
        ("Una lezione generica sul risparmio.", 0.62, True),
        ("Il mercato premia la concorrenza [[material:v1#page=30]].", 0.62, False),
        ("Una lezione su un altro tema.", 0.30, False),
    ],
)
def test_a_draft_that_cites_none_of_the_relevant_passages_is_caught(draft, score, expected):
    from aix.agent.graph.nodes import _ignores_the_documents

    state = {"material_context_items": [dict(PASSAGE, score=score)], "lesson_plan_draft": draft}

    assert _ignores_the_documents(state) is expected


async def test_the_critic_s_approval_does_not_stand_for_a_draft_that_ignores_the_documents(
    monkeypatch,
):
    from aix.agent.graph import nodes

    verdict = SimpleNamespace(
        approved=True,
        summary="ok",
        average_score=3.8,
        scores={},
        strengths=[],
        revision_instructions="",
    )

    class _Critic:
        async def critique(self, **_):
            return verdict

    monkeypatch.setattr(nodes, "get_critic", lambda: _Critic())
    state = {
        "teacher_query": "crea una lezione sulla finanza personale",
        "lesson_plan_draft": "Una lezione generica sul risparmio.",
        "material_context_items": [PASSAGE],
        "revision_count": 0,
        "max_revisions": 1,
    }

    updates = await nodes.critique_node(state)

    assert updates["approved"] is False
    assert "non ne usa né cita nessun passaggio" in updates["revision_instructions"]
    assert updates["revision_count"] == 1

    # Once the revision is spent, the Critic's verdict stands.
    updates = await nodes.critique_node({**state, "revision_count": 1})
    assert updates["approved"] is True


WATER = [
    {
        "score": 0.52,
        # «lavare» is here on purpose: «lava» (from «magma e lava») met it.
        "text": "Il ciclo idrologico: evaporazione, condensazione, precipitazioni. "
        "L'acqua serve per cucinare e lavare i capi di abbigliamento.",
    }
]


@pytest.mark.parametrize(
    ("subject", "turn", "passages", "expected"),
    [
        # 2026-10-20: the water-cycle handout still bound, a lesson on volcanoes
        # asked -- the rewrite turned it into a lesson on the water cycle.
        (
            ["vulcani", "eruzione vulcanica", "magma e lava"],
            "Fammi una lezione sui vulcani",
            WATER,
            False,
        ),
        # The case the check was made for: the document is about the request.
        (["ciclo dell'acqua", "evaporazione"], "Una lezione sul ciclo dell'acqua", WATER, True),
        # No subject from the planner: as before, the passages count.
        ([], "Prepariamo una lezione mirata", WATER, True),
    ],
)
def test_the_documents_send_a_lesson_back_only_when_they_are_about_its_subject(
    subject, turn, passages, expected
):
    from aix.agent.graph.nodes import _ignores_the_documents

    state = {
        "material_context_items": passages,
        "lesson_plan_draft": "# Lezione\n\n## Attività (20 minuti)\n- Senza citare i documenti.",
        "subject_concepts": subject,
        "raw_user_turn": turn,
    }

    assert _ignores_the_documents(state) is expected
