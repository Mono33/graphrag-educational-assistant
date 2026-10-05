"""The lesson as a document, and the snapshot exports read (Fase 3 · PR 7).

The Markdown below has the shape the Writer actually produces — checked against
lessons in the dev database: an Art. 50 marker on top, a title, a bold duration
line, sections with timings in parentheses, nested headings inside "Attività",
a table in the assessment, and a sources footer.

§10.2 gives the adapter four rules, and each has a group of tests: it
recognises the structure, keeps the order, invents nothing, and produces block
ids that stay stable.
"""

from __future__ import annotations

import pytest

from aix.export import adapt_lesson_markdown, build_export_ir, render_outline

pytestmark = pytest.mark.unit

LESSON = """<!-- ai-generated: true, system: agentic-graphrag, trace_id: abc -->

# Memoria di lavoro

**Materia:** Scienze | **Durata:** 60 minuti

## Introduzione *(10 minuti)*

Domanda di attivazione: *che cosa ricordate di ieri?*

## Attività Principali

### Fase 1 — I Do *(20 minuti)*

- Spiegazione con **schede cartacee**
  - una scheda per coppia
- Video: [How We Make Memories](https://youtu.be/bSycdIx-C48)

### Fase 2 — We Do *(20 minuti)*

1. Lavoro in gruppi da 4
2. Restituzione

## Valutazione *(10 minuti)*

| Strumento | Funzione |
| --- | --- |
| Exit ticket | Verifica **immediata** |

*Fonti:* Cowan (2013); Knowledge Graph.
"""


@pytest.fixture
def doc():
    return adapt_lesson_markdown(LESSON, lesson_id="L1")


def _kinds(doc):
    return [b.kind for b in doc.blocks]


# ---------------------------------------------------------------------------
# It recognises the structure
# ---------------------------------------------------------------------------


def test_the_title_and_duration_are_read(doc):
    assert doc.title == "Memoria di lavoro"
    assert doc.duration_minutes == 60


def test_headings_keep_their_level_and_minutes(doc):
    headings = [(b.level, b.text, b.minutes) for b in doc.blocks if b.kind == "heading"]

    assert headings[0] == (1, "Memoria di lavoro", None)
    assert (3, "Fase 1 — I Do (20 minuti)", 20) in headings
    assert (2, "Attività Principali", None) in headings


def test_lists_keep_nesting_and_numbering(doc):
    bullets, numbered = [b for b in doc.blocks if b.kind == "list"]

    assert bullets.ordered is False
    assert bullets.depths == (0, 1, 0), "the nested bullet is one level deeper"
    assert numbered.ordered is True
    assert len(numbered.items) == 2


def test_formatting_and_links_survive_inline(doc):
    bullets = next(b for b in doc.blocks if b.kind == "list")

    assert any(run.bold and run.text == "schede cartacee" for run in bullets.items[0])
    link = next(run for run in bullets.items[2] if run.href)
    assert link.href == "https://youtu.be/bSycdIx-C48"
    assert link.text == "How We Make Memories"


def test_the_table_is_read_with_its_header_row(doc):
    table = next(b for b in doc.blocks if b.kind == "table")

    assert len(table.rows) == 2
    assert "".join(r.text for r in table.rows[0][0]) == "Strumento"
    assert any(run.bold for run in table.rows[1][1]), "bold survives inside a cell"


def test_the_ai_marker_is_not_content(doc):
    assert all("ai-generated" not in b.text for b in doc.blocks)
    assert not doc.warnings or all("HTML" not in w for w in doc.warnings)


# ---------------------------------------------------------------------------
# It keeps the order
# ---------------------------------------------------------------------------


def test_blocks_stay_in_the_order_the_writer_wrote(doc):
    assert _kinds(doc) == [
        "heading",  # title
        "paragraph",  # materia / durata
        "heading",  # introduzione
        "paragraph",
        "heading",  # attività
        "heading",  # fase 1
        "list",
        "heading",  # fase 2
        "list",
        "heading",  # valutazione
        "table",
        "paragraph",  # fonti
    ]


# ---------------------------------------------------------------------------
# It invents nothing
# ---------------------------------------------------------------------------


def test_a_missing_section_is_reported_not_filled():
    doc = adapt_lesson_markdown("# Solo titolo\n\nUn paragrafo.\n")

    assert _kinds(doc) == ["heading", "paragraph"]
    assert any("Durata" in w for w in doc.warnings)


def test_a_lesson_without_a_title_gets_the_fallback_and_a_warning():
    doc = adapt_lesson_markdown("Solo testo.\n", fallback_title="Lezione di prova")

    assert doc.title == "Lezione di prova"
    assert any("titolo" in w for w in doc.warnings)


def test_an_image_becomes_a_link_and_a_warning():
    """§10.10: unknown licence is link-only, and the renderer never fetches."""
    doc = adapt_lesson_markdown("# T\n\n![schema](https://example.org/a.png)\n")
    run = doc.blocks[1].runs[0]

    assert run.href == "https://example.org/a.png"
    assert "immagine" in run.text
    assert any("Immagine non incorporata" in w for w in doc.warnings)


@pytest.mark.parametrize(
    "href",
    [
        # Refused by markdown-it itself: no link token is produced at all, and
        # the Markdown stays literal text — harmless in a PDF.
        "javascript:alert(1)",
        "data:text/html,<b>x</b>",
        "file:///etc/passwd",
        # Accepted by markdown-it, refused by the export's own allowlist.
        "ftp://example.org/x",
        "tel:+390000000",
    ],
)
def test_an_unsafe_link_is_never_clickable(href):
    """Two layers, and the test checks the guarantee rather than the layer."""
    doc = adapt_lesson_markdown(f"# T\n\n[clicca]({href})\n")
    runs = [run for block in doc.blocks for run in block.runs]

    assert all(run.href is None for run in runs)
    assert "clicca" in "".join(run.text for run in runs)


def test_raw_html_is_dropped_with_a_warning():
    doc = adapt_lesson_markdown("# T\n\n<div onclick='x()'>ciao</div>\n")

    assert all("onclick" not in b.text for b in doc.blocks)
    assert any("HTML" in w for w in doc.warnings)


def test_empty_markdown_does_not_raise():
    doc = adapt_lesson_markdown("")

    assert doc.blocks == ()
    assert doc.title == "Lezione"


# ---------------------------------------------------------------------------
# Block ids are stable
# ---------------------------------------------------------------------------


def test_the_same_content_yields_the_same_ids():
    first = [b.block_id for b in adapt_lesson_markdown(LESSON).blocks]
    second = [b.block_id for b in adapt_lesson_markdown(LESSON).blocks]

    assert first == second


def test_inserting_a_paragraph_does_not_renumber_what_follows():
    """Position in the file is not part of an id, or every edit would move them all."""
    before = {b.text or b.kind: b.block_id for b in adapt_lesson_markdown(LESSON).blocks}
    edited = LESSON.replace("# Memoria di lavoro\n", "# Memoria di lavoro\n\nNota aggiunta.\n")
    after = {b.text or b.kind: b.block_id for b in adapt_lesson_markdown(edited).blocks}

    table_id = next(b.block_id for b in adapt_lesson_markdown(LESSON).blocks if b.kind == "table")
    assert table_id in after.values()
    assert before["Valutazione (10 minuti)"] == after["Valutazione (10 minuti)"]


def test_duplicate_content_still_gets_distinct_ids():
    doc = adapt_lesson_markdown("# T\n\nUguale.\n\nUguale.\n")
    ids = [b.block_id for b in doc.blocks if b.kind == "paragraph"]

    assert len(ids) == 2 and len(set(ids)) == 2


def test_the_content_hash_identifies_the_exact_revision():
    assert adapt_lesson_markdown(LESSON).content_hash == adapt_lesson_markdown(LESSON).content_hash
    assert (
        adapt_lesson_markdown(LESSON).content_hash
        != adapt_lesson_markdown(LESSON + " ").content_hash
    )


# ---------------------------------------------------------------------------
# The export snapshot
# ---------------------------------------------------------------------------


def test_sections_group_blocks_under_their_heading(doc):
    ir = build_export_ir(doc)

    titles = [s.title for s in ir.sections]
    assert titles[0] == "Introduzione (10 minuti)"
    assert "Memoria di lavoro" not in titles, "the title is drawn once, as a title"
    valutazione = next(s for s in ir.sections if s.title.startswith("Valutazione"))
    assert [b.kind for b in valutazione.blocks] == ["table", "paragraph"]


def test_text_before_the_first_section_is_kept_as_preamble(doc):
    ir = build_export_ir(doc)

    assert [b.kind for b in ir.preamble] == ["paragraph"]


def test_the_timed_outline_does_not_count_nested_timings_twice(doc):
    ir = build_export_ir(doc)

    assert [m for _t, m in ir.timed_outline] == [10, 20, 20, 10]
    assert ir.outline_total_minutes == 60


def test_the_snapshot_carries_the_disclosure_and_no_rendering_details(doc):
    ir = build_export_ir(doc)

    assert ir.disclosure_short == "Generato dall'IA"
    assert "verificare" in ir.disclosure_long.lower()
    assert ir.content_hash == doc.content_hash


# ---------------------------------------------------------------------------
# The outline
# ---------------------------------------------------------------------------


def test_the_outline_lists_sections_with_minutes_and_a_total(doc):
    outline = render_outline(build_export_ir(doc))

    assert outline.startswith("Memoria di lavoro\n")
    assert " 10′  Introduzione (10 minuti)" in outline
    assert "  Fase 1 — I Do (20 minuti)" in outline, "level 3 is indented under level 2"
    assert " 60′  totale delle sezioni" in outline
    assert "Generato dall'IA" in outline


def test_the_outline_flags_a_total_that_does_not_match():
    short = LESSON.replace("## Valutazione *(10 minuti)*", "## Valutazione *(2 minuti)*")
    outline = render_outline(build_export_ir(adapt_lesson_markdown(short)))

    assert "(la lezione dichiara 60)" in outline
