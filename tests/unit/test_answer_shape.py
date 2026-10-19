"""The Writer's answer loses its packaging before anyone reads it.

On 2026-09-22 a teacher asked for a list of topics for a future lesson. The
Writer wrapped the whole answer in a ```` ```markdown ```` fence -- copied from
the list prompt, which shows its expected format inside one -- and the PDF
export printed ``#`` and ``**`` literally in one grey box. The lesson of the
previous turn, written from the one prompt without a fence, came out clean.

The same day showed that a correction applied at one exit is not applied at the
others: the disclaimer fix lived in the WebUI's renderer, which the PDF export
skips for lessons without private citations and the public API skips entirely.
So the tidying happens in ``write_node``, and these tests pin both the rules and
the places they run.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from aix.agent.validation.answer_shape import tidy_answer, unwrap_whole_answer_fence

pytestmark = pytest.mark.unit

NOTE = (
    "⚠️ **Nota sulle fonti**: Il contenuto disciplinare proviene da fonti "
    "esterne e non è stato verificato dal Knowledge Graph FEM."
)
MARKING = "<!-- ai-generated: true, system: agentic-graphrag -->"


# ---------------------------------------------------------------------------
# The fence around the whole answer
# ---------------------------------------------------------------------------


def test_a_fenced_answer_is_unwrapped():
    answer = "```markdown\n# Nuovi argomenti\n\n1. **Luce e metabolismo**\n```\n"

    assert unwrap_whole_answer_fence(answer) == "# Nuovi argomenti\n\n1. **Luce e metabolismo**\n"


def test_the_marking_comment_in_front_of_a_stored_lesson_is_kept():
    """Stored lessons carry the AI-generated marking before the fence."""
    stored = f"{MARKING}\n\n```markdown\n# Titolo\n\nTesto.\n```"

    out = unwrap_whole_answer_fence(stored)

    assert out.startswith(MARKING), "the EU AI Act marking is not packaging"
    assert "```" not in out and "# Titolo" in out


def test_a_fence_without_a_language_is_packaging_too():
    assert unwrap_whole_answer_fence("```\n# Titolo\n\nTesto.\n```\n") == "# Titolo\n\nTesto.\n"


def test_an_answer_cut_off_before_its_closing_fence_is_unwrapped():
    """Out of tokens mid-answer: the opening fence alone turns it all into code."""
    assert (
        unwrap_whole_answer_fence("```markdown\n# Titolo\n\nTesto tronc")
        == "# Titolo\n\nTesto tronc\n"
    )


@pytest.mark.parametrize(
    "text, why",
    [
        ("Esempio:\n\n```python\nprint(1)\n```\n", "a code example inside prose"),
        ("```python\nprint(1)\n```\n", "a whole answer in Python is code, not packaging"),
        ("```\nuno\n```\n\nTesto.\n\n```\ndue\n```\n", "two blocks, and the text between them"),
        ("# Titolo\n\nNessun recinto.\n", "nothing to unwrap"),
        ("", "an empty answer"),
    ],
)
def test_anything_else_is_left_exactly_as_it_is(text, why):
    assert unwrap_whole_answer_fence(text) == text, why


def test_tidy_unwraps_before_it_counts_disclaimers():
    """Inside a fence the disclaimer is code; the order of the two rules matters."""
    answer = f"```markdown\n# Titolo\n\n{NOTE}\n\n*Fonti:* Wikipedia\n\n---\n{NOTE}\n```\n"

    out = tidy_answer(answer)

    assert "```" not in out
    assert out.count("Nota sulle fonti") == 1


# ---------------------------------------------------------------------------
# Where it runs
# ---------------------------------------------------------------------------


async def test_write_node_tidies_the_draft_at_the_source():
    """From here the draft reaches the WebUI, the exports and the API alike."""
    from aix.agent.graph import nodes as graph_nodes

    fenced = "```markdown\n# Nuovi argomenti\n\nUn elenco.\n```\n"
    writer = AsyncMock(write=AsyncMock(return_value=fenced))
    state = {
        "teacher_query": "dammi una lista di argomenti",
        "plan": {"query_intent": "list"},
        "query_intent": "list",
        "language": "it",
    }

    with patch.object(graph_nodes, "get_writer", return_value=writer):
        updates = await graph_nodes.write_node(state)

    assert updates["lesson_plan_draft"] == "# Nuovi argomenti\n\nUn elenco.\n"


async def test_a_stored_lesson_without_citations_is_tidied_for_export():
    """The gap in the first disclaimer fix.

    ``resolve_for_lesson`` returned early for a lesson with no private
    citations, so the PDF of such a lesson kept what the WebUI had removed.
    """
    from aix.webui.materials.citations import resolve_for_lesson

    stored = f"{MARKING}\n\n```markdown\n# Titolo\n\nTesto.\n\n{NOTE}\n\n---\n{NOTE}\n```"

    out = await resolve_for_lesson(stored, session=None, lesson=None)

    assert "```" not in out
    assert out.count("Nota sulle fonti") == 1


def test_every_intent_prompt_says_not_to_fence_the_answer():
    """The fence is copied from the prompts, so each one that shows it says so."""
    from aix.agent.prompts import writer_prompt

    shows_a_fence = [
        name
        for name, value in vars(writer_prompt).items()
        if name.startswith("WRITER_SYSTEM_PROMPT_") and "```markdown" in value
    ]

    assert shows_a_fence, "the check would pass vacuously"
    for name in shows_a_fence:
        assert "never wrap your answer" in getattr(writer_prompt, name), name


# --- the source note, chosen by the code (2026-10-19) ---------------------------------

_LESSON_BODY = (
    "# Il ciclo dell'acqua\n\n## Attività (20 minuti)\n- Osservare.\n\n*Fonti:* Wikipedia."
)


@pytest.mark.parametrize(
    "written",
    [
        # the version for attached documents, with nothing attached
        "---\n⚠️ **Nota sulle fonti**: Il contenuto disciplinare proviene dai materiali "
        "allegati dalla docente\ne, dove indicato, da fonti esterne.",
        # a note of its own
        "---\n*Nota sulle fonti: la lezione è costruita sui riferimenti forniti dal docente.*",
    ],
)
def test_the_source_note_says_what_the_lesson_was_built_from(written):
    from aix.agent.validation.answer_shape import set_source_note, source_note

    note = source_note(has_material=False, domain="udl")

    out = set_source_note(f"{_LESSON_BODY}\n\n{written}\n", note)

    assert out.count("Nota sulle fonti") == 1 and out.rstrip().endswith(
        "(Universal Design for Learning)."
    )
    assert "allegati" not in out and "forniti dal docente" not in out
    assert out.startswith(_LESSON_BODY + "\n\n---\n⚠️")


def test_with_documents_attached_the_note_says_so():
    from aix.agent.validation.answer_shape import source_note

    assert "materiali allegati dalla docente" in source_note(has_material=True, domain="neuro")
    assert "fonti esterne\n(Wikipedia" in source_note(has_material=False, domain="neuro")


def test_an_answer_without_a_note_gets_none():
    from aix.agent.validation.answer_shape import set_source_note, source_note

    text = "Una definizione breve. La nota sulle fonti non serve qui."

    assert set_source_note(text, source_note(has_material=False)) == text
