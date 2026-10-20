"""Fixing what the judge found, and only that (2026-10-19)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aix.agent.validation.judge import Issue
from aix.agent.validation.repair import (
    apply_replacements,
    locate,
    revise_units,
    unit_bounds,
)

pytestmark = pytest.mark.unit

LESSON = """# Le sillabe

## Introduzione (10 minuti)
- Alla lavagna: la divisione in sillabe di **gatto** è «ga-to».
- Ripetiamo insieme.

## Attività (40 minuti)

**Attività 1 – Coppie di suoni (15 min)**
- Gruppi da 4 con 5 ruoli: lettore, scrittore, controllore, portavoce, cronometrista.
- Coppie di parole: «foglia/moglie».

**Attività 2 – Dettato (25 min)**
- Dettato di dieci parole.

## Valutazione (10 minuti)
- Osservazione.
"""


def _issue(quote, fix, kind="sostituzione", severity="bloccante", why="sbagliato"):
    return Issue(check="esempi_lingua", severity=severity, quote=quote, why=why, fix=fix, kind=kind)


def test_a_quote_is_located_whatever_the_emphasis():
    start, end = locate("la divisione in sillabe di gatto è «ga-to»", LESSON)

    assert LESSON[start:end] == "la divisione in sillabe di **gatto** è «ga-to»"


def test_an_ambiguous_or_absent_quote_is_not_located():
    assert locate("gatto", LESSON + "\nAncora gatto.") is None
    assert locate("«gat-to»", LESSON) is None


def test_a_replacement_changes_the_quote_and_nothing_else():
    result = apply_replacements(LESSON, [_issue("«ga-to»", "«gat-to»")])

    assert "«gat-to»" in result.lesson and "«ga-to»" not in result.lesson
    assert result.lesson.replace("«gat-to»", "«ga-to»") == LESSON
    assert [i.fix for i in result.replaced] == ["«gat-to»"] and not result.left


@pytest.mark.parametrize(
    "issue",
    [
        _issue("«foglia/moglie»", "x" * 900),  # a rewrite, not a replacement
        _issue("sillabe di **gatto", "sillabe di gatto"),  # cuts an emphasis in half
        _issue("«foglia/moglie»", "«foglia/foca»", kind="revisione"),  # the judge said revise
    ],
)
def test_an_unsafe_replacement_is_not_made(issue):
    result = apply_replacements(LESSON, [issue])

    assert result.lesson == LESSON
    assert result.left == [issue]


def test_a_minor_problem_is_corrected_too():
    """2026-10-20: the corrections made in the lesson, not proposed under it."""
    result = apply_replacements(LESSON, [_issue("«ga-to»", "«gat-to»", severity="minore")])

    assert "«gat-to»" in result.lesson and not result.left


@pytest.mark.parametrize(
    ("line", "quote", "fix"),
    [
        # 2026-10-19: half a sentence quoted, the whole sentence written as the fix.
        (
            "- Mostrare il video (5 minuti) oppure, se il tempo non basta, presentare "
            "l'idea in due frasi: la capacità cresce.",
            "Mostrare il video (5 minuti) oppure",
            "Guardare il video prima e presentare agli alunni l'idea in due frasi:",
        ),
        (
            "- con la pentola calda, l'insegnante mostra il vapore e fa toccare il coperchio "
            "freddo. Gli alunni ripetono.",
            "fa toccare il coperchio freddo",
            "mostra il vapore e le gocce sul coperchio tenuto dall'insegnante",
        ),
        (
            "- Spiega che il cervello legge le lettere e che richiede tempo: è normale fare "
            "fatica.",
            "Spiega che il cervello legge le lettere e che richiede tempo",
            "Spiega che il cervello collega le lettere ai suoni, sempre più veloce.",
        ),
    ],
)
def test_a_fix_that_would_break_its_line_goes_to_the_block_revision(line, quote, fix):
    lesson = LESSON + line + "\n"

    result = apply_replacements(lesson, [_issue(quote, fix)])

    assert result.lesson == lesson and [i.quote for i in result.left] == [quote]


def test_a_fix_that_fits_its_line_is_made():
    lesson = LESSON + "- La Bastiglia fu presa nel 1792, dal popolo di Parigi.\n"

    result = apply_replacements(lesson, [_issue("nel 1792", "nel 1789")])

    assert "fu presa nel 1789, dal popolo" in result.lesson


def test_a_block_is_the_activity_around_the_problem():
    start, end = unit_bounds(LESSON, LESSON.index("5 ruoli"))

    assert LESSON[start:end].startswith("**Attività 1 – Coppie di suoni (15 min)**")
    assert "Dettato" not in LESSON[start:end]


def test_the_last_block_stops_where_the_sources_begin():
    """A revision of «Differenziazione» once dropped the whole «Fonti» section."""
    lesson = LESSON + "\n---\n*Fonti:*\n- *Knowledge Graph:* Chunking.\n"

    start, end = unit_bounds(lesson, lesson.index("Osservazione"))

    assert lesson[start:end].startswith("## Valutazione") and "Fonti" not in lesson[start:end]


class _Client:
    def __init__(self, answer: str):
        self.answer, self.calls = answer, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        usage = SimpleNamespace(cost=0.001)
        message = SimpleNamespace(content=self.answer)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


ROLES = _issue(
    "Gruppi da 4 con 5 ruoli",
    "quattro ruoli per gruppi da 4",
    kind="revisione",
    why="5 ruoli, 4 alunni",
)


async def test_a_revised_block_goes_back_in_its_place_and_the_rest_stays():
    fixed = (
        "**Attività 1 – Coppie di suoni (15 min)**\n"
        "- Gruppi da 4 con 4 ruoli: lettore, scrittore, controllore, portavoce.\n"
        "- Coppie di parole: «foglia/moglie»."
    )
    client = _Client(fixed)

    result = await revise_units(LESSON, [ROLES], model="x/y", client=client)

    assert "4 ruoli: lettore, scrittore, controllore, portavoce." in result.lesson
    assert result.lesson.split("**Attività 1")[0] == LESSON.split("**Attività 1")[0]
    assert result.lesson.split("**Attività 2")[1] == LESSON.split("**Attività 2")[1]
    assert result.revised == [ROLES] and not result.left
    # Only the block went to the model, not the lesson.
    sent = client.calls[0]["messages"][1]["content"]
    assert "Dettato" not in sent and "Introduzione" not in sent


@pytest.mark.parametrize(
    "answer",
    [
        "**Attività 1 – Coppie di suoni (20 min)**\n- Gruppi da 4 con 4 ruoli.",  # minutes changed
        "**Attività uno**\n- Gruppi da 4 con 4 ruoli.",  # title changed
        "",  # nothing
    ],
)
async def test_a_revised_block_that_changes_its_title_or_minutes_is_refused(answer):
    result = await revise_units(LESSON, [ROLES], model="x/y", client=_Client(answer))

    assert result.lesson == LESSON
    assert result.left == [ROLES]
