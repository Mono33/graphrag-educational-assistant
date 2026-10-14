"""A lesson read as an outline plus the sections a request is about."""

from __future__ import annotations

import pytest

from aix.webui.agent import lesson_outline as lo

pytestmark = pytest.mark.unit

LESSON = """<!-- ai-generated: true -->
# La Guerra Fredda

**Livello:** terza media · **Durata:** 50 minuti

## Obiettivi di apprendimento
- Spiegare la divisione dell'Europa

## Materiali
- Carta geografica dell'Europa del 1950

## Introduzione (10 min)
Domanda stimolo sul Muro di Berlino.

## Attività principali (30 min)
### Attività 1 — Le due Germanie
Lavoro a coppie sulle fonti.

## Valutazione
Exit ticket di tre domande.
"""


def test_the_sections_are_the_header_and_each_level_two_heading():
    sections = lo.split_sections(LESSON)

    assert [s.heading for s in sections] == [
        "Intestazione",
        "Obiettivi di apprendimento",
        "Materiali",
        "Introduzione (10 min)",
        "Attività principali (30 min)",
        "Valutazione",
    ]
    assert "### Attività 1" in sections[4].text, "a level-three heading stays inside its section"


def test_the_outline_has_one_line_per_section_with_its_first_line():
    text = lo.outline(lo.split_sections(LESSON))

    assert "- **Materiali** — Carta geografica dell'Europa del 1950" in text
    assert text.count("\n- **") == 6


def test_a_request_about_one_section_picks_it_with_header_and_objectives():
    sections = lo.split_sections(LESSON)

    chosen = lo.pick(sections, "rendi più interattiva la valutazione finale")

    assert [sections[i].heading for i in chosen] == [
        "Intestazione",
        "Obiettivi di apprendimento",
        "Valutazione",
    ]


def test_a_request_that_names_no_section_picks_none():
    """«Rendila più breve» is about the whole lesson: the caller sends it whole."""
    assert lo.pick(lo.split_sections(LESSON), "rendila più breve") == []


def test_the_left_out_sections_are_named():
    sections = lo.split_sections(LESSON)

    text = lo.chosen_sections(sections, [0, 5])

    assert "Exit ticket" in text
    assert "Materiali" in text and "non sono riportate per intero" in text


def test_a_lesson_without_sections_is_one_header():
    assert [s.heading for s in lo.split_sections("Solo testo.")] == ["Intestazione"]
