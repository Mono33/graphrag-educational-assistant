"""What every lesson must look like, made true by code (blind read of 2026-10-20)."""

from __future__ import annotations

import pytest

from aix.agent.validation.lesson_shape import (
    drop_category_labels,
    ensure_header,
    keep_used_sources,
)
from aix.agent.validation.rules import rule_group_total

pytestmark = pytest.mark.unit

PROFILE = {
    "group": {"students_number": 22, "grade": "PRIMARIA"},
    "time_available_minutes": 60,
}

LESSON = """# Il ciclo dell'acqua

## Obiettivi
- Ricordare le fasi.

## Attività 1 – Esperimento (20 min)
- Compiti brevi (Focus Tasks): una consegna alla volta.
- Si valorizza l'errore come indizio (EmotionalRegulation).

---
*Fonti:* Knowledge Graph UDL: Focus Tasks, Glossaries, Station Rotation, Peer Tutoring; Wikipedia, «Ciclo dell'acqua» (2024), https://it.wikipedia.org/wiki/Ciclo_dell%27acqua.
"""


def test_the_header_comes_from_the_class_profile_when_the_writer_left_it_out():
    out = ensure_header(LESSON, PROFILE)

    assert out.startswith(
        "# Il ciclo dell'acqua\n\n**Livello:** Primaria · 22 alunni  \n**Durata:** 60 minuti\n"
    )
    assert ensure_header(out, PROFILE) == out


@pytest.mark.parametrize(
    "text",
    [
        LESSON.replace("# Il ciclo dell'acqua\n", "# Il ciclo dell'acqua\n**Livello:** quarta\n"),
        "# Una definizione\n\nLa condensazione è il passaggio da vapore ad acqua.",
    ],
)
def test_a_header_already_there_or_an_answer_that_is_not_a_lesson_is_left_alone(text):
    assert ensure_header(text, PROFILE) == text


def test_a_graph_category_leaves_the_body_and_a_strategy_s_name_stays():
    out = drop_category_labels(LESSON)

    assert "(EmotionalRegulation)" not in out and "(Focus Tasks)" in out
    assert "come indizio." in out


def test_the_sources_keep_the_graph_s_names_the_lesson_uses_and_every_other_source():
    out = keep_used_sources(LESSON)
    line = next(line for line in out.splitlines() if line.startswith("*Fonti:*"))

    assert "Focus Tasks" in line
    assert "Glossaries" not in line and "Station Rotation" not in line
    assert "Wikipedia, «Ciclo dell'acqua» (2024), https://it.wikipedia.org" in line


@pytest.mark.parametrize(
    ("line", "students", "flagged"),
    [
        # 2026-10-20, DeepSeek: 25 pupils for a class of 21.
        ("Gruppi da 4 (cinque gruppi) e un gruppo da 5, così tutti i 21 sono impegnati.", 21, True),
        ("10 coppie di 2 alunni, con un gruppo di 3 che lavora come osservatore.", 21, True),
        ("4 gruppi da 5 e 1 gruppo da 4 (24 alunni).", 24, False),
        ("gruppi da 5 (4 gruppi) e da 4 (1 gruppo)", 24, False),
        ("6 gruppi da 4 (24 alunni) più una coppia di osservatori.", 26, False),
        # Where one pupil goes is not a split of the class.
        ("2 gruppi da 5 e 4 gruppi da 4; lo studente è inserito in un gruppo da 4.", 26, False),
        ("I due alunni con DSA lavorano in coppia.", 22, False),
    ],
)
def test_the_groups_add_up_to_the_class(line, students, flagged):
    profile = {"group": {"students_number": students}}

    assert bool(rule_group_total(f"# L\n\n- {line}\n", profile)) is flagged
