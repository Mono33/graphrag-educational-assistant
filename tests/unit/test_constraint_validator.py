"""Deterministic feasibility rules (Q1 · 1.3).

The lesson that prompted this scored 91% from the Critic while asking
twenty-five device-less students to use Kahoot on their phones, in a room whose
desks do not move. Every rule below turns one of those into something countable.

Each rule is tested twice: a lesson that violates it, and one that does not in
the same classroom. Without the second half a rule that fires on everything
would pass.
"""

from __future__ import annotations

import pytest

from aix.agent.validation import validate_lesson
from aix.agent.validation.models import BLOCKING, WARNING, ValidationReport, Violation
from aix.agent.validation.rules import section_minutes

pytestmark = pytest.mark.unit

#: Devices: none. Desks: fixed. WiFi: none. Twenty-five students, sixty minutes.
STRICT = {
    "group": {"students_number": 25, "disabilities": ["ADHD"]},
    "classroom": {
        "own_device": "NO",
        "has_wifi": False,
        "has_lim": True,
        "pc_station": False,
        "forniture_mobility": "NO",
    },
    "time_available_minutes": 60,
}

GOOD_LESSON = """# Memoria di lavoro

**Durata:** 60 minuti

## Introduzione *(10 minuti)*
Domanda di attivazione alla lavagna.

## Attività Principali
### Fase 1 — I Do *(20 minuti)*
Spiegazione con schede cartacee.
### Fase 2 — We Do *(20 minuti)*
Lavoro in gruppi da 4 con fogli stampati.

## Valutazione *(10 minuti)*
### Exit ticket (5 min)
Su foglietto di carta.
"""


def _rules(lesson: str, profile=STRICT) -> set[str]:
    return {v.rule for v in validate_lesson(lesson, profile).violations}


# ---------------------------------------------------------------------------
# A lesson that fits the classroom raises nothing
# ---------------------------------------------------------------------------


def test_a_feasible_lesson_passes_cleanly():
    report = validate_lesson(GOOD_LESSON, STRICT)

    assert report.passed
    assert report.violations == []


def test_an_empty_draft_is_not_a_violation():
    """Nothing to check yet is not the same as something wrong."""
    assert validate_lesson("", STRICT).passed
    assert validate_lesson("   \n ", STRICT).passed


# ---------------------------------------------------------------------------
# Minutes
# ---------------------------------------------------------------------------


def test_a_heading_with_minutes_absorbs_the_headings_under_it():
    """Summing every number in the document would exceed the lesson."""
    claimed = section_minutes(GOOD_LESSON)

    assert [minutes for _title, minutes in claimed] == [10, 20, 20, 10]
    assert sum(m for _t, m in claimed) == 60, "the exit ticket's 5 is inside the 10"


def test_timings_that_do_not_add_up_block_the_lesson():
    short = GOOD_LESSON.replace("*(20 minuti)*", "*(5 minuti)*")

    violations = validate_lesson(short, STRICT).blocking
    assert [v.rule for v in violations] == ["duration_mismatch"]
    assert "30" in violations[0].remediation and "60" in violations[0].remediation


def test_rounding_within_ten_percent_is_fine():
    """55 against a requested 60 is not a lesson anyone would call broken."""
    close = GOOD_LESSON.replace("## Valutazione *(10 minuti)*", "## Valutazione *(5 minuti)*")

    assert "duration_mismatch" not in _rules(close)


def test_without_a_requested_duration_there_is_nothing_to_check():
    assert "duration_mismatch" not in _rules(GOOD_LESSON, {"classroom": {}})


def test_english_headings_are_read_too():
    english = """# Memory
## Introduction (10 minutes)
## Main Activities
### Activity 1 (25 minutes)
### Activity 2 (25 minutes)
"""
    assert sum(m for _t, m in section_minutes(english)) == 60


# ---------------------------------------------------------------------------
# The classroom
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "Gli studenti rispondono a un quiz Kahoot dal proprio smartphone.",
        "Ogni studente apre Mentimeter sul telefono.",
        "Scansiona il codice QR per aprire il sondaggio.",
    ],
)
def test_an_activity_that_needs_a_student_device_is_blocked(line):
    report = validate_lesson(GOOD_LESSON + "\n" + line, STRICT)

    assert "byod_off" in {v.rule for v in report.blocking}
    assert any(line[:20] in v.evidence for v in report.blocking)


def test_the_same_activity_is_fine_when_the_students_have_devices():
    allowed = {**STRICT, "classroom": {**STRICT["classroom"], "own_device": "YES"}}
    lesson = GOOD_LESSON + "\nGli studenti rispondono a un quiz Kahoot dal proprio smartphone."

    assert "byod_off" not in _rules(lesson, allowed)


def test_an_activity_that_needs_the_internet_is_blocked_without_wifi():
    lesson = GOOD_LESSON + "\nCiascun gruppo fa una ricerca su internet sul tema."

    assert "no_wifi" in _rules(lesson)


def test_rearranging_the_desks_is_blocked_when_they_do_not_move():
    lesson = GOOD_LESSON + "\nDisporre i banchi a isole da quattro."

    assert "furniture_static" in _rules(lesson)


def test_rearranging_is_allowed_when_the_furniture_moves():
    movable = {**STRICT, "classroom": {**STRICT["classroom"], "forniture_mobility": "YES"}}

    assert "furniture_static" not in _rules(GOOD_LESSON + "\nDisporre i banchi a isole.", movable)


# ---------------------------------------------------------------------------
# Warnings: worth saying, not worth regenerating over
# ---------------------------------------------------------------------------


def test_oversized_groups_warn_rather_than_block():
    lesson = GOOD_LESSON.replace("gruppi da 4", "gruppi da 8")
    report = validate_lesson(lesson, STRICT)

    assert [v.rule for v in report.warnings] == ["group_size"]
    assert report.passed, "a warning must not send the lesson back to the Writer"


def test_groups_of_five_are_still_acceptable():
    assert "group_size" not in _rules(GOOD_LESSON.replace("gruppi da 4", "gruppi da 5"))


def test_projection_without_a_projector_only_warns():
    """'slide' may well be the teacher's own material on paper."""
    blind = {**STRICT, "classroom": {**STRICT["classroom"], "has_lim": False}}
    report = validate_lesson(GOOD_LESSON + "\nProietta la slide riassuntiva.", blind)

    assert [v.rule for v in report.warnings] == ["no_projection"]
    assert report.passed


# ---------------------------------------------------------------------------
# The regression sentinel for the profile renderer
# ---------------------------------------------------------------------------


def test_an_undeclared_bes_acronym_blocks_the_lesson():
    """If raw codes go back into the prompt, this fails before a teacher notices."""
    lesson = GOOD_LESSON + "\nPer gli studenti DCGM prevedere una scheda semplificata."

    assert "bes_label_drift" in _rules(lesson)


def test_a_declared_need_may_be_named():
    lesson = GOOD_LESSON + "\nPer gli studenti con ADHD, timer visivo sempre attivo."

    assert "bes_label_drift" not in _rules(lesson)


def test_two_letter_codes_are_not_hunted():
    """``PD`` and ``SA`` match too much ordinary text to block a lesson over."""
    lesson = GOOD_LESSON + "\nLa SA di riferimento e il PD del gruppo."

    assert "bes_label_drift" not in _rules(lesson)


# ---------------------------------------------------------------------------
# The report, and what the retry reads from it
# ---------------------------------------------------------------------------


def test_the_retry_instruction_names_only_what_blocks():
    lesson = GOOD_LESSON.replace("gruppi da 4", "gruppi da 9") + "\nUsate Kahoot dal telefono."
    prompt = validate_lesson(lesson, STRICT).remediation_prompt()

    assert "Kahoot" in prompt
    assert "gruppi da 9" not in prompt, "a warning does not belong in the retry"


def test_a_clean_report_asks_for_nothing():
    assert validate_lesson(GOOD_LESSON, STRICT).remediation_prompt() == ""


def test_violations_survive_a_round_trip_through_the_checkpointer():
    """They live in AgentState, which is serialised."""
    report = validate_lesson(GOOD_LESSON + "\nUsate Kahoot dal telefono.", STRICT)
    rebuilt = ValidationReport([Violation.from_dict(d) for d in report.as_dicts()])

    assert rebuilt.blocking == report.blocking


def test_a_broken_rule_loses_its_rule_not_the_lesson(monkeypatch):
    """The validator sits between a finished draft and the teacher."""
    from aix.agent.validation import validator as validator_module

    def _explodes(_lesson, _profile):
        raise ValueError("regola rotta")

    monkeypatch.setattr(validator_module, "ALL_RULES", (_explodes,))

    assert validate_lesson(GOOD_LESSON, STRICT).passed


@pytest.mark.parametrize("severity", [BLOCKING, WARNING])
def test_severities_are_the_two_the_graph_branches_on(severity):
    assert severity in ("blocking", "warning")
