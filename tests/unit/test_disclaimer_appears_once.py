"""The source disclaimer is printed once, whatever the model does.

Two instructions in the Writer's prompt each claimed the last position --
«Output order: ... -> `*Fonti:*` footer» and «at the end of EVERY response,
include this disclaimer» -- and the model satisfied both the only way it could:
writing the disclaimer before the footer and again after it. A teacher's lesson
carried it twice, a few lines apart, once with the pedagogy references appended
and once bare (2026-09-22).

The prompt was fixed. This pins the net under it, because the same class of
defect -- two instructions asking for one thing twice -- produced two lists of
sources earlier the same week, after being asked away once already.
"""

from __future__ import annotations

import pytest

from aix.webui.materials.citations import once_only_disclaimer

pytestmark = pytest.mark.unit

NOTE = (
    "⚠️ **Nota sulle fonti**: Il contenuto disciplinare (argomento specifico) "
    "proviene da fonti esterne e non è stato verificato dal Knowledge Graph FEM."
)


def test_the_second_copy_goes_and_the_first_stays():
    lesson = f"## Attività\n\nIl corpo della lezione.\n\n{NOTE} (van de Pol, 2010)\n\n*Fonti:* Wikipedia\n\n---\n{NOTE}\n"

    out = once_only_disclaimer(lesson)

    assert out.count("Nota sulle fonti") == 1
    assert "van de Pol" in out, "the copy written in context is the one kept"
    assert "*Fonti:*" in out, "the footer between them is not collateral damage"
    assert "Il corpo della lezione." in out


def test_one_copy_is_left_exactly_as_it_is():
    lesson = f"Una lezione.\n\n*Fonti:* Wikipedia\n\n---\n{NOTE}\n"

    assert once_only_disclaimer(lesson) == lesson


def test_a_lesson_without_the_disclaimer_is_untouched():
    """Not every intent ends on it, and this function does not add one."""
    lesson = "Una risposta breve, senza footer.\n"

    assert once_only_disclaimer(lesson) == lesson
    assert once_only_disclaimer("") == ""


def test_three_copies_still_leave_one():
    lesson = f"Testo.\n\n{NOTE}\n\nAltro.\n\n{NOTE}\n\nFine.\n\n{NOTE}\n"

    out = once_only_disclaimer(lesson)

    assert out.count("Nota sulle fonti") == 1
    assert "Altro." in out and "Fine." in out, "only the disclaimer is removed"


def test_it_runs_where_the_lesson_is_rendered():
    """The guard is worth nothing if the render path does not use it."""
    from aix.webui.materials.citations import with_sources

    lesson = f"Una lezione.\n\n{NOTE}\n\n---\n{NOTE}\n"

    assert with_sources(lesson).count("Nota sulle fonti") == 1
