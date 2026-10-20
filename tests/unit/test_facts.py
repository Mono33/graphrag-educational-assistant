"""What can be computed is computed, and given to the judge as a fact (2026-10-20)."""

from __future__ import annotations

import pytest

from aix.agent.validation.facts import (
    arithmetic_facts,
    calculate,
    count_facts,
    fraction_facts,
    syllabifications,
    syllable_facts,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("word", "split"),
    [
        ("gatto", "gat-to"),
        ("mamma", "mam-ma"),
        ("pasta", "pa-sta"),
        ("capra", "ca-pra"),
        ("foglia", "fo-glia"),
        ("bagno", "ba-gno"),
        ("pesce", "pe-sce"),
        ("acqua", "ac-qua"),
        ("ombrello", "om-brel-lo"),
        ("zucchero", "zuc-che-ro"),
        ("inverno", "in-ver-no"),
        ("paese", "pa-e-se"),
    ],
)
def test_italian_words_split_by_the_rules(word, split):
    assert split in syllabifications(word)


def test_a_vowel_pair_that_may_be_either_is_accepted_both_ways():
    assert {"scuo-la", "scu-o-la"} <= syllabifications("scuola")


def test_a_wrong_split_is_a_fact_with_its_fix_and_the_word_it_belongs_to():
    lesson = "# Le sillabe\n- Alla lavagna, la divisione in sillabe di «gatto»: «ga-to»."

    [fact] = syllable_facts(lesson)

    assert fact.wrong and fact.quote == "ga-to" and fact.fix == "gat-to"


@pytest.mark.parametrize(
    "line",
    [
        "- Esempi corretti: «ca-sa», «pa-sta», «fo-glia», «scuo-la».",
        "- Il gesto suono-segno e il ritmo passo-passo, con un warm-up iniziale.",
    ],
)
def test_right_splits_and_compound_words_are_left_alone(line):
    assert not [f for f in syllable_facts(f"# Le sillabe\n{line}") if f.wrong]


def test_a_quiz_with_the_wrong_key_and_a_false_equivalence_are_facts():
    lesson = (
        "- Domanda: «Quale frazione è equivalente a 2/3? a) 3/4 b) 4/6 c) 6/8». "
        "Risposta corretta: c).\n"
        "- Far notare che 1/2 e 2/3 sono equivalenti.\n"
        "- Bonus: 2/6 con 4/12; 1/2 è uguale a 2/4."
    )

    facts = fraction_facts(lesson)
    wrong = [f for f in facts if f.wrong]

    assert [f.quote for f in wrong] == ["Risposta corretta: c)", "1/2 e 2/3"]
    assert wrong[0].fix == "Risposta corretta: b)"
    assert sum(not f.wrong for f in facts) == 2


def test_operations_are_computed_whole():
    facts = arithmetic_facts("Tempi: 15 + 15 + 10 = 40 minuti. Divisione: 15 : 3 = 6.")

    assert [(f.wrong, f.fix) for f in facts] == [(False, ""), (True, "15 : 3 = 5")]


def test_the_calculator_reads_numbers_and_operators_only():
    assert calculate("2 + 3 × 4") == 14
    assert calculate("__import__('os')") is None


def test_the_groups_are_a_fact_against_the_class():
    profile = {"group": {"students_number": 21}}

    [fact] = count_facts("- Gruppi da 4 (cinque gruppi) e un gruppo da 5.", profile)

    assert fact.wrong and "25 alunni" in fact.statement
