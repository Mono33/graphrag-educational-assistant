"""The judge's plumbing, without a model (2026-10-19).

What a model makes of a lesson is measured by ``scripts/eval/check_judge.py``;
these pin what the code does with its answer: problems without a quote that is
in the lesson are not counted, a malformed answer is a reported error, and the
judge never raises.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from aix.agent.validation.judge import CHECKS, Issue, judge_lesson, parse_issues, quote_in

pytestmark = pytest.mark.unit

LESSON = """# Le sillabe

## Attività 1 (10 minuti)
- Divisione in sillabe alla lavagna: «ga-to», «ca-sa».
- Per distinguere la *p* dalla *b*, guardate da che parte è la pancia.
"""


class _Client:
    def __init__(self, answer: str, fail: bool = False):
        self.answer, self.fail = answer, fail
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        if self.fail:
            raise RuntimeError("provider down")
        usage = SimpleNamespace(prompt_tokens=1200, completion_tokens=300, cost=0.012)
        message = SimpleNamespace(content=self.answer)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def _answer(*issues):
    return json.dumps({"issues": list(issues)}, ensure_ascii=False)


@pytest.mark.parametrize(
    "quote",
    [
        "Divisione in sillabe alla lavagna: «ga-to», «ca-sa».",
        "per distinguere la p dalla b, guardate da che parte è la pancia",  # emphasis, case
        "Divisione in sillabe alla lavagna: … «ca-sa».",  # the model's own ellipsis
    ],
)
def test_a_quote_is_found_whatever_the_emphasis_and_spacing(quote):
    assert quote_in(quote, LESSON)


def test_a_quote_that_is_not_in_the_lesson_is_not_found():
    assert not quote_in("Divisione in sillabe: «gat-to»", LESSON)
    assert not quote_in("", LESSON)


async def test_a_problem_without_its_quote_in_the_lesson_is_not_counted():
    client = _Client(
        _answer(
            {
                "check": "esempi_lingua",
                "severity": "bloccante",
                "quote": "«ga-to»",
                "why": "gatto si divide gat-to",
                "fix": "«gat-to»",
            },
            {
                "check": "fatti",
                "severity": "bloccante",
                "quote": "L'acqua diventa aria quando evapora.",
                "why": "invented",
                "fix": "-",
            },
        )
    )

    report = await judge_lesson(LESSON, client=client, model="anthropic/claude-sonnet-5.5")

    assert [i.check for i in report.issues] == ["esempi_lingua"]
    assert [i.quote for i in report.unquoted] == ["L'acqua diventa aria quando evapora."]
    assert report.blocking and report.cost == 0.012 and report.input_tokens == 1200


async def test_a_malformed_answer_or_a_failure_is_reported_never_raised():
    unreadable = await judge_lesson(LESSON, client=_Client("non so"), model="x/y")
    down = await judge_lesson(LESSON, client=_Client("", fail=True), model="x/y")

    assert unreadable.error and not unreadable.issues
    assert "provider down" in down.error


def test_unknown_checks_and_severities_are_normalised():
    issues = parse_issues(
        _answer({"check": "stile", "severity": "grave", "quote": "q", "why": "w", "fix": "f"})
    )

    assert (issues[0].check, issues[0].severity) == ("altro", "minore")
    assert "esercizi" in CHECKS and "sicurezza" in CHECKS


def test_a_wrong_fact_is_blocking_whatever_the_model_says():
    """«l'acqua si trasforma in aria» came back «minore» and stayed in the lesson."""
    issues = parse_issues(
        _answer(
            {"check": "fatti", "severity": "minore", "quote": "q", "why": "w", "fix": "f"},
            {"check": "classe", "severity": "minore", "quote": "q", "why": "w", "fix": "f"},
        )
    )

    assert [i.severity for i in issues] == ["bloccante", "minore"]


async def test_each_change_gets_its_verdict_and_a_failure_keeps_them_all():
    from aix.agent.validation.judge import verify_changes
    from aix.agent.validation.repair import Change

    issue = Issue("esempi_lingua", "bloccante", "«ga-to»", "gatto si divide gat-to", "«gat-to»")
    changes = [Change("«ga-to»", "«gat-to»", [issue]), Change("1792", "1798", [issue])]
    answer = json.dumps(
        {"verdicts": [{"id": 1, "verdict": "ok"}, {"id": 2, "verdict": "errata", "why": "1789"}]}
    )

    verdict = await verify_changes(changes, client=_Client(answer), model="x/y")
    down = await verify_changes(changes, client=_Client("", fail=True), model="x/y")

    assert verdict.ok == [True, False] and verdict.reasons[1] == "1789"
    assert down.ok == [True, True] and "provider down" in down.error


def test_a_change_is_shown_to_the_verifier_as_the_whole_line_it_sits_in():
    from aix.agent.validation.judge import in_lines

    lesson = "# T\n- l'insegnante mostra il vapore e mostra il vapore e le gocce.\n- Fine.\n"

    before, after = in_lines("fa toccare il coperchio", "mostra il vapore e le gocce", lesson)

    assert after == "- l'insegnante mostra il vapore e mostra il vapore e le gocce."
    assert before == "- l'insegnante mostra il vapore e fa toccare il coperchio."
    assert in_lines("a", "assente", lesson) == ("a", "assente")
