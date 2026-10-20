"""What can be computed is computed: facts the judge is given rather than asked to work out.

A model doing sums or splitting syllables in its head gets them wrong now and then,
and says so with confidence: «ga-to» for «gatto», «1/2 e 2/3 sono equivalenti», a
quiz on fractions whose answer key is wrong (the error bench, 2026-10-19). Here the
code checks what has one right answer and hands the judge the result as a fact:

* **syllables** -- the Italian rules (double consonants split, s + consonant and
  consonant + l/r go together, ch gh gn gl sc inseparable), only in a lesson that
  talks about syllables; where a vowel pair may be a diphthong or a hiatus both
  splits are accepted, so only a certain mistake is reported;
* **fractions** -- an equivalence stated in the text, and the answer key of a quiz
  «Quale frazione è equivalente a …?»;
* **arithmetic** -- an operation written with «=»;
* **minutes and groups** -- the validator's own counts (``validation.rules``).

A fact is ``wrong`` or confirms the text; with a ``fix`` the code knows the
correction. ``judge_lesson`` puts the facts in the judge's prompt and adds, as a
problem, a wrong fact the judge did not report. Nothing here calls a model.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Optional


@dataclass
class Fact:
    """One computed check: where it is in the lesson, what is true, how to fix it."""

    check: str
    quote: str
    statement: str
    wrong: bool
    fix: str = ""


# ---------------------------------------------------------------------------
# Syllables
# ---------------------------------------------------------------------------

_VOWELS = set("aeiouàèéìíòóùú")
_STRONG = set("aeoàèéòó")
_ACCENTED_WEAK = set("ìíùú")
#: What may begin a syllable besides one consonant.
_ONSETS = {
    "bl", "br", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "pl", "pr", "tr", "vr",
    "ch", "gh", "gn", "sc", "qu", "gu",
    "sch", "sgh", "scr", "spr", "str", "spl", "sbr", "sdr", "sfr", "sgr", "squ",
}  # fmt: skip


def _valid_onset(cluster: str) -> bool:
    if len(cluster) <= 1:
        return True
    if cluster in _ONSETS:
        return True
    # s + one consonant: «pa-sta», «ca-sco».
    return len(cluster) == 2 and cluster[0] == "s" and cluster[1] not in _VOWELS


def _units(word: str) -> list[str]:
    """Letters, with a glide written as part of its consonant: «gi» in «gioco», «qu»."""
    units: list[str] = []
    i = 0
    while i < len(word):
        ch = word[i]
        nxt = word[i + 1] if i + 1 < len(word) else ""
        after = word[i + 2] if i + 2 < len(word) else ""
        # «ci», «gi», «sci», «gli» before a vowel: the i is a sign, not a vowel.
        if ch in "cg" and nxt == "i" and after in _VOWELS:
            units.append(ch + "i")
            i += 2
            continue
        # «glia», «glio»: after «gl» the i is a sign too.
        if ch == "l" and units and units[-1] == "g" and nxt == "i" and after in _VOWELS:
            units.append("li")
            i += 2
            continue
        if ch in "qg" and nxt == "u" and after in _VOWELS:
            units.append(ch + "u")
            i += 2
            continue
        units.append(ch)
        i += 1
    return units


def _is_vowel(unit: str) -> bool:
    return unit in _VOWELS


def syllabifications(word: str) -> set[str]:
    """Every acceptable split of ``word`` into syllables, joined with «-»."""
    word = word.lower()
    units = _units(word)
    # Mandatory split points (between units i-1 and i) and optional ones.
    mandatory: set[int] = set()
    optional: set[int] = set()
    vowel_positions = [i for i, u in enumerate(units) if _is_vowel(u)]
    for left, right in zip(vowel_positions, vowel_positions[1:], strict=False):
        cluster_units = units[left + 1 : right]
        if not cluster_units:
            # Two vowels in a row: a hiatus splits, a diphthong does not.
            a, b = units[left], units[right]
            if (a in _STRONG and b in _STRONG) or a in _ACCENTED_WEAK or b in _ACCENTED_WEAK:
                mandatory.add(right)
            else:
                optional.add(right)
            continue
        # The longest valid onset, never splitting a double consonant on one side.
        for k in range(len(cluster_units)):
            onset_units = cluster_units[k:]
            plain = "".join(u[0] for u in onset_units)
            if len(plain) >= 2 and plain[0] == plain[1]:
                continue  # «gat-to»: a double consonant always splits
            if _valid_onset(plain) or (len(onset_units) == 1):
                mandatory.add(left + 1 + k)
                break
        else:
            mandatory.add(right)
    results = set()
    optional_list = sorted(optional)
    for chosen in itertools.product([False, True], repeat=len(optional_list)):
        points = mandatory | {p for p, on in zip(optional_list, chosen, strict=True) if on}
        pieces, last = [], 0
        for point in sorted(points):
            pieces.append("".join(units[last:point]))
            last = point
        pieces.append("".join(units[last:]))
        results.add("-".join(p for p in pieces if p))
    return results


_HYPHENATED = re.compile(r"(?<![\w-])([a-zàèéìíòóùú]{1,5}(?:-[a-zàèéìíòóùú]{1,5})+)(?![\w-])")
_WORDS = re.compile(r"[a-zàèéìíòóùú]{3,}", re.IGNORECASE)


def _close(a: str, b: str) -> bool:
    """One letter apart: «gato» for «gatto»."""
    if abs(len(a) - len(b)) != 1:
        return False
    short, long_ = sorted((a, b), key=len)
    return any(long_[:i] + long_[i + 1 :] == short for i in range(len(long_)))


def syllable_facts(lesson: str) -> list[Fact]:
    """Syllable splits written in a lesson about syllables, checked."""
    if "sillab" not in lesson.lower():
        return []
    facts = []
    for line in lesson.splitlines():
        for match in _HYPHENATED.finditer(line):
            token = match.group(1)
            joined = token.replace("-", "")
            pieces = token.split("-")
            if len(joined) < 3 or len(pieces) < 2:
                continue
            # A syllable split, not a compound («suono-segno», «warm-up»): short
            # pieces with one vowel group each, quoted or in a line about syllables.
            if any(len(p) > 4 or len(re.findall(r"[aeiouàèéìíòóùú]+", p)) != 1 for p in pieces):
                continue
            if re.search(r"[wkyjx]", joined):  # «warm-up»: not an Italian word
                continue
            around = line[max(0, match.start() - 1) : match.end() + 1]
            if not (re.match(r"[«\"“'*]", around) or "sillab" in line.lower()):
                continue
            words = {w.lower() for w in _WORDS.findall(line)} - {joined}
            target = next((w for w in words if _close(w, joined)), joined)
            accepted = syllabifications(target)
            if token in accepted:
                continue
            right = sorted(accepted, key=len)[0]
            facts.append(
                Fact(
                    check="esempi_lingua",
                    quote=token,
                    statement=f"la divisione in sillabe di «{target}» è «{right}», non «{token}»",
                    wrong=True,
                    fix=right,
                )
            )
    return facts


# ---------------------------------------------------------------------------
# Fractions and arithmetic
# ---------------------------------------------------------------------------

_FRACTION = r"(\d{1,3})\s*/\s*(\d{1,3})"
#: «1/2 e 2/3 sono equivalenti», «1/2 è uguale a 2/4», «2/6 = 4/12», «(es. 2/6 con 4/12)».
_EQUIVALENCE = re.compile(
    rf"{_FRACTION}\s*(?:=|e|è uguale a|equivale a|con)\s*{_FRACTION}", re.IGNORECASE
)
_QUIZ = re.compile(rf"equivalente\s+a\s+{_FRACTION}", re.IGNORECASE)
_OPTION = re.compile(rf"\b([a-d])\)\s*{_FRACTION}")
_KEY = re.compile(r"(?:risposta(?:\s+corretta)?|soluzione)\s*:?\s*([a-d])\)", re.IGNORECASE)


def _frac(n: str, d: str) -> Optional[Fraction]:
    return Fraction(int(n), int(d)) if int(d) else None


def fraction_facts(lesson: str) -> list[Fact]:
    """Equivalences stated in the text, and the answer key of a fraction quiz."""
    facts = []
    for line in lesson.splitlines():
        quiz = _QUIZ.search(line)
        if quiz:
            target = _frac(*quiz.groups())
            options = {o: _frac(n, d) for o, n, d in _OPTION.findall(line)}
            key = _KEY.search(line)
            right = [o for o, f in options.items() if f is not None and f == target]
            if key and options and target is not None:
                given = key.group(1).lower()
                ok = given in right
                fix = key.group(0).replace(f"{given})", f"{right[0]})") if right else ""
                facts.append(
                    Fact(
                        check="esercizi",
                        quote=key.group(0),
                        statement=(
                            f"tra le opzioni, è equivalente a {quiz.group(1)}/{quiz.group(2)} "
                            + (", ".join(f"{o})" for o in right) if right else "nessuna")
                            + ("" if ok else f": la risposta indicata {given}) è sbagliata")
                        ),
                        wrong=not ok,
                        fix="" if ok else fix,
                    )
                )
            continue
        if not re.search(r"equivalent|uguale|equivale|=", line, re.IGNORECASE):
            continue
        negated = bool(re.search(r"\bnon\s+(?:sono|è)\s+(?:equivalent|ugual)", line, re.I))
        for match in _EQUIVALENCE.finditer(line):
            a, b = _frac(match.group(1), match.group(2)), _frac(match.group(3), match.group(4))
            if a is None or b is None:
                continue
            equal = a == b
            claimed = not negated
            left = f"{match.group(1)}/{match.group(2)}"
            right = f"{match.group(3)}/{match.group(4)}"
            facts.append(
                Fact(
                    check="fatti",
                    quote=match.group(0),
                    statement=(
                        f"{left} e {right} {'sono' if equal else 'non sono'} equivalenti "
                        f"({left} = {_say(float(a))}, {right} = {_say(float(b))})"
                    ),
                    wrong=equal != claimed,
                )
            )
    return facts


_NUMBER = r"\d+(?:[.,]\d+)?"
#: A whole expression and its result: «15 + 15 + 10 = 40», «3 × 4 = 12».
_OPERATION = re.compile(
    rf"(?<![\w/.,])"
    rf"(?P<expr>{_NUMBER}(?:\s*[+\-−×x*:÷]\s*{_NUMBER})+)\s*=\s*(?P<result>{_NUMBER})(?![\w/]|[.,]\d)"
)
_TO_PYTHON = str.maketrans({"−": "-", "×": "*", "x": "*", ":": "/", "÷": "/", ",": "."})


def calculate(expression: str) -> Optional[float]:
    """The value of an arithmetic expression, or ``None``: numbers and + - × : only.

    Parsed, never run: a lesson's text is not code.
    """
    import ast
    import operator

    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def value(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](value(node.left), value(node.right))
        raise ValueError("not arithmetic")

    try:
        return value(ast.parse(expression.translate(_TO_PYTHON), mode="eval").body)
    except (ValueError, SyntaxError, ZeroDivisionError):
        return None


def _num(text: str) -> float:
    return float(text.replace(",", "."))


def _say(value: float) -> str:
    return str(int(value)) if value == int(value) else f"{value:.4g}".replace(".", ",")


def arithmetic_facts(lesson: str) -> list[Fact]:
    """Each operation written with «=»: the result in the text against the computed one."""
    facts = []
    for match in _OPERATION.finditer(lesson):
        expression, written = match.group("expr"), match.group("result")
        result = calculate(expression)
        if result is None:
            continue
        ok = abs(result - _num(written)) < 1e-9
        facts.append(
            Fact(
                check="esercizi",
                quote=match.group(0),
                statement=f"{expression} = {_say(result)}",
                wrong=not ok,
                fix="" if ok else f"{expression} = {_say(result)}",
            )
        )
    return facts


# ---------------------------------------------------------------------------
# Minutes and groups: the validator's counts, as facts
# ---------------------------------------------------------------------------


def count_facts(lesson: str, profile: Any) -> list[Fact]:
    """The section minutes against the duration, the groups against the class."""
    from aix.agent.validation.rules import _pupils_in, _sentences, section_minutes

    if hasattr(profile, "model_dump"):
        profile = profile.model_dump(mode="json")
    profile = profile if isinstance(profile, dict) else {}
    facts = []
    requested = profile.get("time_available_minutes") or profile.get("lesson_duration")
    sections = section_minutes(lesson)
    if isinstance(requested, int) and requested > 0 and sections:
        total = sum(minutes for _title, minutes in sections)
        title = sections[0][0]
        facts.append(
            Fact(
                check="classe",
                quote=title,
                statement=f"i minuti delle sezioni sommano {total}, la durata richiesta è {requested}",
                wrong=abs(total - requested) > max(1, round(requested * 0.10)),
            )
        )
    students = (profile.get("group") or {}).get("students_number")
    if isinstance(students, int) and students > 0:
        for line in _sentences(lesson):
            pupils = _pupils_in(line)
            if pupils >= 0.7 * students:
                facts.append(
                    Fact(
                        check="classe",
                        quote=line.strip()[:200],
                        statement=f"i gruppi di questa riga contano {pupils} alunni, la classe ne ha {students}",
                        wrong=pupils != students,
                    )
                )
    return facts


def computed_facts(lesson: str, profile: Any = None) -> list[Fact]:
    """Every computed check on ``lesson``."""
    return (
        syllable_facts(lesson)
        + fraction_facts(lesson)
        + arithmetic_facts(lesson)
        + count_facts(lesson, profile)
    )


def render_facts(facts: list[Fact]) -> str:
    """The facts as the judge reads them: certain, not to be worked out again."""
    lines = []
    for fact in facts:
        mark = "ERRATO" if fact.wrong else "corretto"
        fix = f" → correzione: «{fact.fix}»" if fact.wrong and fact.fix else ""
        lines.append(f"- [{mark}] «{fact.quote}»: {fact.statement}{fix}")
    return "\n".join(lines)
