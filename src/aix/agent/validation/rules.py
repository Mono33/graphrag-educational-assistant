"""The rules themselves. Each one is a pure function of (lesson, profile).

Every rule answers a question with a countable answer. None of them asks whether
the lesson is *good* — that is the Critic's job, and mixing the two is what
produced a 91% score on a lesson four fifths of which could not be run.

Two limits are worth stating plainly, because they decide which rules block and
which only warn:

* **There is no tool registry yet** (it is 1.2, out of scope for the test
  milestone). So "does this activity need a student device" is answered by a
  curated list of names and phrases, not by a lookup. The list is deliberately
  narrow — a tool that is actually named, or an explicit mention of a personal
  device — so that a false positive needs the lesson to literally say it.
* **The lesson is a Markdown string**, not a structure. Minutes are read from
  headings with a regex, the way ``retriever_agent._parse_lesson_minutes``
  already reads the requested duration. In Fase 3 both become lookups on a
  document model and these regexes go away.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from aix.agent.validation.models import BLOCKING, WARNING, Violation

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

#: Tools that cannot run without a device in each student's hands. Named tools
#: only: the point is to catch "students open Kahoot on their phones", not to
#: guess at every activity that might involve a screen.
_STUDENT_DEVICE_TOOLS = (
    "kahoot",
    "mentimeter",
    "menti.com",
    "padlet",
    "wooclap",
    "quizizz",
    "socrative",
    "nearpod",
    "edpuzzle",
    "jamboard",
    "google moduli",
    "google forms",
    "plickers",
    "slido",
)

#: Explicit mentions of a personal device, in both languages the Writer uses.
_STUDENT_DEVICE_PHRASES = (
    "smartphone",
    "cellulare",
    "telefono",
    "proprio dispositivo",
    "propri dispositivi",
    "own device",
    "their phones",
    "qr code",
    "codice qr",
    "scansiona il codice",
    "tablet personale",
)

#: Activities that cannot happen without a connection.
_ONLINE_PHRASES = (
    "online",
    "in rete",
    "sul web",
    "ricerca in internet",
    "ricerca su internet",
    "cerca su internet",
    "streaming",
    "https://",
    "http://",
)

#: Moving the room around.
_FURNITURE_PHRASES = (
    "disporre i banchi",
    "disporre le sedie",
    "spostare i banchi",
    "spostare le sedie",
    "riorganizzare l'aula",
    "riorganizzare i banchi",
    "in cerchio",
    "a isole",
    "isole di banchi",
    "rearrange the desks",
    "move the desks",
    "in a circle",
)

#: Needs something to project onto.
_PROJECTION_PHRASES = ("proietta", "proiezione", "alla lim", "sulla lim", "project", "slide")

#: BES codes long enough to be unambiguous. ``PD``, ``SA`` and ``DF`` are left
#: out on purpose: two letters match too much ordinary text to be worth a
#: blocking rule, and a false block is more expensive than a missed one.
_BES_CODES = ("DSA", "ADHD", "DOP", "DCGL", "DCGM", "DCGS", "DLDS")

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_MINUTES = re.compile(r"(\d{1,3})\s*(?:min\b|min\.|minut|minute)", re.IGNORECASE)
_GROUP_SIZE = re.compile(r"gruppi?\s+(?:da|di)\s+(\d{1,2})|groups?\s+of\s+(\d{1,2})", re.IGNORECASE)


def _sentences(text: str) -> list[str]:
    """Lines, which in a Markdown lesson are the unit a teacher would quote."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def _find(text: str, needles: tuple[str, ...]) -> list[tuple[str, str]]:
    """Every (line, needle) where a line mentions one of ``needles``."""
    hits = []
    for line in _sentences(text):
        lowered = line.lower()
        for needle in needles:
            if needle in lowered:
                hits.append((line, needle))
                break
    return hits


def _trim(line: str, limit: int = 160) -> str:
    clean = re.sub(r"\s+", " ", line).strip()
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


# ---------------------------------------------------------------------------
# Minutes
# ---------------------------------------------------------------------------


def section_minutes(lesson: str) -> list[tuple[str, int]]:
    """The minutes each section claims, without counting any of them twice.

    A lesson nests: "Main activities" holds three activities with their own
    timings, and "Assessment (8 minutes)" holds an exit ticket that says 5. So a
    heading that states its own duration absorbs everything under it, and only
    headings that state nothing are descended into. Summing every number in the
    document instead would produce a total larger than the lesson.
    """
    claimed: list[tuple[str, int]] = []
    skip_deeper_than: Optional[int] = None

    for line in lesson.splitlines():
        match = _HEADING.match(line.strip())
        if not match:
            continue
        level, title = len(match.group(1)), match.group(2).strip()

        if skip_deeper_than is not None and level > skip_deeper_than:
            continue
        skip_deeper_than = None

        minutes = _MINUTES.search(title)
        if minutes:
            claimed.append((title, int(minutes.group(1))))
            skip_deeper_than = level

    return claimed


def rule_duration_mismatch(lesson: str, profile: dict[str, Any]) -> list[Violation]:
    """The section timings have to add up to the duration the teacher asked for.

    Ten per cent either way: the Writer rounds, and a lesson of 55 minutes
    against a requested 60 is not a lesson anyone would call broken.
    """
    requested = profile.get("time_available_minutes") or profile.get("lesson_duration")
    if not isinstance(requested, int) or requested <= 0:
        return []

    sections = section_minutes(lesson)
    if not sections:
        return []

    total = sum(minutes for _title, minutes in sections)
    if abs(total - requested) <= max(1, round(requested * 0.10)):
        return []

    breakdown = ", ".join(f"{title} = {minutes}′" for title, minutes in sections)
    direction = "superano" if total > requested else "non raggiungono"
    return [
        Violation(
            severity=BLOCKING,
            rule="duration_mismatch",
            evidence=_trim(breakdown, 220),
            remediation=(
                f"I tempi delle sezioni sommano {total} minuti e {direction} i "
                f"{requested} richiesti: ribilancia le durate perché il totale sia "
                f"esattamente {requested} minuti."
            ),
        )
    ]


# ---------------------------------------------------------------------------
# The classroom
# ---------------------------------------------------------------------------


def rule_byod_off(lesson: str, profile: dict[str, Any]) -> list[Violation]:
    """No activity may need a device the students do not have."""
    room = profile.get("classroom") or {}
    if str(room.get("own_device") or "") != "NO":
        return []

    violations = []
    for line, needle in _find(lesson, _STUDENT_DEVICE_TOOLS + _STUDENT_DEVICE_PHRASES):
        violations.append(
            Violation(
                severity=BLOCKING,
                rule="byod_off",
                evidence=_trim(line),
                remediation=(
                    f"Gli studenti non hanno dispositivi personali: sostituisci "
                    f"l'attività che richiede «{needle}» con una che funzioni "
                    f"senza dispositivi in mano agli studenti."
                ),
            )
        )
    return violations


def rule_no_wifi(lesson: str, profile: dict[str, Any]) -> list[Violation]:
    """No activity may depend on a connection that is not there."""
    room = profile.get("classroom") or {}
    if room.get("has_wifi") is not False:
        return []

    violations = []
    for line, needle in _find(lesson, _ONLINE_PHRASES + _STUDENT_DEVICE_TOOLS):
        violations.append(
            Violation(
                severity=BLOCKING,
                rule="no_wifi",
                evidence=_trim(line),
                remediation=(
                    f"In aula non c'è connessione: sostituisci l'attività che "
                    f"dipende da «{needle}» con una utilizzabile offline."
                ),
            )
        )
    return violations


def rule_furniture_static(lesson: str, profile: dict[str, Any]) -> list[Violation]:
    """Desks that cannot move cannot be rearranged."""
    room = profile.get("classroom") or {}
    if str(room.get("forniture_mobility") or "") != "NO":
        return []

    return [
        Violation(
            severity=BLOCKING,
            rule="furniture_static",
            evidence=_trim(line),
            remediation=(
                "Gli arredi non sono spostabili: riscrivi l'attività perché "
                "funzioni con i banchi nella disposizione attuale."
            ),
        )
        for line, _needle in _find(lesson, _FURNITURE_PHRASES)
    ]


def rule_no_projection(lesson: str, profile: dict[str, Any]) -> list[Violation]:
    """Nothing to project onto, and nothing to project from.

    A warning rather than a block: "slide" in a lesson may be the teacher's own
    material shown on paper, and this one cannot tell the difference.
    """
    room = profile.get("classroom") or {}
    if room.get("has_lim") is not False or room.get("pc_station") is not False:
        return []

    return [
        Violation(
            severity=WARNING,
            rule="no_projection",
            evidence=_trim(line),
            remediation=(
                "L'aula non ha LIM né postazioni PC: verifica che il materiale "
                "sia utilizzabile senza proiezione."
            ),
        )
        for line, _needle in _find(lesson, _PROJECTION_PHRASES)
    ]


def rule_group_size(lesson: str, profile: dict[str, Any]) -> list[Violation]:
    """Cooperative groups above five stop being cooperative.

    A warning: the number in the text may be describing something other than a
    group, and rewriting a lesson over it would be the wrong trade.
    """
    violations = []
    for line in _sentences(lesson):
        match = _GROUP_SIZE.search(line)
        if not match:
            continue
        size = int(match.group(1) or match.group(2))
        if size > 5:
            violations.append(
                Violation(
                    severity=WARNING,
                    rule="group_size",
                    evidence=_trim(line),
                    remediation=(
                        f"Gruppi da {size} sono troppo grandi per il lavoro "
                        f"cooperativo: portali a un massimo di 5 membri."
                    ),
                )
            )
    return violations


def rule_bes_label_drift(lesson: str, profile: dict[str, Any]) -> list[Violation]:
    """No BES acronym in the lesson that the teacher did not declare.

    This is the regression sentinel for the profile renderer. If someone puts
    raw codes back into the prompt, the model starts echoing them, and this
    rule fails before a teacher has to notice.
    """
    declared = {
        str(getattr(code, "value", code)).upper()
        for code in ((profile.get("group") or {}).get("disabilities") or [])
    }

    # Nothing declared means there is nothing to drift *from*. The rule exists to
    # catch raw codes leaking back into the prompt for a class whose needs the
    # teacher described; with no profile it blocked ordinary Italian — a lesson
    # about inclusive teaching cannot mention DSA at all (found 2026-09-15).
    if not declared:
        return []

    violations = []
    for code in _BES_CODES:
        if code in declared:
            continue
        pattern = re.compile(rf"\b{code}\b")
        for line in _sentences(lesson):
            if pattern.search(line):
                violations.append(
                    Violation(
                        severity=BLOCKING,
                        rule="bes_label_drift",
                        evidence=_trim(line),
                        remediation=(
                            f"La sigla «{code}» non è fra i bisogni dichiarati dalla "
                            f"docente: togli il riferimento o usa la formulazione estesa "
                            f"dei bisogni effettivamente presenti."
                        ),
                    )
                )
                break
    return violations


#: Every rule, in the order their violations are reported.
ALL_RULES = (
    rule_duration_mismatch,
    rule_byod_off,
    rule_no_wifi,
    rule_furniture_static,
    rule_no_projection,
    rule_group_size,
    rule_bes_label_drift,
)
