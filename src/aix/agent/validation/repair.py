"""Fixing what the judge found, and only that (2026-10-19).

Until now a rejected lesson went back to the Writer whole: every revision
rewrote everything -- the parts that were wrong and the parts that were fine,
which could come back with errors of their own -- and paid for a whole lesson.
The judge (``validation.judge``) says where each problem is, quoting the
lesson. So the repair goes where the quote is:

* :func:`apply_replacements` -- a problem the judge marks «sostituzione» (an
  example, a date, an answer key: «ga-to» → «gat-to») is fixed by putting its
  fix in place of its quote. Code, no model.
* :func:`revise_units` -- a problem that needs more than that (roles that do not
  fit the groups) is sent to the model with only the block around it -- the
  activity, or the section -- and the corrected block goes back in its place.

Both are cautious, because a repair that breaks something else is worse than
the note it would have been:

* a replacement needs its quote to be in the lesson once, and not to cut an
  emphasis in half;
* a revised block must keep its title -- minutes included, so the lesson's
  timing still adds up -- and a sensible length; otherwise the block stays as it
  was and the problem becomes a note for the teacher.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from aix.agent.validation.judge import Issue

logger = logging.getLogger(__name__)

#: A fix much longer than its quote is not a replacement: it is a rewrite.
_MAX_FIX_CHARS = 400
_QUOTE_MAP = {"’": "'", "“": '"', "”": '"'}
_MINUTES = re.compile(r"(\d{1,3})\s*(?:min|′)", re.IGNORECASE)
_HEADING = re.compile(r"^#{1,6}\s")
#: A bold line standing alone, as the Writer titles its activities.
_BOLD_TITLE = re.compile(r"^\*\*[^*\n]+\*\*\s*$")


def repair_model() -> str:
    from aix.core.config import config as app_config

    return os.getenv("AIX_REPAIR_MODEL") or app_config.openai.model


def _flat_with_index(text: str) -> tuple[str, list[int]]:
    """``judge._flat`` of ``text``, with the original index of every kept character."""
    chars: list[str] = []
    index: list[int] = []
    space = True
    for i, ch in enumerate(text):
        if ch in "*_`":
            continue
        ch = _QUOTE_MAP.get(ch, ch).lower()
        if ch.isspace():
            if space:
                continue
            ch, space = " ", True
        else:
            space = False
        chars.append(ch)
        index.append(i)
    if chars and chars[-1] == " ":
        chars.pop()
        index.pop()
    return "".join(chars), index


def locate(quote: str, lesson: str) -> Optional[tuple[int, int]]:
    """Where ``quote`` is in ``lesson`` (start, end), or ``None`` when it is not
    there exactly once."""
    from aix.agent.validation.judge import _flat

    needle = _flat(quote)
    flat, index = _flat_with_index(lesson)
    if len(needle) < 3:
        return None
    first = flat.find(needle)
    if first < 0 or flat.find(needle, first + 1) >= 0:
        return None
    return index[first], index[first + len(needle) - 1] + 1


@dataclass
class RepairResult:
    lesson: str
    replaced: list[Issue] = field(default_factory=list)
    revised: list[Issue] = field(default_factory=list)
    #: Blocking problems the repair did not fix: they become notes.
    left: list[Issue] = field(default_factory=list)
    cost: float = 0.0


def apply_replacements(lesson: str, issues: list[Issue]) -> RepairResult:
    """Put each blocking «sostituzione» fix in place of its quote, where that is safe."""
    spans: list[tuple[int, int, Issue]] = []
    left: list[Issue] = []
    for issue in issues:
        if issue.severity != "bloccante":
            continue
        fix = issue.fix.strip()
        span = locate(issue.quote, lesson) if issue.kind == "sostituzione" else None
        safe = (
            span is not None
            and fix
            and fix != issue.quote.strip()
            and len(fix) <= max(_MAX_FIX_CHARS, 3 * len(issue.quote))
            # An emphasis cut in half would leave a stray «**» in the lesson.
            and lesson[span[0] : span[1]].count("**") % 2 == 0
            and not any(span[0] < end and start < span[1] for start, end, _ in spans)
        )
        if safe:
            spans.append((span[0], span[1], issue))
        else:
            left.append(issue)
    result = RepairResult(lesson=lesson, left=left)
    for start, end, issue in sorted(spans, key=lambda s: s[0], reverse=True):
        result.lesson = result.lesson[:start] + issue.fix.strip() + result.lesson[end:]
        result.replaced.append(issue)
    result.replaced.reverse()
    return result


def unit_bounds(lesson: str, position: int) -> tuple[int, int]:
    """The block around ``position``: from the title above it (a heading, or a bold
    line standing alone, as activities are titled) to the next one."""
    starts = [0]
    offset = 0
    for line in lesson.splitlines(keepends=True):
        stripped = line.strip()
        if _HEADING.match(stripped) or _BOLD_TITLE.match(stripped):
            starts.append(offset)
        offset += len(line)
    starts = sorted(set(starts))
    start = max(s for s in starts if s <= position)
    later = [s for s in starts if s > start]
    return start, (later[0] if later else len(lesson))


_REVISE_SYSTEM = """Sei l'autore di una lezione per una docente. Ricevi UN SOLO \
brano della lezione e i problemi trovati in quel brano da un revisore. Correggi \
quei problemi e nient'altro.

Regole:
- restituisci solo il brano corretto, in Markdown, senza commenti né spiegazioni;
- la prima riga (il titolo, con i suoi minuti) resta identica;
- cambia il meno possibile: stessi tempi, stessi materiali, stessi nomi delle \
strategie, stesso stile; non aggiungere sezioni;
- se un problema non si può correggere dentro questo brano, lascia il brano com'è."""

_REVISE_USER = """## Richiesta della docente
{request}

## Problemi da correggere in questo brano
{problems}

## Brano
{unit}"""


def _acceptable(old: str, new: str) -> bool:
    """A revised block that keeps its title and minutes, and a sensible length."""
    if not new.strip():
        return False
    old_title, new_title = old.strip().splitlines()[0], new.strip().splitlines()[0]
    if old_title.strip() != new_title.strip():
        return False
    if _MINUTES.findall(old_title) != _MINUTES.findall(new_title):
        return False
    return 0.5 * len(old) <= len(new) <= 2.0 * len(old) + 300


async def revise_units(
    lesson: str,
    issues: list[Issue],
    *,
    request: str = "",
    model: Optional[str] = None,
    client: Any = None,
) -> RepairResult:
    """Revise the block around each blocking problem; the rest of the lesson stays."""
    from aix.core.config import config as app_config
    from aix.core.config import reasoning_extra_body

    result = RepairResult(lesson=lesson)
    units: dict[tuple[int, int], list[Issue]] = {}
    for issue in issues:
        if issue.severity != "bloccante":
            continue
        span = locate(issue.quote, lesson)
        if span is None:
            result.left.append(issue)
            continue
        units.setdefault(unit_bounds(lesson, span[0]), []).append(issue)

    model = model or repair_model()
    extra = dict(reasoning_extra_body(model) or {})
    extra["usage"] = {"include": True}
    for (start, end), group in sorted(units.items(), key=lambda u: u[0][0], reverse=True):
        unit = result.lesson[start:end]
        problems = "\n".join(f"- «{i.quote}»: {i.why} Correzione: {i.fix}" for i in group)
        try:
            client = client or app_config.openai.get_async_client()
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _REVISE_SYSTEM},
                    {
                        "role": "user",
                        "content": _REVISE_USER.format(
                            request=request or "(non indicata)", problems=problems, unit=unit
                        ),
                    },
                ],
                max_tokens=2500,
                temperature=0.2,
                extra_body=extra,
            )
            new = (response.choices[0].message.content or "").strip()
            cost = getattr(getattr(response, "usage", None), "cost", None)
            result.cost += float(cost) if isinstance(cost, (int, float)) else 0.0
        except Exception as exc:  # noqa: BLE001 — the block stays as it was
            logger.warning("[repair] block not revised (%s: %s)", type(exc).__name__, exc)
            result.left.extend(group)
            continue
        if not _acceptable(unit, new):
            logger.info("[repair] revised block refused (title, minutes or length changed)")
            result.left.extend(group)
            continue
        # Keep the block's own spacing to the next one.
        trailing = unit[len(unit.rstrip()) :]
        result.lesson = result.lesson[:start] + new.rstrip() + trailing + result.lesson[end:]
        result.revised.extend(group)
    return result
