"""The lesson as an outline plus the sections a request is about: progressive disclosure.

A light turn used to receive the whole lesson (up to 60k characters) whatever the
request. Most requests are about one part of it -- "amplia la fase We Do",
"aggiungi una verifica" -- so the model gets the outline of every section (heading
and first line) and only the full text of the sections the request is about. The
outline is the same on every turn of the lesson, so it can be cached; the chosen
sections change with the request.

The choice is lexical and cheap: no model call. When nothing in the request
points to a section ("rendila più breve"), the whole lesson is sent, as before.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Words that say nothing about which section is meant.
_STOP = frozenset(
    """a ad al alla alle anche che chi ci col come con da dal dalla dei del della delle di
    e ed fa fai gli ha ho i il in io la le lei li lo ma mi ne nei nel nella nelle noi non o
    per piu più po poi puoi qui se si sia su sul sulla sono ti tra tu un una uno vorrei
    voglio puoi potresti fammi fare dammi aggiungi rendi metti togli lezione questa questo
    quella quello sezione parte the and of to for with""".split()
)

_SECTION_START = re.compile(r"^## ", re.M)


@dataclass(frozen=True)
class Section:
    heading: str
    text: str  # the whole section, heading included

    @property
    def first_line(self) -> str:
        for line in self.text.splitlines()[1:]:
            line = line.strip(" -*#>\t")
            if line:
                return line[:140]
        return ""


def split_sections(markdown: str) -> list[Section]:
    """``##`` sections in order; what comes before the first one is the "Intestazione"."""
    md = markdown or ""
    starts = [m.start() for m in _SECTION_START.finditer(md)]
    sections: list[Section] = []
    head = md[: starts[0]] if starts else md
    if head.strip():
        sections.append(Section("Intestazione", head.strip()))
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(md)
        text = md[start:end].strip()
        sections.append(Section(text.splitlines()[0][3:].strip(), text))
    return sections


def _stems(text: str) -> set[str]:
    words = re.findall(r"[a-zàèéìòù]{3,}", (text or "").lower())
    return {w[:5] for w in words if w not in _STOP}


def pick(sections: list[Section], request: str, k: int = 2) -> list[int]:
    """Indexes of the sections the request is about; empty when it points to none.

    A word of the request in a heading weighs three times a word in the body.
    The header (title, level, duration) and the objectives always come along.
    """
    asked = _stems(request)
    scored = []
    for i, s in enumerate(sections):
        score = 3 * len(asked & _stems(s.heading)) + len(asked & _stems(s.text))
        scored.append((score, i))
    chosen = [i for score, i in sorted(scored, reverse=True)[:k] if score > 0]
    if not chosen:
        return []
    keep = {
        i
        for i, s in enumerate(sections)
        if s.heading == "Intestazione" or "obiettiv" in s.heading.lower()
    }
    return sorted(keep | set(chosen))


def outline(sections: list[Section]) -> str:
    """One line per section: heading and its first line."""
    lines = ["## La lezione: scaletta (ogni sezione e la sua prima riga)", ""]
    for s in sections:
        lines.append(
            f"- **{s.heading}** — {s.first_line}" if s.first_line else f"- **{s.heading}**"
        )
    return "\n".join(lines) + "\n"


def chosen_sections(sections: list[Section], indexes: list[int]) -> str:
    """The full text of the chosen sections, and the names of those left out."""
    left_out = [s.heading for i, s in enumerate(sections) if i not in indexes]
    body = "\n\n".join(sections[i].text for i in indexes)
    note = (
        "\n\n(Le altre sezioni non sono riportate per intero: "
        + "; ".join(left_out)
        + ". Se per rispondere ti serve il loro testo, dillo nella risposta.)"
        if left_out
        else ""
    )
    return "## Le sezioni della lezione che riguardano la richiesta\n\n" + body + note + "\n"
