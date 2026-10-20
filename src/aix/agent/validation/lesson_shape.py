"""What every lesson must look like, made true by code rather than asked of the model.

The blind read of 2026-10-20 (Sonnet as Writer against every role on DeepSeek
V4.1 Flash) found DeepSeek close to Sonnet, and most of the gap in things a rule
can check: the header with class and duration missing in five lessons out of
eight, the sources listing every node retrieved (17 in one lesson) rather than
the ones used, and graph categories in the teacher's text («(EmotionalRegulation)»).
They hold for any Writer, so they are corrected here, after the draft, in
``write_node``: the header from the class profile, the sources cut to what the
body uses, the category labels taken out of the body.

Each correction is narrow: it changes the text only when it recognises the shape
it is fixing, and leaves it alone otherwise.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_TITLE = re.compile(r"^# .+$", re.MULTILINE)
#: A header line the Writer already wrote: «**Livello:** …», «**Classe:** …», «**Durata:** …».
_HEADER_LINE = re.compile(r"^\s*\*\*(?:Livello|Classe|Durata|Destinatari)\b", re.IGNORECASE)
_SECTION_MINUTES = re.compile(r"^#{2,4} .*\(\s*\d+\s*(?:min|′)", re.IGNORECASE | re.MULTILINE)


def _profile(profile: Any) -> dict[str, Any]:
    if hasattr(profile, "model_dump"):
        profile = profile.model_dump(mode="json")
    return profile if isinstance(profile, dict) else {}


def lesson_header(profile: Any) -> str:
    """«**Livello:** Primaria · 22 alunni» and «**Durata:** 60 minuti», from the profile."""
    from aix.api.schemas.educational_profile import GRADE_LABELS

    ep = _profile(profile)
    group = ep.get("group") or {}
    grade = GRADE_LABELS.get(str(getattr(group.get("grade"), "value", group.get("grade")) or ""))
    students = group.get("students_number")
    level = " · ".join(
        part
        for part in (
            grade,
            f"{students} alunni" if isinstance(students, int) and students else None,
        )
        if part
    )
    duration = ep.get("time_available_minutes") or ep.get("lesson_duration")
    lines = []
    if level:
        lines.append(f"**Livello:** {level}")
    if isinstance(duration, int) and duration > 0:
        lines.append(f"**Durata:** {duration} minuti")
    return "  \n".join(lines)


def ensure_header(lesson: str, profile: Any) -> str:
    """The class and the duration under the title, when the Writer left them out.

    Only for a lesson plan: a title and timed sections. An answer that is not a
    lesson, or a lesson whose first lines already state them, is left as it is.
    """
    title = _TITLE.search(lesson or "")
    header = lesson_header(profile)
    if not title or not header or not _SECTION_MINUTES.search(lesson):
        return lesson
    after = lesson[title.end() :].lstrip("\n").splitlines()[:8]
    if any(_HEADER_LINE.match(line) for line in after):
        return lesson
    logger.info("[lesson-shape] header added from the class profile")
    return f"{lesson[: title.end()]}\n\n{header}\n{lesson[title.end() :]}"


#: A graph category in brackets: «(EmotionalRegulation)», «(StructuredCollaboration)».
#: One word in CamelCase, never a strategy's name, which has spaces or is one word.
_CATEGORY = re.compile(r"\s*\(\s*\*?([A-Z][a-z]+(?:[A-Z][a-z]+)+)\*?\s*\)")


def drop_category_labels(lesson: str) -> str:
    """The graph's category names out of the lesson body; the sources keep them."""
    from aix.agent.kg_evidence import _SOURCES

    match = _SOURCES.search(lesson or "")
    body, rest = (lesson[: match.start()], lesson[match.start() :]) if match else (lesson, "")
    cleaned = _CATEGORY.sub("", body)
    if cleaned != body:
        logger.info("[lesson-shape] graph category labels removed from the body")
    return cleaned + rest


#: A sources line about the graph: «- *Knowledge Graph:* A; B», «*Fonti: Knowledge Graph — A, B*».
_KG_LINE = re.compile(
    r"^(?P<head>.*?Knowledge Graph[^:—–\n]*[:—–][*_]*\s*)(?P<names>.+)$", re.MULTILINE
)


#: Where the graph's names end in a sources line and another kind of source begins.
_OTHER_SOURCE = re.compile(r":|\b(?:19|20)\d\d\b|https?://|\.\s+\S")


def keep_used_sources(lesson: str) -> str:
    """The graph's names in the sources cut to those the lesson body uses.

    A name stays when the body names it (the Writer writes the graph's name in
    brackets after the Italian one). A line where no name would be left, or
    that cannot be read as a list, is left as it is: better a long list than a
    source removed by mistake.
    """
    from aix.agent.kg_evidence import _SOURCES, _in_lesson, _norm, lesson_body

    match = _SOURCES.search(lesson or "")
    if not match:
        return lesson
    body = _norm(lesson_body(lesson))
    head, sources = lesson[: match.start()], lesson[match.start() :]

    def used(item: str) -> bool:
        name = re.sub(r"\s*\([^)]*\)", "", item).strip(" *_.;,")
        # A phrase is not a node's name («nodi dell'ambiente di apprendimento …»).
        return len(name.split()) > 6 or _in_lesson(name, body)

    def cut(line: re.Match[str]) -> str:
        names = line.group("names")
        # The list ends where another kind of source begins -- a label («Video:»),
        # a year, a link, a new sentence -- and what follows is never cut: the
        # papers and videos after the graph's names in one line (2026-10-20).
        other = _OTHER_SOURCE.search(names)
        if other:
            start = names.rfind(";", 0, other.start())
            start = max(start, names.rfind(",", 0, other.start())) if start < 0 else start
            if start <= 0:
                return line.group(0)
            protected, names = names[start:], names[:start]
        else:
            protected = ""
        # The items with their separators, a comma inside brackets not being one.
        parts = re.split(r"(;\s*|,\s*(?![^()]*\)))", names)
        items, separators = parts[0::2], parts[1::2]
        # What closes the line (a full stop, an italic opened before the list)
        # stays whatever is cut: the last item's own emphasis is its own.
        tail = re.search(r"[*_.\s]*$", items[-1]).group(0)
        own = 1 if items[-1].lstrip()[:1] in "*_" and tail.strip(" .")[:1] in "*_" else 0
        closing = tail[own:] if own else tail
        items[-1] = items[-1][: len(items[-1]) - len(tail)] + tail[:own]
        keep = [used(i) for i in items]
        if len(items) < 3 or not any(keep) or all(keep):
            return line.group(0)
        out = ""
        for n, (item, ok) in enumerate(zip(items, keep, strict=True)):
            if ok:
                out += (separators[n - 1] if out and n else "") + item
        return line.group("head") + out.rstrip() + closing + protected

    trimmed = _KG_LINE.sub(cut, sources)
    if trimmed != sources:
        logger.info("[lesson-shape] sources cut to the graph's names the lesson uses")
    return head + trimmed


def shape_lesson(lesson: str, profile: Any) -> str:
    """All three, in the order that lets each see the text it expects."""
    return keep_used_sources(drop_category_labels(ensure_header(lesson, profile)))
