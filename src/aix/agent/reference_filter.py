"""Scientific references that cannot support a lesson, dropped before the Writer sees them.

The curated media and the outside search return papers found from the graph's
concept names, not from the lesson. On 2026-10-16 (``scripts/eval/quality_ab.py``,
read blind) lessons for primary and middle school cited recognition memory in
rodents, a study of ChatGPT, motivation in college students and postsecondary
STEM -- and the curated ones came with «Includi questi riferimenti nella sezione
Fonti». A reference that does not back a choice of the lesson is better left out
before writing than flagged after (``docs/product/Specifica_Risposta_Mediana.md``
§2.7).

Only what is plainly about something else is dropped -- animal studies, language
models, and, for a school class, university or adult learners. Anything doubtful
stays: the Writer is told to cite a reference only where it supports a choice.
"""

from __future__ import annotations

import re
from typing import Any, Optional

#: About animals or machines, not pupils.
_NOT_PUPILS = re.compile(
    r"\b(?:rats?|mice|mouse|rodents?|monkeys?|primates?|zebrafish|animal model|in vivo"
    r"|chatgpt|gpt-\d|large language models?|llms?)\b",
    re.IGNORECASE,
)
#: About learners past school.
_PAST_SCHOOL = re.compile(
    r"\b(?:college|undergraduates?|university students?|postsecondary|post-secondary"
    r"|higher education|adult learners?|graduate students?|medical students?"
    r"|nursing students?|employees?|workplace)\b",
    re.IGNORECASE,
)
#: Grades where a study of learners past school says nothing about the class.
SCHOOL_GRADES = frozenset({"INFANZIA", "PRIMARIA", "SECONDARIA_I_GRADO", "SECONDARIA_II_GRADO"})


def _grade(profile: Any) -> str:
    if hasattr(profile, "model_dump"):
        profile = profile.model_dump(mode="json")
    group = (profile or {}).get("group") if isinstance(profile, dict) else None
    grade = (group or {}).get("grade") if isinstance(group, dict) else None
    return str(getattr(grade, "value", grade) or "").upper()


def off_topic(reference: dict[str, Any], profile: Any = None) -> Optional[str]:
    """Why ``reference`` cannot support this lesson, or ``None`` when it may."""
    text = " ".join(
        str(reference.get(key) or "") for key in ("title", "journal", "venue", "abstract")
    )
    if _NOT_PUPILS.search(text):
        return "not about pupils"
    grade = _grade(profile)
    if (not grade or grade in SCHOOL_GRADES) and _PAST_SCHOOL.search(text):
        return "about learners past school"
    return None


def keep_references(items: Any, profile: Any = None) -> list[dict[str, Any]]:
    """``items`` without the references :func:`off_topic` rules out."""
    return [item for item in items or [] if isinstance(item, dict) and not off_topic(item, profile)]
