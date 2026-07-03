"""
Lesson display helpers (CORE 2 #6.6 P5 — warm-academic brand pass).

Pure formatters for turning a ``Lesson`` ORM row into the template-friendly
dicts consumed by the Dashboard / Library / Workspace templates. Lives in
the lessons subpackage (alongside ``Lesson`` model) so both ``aix.webui.routes``
(home/dashboard) and ``aix.webui.lessons.routes`` (library/workspace) can
import from here without circular dependencies.

Nothing here touches the database, runs an agent, or mutates state — these
are read-only formatters. Adding / changing a status label is a single-edit
operation in :data:`STATUS_DISPLAY`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from aix.webui.lessons.models import Lesson

# ---------------------------------------------------------------------------
# Status → display tuple
# ---------------------------------------------------------------------------
#
# Each entry maps a ``Lesson.status`` enum value to a (label, pill_class, dot_class)
# triple used across the brand pages:
#
#     - label      : Italian copy shown inside the pill / table cell
#     - pill_class : .aix-pill-* utility class for the badge styling
#     - dot_class  : .aix-dot-* class for the small status dot in compact rows
#
# Keep this in lockstep with the lesson lifecycle defined in
# ``aix.webui.lessons.models``. Adding a new status only requires a new entry.
# ---------------------------------------------------------------------------
STATUS_DISPLAY: dict[str, tuple[str, str, str]] = {
    "draft": ("Bozza", "aix-pill-warning", "aix-dot-warning"),
    "running": ("In corso", "aix-pill-info", "aix-dot-info"),
    "complete": ("✓ Completata", "aix-pill-success", "aix-dot-success"),
    "error": ("Errore", "aix-pill-error", "aix-dot-error"),
}


def status_display(status: str) -> tuple[str, str, str]:
    """Look up the display triple for a lesson status (safe default)."""
    return STATUS_DISPLAY.get(
        status or "",
        ("—", "aix-pill-neutral", "aix-dot-neutral"),
    )


# ---------------------------------------------------------------------------
# Date / time formatters (Italian)
# ---------------------------------------------------------------------------

_MONTHS_IT = [
    "gen",
    "feb",
    "mar",
    "apr",
    "mag",
    "giu",
    "lug",
    "ago",
    "set",
    "ott",
    "nov",
    "dic",
]

_DAYS_IT = [
    "Lunedì",
    "Martedì",
    "Mercoledì",
    "Giovedì",
    "Venerdì",
    "Sabato",
    "Domenica",
]

_MONTHS_FULL_IT = [
    "gennaio",
    "febbraio",
    "marzo",
    "aprile",
    "maggio",
    "giugno",
    "luglio",
    "agosto",
    "settembre",
    "ottobre",
    "novembre",
    "dicembre",
]


def short_date_it(dt: Optional[datetime]) -> str:
    """Render a datetime as a compact Italian label (e.g. '10 mag')."""
    if dt is None:
        return ""
    return f"{dt.day} {_MONTHS_IT[dt.month - 1]}"


def full_date_it(dt: Optional[datetime]) -> str:
    """Render as 'DD/MM/YYYY HH:MM' — the format used on Library cards."""
    if dt is None:
        return ""
    return dt.strftime("%d/%m/%Y %H:%M")


def today_label_it() -> str:
    """Render today as e.g. 'Mercoledì 10 maggio' for the dashboard greeting."""
    today = datetime.now()
    return f"{_DAYS_IT[today.weekday()]} {today.day} {_MONTHS_FULL_IT[today.month - 1]}"


def relative_time_it(dt: Optional[datetime]) -> str:
    """Render a datetime as 'N min/ore/giorni fa' or 'oggi alle HH:MM'."""
    if dt is None:
        return ""
    now = datetime.now(timezone.utc)
    # Coerce naive datetimes (SQLite-stored UTC) into aware UTC so the diff works.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = now - dt
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return "pochi secondi fa"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min fa"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} or{'a' if hours == 1 else 'e'} fa"
    days = hours // 24
    if days < 7:
        return f"{days} giorn{'o' if days == 1 else 'i'} fa"
    return short_date_it(dt)


# ---------------------------------------------------------------------------
# EducationalProfile JSON → template-friendly summary
# ---------------------------------------------------------------------------


def profile_summary(profile: dict[str, Any] | None) -> dict[str, Any]:
    """
    Pull a small, template-friendly subset out of the ``educational_profile_json``
    blob so templates don't have to walk a deeply nested dict for each row.

    Tolerant to missing keys — older lesson rows from earlier phases may have
    a partial profile. Anything missing renders as ``''`` / ``[]``.
    """
    profile = profile or {}
    group = profile.get("group") or {}
    disabilities = group.get("disabilities") or []
    if isinstance(disabilities, str):
        disabilities = [disabilities]
    class_features = group.get("class_features") or []
    if isinstance(class_features, str):
        class_features = [class_features]
    return {
        "subject_area": profile.get("subject_area") or "",
        "specific_topic": profile.get("specific_topic") or "",
        "duration_min": profile.get("time_available_minutes"),
        "group_title": group.get("title") or "",
        "students_number": group.get("students_number"),
        "disabilities": [d for d in disabilities if d],
        "class_features": [c for c in class_features if c],
    }


# ---------------------------------------------------------------------------
# Lesson → row dict
# ---------------------------------------------------------------------------


def lesson_to_row(lesson: Lesson) -> dict[str, Any]:
    """
    Shape a ``Lesson`` row into the dict consumed by Dashboard + Library
    templates. One representation, two surfaces.

    Returned shape (every key always present, '' / [] / None for missing):
        id              UUID
        title           str        — fallback to "Lezione senza titolo"
        domain          str        — neuro / udl / all
        subject         str
        topic           str
        duration_min    int | None
        group_title     str
        students_number int | None
        disabilities    list[str]
        class_features  list[str]
        status          str        — raw lifecycle enum
        status_label    str        — Italian display ("Bozza", "✓ Completata", ...)
        pill_class      str        — .aix-pill-* class
        dot_class       str        — .aix-dot-* class
        created_at      datetime
        updated_at      datetime
        created_full    str        — DD/MM/YYYY HH:MM
        updated_full    str
        updated_short   str        — "10 mag"
        updated_rel     str        — "2 ore fa"
    """
    label, pill_cls, dot_cls = status_display(lesson.status)
    summary = profile_summary(lesson.educational_profile_json)
    return {
        "id": lesson.id,
        "title": lesson.title or "Lezione senza titolo",
        "domain": lesson.domain or "",
        "subject": summary["subject_area"],
        "topic": summary["specific_topic"],
        "duration_min": summary["duration_min"],
        "group_title": summary["group_title"],
        "students_number": summary["students_number"],
        "disabilities": summary["disabilities"],
        "class_features": summary["class_features"],
        "status": lesson.status or "",
        "status_label": label,
        "pill_class": pill_cls,
        "dot_class": dot_cls,
        "created_at": lesson.created_at,
        "updated_at": lesson.updated_at,
        "created_full": full_date_it(lesson.created_at),
        "updated_full": full_date_it(lesson.updated_at),
        "updated_short": short_date_it(lesson.updated_at),
        "updated_rel": relative_time_it(lesson.updated_at),
    }


# ---------------------------------------------------------------------------
# Activity feed derivation (Dashboard "Attività recente")
# ---------------------------------------------------------------------------


def activity_event_for_lesson(lesson: Lesson) -> dict[str, Any]:
    """
    Synthesise an "Attività recente" row from a Lesson's lifecycle. This is
    purely derived from ``status`` + ``updated_at`` — no event log table is
    needed in this brand pass; a real activity feed lands later.
    """
    if lesson.status == "draft":
        verb = "creata in bozza"
    elif lesson.status == "running":
        verb = "in elaborazione"
    elif lesson.status == "complete":
        verb = "approvata dal critic"
    elif lesson.status == "error":
        verb = "interrotta con errore"
    else:
        verb = "aggiornata"
    title = lesson.title or "Lezione senza titolo"
    return {
        "lesson_id": lesson.id,
        "rel": relative_time_it(lesson.updated_at),
        "description": f"Lezione {title} {verb}",
    }
