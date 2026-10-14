"""Live subject resources for a lesson: papers, Wikipedia, videos (Phase 1b; shared 2026-10-14).

The WebUI's right panel loads them after the lesson, off the generation path
(``GET /webui/lesson/{id}/media-live``); the API returns the same panel for a
session (``GET /api/v1/sessions/{id}/media``). One function decides what is
searched and how it is ranked, so the two show the same resources.

Best effort by design: anything missing or failing gives an empty panel, never
an error.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from aix.webui.lessons.models import Lesson

logger = logging.getLogger(__name__)


def _media_live_enabled() -> bool:
    """True when the dynamic live-media layer is switched on (Phase 1b).

    Cheap, env-only check (no network). Defaults to False so the media panel
    renders byte-identically to before unless explicitly enabled.
    """
    try:
        from aix.agent.media import MediaConfig

        return MediaConfig.from_env().live_enabled
    except Exception:
        return False


def _lesson_has_live_media_context(lesson: Lesson | None) -> bool:
    """True when a lesson has a real teacher query/run context for live media.

    Live media is intended to enrich *a specific teacher request*, not to fire
    just because a draft profile has a subject/topic filled in. This keeps the
    initial draft panel aligned with Angelo's original behavior and only starts
    the dynamic layer after a run/query exists.
    """
    if lesson is None:
        return False
    if lesson.status == "draft":
        return False
    query = (getattr(lesson, "teacher_query", None) or "").strip()
    return bool(query)


# Bound the lesson-content slice we feed the Phase 3 ranker (token-cost guard).
_RANK_CONTENT_MAX_CHARS = 2000

# Phase 3.1 — strip a leading "crea una lezione su …" style instruction so the
# topical remainder ("disturbi da deficit di attenzione") is what we search,
# not the imperative. Conservative: applied at most once, with a length guard.
_LESSON_QUERY_LEAD_IN_RE = re.compile(
    r"^\s*"
    # optional politeness / modal lead-in ("vorrei", "puoi", "mi serve" …)
    r"(?:(?:per favore|per cortesia|puoi|potresti|vorrei|mi serve|mi servirebbe|ho bisogno di)\s+)?"
    r"(?:che\s+tu\s+|di\s+)?"
    # optional imperative verb ("crea", "prepara", "spiega" …)
    r"(?:(?:crea(?:mi)?|fa(?:i|mmi)?|prepara(?:mi)?|genera(?:mi)?|progetta|scrivi(?:mi)?|"
    r"costruisci|sviluppa|realizza|imposta|elabora|produci|"
    r"spiega(?:mi)?|parla(?:mi)?|illustra(?:mi)?|descrivi(?:mi)?|mostra(?:mi)?)\s+)?"
    # optional article
    r"(?:(?:una|un|il|lo|la|i|gli|le|dei|degli|delle)\s+)?"
    # required lesson-ish noun
    r"(?:lezione|lezioni|attivit[aà]|unit[aà](?:\s+didattica)?|modulo|percorso|"
    r"spiegazione|presentazione|introduzione|ripasso)\s+"
    # required connector preposition before the topic (handles elided "sull'…")
    r"(?:"
    r"(?:su(?:i|l|lo|lla|lle|gli)?|di|del(?:la|lo|le|i|gli)?|riguardo(?:\s+a)?|circa|"
    r"in\s+merito\s+a|sul\s+tema\s+di|a\s+proposito\s+di)\s+"
    r"|(?:sull|dell|dall|nell|all|d)['\u2019]"
    r")",
    re.IGNORECASE,
)


def _clean_lesson_query(text: str) -> str:
    """Return the topical core of a teacher query (Phase 3.1).

    'crea una lezione sui disturbi da deficit di attenzione'
        → 'disturbi da deficit di attenzione'
    Conservative: if nothing matches or the remainder is too short to be a real
    topic, the original (trimmed) text is returned unchanged.
    """
    if not text:
        return text
    cleaned = _LESSON_QUERY_LEAD_IN_RE.sub("", text, count=1).strip()
    cleaned = cleaned.strip(" .:;-—–")
    return cleaned if len(cleaned) >= 3 else text.strip()


def _live_media_ranking_content(lesson: Lesson) -> Optional[str]:
    """Return a bounded slice of the generated lesson for Phase 3 re-ranking.

    The live layer ranks media against the teacher's query *and* the lesson it
    produced. We pass a capped prefix of ``lesson_plan_md`` (plain markdown is
    fine for embedding) so ranking stays sharp without unbounded token cost.
    Returns ``None`` when there is no lesson text yet (e.g. mid-draft).
    """
    text = (getattr(lesson, "lesson_plan_md", None) or "").strip()
    if not text:
        return None
    return text[:_RANK_CONTENT_MAX_CHARS]


def _live_media_concepts(lesson: Lesson) -> list[str]:
    """Derive the live-media SEARCH concepts from a lesson (Phase 3.1).

    Each concept becomes a live API search (quota-bearing), so we favor the most
    *specific* signals and demote broad ones:
      1. ``specific_topic`` (argomento) — the cleanest topical anchor (e.g. "adhd").
      2. the teacher query, stripped of "crea una lezione su …" instruction noise.
      3. ``subject_area`` (materia) — DEMOTED to a fallback: broad terms like
         "Scienze" waste a search and dilute relevance, so it is only added when
         we still have fewer than 2 specific concepts.
    Returns a de-duplicated, ordered list (possibly empty); the service caps it.
    """
    profile = lesson.educational_profile_json or {}
    cleaned_query = _clean_lesson_query((getattr(lesson, "teacher_query", None) or "").strip())

    concepts: list[str] = []
    seen: set[str] = set()

    def _add(raw: Optional[str]) -> None:
        if not raw:
            return
        value = str(raw).strip()
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            concepts.append(value)

    # Specific signals first (argomento, then the cleaned query)…
    _add(profile.get("specific_topic"))
    _add(cleaned_query)
    # …broad subject area only as a fallback when specifics are thin.
    if len(concepts) < 2:
        _add(profile.get("subject_area"))

    return concepts


def _live_media_ranking_query(lesson: Lesson) -> Optional[str]:
    """Build the Phase 3 ranking 'query' side from structured profile signals.

    Unlike the fetch concepts (which cost API quota), this is pure embedding
    context, so we *can* include the broad subject + school grade — they help the
    reranker favor on-topic, age-appropriate items at no quota cost. Most specific
    signal first. Returns ``None`` when there is nothing to rank against.
    """
    profile = lesson.educational_profile_json or {}
    group = profile.get("group") or {}
    parts = [
        profile.get("specific_topic"),  # argomento (most specific)
        _clean_lesson_query((getattr(lesson, "teacher_query", None) or "").strip()),
        profile.get("subject_area"),  # materia (context only)
        group.get("grade"),  # school level (context only)
    ]
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in parts:
        if not raw:
            continue
        value = str(raw).strip()
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            ordered.append(value)
    return " · ".join(ordered) or None


async def live_media_for_lesson(lesson: Lesson) -> dict[str, Any]:
    """The panel's buckets (``videos``, ``citations``, ``wikipedia``), or ``{}``."""
    if not _media_live_enabled() or not _lesson_has_live_media_context(lesson):
        return {}
    concepts = _live_media_concepts(lesson)
    if not concepts:
        return {}
    try:
        from aix.agent.media import fetch_live_subject_resources, to_panel_media

        # A structured ranking query (argomento · cleaned teacher query · materia ·
        # grade) and a bounded slice of the generated lesson drive the semantic
        # re-ranking of the live items (a no-op when the rerank flag is off).
        live = await fetch_live_subject_resources(
            concepts=concepts,
            language="it",
            query=_live_media_ranking_query(lesson),
            content=_live_media_ranking_content(lesson),
        )
        panel = to_panel_media(live) or {}
    except Exception as exc:  # noqa: BLE001 — an optional panel never breaks anything
        logger.warning("[media-live] fetch/map failed for lesson_id=%s: %s", lesson.id, exc)
        return {}
    logger.info(
        "[media-live] lesson_id=%s concepts=%s → %d videos, %d citations, %d wikipedia",
        lesson.id,
        concepts,
        len(panel.get("videos") or []),
        len(panel.get("citations") or []),
        len(panel.get("wikipedia") or []),
    )
    return panel
