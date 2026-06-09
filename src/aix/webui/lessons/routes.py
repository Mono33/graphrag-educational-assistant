"""
HTML routes for lessons (CORE 2 #6.6 P1 + P2 + P3).

Routes (all under the ``/webui`` prefix):

    GET  /webui/lesson/new            — render the EducationalProfile form (P1)
    POST /webui/lesson                — validate, persist, redirect to show (P1)
                                        The form captures only structural
                                        profile data; the free-text agent
                                        query lives on the chat workspace
                                        (P2 phase 2).
    GET  /webui/lesson/{id}           — chat workspace, state-driven (P2)
    POST /webui/lesson/{id}/run       — start a run; goes straight to
                                        ``running`` and the SSE pane (P2).
                                        Same flow as P2 — P3 just adds the
                                        chat-attachment uploads as Writer
                                        context, nothing else.
    POST /webui/lesson/{id}/upload    — P3: chat attachment add (multipart)
    DELETE /webui/lesson/{id}/upload/{file_id}
                                      — P3: chat attachment remove
    GET  /webui/lesson/{id}/stream    — SSE feed for an in-flight run (P2)

    GET  /webui/lesson/{id}/profile       — read-only sidebar partial (Annulla)
    GET  /webui/lesson/{id}/profile/edit  — editable sidebar form (Aggiorna profilo)
    POST /webui/lesson/{id}/profile       — validate + persist profile inline

Auth posture:
    All routes require an authenticated user. Anonymous browsers get bounced
    to ``/auth/login?next=…`` so they come back to where they were after login.

Design notes:
    - POST ``/run`` is the *only* trigger that opens the SSE pane — it sets
      ``status="draft"→"running"`` (the actual flip happens inside the
      service layer, behind the SSE generator) and returns
      ``partials/chat_conversation.html`` with the live chat pane embedded.
    - GET stays idempotent.
    - GET ``/stream`` is the SSE endpoint htmx subscribes to. It runs the
      agent in-process, persists status transitions, and emits per-agent
      card events as HTML fragments rendered by the ``partials/chat_*``
      template family.
    - P3 file uploads happen *inside the chat input* via a small attachment
      tray (``partials/chat_attachments.html``); they are read by the
      service layer and joined into the Writer's prompt as
      ``teacher_provided_context``. They are never sent to the Knowledge
      Graph and do not gate the run.

SSE event vocabulary (single ``card`` event, terminating ``final`` / ``error``):
    event: card     (per-agent card; appended to #chat-cards via beforeend)
    event: final    (final lesson card; appended after the critic card)
    event: error    (error card; appended in place of the lesson card)
    event: end      (terminator — sse-close="end" on the client closes the
                    EventSource, preventing browser auto-reconnect)

    The ``writer`` card's ``hx-swap-oob="outerHTML"`` on the wrapper element
    (id ``writer-card-rev{N}``) makes htmx replace the matching pending card
    in place rather than appending; the retriever card's payload also
    carries an OOB ``<aside id="media-panel">`` that updates the right
    sidebar without a separate event.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Optional

import markdown as md
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from aix.api.schemas.educational_profile import (
    CLASS_FEATURE_LABELS,
    DISABILITY_LABELS,
    FORNITURE_MOBILITY_LABELS,
    GRADE_LABELS,
    OWN_DEVICE_LABELS,
    PEDAGOGICAL_INTENT_OPTIONS,
    STUDENT_ATTR_LABELS,
)
from aix.webui.agent import run_agent_stream
from aix.webui.auth.dependencies import optional_current_user
from aix.webui.auth.models import User
from aix.webui.db import get_async_session
from aix.webui.lessons.display import lesson_to_row
from aix.webui.lessons.models import Lesson, LessonMessage, SavedProfile
from aix.webui.lessons.schemas import form_to_profile_dict, profile_to_form_values
from aix.webui.lessons.uploads import delete_upload, save_upload

logger = logging.getLogger(__name__)


_PACKAGE_DIR = Path(__file__).resolve().parents[1]
_TEMPLATES_DIR = _PACKAGE_DIR / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


router = APIRouter(prefix="/webui", tags=["webui-lessons"])


# Per-process registry of in-flight agent runs. Used by the SSE endpoint to
# detect when a second client (a duplicate browser tab, an aggressive
# auto-reconnect) tries to attach to a lesson we're already streaming, and
# to gracefully surface that as an error rather than silently kicking off a
# second concurrent run.
#
# Limitations (acknowledged):
#   • Per-process only. With multiple workers (gunicorn, multi-replica
#     deploys) two workers wouldn't see each other's runs. Fine for dev,
#     replaced by a real run registry (DB row + heartbeat) in CORE 6.
#   • Doesn't survive a server restart. That's intentional: a stale
#     ``lesson.status == "running"`` row should *not* keep us out of
#     re-running the lesson after a crash — without it we'd never recover.
_ACTIVE_RUNS: set[uuid.UUID] = set()


def _label_dicts() -> dict[str, Any]:
    """Bundle of Italian labels passed into every lesson template."""
    return {
        "GRADE_LABELS": GRADE_LABELS,
        "DISABILITY_LABELS": DISABILITY_LABELS,
        "CLASS_FEATURE_LABELS": CLASS_FEATURE_LABELS,
        "STUDENT_ATTR_LABELS": STUDENT_ATTR_LABELS,
        "FORNITURE_MOBILITY_LABELS": FORNITURE_MOBILITY_LABELS,
        "OWN_DEVICE_LABELS": OWN_DEVICE_LABELS,
        "PEDAGOGICAL_INTENT_OPTIONS": PEDAGOGICAL_INTENT_OPTIONS,
    }


async def _load_chat_messages(
    session: AsyncSession,
    lesson_id: uuid.UUID,
) -> list[dict[str, Any]]:
    """
    Load all ``LessonMessage`` rows for a lesson and shape them into the
    template-friendly form expected by ``chat_history.html``.

    Returns rows ordered by ``(turn_index, created_at)`` ASC. Each row is
    flattened to a plain dict with markdown pre-rendered to HTML for
    assistant turns (Jinja can't call markdown.markdown directly without
    a registered filter). User turns keep ``content_html=None`` — their
    content is rendered as plain text inside the bubble.

    Returns an empty list when the lesson has no messages yet (legacy
    pre-#10.3 lessons or fresh draft lessons that haven't run a turn).
    The template falls back to the legacy single-bubble layout in that
    case — no breaking change for existing rows.
    """
    result = await session.execute(
        select(LessonMessage)
        .where(LessonMessage.lesson_id == lesson_id)
        .order_by(LessonMessage.turn_index, LessonMessage.created_at)
    )
    out: list[dict[str, Any]] = []
    for msg in result.scalars().all():
        out.append(
            {
                "id": msg.id,
                "role": msg.role,
                "content_md": msg.content_md or "",
                # Pre-render assistant markdown — Jinja sees ready-to-paint HTML.
                # User messages render as plain text inside the bubble (no
                # markdown — teacher's typed query, escape-safe via Jinja's
                # default autoescape).
                "content_html": (
                    _markdown_to_html(msg.content_md or "") if msg.role == "assistant" else None
                ),
                "turn_index": msg.turn_index,
                "agent_kind": msg.agent_kind,
                "meta_json": msg.meta_json or {},
                "created_at": msg.created_at,
            }
        )
    return out


def _bounce_to_login(target_path: str) -> RedirectResponse:
    """Redirect anonymous users to /auth/login with ?next= so they come back."""
    return RedirectResponse(f"/auth/login?next={target_path}", status_code=303)


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
        profile.get("subject_area"),     # materia (context only)
        group.get("grade"),              # school level (context only)
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


# ----------------------------------------------------------------------------
# GET /webui/lessons — lesson history list
# ----------------------------------------------------------------------------


@router.get(
    "/lessons",
    response_class=HTMLResponse,
    name="webui_lessons_list",
)
async def lesson_list(
    request: Request,
    user: Optional[User] = Depends(optional_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> Response:
    """
    List all lessons for the current user, newest first.

    Brand-pass (CORE 2 #6.6 P5) additions to the context passed to the
    template — all *derived* from the same query result, no extra DB hits:

        rows     : list[row_dict]  — flattened lesson rows (display.lesson_to_row)
        stats    : dict[str, int]  — counts per status (total, draft, complete, …)
        filters  : dict[str, list] — distinct values for sidebar filters
                                     (subjects, classes, disabilities)
        active_nav : str           — highlights "Le mie lezioni" in the navbar

    The legacy ``lessons`` context var (list of ORM rows) is kept around for
    backward compatibility — the new template doesn't consume it, but any
    yet-unmigrated partial that imports this route's context still works.
    """
    if user is None:
        return _bounce_to_login("/webui/lessons")

    result = await session.execute(
        select(Lesson).where(Lesson.owner_id == user.id).order_by(Lesson.updated_at.desc())
    )
    lessons = list(result.scalars().all())

    rows = [lesson_to_row(lesson) for lesson in lessons]

    # Aggregate stats for the one-liner subtitle ("7 lezioni · 0 bozze · …").
    stats = {
        "total": len(rows),
        "draft": sum(1 for r in rows if r["status"] == "draft"),
        "running": sum(1 for r in rows if r["status"] == "running"),
        "complete": sum(1 for r in rows if r["status"] == "complete"),
        "error": sum(1 for r in rows if r["status"] == "error"),
    }

    # Distinct filter values, derived from the rows we already have. Sorted
    # so the sidebar checkboxes render deterministically. Empty strings are
    # dropped so a row with no subject doesn't get a blank checkbox.
    def _distinct(values: list[str]) -> list[str]:
        return sorted({v for v in values if v})

    filters = {
        "subjects": _distinct(r["subject"] for r in rows),
        "classes": _distinct(r["group_title"] for r in rows),
        "disabilities": _distinct(d for r in rows for d in (r["disabilities"] or [])),
    }

    return templates.TemplateResponse(
        "pages/lesson_list.html",
        {
            "request": request,
            "user": user,
            "lessons": lessons,  # legacy context var, kept for compat
            "rows": rows,  # new shape for the brand template
            "stats": stats,
            "filters": filters,
            "active_nav": "lessons",
            "title": "Le mie lezioni · AixLearning",
        },
    )


# ----------------------------------------------------------------------------
# DELETE /webui/lesson/{id} — delete a lesson
# ----------------------------------------------------------------------------


@router.delete(
    "/lesson/{lesson_id}",
    name="webui_lesson_delete",
)
async def lesson_delete(
    lesson_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
    user: Optional[User] = Depends(optional_current_user),
) -> Response:
    """Permanently delete a lesson owned by the current user."""
    if user is None:
        raise HTTPException(status_code=401)

    result = await session.execute(
        select(Lesson).where(Lesson.id == lesson_id, Lesson.owner_id == user.id)
    )
    lesson = result.scalar_one_or_none()
    if lesson is None:
        raise HTTPException(status_code=404, detail="Lezione non trovata")

    await session.delete(lesson)
    await session.commit()

    return Response(
        status_code=204,
        headers={"HX-Redirect": "/webui/lessons"},
    )


# ----------------------------------------------------------------------------
# GET /webui/lesson/{id}/card-fragment — full lesson card (non-SSE)
# ----------------------------------------------------------------------------


@router.get(
    "/lesson/{lesson_id}/card-fragment",
    response_class=HTMLResponse,
    name="webui_lesson_card_fragment",
)
async def lesson_card_fragment(
    request: Request,
    lesson_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
    user: Optional[User] = Depends(optional_current_user),
) -> Response:
    """
    Return the full lesson card HTML via a regular GET (no SSE).

    Called by the htmx:sseClose handler in chat_pane.html to replace the small
    'done' placeholder with the complete lesson card fetched from the DB.
    This sidesteps the htmx-ext-sse single-line buffer limit that truncates
    large lesson plans when sent as SSE event data.
    """
    if user is None:
        logger.warning("[card-fragment] 401 — unauthenticated request for lesson_id=%s", lesson_id)
        raise HTTPException(status_code=401)

    result = await session.execute(
        select(Lesson).where(Lesson.id == lesson_id, Lesson.owner_id == user.id)
    )
    lesson = result.scalar_one_or_none()
    if lesson is None:
        logger.warning("[card-fragment] 404 lesson_id=%s user_id=%s", lesson_id, user.id)
        raise HTTPException(status_code=404, detail="Lezione non trovata")

    # After phase 1 (planner-only run) the lesson is reset to "draft".
    # The sseClose handler would try to fetch the lesson card, but there's
    # nothing to show yet — the intent selection card is already in the DOM
    # from the SSE events.  Return empty so nothing is appended.
    if lesson.status == "draft":
        return HTMLResponse(content="", status_code=200)

    logger.info(
        "[card-fragment] rendering lesson_id=%s status=%s plan_len=%s",
        lesson_id,
        lesson.status,
        len(lesson.lesson_plan_md or ""),
    )
    return templates.TemplateResponse(
        "partials/chat_lesson_card.html",
        {
            "request": request,
            "lesson": lesson,
            "lesson_plan_html": _markdown_to_html(lesson.lesson_plan_md or ""),
            "meta": {"approved": lesson.status == "complete", "revision_count": 0},
        },
    )


# ----------------------------------------------------------------------------
# GET /webui/lesson/{id}/media-live — live (auto-retrieved) media fragment
# ----------------------------------------------------------------------------


@router.get(
    "/lesson/{lesson_id}/media-live",
    response_class=HTMLResponse,
    name="webui_lesson_media_live",
)
async def lesson_media_live(
    request: Request,
    lesson_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
    user: Optional[User] = Depends(optional_current_user),
) -> Response:
    """Off-critical-path live media fragment for the right media panel (Phase 1b).

    Lazy-loaded by the ``#media-live-slot`` placeholder (``hx-trigger="load"``).
    Fetches live papers (OpenAlex) + Wikipedia for the lesson's topic via
    :class:`LiveMediaService` (cache-first, bounded), maps them into the panel's
    bucket shape, and returns the replacement slot fragment.

    This endpoint is **never** on the lesson-generation critical path — the
    planner→retriever→writer→critic pipeline does not wait on it. It degrades to
    an empty slot when the live layer is disabled, the lesson/topic is unknown,
    nothing is found, or anything fails (so it can never break the panel).
    """
    empty = HTMLResponse('<div id="media-live-slot"></div>')

    # Optional, best-effort panel — no auth noise, no errors bubble to the UI.
    if user is None or not _media_live_enabled():
        return empty

    result = await session.execute(
        select(Lesson).where(Lesson.id == lesson_id, Lesson.owner_id == user.id)
    )
    lesson = result.scalar_one_or_none()
    if lesson is None:
        return empty

    if not _lesson_has_live_media_context(lesson):
        return empty

    concepts = _live_media_concepts(lesson)
    if not concepts:
        return empty

    try:
        from aix.agent.media import fetch_live_subject_resources, to_panel_media

        # Phase 3 / 3.1 inputs: a structured ranking query (argomento · cleaned
        # teacher query · materia · grade) + a bounded slice of the generated
        # lesson drive the semantic re-ranking of the live items (no-op when the
        # rerank flag is off — see MediaConfig.rerank_enabled).
        query = _live_media_ranking_query(lesson)
        content = _live_media_ranking_content(lesson)
        live = await fetch_live_subject_resources(
            concepts=concepts, language="it", query=query, content=content
        )
        panel = to_panel_media(live)
    except Exception as exc:
        logger.warning("[media-live] fetch/map failed for lesson_id=%s: %s", lesson_id, exc)
        return empty

    if not panel:
        return empty

    logger.info(
        "[media-live] lesson_id=%s concepts=%s → %d videos, %d citations, %d wikipedia",
        lesson_id,
        concepts,
        len(panel.get("videos") or []),
        len(panel.get("citations") or []),
        len(panel.get("wikipedia") or []),
    )
    return templates.TemplateResponse(
        "partials/media_live_sections.html",
        {
            "request": request,
            "videos": panel.get("videos") or [],
            "citations": panel.get("citations") or [],
            "wikipedia": panel.get("wikipedia") or [],
        },
    )


# ----------------------------------------------------------------------------
# GET /webui/lesson/{id}/chat-input-fragment — chat input partial (state-driven)
# ----------------------------------------------------------------------------


@router.get(
    "/lesson/{lesson_id}/chat-input-fragment",
    response_class=HTMLResponse,
    name="webui_lesson_chat_input_fragment",
)
async def lesson_chat_input_fragment(
    request: Request,
    lesson_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
    user: Optional[User] = Depends(optional_current_user),
) -> Response:
    """
    Return just the chat-input partial for ``lesson_id`` rendered in its
    current state ("draft" / "running" / "complete" / "error").

    Called by chat_pane.html's htmx:sseClose handler as a defensive backup
    to the OOB chat_input swap that the kind="done" / kind="error" SSE
    events also emit. On follow-up turns the OOB swap can silently lose
    the race against /run's innerHTML swap that just rebuilt the
    #chat-input-wrapper element, leaving the user stuck on the disabled
    "Generazione in corso..." state. This endpoint is the safety net:
    after the lesson card is fetched, the JS handler re-fetches the input
    here so it always lands on the correct active follow-up state.

    Idempotent — if OOB already succeeded this is a same-content
    outerHTML re-swap (no user-visible difference, ~1KB roundtrip).
    """
    if user is None:
        logger.warning(
            "[chat-input-fragment] 401 — unauthenticated request for lesson_id=%s",
            lesson_id,
        )
        raise HTTPException(status_code=401)

    result = await session.execute(
        select(Lesson).where(Lesson.id == lesson_id, Lesson.owner_id == user.id)
    )
    lesson = result.scalar_one_or_none()
    if lesson is None:
        logger.warning("[chat-input-fragment] 404 lesson_id=%s user_id=%s", lesson_id, user.id)
        raise HTTPException(status_code=404, detail="Lezione non trovata")

    logger.info(
        "[chat-input-fragment] rendering lesson_id=%s status=%s",
        lesson_id,
        lesson.status,
    )
    return templates.TemplateResponse(
        "partials/chat_input.html",
        {
            "request": request,
            "lesson": lesson,
            # _oob=False because this response is targeted explicitly via
            # htmx.ajax(target: '#chat-input-wrapper', swap: 'outerHTML') —
            # we don't want the duplicate hx-swap-oob attribute that would
            # turn the JS-driven swap into an OOB fallback.
            "_oob": False,
        },
    )


# ----------------------------------------------------------------------------
# GET /webui/lesson/new — render the form
# ----------------------------------------------------------------------------


@router.get(
    "/lesson/new",
    response_class=HTMLResponse,
    name="webui_lesson_new",
)
async def lesson_new_get(
    request: Request,
    user: Optional[User] = Depends(optional_current_user),
    session: AsyncSession = Depends(get_async_session),
    topic: Optional[str] = None,
    domain: Optional[str] = None,
    profile_id: Optional[uuid.UUID] = None,
) -> Response:
    """Show the EducationalProfile form. Anon users are bounced to login."""
    if user is None:
        return _bounce_to_login("/webui/lesson/new")

    # F6: "What's Next?" pre-fill — ?topic=X&domain=Y carries context forward
    prefill: dict[str, Any] = {}
    if topic:
        prefill["specific_topic"] = topic.strip()
    if domain and domain in {"neuro", "udl", "all"}:
        prefill["domain"] = domain

    # SavedProfile pre-fill — ?profile_id=UUID loads a saved profile and
    # expands it into the flat form_values dict via profile_to_form_values().
    if profile_id:
        sp_result = await session.execute(
            select(SavedProfile).where(
                SavedProfile.id == profile_id,
                SavedProfile.owner_id == user.id,
            )
        )
        sp = sp_result.scalar_one_or_none()
        if sp is not None:
            prefill = {**profile_to_form_values(sp.profile_json), **prefill}

    # Load all saved profiles for the selector at the top of the form.
    sp_all = await session.execute(
        select(SavedProfile)
        .where(SavedProfile.owner_id == user.id)
        .order_by(SavedProfile.created_at.desc())
    )
    saved_profiles = list(sp_all.scalars().all())

    return templates.TemplateResponse(
        "pages/lesson_new.html",
        {
            "request": request,
            "title": "Nuova lezione · AixLearning",
            "phase": "P3 — Chat workspace + allegati",
            "user": user,
            # CORE 2 #6.6 P5.3 — drives the navbar's underline on "Crea lezione".
            "active_nav": "new",
            "form_errors": None,
            "form_values": prefill,
            "saved_profiles": saved_profiles,
            **_label_dicts(),
        },
    )


# ----------------------------------------------------------------------------
# POST /webui/lesson — validate + persist
# ----------------------------------------------------------------------------


@router.post(
    "/lesson",
    name="webui_lesson_create",
)
async def lesson_create(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
    user: Optional[User] = Depends(optional_current_user),
) -> Response:
    """
    Validate the EducationalProfile form, persist a Lesson row, redirect to
    the chat workspace. Re-renders the form with errors on failure.
    """
    if user is None:
        return _bounce_to_login("/webui/lesson/new")

    form = await request.form()

    # Capture the raw form values so we can re-render them on error. The
    # free-text agent query is *not* on this form anymore — it's captured
    # by the chat input on the lesson workspace (P2 phase 2 UX), so the
    # form is profile-only.
    form_values_for_redisplay = {
        "title": form.get("title", ""),
        "domain": form.get("domain", "neuro"),
        "group_title": form.get("group_title", ""),
        "group_students_number": form.get("group_students_number", ""),
        "group_grade": form.get("group_grade", ""),
        "group_disabilities": list(form.getlist("group_disabilities")),
        "group_class_features": list(form.getlist("group_class_features")),
        "group_student_attributes": list(form.getlist("group_student_attributes")),
        "classroom_title": form.get("classroom_title", ""),
        "classroom_forniture_mobility": form.get("classroom_forniture_mobility", "NO"),
        "classroom_has_lim": "classroom_has_lim" in form,
        "classroom_has_wifi": "classroom_has_wifi" in form,
        "classroom_has_suite": "classroom_has_suite" in form,
        "classroom_pc_station": "classroom_pc_station" in form,
        "classroom_own_device": form.get("classroom_own_device", "NO"),
        "time_available_minutes": form.get("time_available_minutes", ""),
        "subject_area": form.get("subject_area", ""),
        "specific_topic": form.get("specific_topic", ""),
    }

    try:
        profile_dict = form_to_profile_dict(form)
    except ValidationError as exc:
        logger.info("Lesson form validation failed: %s", exc.errors())
        sp_all_err = await session.execute(
            select(SavedProfile)
            .where(SavedProfile.owner_id == user.id)
            .order_by(SavedProfile.created_at.desc())
        )
        return templates.TemplateResponse(
            "pages/lesson_new.html",
            {
                "request": request,
                "title": "Nuova lezione · AixLearning",
                "phase": "P3 — Chat workspace + allegati",
                "user": user,
                # CORE 2 #6.6 P5.3 — keep the navbar consistent on the
                # error-redisplay path (the user is still on /lesson/new).
                "active_nav": "new",
                "form_errors": [
                    f"{'.'.join(str(p) for p in err.get('loc', ()))}: "
                    f"{err.get('msg', 'campo non valido')}"
                    for err in exc.errors()
                ],
                "form_values": form_values_for_redisplay,
                "saved_profiles": list(sp_all_err.scalars().all()),
                **_label_dicts(),
            },
            status_code=422,
        )

    title_raw = form.get("title")
    title = title_raw.strip() if isinstance(title_raw, str) and title_raw.strip() else None
    domain_raw = form.get("domain") or "neuro"
    domain = str(domain_raw).strip() or "neuro"

    # ``teacher_query`` is intentionally left ``None`` here — the chat
    # input on the workspace captures it on the user's first turn (P2
    # phase 2 UX). The lesson lands in ``draft`` status and the chat
    # input is the affordance to start the run.
    lesson = Lesson(
        owner_id=user.id,
        title=title,
        domain=domain,
        educational_profile_json=profile_dict,
        teacher_query=None,
        status="draft",
    )
    session.add(lesson)
    await session.commit()
    await session.refresh(lesson)

    logger.info(
        "📝 Lesson created: id=%s owner=%s domain=%s title=%r",
        lesson.id,
        user.id,
        lesson.domain,
        lesson.title,
    )

    return RedirectResponse(url=f"/webui/lesson/{lesson.id}", status_code=303)


# ----------------------------------------------------------------------------
# GET /webui/lesson/{lesson_id} — chat workspace
# ----------------------------------------------------------------------------


@router.get(
    "/lesson/{lesson_id}",
    response_class=HTMLResponse,
    name="webui_lesson_show",
)
async def lesson_show(
    request: Request,
    lesson_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
    user: Optional[User] = Depends(optional_current_user),
) -> Response:
    """
    Show a single Lesson as the 3-pane chat workspace (P2 phase 2).

    The page is *state-driven* — what fills #chat-stream depends on
    ``lesson.status``:

        draft     → chat input + attachments tray; POST /run starts the
                    full pipeline live (planner → retriever → writer →
                    critic) over SSE.
        running   → live SSE chat pane (re-attaches if user reloaded mid-run).
        complete  → final lesson card + summary (replay path).
        error     → error card with persisted ``error_message`` + Riprova.

    Owner-only: viewing someone else's lesson returns 404 to avoid leaking
    existence across user accounts.
    """
    if user is None:
        return _bounce_to_login(f"/webui/lesson/{lesson_id}")

    result = await session.execute(
        select(Lesson).where(Lesson.id == lesson_id, Lesson.owner_id == user.id)
    )
    lesson = result.scalar_one_or_none()
    if lesson is None:
        raise HTTPException(status_code=404, detail="Lezione non trovata")

    # Pre-render the persisted lesson plan to HTML so the lesson card
    # template doesn't have to know which markdown extensions we use.
    # Only the terminal "complete" state needs this — running/error/draft
    # never paint it.
    lesson_plan_html = (
        _markdown_to_html(lesson.lesson_plan_md)
        if lesson.status == "complete" and lesson.lesson_plan_md
        else None
    )

    # Build a synthetic ``meta`` for the lesson card on the replay path —
    # we don't persist scores / duration on the row today, so the card
    # surfaces only what we know (approval-by-presence, lesson length).
    # P4 will add proper run metadata persistence.
    meta_for_replay = {"approved": True, "revision_count": 0} if lesson.status == "complete" else {}

    # Multi-turn history for the chat transcript (#10.3d). Returns []
    # for legacy / fresh lessons → templates fall back to the legacy
    # single-bubble layout (chat_user_message.html + state-driven block).
    messages = await _load_chat_messages(session, lesson.id)

    # F6: "What's Next?" — adjacent KG concepts for completed lessons.
    # Runs in a sync threadpool to keep the async route non-blocking.
    # Returns [] gracefully on Neo4j errors or missing topic.
    adjacent_concepts: list[dict] = []
    if lesson.status == "complete":
        profile_j = lesson.educational_profile_json or {}
        concept_name = profile_j.get("specific_topic") or profile_j.get("subject_area")
        if concept_name:
            import asyncio

            try:
                import os

                from neo4j import GraphDatabase

                from aix.retrieval.graph_retriever import HybridGraphRetriever

                neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
                neo4j_user = os.environ.get("NEO4J_USERNAME", "neo4j")
                neo4j_pass = os.environ.get("NEO4J_PASSWORD", "")
                _driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
                _retriever = HybridGraphRetriever(_driver, domain=lesson.domain or "all")
                adjacent_concepts = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: _retriever.get_concept_neighbors(
                        concept_name, domain=lesson.domain or "all", limit=5
                    ),
                )
                _driver.close()
            except Exception as _e:
                logger.warning("[lesson_show] what_next failed: %s", _e)

    return templates.TemplateResponse(
        "pages/lesson_show.html",
        {
            "request": request,
            "title": (lesson.title or "Lezione") + " · AixLearning",
            "phase": "P3 — Chat workspace + allegati",
            "user": user,
            "lesson": lesson,
            "messages": messages,
            "lesson_plan_html": lesson_plan_html,
            "meta": meta_for_replay,
            "media": None,  # not persisted yet — empty placeholder on reload
            # Dynamic Media Retrieval Phase 1b — the live layer must mirror the
            # curated panel: it only appears *during an active run* (via the SSE
            # retriever swap below), never on a passive page open/reload. On a
            # plain GET the curated media is None (empty panel), so the live slot
            # is suppressed here to stay consistent and avoid firing for a
            # previously-completed lesson that still carries a persisted query.
            "media_live_enabled": _media_live_enabled(),
            "media_live_ready": False,
            "lesson_id": lesson.id,
            # P5.4 — workspace is a "leaf" of the Library tab. Highlighting
            # "Le mie lezioni" in the top nav matches the user's mental model
            # ("I'm inside one of my lessons") and is the same convention the
            # Library list uses (active_nav="lessons").
            "active_nav": "lessons",
            "adjacent_concepts": adjacent_concepts,
            **_label_dicts(),
        },
    )


# ----------------------------------------------------------------------------
# POST /webui/lesson/{id}/run — start a new agent run (P2)
# ----------------------------------------------------------------------------


@router.post(
    "/lesson/{lesson_id}/run",
    response_class=HTMLResponse,
    name="webui_lesson_run",
)
async def lesson_run(
    request: Request,
    lesson_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
    user: Optional[User] = Depends(optional_current_user),
) -> Response:
    """
    Open a new agent run for ``lesson_id``.

    Multi-turn semantics (CORE 2 #10.3c):
        ``mode`` is auto-detected from ``lesson.status``:
            draft              → mode="new"        (first turn)
            complete | error   → mode="follow_up"  (continuation)
            running            → 409 (already in progress)

        Each turn writes a ``LessonMessage(role="user", turn_index=N)``
        on entry; the service layer writes the matching
        ``LessonMessage(role="assistant", turn_index=N)`` on success.
        The chat history is rendered from these rows (CQRS pattern with
        the LangGraph checkpointer — see lessons/models.py docstring).

    Backward compat for pre-#10.3 lessons:
        Lessons created BEFORE this code lands have no ``LessonMessage``
        rows but DO have ``lesson.teacher_query`` + ``lesson.lesson_plan_md``
        from the legacy single-turn flow. On the first follow-up we backfill
        turn 1 from those legacy fields so the rendered chat history is
        contiguous and the agent's history-injection logic sees the prior
        exchange. One-time, idempotent.

    Form data:
        query   Optional[str] — the teacher's new query for this turn.
                Required for the first turn (``draft`` lesson). Optional
                for follow-ups when implicitly re-using the latest query
                (post-#10.3 the chat input always submits a non-empty
                query, so this fallback is purely a safety net for the
                legacy "Rigenera" / "Riprova" buttons).

    Returns ``partials/chat_conversation.html`` so htmx swaps the running
    pane (input → spinner + SSE stream) into ``#chat-conversation``. The
    response includes the just-persisted user bubble at the bottom of the
    history so the teacher sees their message immediately.
    """
    if user is None:
        # POSTs from htmx don't follow redirects sensibly; return a small
        # fragment that tells htmx to bounce the *whole* page to login.
        # (HX-Redirect is the standard htmx response header for this.)
        return HTMLResponse(
            content="",
            status_code=401,
            headers={"HX-Redirect": f"/auth/login?next=/webui/lesson/{lesson_id}"},
        )

    result = await session.execute(
        select(Lesson).where(Lesson.id == lesson_id, Lesson.owner_id == user.id)
    )
    lesson = result.scalar_one_or_none()
    if lesson is None:
        raise HTTPException(status_code=404, detail="Lezione non trovata")

    # Capture the user's new query if present.
    form = await request.form()
    query_raw = form.get("query")
    new_query = query_raw.strip() if isinstance(query_raw, str) and query_raw.strip() else None

    if new_query is not None:
        # Server-side validation mirroring the textarea ``minlength=3`` /
        # ``maxlength=2000``. Defensive — clients can be tampered with.
        if len(new_query) < 3:
            return HTMLResponse(
                content=(
                    '<div class="rounded-lg border border-amber-200 bg-amber-50 '
                    'text-amber-900 px-3 py-2 text-sm">'
                    "La richiesta deve avere almeno 3 caratteri."
                    "</div>"
                ),
                status_code=422,
            )
        if len(new_query) > 2000:
            new_query = new_query[:2000]
        lesson.teacher_query = new_query

    # Defensive: if the teacher posts /run without ever having set a
    # query (no first-turn submit, then somehow triggered Rigenera), bail
    # with a friendly error rather than running on an empty prompt.
    if not (lesson.teacher_query or "").strip():
        return HTMLResponse(
            content=(
                '<div class="rounded-lg border border-amber-200 bg-amber-50 '
                'text-amber-900 px-3 py-2 text-sm">'
                "Scrivi prima la tua richiesta nel campo della chat, poi premi Invia."
                "</div>"
            ),
            status_code=422,
        )

    if lesson.status == "running":
        return HTMLResponse(
            content=(
                '<div class="rounded-lg border border-amber-200 bg-amber-50 '
                'text-amber-900 px-3 py-2 text-sm">'
                "Una generazione è già in corso. Attendi il termine."
                "</div>"
            ),
            status_code=409,
        )

    # ── Multi-turn mode detection (#10.3c) ────────────────────────────
    # Computed early so the phase-1/phase-2 logic below can branch on it.
    is_follow_up = lesson.status in ("complete", "error")
    mode = "follow_up" if is_follow_up else "new"

    # ── Chat-based pedagogical intent — two-phase flow ───────────────
    # Phase 1 (planner-only): first new-run on a draft lesson, before the
    # teacher has confirmed an intent from the chat card.  We flag this
    # with __planner_only__ in profile_json so the SSE generator knows to
    # stop after the planner and render the intent selection card.
    #
    # Phase 2 (full run): the teacher clicked a chip in the intent card
    # and the POST carries intent_confirmed=1.  We store the chosen intent
    # in profile_json and run the full pipeline.  No new LessonMessage is
    # written for phase 2 — the user turn was already recorded in phase 1.
    #
    # Refinement runs and follow-up turns bypass phase 1 entirely.
    # ─────────────────────────────────────────────────────────────────────
    intent_confirmed = form.get("intent_confirmed") == "1"
    is_phase1 = (mode == "new") and (not intent_confirmed)

    if is_phase1:
        _prof = dict(lesson.educational_profile_json or {})
        _prof["__planner_only__"] = True
        lesson.educational_profile_json = _prof
    elif intent_confirmed:
        _intent_code_raw = form.get("pedagogical_intent_code")
        _intent_code = (
            _intent_code_raw.strip()
            if isinstance(_intent_code_raw, str) and _intent_code_raw.strip()
            else None
        )
        if _intent_code and _intent_code != "skip":
            _intent_detail_raw = form.get("pedagogical_intent_detail")
            _intent_detail = (
                _intent_detail_raw.strip()
                if isinstance(_intent_detail_raw, str) and _intent_detail_raw.strip()
                else None
            )
            _prof = dict(lesson.educational_profile_json or {})
            _prof.pop("__planner_only__", None)
            _prof["pedagogical_intent"] = (
                f"{_intent_code}: {_intent_detail}" if _intent_detail else _intent_code
            )
            lesson.educational_profile_json = _prof

    # ── F3b: SAM guided refinement instruction ────────────────────────
    # Optional: refinement_code (one of the predefined options) +
    # refinement_detail (freetext). Stored transiently in profile_json
    # as "__refinement__" so the service layer can pop it without a
    # DB schema change. Only set on follow-up "Raffina" runs.
    refinement_code_raw = form.get("refinement_code")
    refinement_detail_raw = form.get("refinement_detail")
    refinement_code = (
        refinement_code_raw.strip()
        if isinstance(refinement_code_raw, str) and refinement_code_raw.strip()
        else None
    )
    refinement_detail = (
        refinement_detail_raw.strip()
        if isinstance(refinement_detail_raw, str) and refinement_detail_raw.strip()
        else None
    )
    if refinement_code:
        _REFINEMENT_PROMPTS = {
            "simplify": "Reduce length by ~30%, simplify vocabulary, keep only essential concepts",
            "deepen": "Add scientific depth, more precise terminology, nuanced examples",
            "more_activities": "Replace passive sections with at least 2 interactive activities or exercises",
            "adapt_class": "Re-calibrate to the class profile: BES accommodations, time constraint, available tech",
        }
        if refinement_code == "custom" and refinement_detail:
            refinement_instruction = refinement_detail
        else:
            base = _REFINEMENT_PROMPTS.get(refinement_code, "")
            refinement_instruction = f"{base} ({refinement_detail})" if refinement_detail else base
        # Store transiently in profile_json so service.py can read it
        profile = dict(lesson.educational_profile_json or {})
        profile["__refinement__"] = refinement_instruction
        lesson.educational_profile_json = profile

    # Backfill turn 1 from legacy fields if this is a follow-up on a
    # pre-#10.3 lesson. One-time, idempotent — guarded by a count probe
    # so subsequent follow-ups skip cleanly.
    if is_follow_up:
        existing_msg_count = await session.scalar(
            select(func.count(LessonMessage.id)).where(LessonMessage.lesson_id == lesson.id)
        )
        if (
            (existing_msg_count or 0) == 0
            and (lesson.teacher_query or "").strip()
            and (lesson.lesson_plan_md or "").strip()
        ):
            logger.info(
                "[lesson_run] backfilling turn 1 from legacy fields lesson_id=%s",
                lesson.id,
            )
            # NOTE: we use the OLD lesson.teacher_query for the backfilled
            # user message, NOT the freshly submitted new_query. The new
            # query is the CURRENT turn — the backfill represents a PRIOR
            # turn that we never persisted to lesson_messages.
            #
            # Tricky bit: by this point ``lesson.teacher_query`` may already
            # have been overwritten with ``new_query`` above. We undo that
            # for the backfill window only — see the assignment guard below.
            backfill_user_query = (
                new_query is None
                and lesson.teacher_query
                or _legacy_query_for_backfill(lesson, new_query)
            )
            session.add(
                LessonMessage(
                    lesson_id=lesson.id,
                    role="user",
                    content_md=backfill_user_query,
                    turn_index=1,
                )
            )
            session.add(
                LessonMessage(
                    lesson_id=lesson.id,
                    role="assistant",
                    content_md=lesson.lesson_plan_md,
                    turn_index=1,
                    agent_kind="writer",
                    meta_json={"approved": True, "backfilled": True},
                )
            )
            await session.flush()

    # ── Compute the next turn_index for the user message we're about to
    # persist. MAX(turn_index) + 1 keeps the ordering invariant. For a
    # brand-new lesson with no rows yet, MAX returns NULL → 0 + 1 = 1.
    latest_turn = (
        await session.scalar(
            select(func.max(LessonMessage.turn_index)).where(LessonMessage.lesson_id == lesson.id)
        )
        or 0
    )
    new_turn_index = latest_turn + 1

    # The query we persist for THIS turn. ``new_query`` is preferred (the
    # user just typed it); we fall back to lesson.teacher_query only as a
    # safety net for legacy buttons that POST with no body. The validation
    # above already guarantees at least one is non-empty.
    user_msg_content = new_query or lesson.teacher_query

    # Phase 2 (intent_confirmed) is not a new user turn — the user message
    # was already persisted in phase 1. Skip to avoid a duplicate row that
    # would cause _persist_assistant_turn to match the wrong turn_index.
    if not intent_confirmed:
        session.add(
            LessonMessage(
                lesson_id=lesson.id,
                role="user",
                content_md=user_msg_content,
                turn_index=new_turn_index,
            )
        )

    # Flip status optimistically so the chat conversation partial we
    # return immediately renders the live SSE pane (rather than the input
    # form). The service layer's setup phase commits this again — that's
    # idempotent and cheaper than a second roundtrip.
    #
    # We DO NOT clear lesson.lesson_plan_md anymore (#10.3): for follow-up
    # turns the previous lesson stays visible in the chat history (rendered
    # via lesson_messages — see chat_conversation.html post-#10.3d). For
    # mode=new this is the first turn so lesson_plan_md is None already.
    lesson.status = "running"
    lesson.error_message = None
    if mode == "new":
        # First turn — keep the existing semantics (clear stale state
        # from any aborted previous attempt on a draft lesson).
        lesson.lesson_plan_md = None
    await session.commit()
    await session.refresh(lesson)

    logger.info(
        "▶️  Lesson run accepted: id=%s owner=%s mode=%s turn=%s has_new_query=%s uploads=%s",
        lesson.id,
        user.id,
        mode,
        new_turn_index,
        bool(new_query),
        len(lesson.uploaded_files_json or []),
    )

    # Reload the chat transcript so the response includes the just-persisted
    # user turn (and any backfilled prior turn) at the bottom of the
    # history. The chat_pane below (running state) will append per-agent
    # cards live as the SSE stream fires; the final assistant message is
    # written by the service on success.
    messages = await _load_chat_messages(session, lesson.id)

    return templates.TemplateResponse(
        "partials/chat_conversation.html",
        {
            "request": request,
            "lesson": lesson,
            "user": user,
            "messages": messages,
            "lesson_plan_html": None,
            "meta": {},
            "media": None,
        },
    )


def _legacy_query_for_backfill(lesson: Lesson, new_query: Optional[str]) -> str:
    """
    Recover the previous turn's user query for the backfill path when
    ``lesson.teacher_query`` has already been overwritten by the route's
    new-query write-through.

    Returns the best available recovery string. Pre-#10.3 lessons that
    DO have a ``lesson_plan_md`` but somehow lost their ``teacher_query``
    fall back to a synthesized "richiesta precedente" placeholder so the
    backfill row stays non-empty (NOT NULL constraint).
    """
    # If the route has overwritten lesson.teacher_query with new_query,
    # we can't recover the original. Fall back to a documented placeholder
    # — better than a NULL row that breaks the rendering invariant.
    if new_query and (lesson.teacher_query or "").strip() == new_query.strip():
        return "[richiesta precedente non persistita]"
    return (lesson.teacher_query or "[richiesta precedente non persistita]").strip()


# ----------------------------------------------------------------------------
# POST /webui/lesson/{id}/upload — chat attachment add (P3)
# ----------------------------------------------------------------------------


@router.post(
    "/lesson/{lesson_id}/upload",
    response_class=HTMLResponse,
    name="webui_lesson_upload",
)
async def lesson_upload(
    request: Request,
    lesson_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
    user: Optional[User] = Depends(optional_current_user),
) -> Response:
    """
    Attach a file to the chat. The text excerpt will be passed to the Writer
    as ``teacher_provided_context`` on the next run; the file is *not* sent
    to the Knowledge Graph and does not trigger anything by itself.

    Returns ``partials/chat_attachments.html`` so the chat-input attachment
    tray re-renders with the new chip in place.
    """
    if user is None:
        raise HTTPException(status_code=401)

    result = await session.execute(
        select(Lesson).where(Lesson.id == lesson_id, Lesson.owner_id == user.id)
    )
    lesson = result.scalar_one_or_none()
    if lesson is None:
        raise HTTPException(status_code=404, detail="Lezione non trovata")

    # Uploads are only meaningful before a run starts. Once we're running
    # the agent, the writer has already been seeded with whatever was
    # attached at run-start; we tell the user instead of silently
    # accepting and ignoring.
    if lesson.status not in ("draft", "complete", "error"):
        return HTMLResponse(
            content=(
                '<p class="text-sm text-amber-800 px-2 py-1">'
                "Allegati disabilitati durante la generazione. Riprova quando il run finisce."
                "</p>"
            ),
            status_code=409,
        )

    form = await request.form()
    up = form.get("file")
    if up is None or not hasattr(up, "read"):
        return HTMLResponse(
            content='<p class="text-sm text-rose-700 px-2 py-1">Nessun file selezionato.</p>',
            status_code=422,
        )
    content = await up.read()  # type: ignore[union-attr]
    filename = getattr(up, "filename", None) or "upload"

    try:
        manifest, _entry = save_upload(lesson_id, filename, content, lesson.uploaded_files_json)
        lesson.uploaded_files_json = manifest
        await session.commit()
        await session.refresh(lesson)
    except ValueError as exc:
        return HTMLResponse(
            content=f'<p class="text-sm text-rose-700 px-2 py-1">{exc}</p>',
            status_code=422,
        )

    return templates.TemplateResponse(
        "partials/chat_attachments.html",
        {"request": request, "lesson": lesson},
    )


@router.delete(
    "/lesson/{lesson_id}/upload/{file_id}",
    response_class=HTMLResponse,
    name="webui_lesson_upload_delete",
)
async def lesson_upload_delete(
    request: Request,
    lesson_id: uuid.UUID,
    file_id: str,
    session: AsyncSession = Depends(get_async_session),
    user: Optional[User] = Depends(optional_current_user),
) -> Response:
    """Remove a chat attachment chip; returns the updated tray partial."""
    if user is None:
        raise HTTPException(status_code=401)

    result = await session.execute(
        select(Lesson).where(Lesson.id == lesson_id, Lesson.owner_id == user.id)
    )
    lesson = result.scalar_one_or_none()
    if lesson is None:
        raise HTTPException(status_code=404, detail="Lezione non trovata")

    if lesson.status == "running":
        return HTMLResponse(
            content=(
                '<p class="text-sm text-amber-800 px-2 py-1">'
                "Eliminazione disabilitata durante la generazione."
                "</p>"
            ),
            status_code=409,
        )

    try:
        manifest = delete_upload(lesson_id, file_id, lesson.uploaded_files_json)
        lesson.uploaded_files_json = manifest
        await session.commit()
        await session.refresh(lesson)
    except ValueError as exc:
        return HTMLResponse(
            content=f'<p class="text-sm text-rose-700 px-2 py-1">{exc}</p>',
            status_code=404,
        )

    return templates.TemplateResponse(
        "partials/chat_attachments.html",
        {"request": request, "lesson": lesson},
    )


# ----------------------------------------------------------------------------
# GET /webui/lesson/{id}/stream — SSE feed of agent events (P2)
# ----------------------------------------------------------------------------


@router.get(
    "/lesson/{lesson_id}/stream",
    name="webui_lesson_stream",
)
async def lesson_stream(
    request: Request,
    lesson_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
    user: Optional[User] = Depends(optional_current_user),
) -> Response:
    """
    Server-Sent Events feed for an in-flight (or already terminal) lesson run.

    Behaviour by ``lesson.status`` at connection time:
        draft     → emit a single 'error' event ("Run non avviato. Premi
                    'Genera lezione'.") and close. Defensive: only fires
                    when a user hand-types the stream URL.
        running   → drive ``run_agent_stream`` and forward each StreamEvent
                    as an SSE event. Mutates ``lesson.status`` to "complete"
                    or "error" via the service.
        complete  → emit a single 'final' event with the persisted lesson
                    card so a reload after the run finished still paints
                    the result without re-running the agent.
        error     → emit a single 'error' event with the persisted message.

    Each event maps 1:1 to a per-event partial in ``partials/chat_*.html``.
    htmx-SSE on the client uses ``sse-swap="card,final,error"`` with
    ``hx-swap="beforeend"`` to dispatch each fragment into #chat-cards.
    """
    if user is None:
        # SSE clients can't follow HTML redirects mid-handshake — return 401
        # and let the page-level auth bounce handle the user's next click.
        raise HTTPException(status_code=401, detail="Auth required")

    result = await session.execute(
        select(Lesson).where(Lesson.id == lesson_id, Lesson.owner_id == user.id)
    )
    lesson = result.scalar_one_or_none()
    if lesson is None:
        raise HTTPException(status_code=404, detail="Lezione non trovata")

    async def event_generator():
        # The terminating ``end`` marker that lets the htmx-SSE client
        # close its EventSource via ``sse-close="end"``. Without it the
        # browser's default 3-second auto-reconnect kicks in and we'd
        # replay the whole run forever.
        terminal_marker = {"event": "end", "data": "ok"}

        # ── REPLAY paths ─────────────────────────────────────────────
        if lesson.status == "complete":
            # Send the same small placeholder used by the live run path.
            # chat_pane.html's htmx:sseClose handler fetches the full card
            # via GET /card-fragment after the stream closes.
            str(lesson.id)
            yield {
                "event": "final",
                "data": (
                    '<div id="lesson-card-loading" class="flex items-start gap-3">'
                    '<div class="flex-shrink-0 w-9 h-9 rounded-full bg-slate-800 text-white'
                    ' flex items-center justify-center ring-2 ring-white shadow-sm">'
                    '<wa-icon name="book-open" style="font-size:1rem;"></wa-icon></div>'
                    '<div class="flex-1 min-w-0"><div class="rounded-xl border border-slate-200'
                    ' bg-white shadow-sm px-4 py-3 text-sm text-slate-500 animate-pulse">'
                    "Caricamento lezione…</div></div></div>"
                ),
            }
            yield terminal_marker
            return

        if lesson.status == "error":
            yield {
                "event": "error",
                "data": _render_partial(
                    request,
                    "partials/chat_error.html",
                    {"lesson": lesson, "error": lesson.error_message or "Errore sconosciuto"},
                ),
            }
            yield terminal_marker
            return

        if lesson.status == "draft":
            yield {
                "event": "error",
                "data": _render_partial(
                    request,
                    "partials/chat_error.html",
                    {
                        "lesson": lesson,
                        "error": ("Esecuzione non avviata. Invia prima un messaggio dalla chat."),
                    },
                ),
            }
            yield terminal_marker
            return

        # status == "running". Guard against a duplicate concurrent attach
        # (second tab, aggressive reconnect) — only one connection drives
        # the agent at a time, the rest get a friendly error and close.
        if lesson.id in _ACTIVE_RUNS:
            yield {
                "event": "error",
                "data": _render_partial(
                    request,
                    "partials/chat_error.html",
                    {
                        "lesson": lesson,
                        "error": (
                            "Una generazione è già in corso in un'altra finestra. "
                            "Attendi il termine e ricarica la pagina."
                        ),
                    },
                ),
            }
            yield terminal_marker
            return

        # ── Phase 1: planner-only run + intent selection card ────────────
        # Triggered when the teacher submits their first query on a draft
        # lesson.  We run only the Planner, show its card, then show the
        # intent selection card and let the SSE stream end.  The teacher
        # picks an intent from the card → POST /run with intent_confirmed=1
        # → Phase 2 (full pipeline) starts.
        _profile_j = lesson.educational_profile_json or {}
        if _profile_j.get("__planner_only__"):
            _ACTIVE_RUNS.add(lesson.id)
            try:
                from aix.agent.agents.planner_agent import PlannerAgent
                from aix.api.schemas.educational_profile import (
                    PEDAGOGICAL_INTENT_OPTIONS as _INTENT_OPTS,
                )

                # Pop the flag so a reconnect doesn't re-trigger phase 1
                _updated_profile = dict(_profile_j)
                _updated_profile.pop("__planner_only__", None)
                lesson.educational_profile_json = _updated_profile

                _planner = PlannerAgent()
                _plan = await _planner.plan(
                    query=lesson.teacher_query or "",
                    domain=lesson.domain or "neuro",
                    language="it",
                )

                _planner_payload = {
                    "query_intent": _plan.query_intent,
                    "intent_label": _plan.query_intent,
                    "key_concepts": _plan.key_concepts,
                    "search_queries": _plan.search_queries,
                    "lesson_type": _plan.lesson_type,
                    "target_grade": _plan.target_grade,
                    "reasoning": _plan.reasoning,
                    "nodes_count": 0,
                    "scope_status": _plan.scope_status,
                }
                yield {
                    "event": "card",
                    "data": _render_partial(
                        request,
                        "partials/chat_planner_card.html",
                        {"payload": _planner_payload, "lesson": lesson},
                    ),
                }
                yield {
                    "event": "card",
                    "data": _render_partial(
                        request,
                        "partials/chat_intent_card.html",
                        {
                            "lesson": lesson,
                            "PEDAGOGICAL_INTENT_OPTIONS": _INTENT_OPTS,
                        },
                    ),
                }

                # Reset to draft — teacher needs to confirm intent before
                # the full run proceeds.
                lesson.status = "draft"
                await session.commit()

            except Exception as _exc:
                logger.exception("[stream] phase1 failed lesson_id=%s", lesson.id)
                _msg = str(_exc)[:480]
                lesson.status = "error"
                lesson.error_message = _msg
                await session.commit()
                yield {
                    "event": "error",
                    "data": _render_partial(
                        request,
                        "partials/chat_error.html",
                        {"lesson": lesson, "error": _msg},
                    ),
                }
            finally:
                _ACTIVE_RUNS.discard(lesson.id)
                yield terminal_marker
            return

        _ACTIVE_RUNS.add(lesson.id)
        try:
            async for event in run_agent_stream(lesson, session):
                # Bail early if the client disconnected (closed tab,
                # navigated away). The agent in run_agent_stream still
                # progresses to a terminal state in the background; the
                # status row gets updated even though no SSE listener
                # consumes the rest of the events.
                if await request.is_disconnected():
                    logger.info(
                        "[stream] client disconnected lesson_id=%s; "
                        "agent continues to terminal state",
                        lesson.id,
                    )
                    break

                sse_message = _stream_event_to_sse(request, lesson, event)
                if sse_message is not None:
                    logger.info(
                        "[stream] → SSE lesson_id=%s kind=%s event=%s",
                        lesson.id,
                        event.kind,
                        sse_message["event"],
                    )
                    yield sse_message
        finally:
            _ACTIVE_RUNS.discard(lesson.id)
            # End-of-stream marker. Triggers ``sse-close="end"`` on the
            # client, which calls ``eventSource.close()`` and prevents
            # the browser's default auto-reconnect.
            yield terminal_marker

    return EventSourceResponse(event_generator())


# ----------------------------------------------------------------------------
# Writer token stream — live typewriter endpoint
# ----------------------------------------------------------------------------


@router.get(
    "/lesson/{lesson_id}/writer-stream",
    name="webui_lesson_writer_stream",
)
async def lesson_writer_stream(
    request: Request,
    lesson_id: uuid.UUID,
    user: Optional[User] = Depends(optional_current_user),
) -> Response:
    """Thin SSE endpoint that forwards live Writer tokens to the browser.

    Opened by ``chat_writer_pending.html`` as a dedicated EventSource the
    moment the pending card mounts.  The main agent SSE stream is blocked
    inside ``graph.astream()`` while the write node runs, so tokens cannot
    flow through it.  Instead, ``write_node`` puts each delta into a
    per-session ``asyncio.Queue`` (the token bus registered in service.py);
    this endpoint drains that queue and re-emits each delta as a
    ``writer_chunk`` SSE event so the user sees a typewriter effect.
    """
    if user is None:
        return Response(status_code=401)

    from aix.agent.graph import write_stream as _ws

    async def token_gen():
        sid = str(lesson_id)
        bus = _ws.get_bus(sid)
        # The frontend may open this connection before the write node starts;
        # poll briefly (up to 5 s) for the bus to be registered.
        for _ in range(50):
            if bus is not None:
                break
            await asyncio.sleep(0.1)
            bus = _ws.get_bus(sid)
        if bus is None:
            yield {"event": "end", "data": "no-bus"}
            return
        while True:
            try:
                token = await asyncio.wait_for(bus.get(), timeout=120.0)
            except asyncio.TimeoutError:
                break
            if token is None:
                break
            if await request.is_disconnected():
                break
            if isinstance(token, tuple):
                event_type, data = token
                event_name = "think_chunk" if event_type == "think" else "writer_chunk"
            else:
                event_name = "writer_chunk"
                data = token
            yield {"event": event_name, "data": data}
        yield {"event": "end", "data": "ok"}

    return EventSourceResponse(token_gen())


# ----------------------------------------------------------------------------
# Inline profile editing (P2 phase 2)
# ----------------------------------------------------------------------------


@router.get(
    "/lesson/{lesson_id}/profile",
    response_class=HTMLResponse,
    name="webui_lesson_profile",
)
async def lesson_profile(
    request: Request,
    lesson_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
    user: Optional[User] = Depends(optional_current_user),
) -> Response:
    """Return the read-only profile sidebar partial (used by the Annulla button)."""
    if user is None:
        raise HTTPException(status_code=401)

    result = await session.execute(
        select(Lesson).where(Lesson.id == lesson_id, Lesson.owner_id == user.id)
    )
    lesson = result.scalar_one_or_none()
    if lesson is None:
        raise HTTPException(status_code=404, detail="Lezione non trovata")

    return templates.TemplateResponse(
        "partials/profile_sidebar.html",
        {"request": request, "lesson": lesson, **_label_dicts()},
    )


@router.get(
    "/lesson/{lesson_id}/profile/edit",
    response_class=HTMLResponse,
    name="webui_lesson_profile_edit",
)
async def lesson_profile_edit(
    request: Request,
    lesson_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
    user: Optional[User] = Depends(optional_current_user),
) -> Response:
    """Return the editable profile sidebar form (used by Aggiorna profilo)."""
    if user is None:
        raise HTTPException(status_code=401)

    result = await session.execute(
        select(Lesson).where(Lesson.id == lesson_id, Lesson.owner_id == user.id)
    )
    lesson = result.scalar_one_or_none()
    if lesson is None:
        raise HTTPException(status_code=404, detail="Lezione non trovata")

    return templates.TemplateResponse(
        "partials/profile_sidebar_edit.html",
        {
            "request": request,
            "lesson": lesson,
            "profile_errors": None,
            **_label_dicts(),
        },
    )


@router.post(
    "/lesson/{lesson_id}/profile",
    response_class=HTMLResponse,
    name="webui_lesson_profile_save",
)
async def lesson_profile_save(
    request: Request,
    lesson_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
    user: Optional[User] = Depends(optional_current_user),
) -> Response:
    """
    Validate the inline profile form and persist the updated profile on the
    lesson row. Returns the read-only sidebar on success, or re-renders the
    edit form with errors.
    """
    if user is None:
        raise HTTPException(status_code=401)

    result = await session.execute(
        select(Lesson).where(Lesson.id == lesson_id, Lesson.owner_id == user.id)
    )
    lesson = result.scalar_one_or_none()
    if lesson is None:
        raise HTTPException(status_code=404, detail="Lezione non trovata")

    form = await request.form()

    try:
        profile_dict = form_to_profile_dict(form)
    except ValidationError as exc:
        logger.info("Profile edit validation failed: %s", exc.errors())
        return templates.TemplateResponse(
            "partials/profile_sidebar_edit.html",
            {
                "request": request,
                "lesson": lesson,
                "profile_errors": [
                    f"{'.'.join(str(p) for p in err.get('loc', ()))}: "
                    f"{err.get('msg', 'campo non valido')}"
                    for err in exc.errors()
                ],
                **_label_dicts(),
            },
            status_code=422,
        )

    lesson.educational_profile_json = profile_dict

    new_domain = str(form.get("domain", lesson.domain) or lesson.domain).strip()
    if new_domain in {"neuro", "udl", "all"}:
        lesson.domain = new_domain

    lesson_title = str(form.get("lesson_title", "") or "").strip()
    lesson.title = lesson_title or None

    await session.commit()
    await session.refresh(lesson)

    logger.info(
        "✏️  Profile updated inline: lesson_id=%s owner=%s domain=%s title=%r",
        lesson.id,
        user.id,
        lesson.domain,
        lesson.title,
    )

    return templates.TemplateResponse(
        "partials/profile_sidebar.html",
        {"request": request, "lesson": lesson, **_label_dicts()},
    )


# ----------------------------------------------------------------------------
# GET /webui/lesson/{id}/export — download lesson as MD or TXT
# GET /webui/lesson/{id}/print  — print-friendly page (browser → PDF)
# ----------------------------------------------------------------------------


@router.get(
    "/lesson/{lesson_id}/export",
    name="webui_lesson_export",
)
async def lesson_export(
    lesson_id: uuid.UUID,
    format: str = "md",
    session: AsyncSession = Depends(get_async_session),
    user: Optional[User] = Depends(optional_current_user),
) -> Response:
    """Download the lesson plan as markdown or plain text."""
    if user is None:
        raise HTTPException(status_code=401)

    result = await session.execute(
        select(Lesson).where(Lesson.id == lesson_id, Lesson.owner_id == user.id)
    )
    lesson = result.scalar_one_or_none()
    if lesson is None:
        raise HTTPException(status_code=404, detail="Lezione non trovata")
    if not lesson.lesson_plan_md:
        raise HTTPException(status_code=404, detail="Nessun contenuto disponibile")

    slug = (lesson.title or f"lezione-{lesson.id}").lower()
    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in slug).strip("-")[:60]

    if format == "md":
        return Response(
            content=lesson.lesson_plan_md,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{slug}.md"'},
        )

    if format == "txt":
        import re

        text = lesson.lesson_plan_md
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # links
        text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)  # images
        text = re.sub(r"#{1,6}\s*", "", text)  # headings
        text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)  # bold/italic
        text = re.sub(r"`{1,3}[^`]*`{1,3}", lambda m: m.group().strip("`"), text)  # code
        text = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", text)  # underscores
        text = re.sub(r"^\s*[-*+]\s+", "• ", text, flags=re.MULTILINE)  # bullets
        return Response(
            content=text,
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{slug}.txt"'},
        )

    raise HTTPException(
        status_code=400, detail="Formato non supportato. Usa ?format=md o ?format=txt"
    )


@router.get(
    "/lesson/{lesson_id}/print",
    response_class=HTMLResponse,
    name="webui_lesson_print",
)
async def lesson_print(
    request: Request,
    lesson_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
    user: Optional[User] = Depends(optional_current_user),
) -> Response:
    """Print-friendly page — the browser's Print → Save as PDF handles PDF export."""
    if user is None:
        return _bounce_to_login(f"/webui/lesson/{lesson_id}/print")

    result = await session.execute(
        select(Lesson).where(Lesson.id == lesson_id, Lesson.owner_id == user.id)
    )
    lesson = result.scalar_one_or_none()
    if lesson is None:
        raise HTTPException(status_code=404, detail="Lezione non trovata")

    return templates.TemplateResponse(
        "pages/lesson_print.html",
        {
            "request": request,
            "lesson": lesson,
            "lesson_plan_html": _markdown_to_html(lesson.lesson_plan_md or ""),
        },
    )


# ----------------------------------------------------------------------------
# SSE event translation (P2 phase 2)
# ----------------------------------------------------------------------------


def _stream_event_to_sse(
    request: Request,
    lesson: Lesson,
    event: Any,  # StreamEvent — typed Any to avoid a circular import
) -> Optional[dict]:
    """
    Translate a ``StreamEvent`` from the agent service into the SSE wire
    format consumed by htmx-SSE.

    Returns ``None`` if the event has no client-visible representation
    (currently never — every kind has a card). The caller logs and yields
    the dict.

    The ``card`` event is the workhorse: planner / retriever / writer /
    critic all use it. The retriever event also bundles the media-panel
    OOB swap into the same ``card`` payload so the right sidebar updates
    in lockstep with the retriever card landing in the chat. The writer
    card uses ``hx-swap-oob`` on its root id to replace the matching
    pending card in place.

    Terminal events ('done' → ``final``, 'error' → ``error``) get their
    own SSE event names so the chat pane's ``sse-close="end"`` semantics
    behave consistently regardless of how the run finished.
    """
    if event.kind == "planner":
        return {
            "event": "card",
            "data": _render_partial(
                request,
                "partials/chat_planner_card.html",
                {"payload": event.payload, "lesson": lesson},
            ),
        }

    if event.kind == "retriever_pending":
        return {
            "event": "card",
            "data": _render_partial(
                request,
                "partials/chat_retriever_pending.html",
                {"payload": event.payload, "lesson": lesson},
            ),
        }

    if event.kind == "retriever":
        # Two root elements in one payload: the chat card (no OOB → lands
        # in #chat-cards via beforeend) AND the media panel (OOB → htmx
        # extracts and replaces #media-panel separately).
        retriever_html = _render_partial(
            request,
            "partials/chat_retriever_card.html",
            {"payload": event.payload},
        )
        media_html = _render_partial(
            request,
            "partials/media_panel.html",
            {
                "media": (event.payload or {}).get("media") or {},
                "oob": True,
                # Phase 1b — this OOB swap happens *during an active run* (the
                # retriever just produced curated media), which is the only moment
                # the live layer should enrich the panel. Reaching this card means
                # a real teacher query is in flight, so the slot is enabled here
                # (and only here) to lazy-load live resources off the critical path.
                "media_live_enabled": _media_live_enabled(),
                "media_live_ready": True,
                "lesson_id": lesson.id if lesson else None,
            },
        )
        return {
            "event": "card",
            "data": retriever_html + " " + media_html,
        }

    if event.kind == "writer_chunk":
        # Raw token from the streaming Writer. Sent as a lightweight
        # ``writer_chunk`` SSE event (NOT ``card``) so the client-side JS
        # handler can append directly to the pending card's stream div
        # without htmx trying to parse it as an HTML fragment.
        return {
            "event": "writer_chunk",
            "data": (event.payload or {}).get("token", ""),
        }

    if event.kind == "writer_pending":
        return {
            "event": "card",
            "data": _render_partial(
                request,
                "partials/chat_writer_pending.html",
                {"payload": event.payload, "lesson": lesson},
            ),
        }

    if event.kind == "critic_pending":
        return {
            "event": "card",
            "data": _render_partial(
                request,
                "partials/chat_critic_pending.html",
                {"payload": event.payload, "lesson": lesson},
            ),
        }

    if event.kind == "writer":
        # The writer card's wrapper carries ``hx-swap-oob="outerHTML"`` and
        # an id matching the pending card — htmx replaces that pending
        # card in place rather than appending a new one.
        return {
            "event": "card",
            "data": _render_partial(
                request,
                "partials/chat_writer_card.html",
                {
                    "payload": event.payload,
                    "lesson_plan_md": event.lesson_plan_md or "",
                    "lesson_plan_html": _markdown_to_html(event.lesson_plan_md or ""),
                },
            ),
        }

    if event.kind == "critic":
        return {
            "event": "card",
            "data": _render_partial(
                request,
                "partials/chat_critic_card.html",
                {"payload": event.payload},
            ),
        }

    if event.kind == "done":
        # Send a small placeholder; chat_pane.html's htmx:sseClose handler
        # fetches the full lesson card via GET /card-fragment (no SSE size limit).
        #
        # CORE 2 #10.1 — also append an OOB-marked re-render of chat_input
        # so the input switches from the disabled "running" state to the
        # active "complete" state without a page reload. ``lesson.status``
        # has already been mutated to "complete" by run_agent_stream BEFORE
        # this event was yielded (see service.run_agent_stream lifecycle
        # contract), so the partial renders the right state inside the OOB
        # wrapper.
        placeholder = (
            '<div id="lesson-card-loading" class="flex items-start gap-3">'
            '<div class="flex-shrink-0 w-9 h-9 rounded-full bg-slate-800 text-white'
            ' flex items-center justify-center ring-2 ring-white shadow-sm">'
            '<wa-icon name="book-open" style="font-size:1rem;"></wa-icon></div>'
            '<div class="flex-1 min-w-0"><div class="rounded-xl border border-slate-200'
            ' bg-white shadow-sm px-4 py-3 text-sm text-slate-500 animate-pulse">'
            "Caricamento lezione finalizzata…</div></div></div>"
        )
        oob_input = _render_partial(
            request,
            "partials/chat_input.html",
            {"lesson": lesson, "_oob": True},
        )
        return {
            "event": "final",
            "data": placeholder + oob_input,
        }

    if event.kind == "error":
        # CORE 2 #10.1 — same OOB rationale as the ``done`` branch: the
        # error path mutates ``lesson.status = "error"`` BEFORE yielding,
        # so re-rendering the input now lands the user back on the active
        # "Riprova" affordance.
        error_card = _render_partial(
            request,
            "partials/chat_error.html",
            {"lesson": lesson, "error": event.error or "Errore sconosciuto"},
        )
        oob_input = _render_partial(
            request,
            "partials/chat_input.html",
            {"lesson": lesson, "_oob": True},
        )
        return {
            "event": "error",
            "data": error_card + oob_input,
        }

    # Unknown kind — log and skip rather than 500-ing the stream.
    logger.warning("[stream] unhandled StreamEvent.kind=%r", event.kind)
    return None


# ----------------------------------------------------------------------------
# GET /webui/profiles — saved-profile selector partial (htmx target)
# POST /webui/profiles — create a named saved profile from form data
# DELETE /webui/profiles/{profile_id} — delete a saved profile
# ----------------------------------------------------------------------------


@router.get(
    "/profiles",
    response_class=HTMLResponse,
    name="webui_profiles_list",
)
async def saved_profiles_list(
    request: Request,
    user: Optional[User] = Depends(optional_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> Response:
    """Return the saved-profiles selector partial (used by htmx after create/delete)."""
    if user is None:
        raise HTTPException(status_code=401)

    result = await session.execute(
        select(SavedProfile)
        .where(SavedProfile.owner_id == user.id)
        .order_by(SavedProfile.created_at.desc())
    )
    saved_profiles = list(result.scalars().all())
    return templates.TemplateResponse(
        "partials/saved_profiles.html",
        {"request": request, "saved_profiles": saved_profiles},
    )


@router.post(
    "/profiles",
    response_class=HTMLResponse,
    name="webui_profiles_create",
)
async def saved_profiles_create(
    request: Request,
    user: Optional[User] = Depends(optional_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> Response:
    """Save the current form state as a named profile."""
    if user is None:
        raise HTTPException(status_code=401)

    form = await request.form()
    name_raw = form.get("profile_name")
    name = name_raw.strip() if isinstance(name_raw, str) and name_raw.strip() else None
    if not name:
        raise HTTPException(status_code=422, detail="Nome profilo obbligatorio")

    try:
        profile_dict = form_to_profile_dict(form)
    except Exception as exc:
        logger.warning("SavedProfile form parse failed: %s", exc)
        raise HTTPException(status_code=422, detail="Profilo non valido") from exc

    sp = SavedProfile(owner_id=user.id, name=name, profile_json=profile_dict)
    session.add(sp)
    await session.commit()

    result = await session.execute(
        select(SavedProfile)
        .where(SavedProfile.owner_id == user.id)
        .order_by(SavedProfile.created_at.desc())
    )
    saved_profiles = list(result.scalars().all())
    return templates.TemplateResponse(
        "partials/saved_profiles.html",
        {"request": request, "saved_profiles": saved_profiles},
    )


@router.delete(
    "/profiles/{profile_id}",
    name="webui_profiles_delete",
)
async def saved_profiles_delete(
    profile_id: uuid.UUID,
    user: Optional[User] = Depends(optional_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> Response:
    """Delete a saved profile and return the updated partial."""
    if user is None:
        raise HTTPException(status_code=401)

    result = await session.execute(
        select(SavedProfile).where(
            SavedProfile.id == profile_id,
            SavedProfile.owner_id == user.id,
        )
    )
    sp = result.scalar_one_or_none()
    if sp is None:
        raise HTTPException(status_code=404)

    await session.delete(sp)
    await session.commit()
    return Response(status_code=204)


# ----------------------------------------------------------------------------
# Helpers (P2)
# ----------------------------------------------------------------------------

# Markdown → HTML rendering. We pick a small, predictable set of extensions:
#   - fenced_code  : ``` ... ``` blocks (writer often uses these)
#   - tables       : pipe tables for activity timetables
#   - sane_lists   : avoids Markdown's surprising list quirks
# Everything else is intentionally off — keep the surface small and the
# output predictable for the prose-styling Tailwind classes in the template.
_MARKDOWN_EXTENSIONS = ["fenced_code", "tables", "sane_lists"]


def _markdown_to_html(text: str) -> str:
    """Render ``text`` as HTML using our pinned extension set."""
    return md.markdown(text or "", extensions=_MARKDOWN_EXTENSIONS, output_format="html5")


def _render_partial(request: Request, template_name: str, ctx: dict) -> str:
    """
    Render a Jinja2 partial to a single line of HTML suitable for an SSE
    ``data:`` field.

    SSE multiline payloads require each line to be prefixed with ``data: ``;
    most HTML fragments contain newlines (one per element on a pretty render)
    and ``sse-starlette`` handles the prefixing for us, but htmx-SSE has
    historically had quirks with multiline payloads. We squash all newlines
    here so each event is exactly one ``data:`` line — robust against both
    the spec and every htmx-SSE version we've tested.
    """
    rendered = templates.get_template(template_name).render({"request": request, **ctx})
    # Collapse the raw HTML to a single line. This loses indentation but
    # preserves text node whitespace via the elements' own block layout.
    return " ".join(rendered.split())
