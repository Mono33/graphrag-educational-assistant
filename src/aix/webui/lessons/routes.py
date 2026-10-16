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
import html
import logging
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

# A JSON column mutated in place is invisible to the unit of work: the list
# object is the same one it already knows about, so nothing looks dirty. The
# shadow annotation edits an entry inside ``uploaded_files_json`` and has to say
# so explicitly.
from sqlalchemy.orm.attributes import flag_modified
from sse_starlette.sse import EventSourceResponse

from aix.api.schemas.educational_profile import (
    CLASS_FEATURE_LABELS,
    DISABILITY_LABELS,
    FORNITURE_MOBILITY_LABELS,
    GRADE_LABELS,
    LEGACY_DISABILITY_CODES,
    OWN_DEVICE_LABELS,
    PEDAGOGICAL_INTENT_OPTIONS,
    STUDENT_ATTR_LABELS,
)
from aix.core.ai_marking import (
    AI_GENERATED_HEADER,
    AI_GENERATED_HEADER_VALUE,
    DISCLOSURE_LONG,
    DISCLOSURE_SHORT,
)
from aix.core.run_registry import get_run_registry
from aix.export.service import ExportRefused, render_lesson_export
from aix.webui.agent.turns import (
    TurnRefused,
    accept_turn,
    drive_turn,
    refinement_label,  # noqa: F401 — re-exported, the tests import it from here
    remember_graph_search,  # noqa: F401 — re-exported, the tests import it from here
)
from aix.webui.auth.csrf import token_for as _csrf_token_for
from aix.webui.auth.dependencies import optional_current_user
from aix.webui.auth.models import User
from aix.webui.db import get_async_session
from aix.webui.errors import report
from aix.webui.features import (
    material_library_enabled,
)
from aix.webui.features import (
    register as _register_feature_flags,
)
from aix.webui.lessons.display import lesson_to_row
from aix.webui.lessons.live_media import _media_live_enabled, live_media_for_lesson
from aix.webui.lessons.media_snapshots import (
    latest_display_snapshot_safe,
    latest_user_turn,
    save_live_snapshot,
    snapshot_for_turn_safe,
)
from aix.webui.lessons.models import Lesson, LessonMessage, SavedProfile
from aix.webui.lessons.schemas import form_to_profile_dict, profile_to_form_values
from aix.webui.lessons.uploads import (
    MAX_FILE_BYTES,
    delete_upload,
    save_upload,
    too_large_message,
)
from aix.webui.materials.citations import (
    labels_for_lesson_safe,
    with_sources,
)
from aix.webui.materials.readiness import states_for_lesson_safe
from aix.webui.materials.repository import BindingRepository
from aix.webui.materials.saved_resources import saved_keys_safe
from aix.webui.materials.shadow import record_upload, retire_material
from aix.webui.sanitize import sanitize_html

logger = logging.getLogger(__name__)


_PACKAGE_DIR = Path(__file__).resolve().parents[1]
_TEMPLATES_DIR = _PACKAGE_DIR / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

# The webui keeps three Jinja environments over the same templates directory
# (``aix.webui.routes``, ``aix.webui.lessons.routes``, ``aix.webui.auth.routes``),
# so the CSRF global has to be registered on each of them. A global rather than
# a per-route context entry: a page added without the token would fail its first
# POST with a 403, and the author would find out in production.
templates.env.globals["csrf_token"] = _csrf_token_for

# §5.2: what the templates offer must be what the server accepts, and both must
# follow the flags. A hard-coded accept list in the HTML is how the paperclip
# came to advertise formats the previous product never took.
_register_feature_flags(templates)


router = APIRouter(prefix="/webui", tags=["webui-lessons"])


# In-flight agent run registry (CORE 6 #37). Used by the SSE endpoint to
# detect when a second client (a duplicate browser tab, an aggressive
# auto-reconnect) tries to attach to a lesson we're already streaming, and to
# gracefully surface that as an error rather than silently kicking off a
# second concurrent run.
#
# Backend is chosen at runtime by ``get_run_registry()``:
#   • SQLite / single-worker dev → in-memory set (the original behaviour;
#     nothing about the dev workflow changes, and a restart simply clears it —
#     so a crash never permanently blocks re-running a lesson).
#   • Postgres / multi-worker prod → shared ``agent_run`` table + heartbeat, so
#     workers/replicas see each other's in-flight runs and stale rows from a
#     crashed worker self-heal after one TTL. This is what unblocks #39.
#
# Both backends expose the same async surface (claim / heartbeat / release /
# is_active), so the lifecycle below is identical regardless of backend.


def _label_dicts() -> dict[str, Any]:
    """Bundle of Italian labels passed into every lesson template."""
    return {
        "GRADE_LABELS": GRADE_LABELS,
        "DISABILITY_LABELS": DISABILITY_LABELS,
        # Pickers offer the platform's current codes; a legacy one only where a
        # profile already has it, so saving the form does not drop it.
        "LEGACY_DISABILITY_CODES": LEGACY_DISABILITY_CODES,
        "CLASS_FEATURE_LABELS": CLASS_FEATURE_LABELS,
        "STUDENT_ATTR_LABELS": STUDENT_ATTR_LABELS,
        "FORNITURE_MOBILITY_LABELS": FORNITURE_MOBILITY_LABELS,
        "OWN_DEVICE_LABELS": OWN_DEVICE_LABELS,
        "PEDAGOGICAL_INTENT_OPTIONS": PEDAGOGICAL_INTENT_OPTIONS,
    }


async def _load_chat_messages(
    session: AsyncSession,
    lesson_id: uuid.UUID,
    labels: Optional[dict[str, str]] = None,
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
                    _markdown_to_html(msg.content_md or "", labels)
                    if msg.role == "assistant"
                    else None
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
        request,
        "pages/lesson_list.html",
        {
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

    # A light turn answered: show the answer, not the lesson again.
    latest = (
        await session.execute(
            select(LessonMessage)
            .where(LessonMessage.lesson_id == lesson.id)
            .order_by(LessonMessage.turn_index.desc(), LessonMessage.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if latest is not None and latest.role == "assistant" and latest.agent_kind == "tutor":
        return templates.TemplateResponse(
            request,
            "partials/chat_light_card.html",
            {
                "lesson": lesson,
                "pending": False,
                "turn": latest.turn_index,
                "meta": latest.meta_json or {},
                "answer_html": _markdown_to_html(
                    latest.content_md or "", await labels_for_lesson_safe(session, lesson)
                ),
            },
        )

    # The run's own verdict, as recorded with its answer. This card used to be
    # handed ``{"approved": status == "complete", "revision_count": 0}``, so
    # every finished lesson read "approvata dal critic · 0 revisioni" -- one the
    # Critic had sent back twice and never approved included, while the log
    # said ``approved=False revisions=2`` (2026-09-25). No record, no badge:
    # never an invented one.
    run_meta = await session.scalar(
        select(LessonMessage.meta_json)
        .where(
            LessonMessage.lesson_id == lesson.id,
            LessonMessage.role == "assistant",
            func.coalesce(LessonMessage.agent_kind, "") != "tutor",
        )
        .order_by(LessonMessage.turn_index.desc(), LessonMessage.created_at.desc())
        .limit(1)
    )
    return templates.TemplateResponse(
        request,
        "partials/chat_lesson_card.html",
        {
            "lesson": lesson,
            "lesson_plan_html": _markdown_to_html(
                lesson.lesson_plan_md or "",
                await labels_for_lesson_safe(session, lesson),
            ),
            "meta": run_meta if isinstance(run_meta, dict) else {},
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
    if user is None:
        return empty

    result = await session.execute(
        select(Lesson).where(Lesson.id == lesson_id, Lesson.owner_id == user.id)
    )
    lesson = result.scalar_one_or_none()
    if lesson is None:
        return empty

    # Passive opens replay what the teacher already saw.  They never call an
    # external provider: this keeps results stable and makes reload/logout free.
    if lesson.status != "running":
        saved = await latest_display_snapshot_safe(
            session,
            lesson_id=lesson.id,
            owner_id=user.id,
        )
        if saved is None or not saved.live_fetched:
            return empty
        panel = saved.live_json or {}
        if not panel:
            return empty
        return templates.TemplateResponse(
            request,
            "partials/media_live_sections.html",
            {
                "videos": panel.get("videos") or [],
                "citations": panel.get("citations") or [],
                "wikipedia": panel.get("wikipedia") or [],
                "lesson_id": lesson.id,
                "saved_keys": await saved_keys_safe(session, user.id),
            },
        )

    turn_index = await latest_user_turn(session, lesson.id)
    current = await snapshot_for_turn_safe(
        session,
        lesson_id=lesson.id,
        owner_id=user.id,
        turn_index=turn_index,
    )
    if current is not None and current.live_fetched:
        panel = current.live_json or {}
        if not panel:
            return empty
        return templates.TemplateResponse(
            request,
            "partials/media_live_sections.html",
            {
                "videos": panel.get("videos") or [],
                "citations": panel.get("citations") or [],
                "wikipedia": panel.get("wikipedia") or [],
                "lesson_id": lesson.id,
                "saved_keys": await saved_keys_safe(session, user.id),
            },
        )

    # Only an active run may go to live providers (``live_media_for_lesson``
    # checks the flag, the topic and the concepts). ``None``: nothing searched.
    panel = await live_media_for_lesson(lesson)
    if panel is None:
        return empty

    try:
        async with session.begin_nested():
            await save_live_snapshot(
                session,
                lesson_id=lesson.id,
                owner_id=user.id,
                turn_index=turn_index,
                media=panel,
            )
        await session.commit()
    except Exception as exc:  # noqa: BLE001 — live media must stay best effort
        await session.rollback()
        logger.warning("[media-live] snapshot failed for lesson_id=%s: %s", lesson_id, exc)

    if not panel:
        return empty

    return templates.TemplateResponse(
        request,
        "partials/media_live_sections.html",
        {
            "videos": panel.get("videos") or [],
            "citations": panel.get("citations") or [],
            "wikipedia": panel.get("wikipedia") or [],
            "lesson_id": lesson.id,
            "saved_keys": await saved_keys_safe(session, user.id),
        },
    )


# ----------------------------------------------------------------------------
# GET /webui/lesson/{id}/attachments-fragment — the tray, with pipeline state
# ----------------------------------------------------------------------------


@router.get(
    "/lesson/{lesson_id}/attachments-fragment",
    response_class=HTMLResponse,
    name="webui_lesson_attachments_fragment",
)
async def lesson_attachments_fragment(
    request: Request,
    lesson_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
    user: Optional[User] = Depends(optional_current_user),
) -> Response:
    """The attachment tray, told what the pipeline is doing with each document.

    The tray asks for this itself while anything is still being prepared (§6.1),
    which is why the state is computed here and not at the five places that
    render the tray: one of them would have been forgotten, and a chip that
    lies about readiness is worse than a chip that says nothing.

    The response also carries the send button out of band, so it unlocks by
    itself when the last document turns ready — the teacher never has to guess
    whether it is safe to press Invia (decision U1). The button, not the whole
    chat input: swapping the input every two seconds wiped whatever the teacher
    was typing while a document was prepared (2026-09-29).
    """
    if user is None:
        raise HTTPException(status_code=401)

    result = await session.execute(
        select(Lesson).where(Lesson.id == lesson_id, Lesson.owner_id == user.id)
    )
    lesson = result.scalar_one_or_none()
    if lesson is None:
        raise HTTPException(status_code=404, detail="Lezione non trovata")

    states = await states_for_lesson_safe(session, lesson_id=lesson.id, owner_id=user.id)
    preparing = [state for state in states.values() if state.preparing]

    tray = _render_partial(
        request,
        "partials/chat_attachments.html",
        {"lesson": lesson, "material_states": states},
    )
    send_button = _render_partial(
        request,
        "partials/chat_send_button.html",
        {"lesson": lesson, "materials_preparing": bool(preparing), "_send_oob": True},
    )
    return HTMLResponse(content=tray + send_button)


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
        request,
        "partials/chat_input.html",
        {
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
        request,
        "pages/lesson_new.html",
        {
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
            request,
            "pages/lesson_new.html",
            {
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

    # §8.6 / U3 — the names of the documents this lesson may cite, looked up once
    # for the whole page: the plan below and every assistant turn of the history.
    material_labels = await labels_for_lesson_safe(session, lesson)

    # Pre-render the persisted lesson plan to HTML so the lesson card
    # template doesn't have to know which markdown extensions we use.
    # Only the terminal "complete" state needs this — running/error/draft
    # never paint it.
    lesson_plan_html = (
        _markdown_to_html(lesson.lesson_plan_md, material_labels)
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
    messages = await _load_chat_messages(session, lesson.id, material_labels)

    # The right panel is product state, not a disposable trace.  Rehydrate the
    # latest completed full-retrieval turn; light follow-ups intentionally keep
    # the previous snapshot.  Legacy lessons have no row and retain the old
    # empty-panel behaviour.
    media_snapshot = await latest_display_snapshot_safe(
        session,
        lesson_id=lesson.id,
        owner_id=user.id,
    )
    persisted_media = media_snapshot.curated_json if media_snapshot is not None else None
    persisted_live_ready = bool(media_snapshot and media_snapshot.live_fetched)

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
        request,
        "pages/lesson_show.html",
        {
            "title": (lesson.title or "Lezione") + " · AixLearning",
            "phase": "P3 — Chat workspace + allegati",
            "user": user,
            "lesson": lesson,
            "messages": messages,
            "lesson_plan_html": lesson_plan_html,
            "meta": meta_for_replay,
            "media": persisted_media,
            "saved_keys": await saved_keys_safe(session, user.id),
            # A persisted live snapshot is replayed through the existing htmx
            # fragment endpoint.  That endpoint returns stored data and never
            # calls a provider on a passive page open.
            "media_live_enabled": _media_live_enabled() or persisted_live_ready,
            "media_live_ready": persisted_live_ready,
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

    form = await request.form()
    outcome = await accept_turn(session, lesson, form, rate_key=f"webui:{user.id}")
    if isinstance(outcome, TurnRefused):
        message = html.escape(outcome.message, quote=False)
        if outcome.code == "materials_preparing":
            names = ", ".join(html.escape(name) for name in outcome.preparing)
            message = f"Sto ancora preparando {names}. Parte da solo appena è pronto."
        return HTMLResponse(
            content=(
                '<div class="rounded-lg border border-amber-200 bg-amber-50 '
                f'text-amber-900 px-3 py-2 text-sm">{message}</div>'
            ),
            status_code=outcome.status_code,
            headers=({"Retry-After": str(outcome.retry_after)} if outcome.retry_after else None),
        )

    # Reload the chat transcript so the response includes the just-persisted
    # user turn (and any backfilled prior turn) at the bottom of the
    # history. The chat_pane below (running state) will append per-agent
    # cards live as the SSE stream fires; the final assistant message is
    # written by the service on success.
    messages = await _load_chat_messages(
        session, lesson.id, await labels_for_lesson_safe(session, lesson)
    )

    return templates.TemplateResponse(
        request,
        "partials/chat_conversation.html",
        {
            "lesson": lesson,
            "user": user,
            "messages": messages,
            "lesson_plan_html": None,
            "meta": {},
            "media": None,
        },
    )


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
    filename = getattr(up, "filename", None) or "upload"

    # Measured before it is read, not after. ``save_upload`` checks the size
    # too, but by then the whole file is a bytes object in RAM: with the limit
    # at 100 MB a teacher who drags a 400 MB video would cost 400 MB of the
    # pilot VM's 4 GB before being told no. Starlette fills ``size`` from the
    # multipart part, and a client that sends none falls through to the check
    # inside ``save_upload``, which is authoritative either way.
    declared = getattr(up, "size", None)
    if isinstance(declared, int) and declared > MAX_FILE_BYTES:
        logger.info(
            "[upload] refused %s (%.1f MB) for lesson %s before reading it: over the limit",
            filename,
            declared / (1024 * 1024),
            lesson_id,
        )
        return HTMLResponse(
            content=f'<p class="text-sm text-rose-700 px-2 py-1">{html.escape(too_large_message())}</p>',
            status_code=422,
        )

    content = await up.read()  # type: ignore[union-attr]

    try:
        # In a thread: parsing a scan runs OCR, and that must not stall the
        # event loop every other teacher's chat is streaming on.
        manifest, entry = await asyncio.to_thread(
            save_upload, lesson_id, filename, content, lesson.uploaded_files_json
        )
        lesson.uploaded_files_json = manifest
        await session.commit()
        await session.refresh(lesson)
    except ValueError as exc:
        # Logged, because a refusal that only exists in the browser is a
        # support call. A teacher uploaded a 13.5 MB magazine issue and the
        # server recorded nothing but "422 Unprocessable Entity" — from the
        # log alone there was no telling a rejected file from a broken
        # attachment, and finding out took reading the code (2026-09-22).
        #
        # The name and the size, never the bytes: the file is the teacher's.
        logger.info(
            "[upload] refused %s (%.1f MB) for lesson %s: %s",
            filename,
            len(content) / (1024 * 1024),
            lesson_id,
            exc,
        )
        return HTMLResponse(
            content=f'<p class="text-sm text-rose-700 px-2 py-1">{html.escape(str(exc))}</p>',
            status_code=422,
        )

    # Fase 1 · PR 4 — shadow mode. Everything above is untouched and already
    # committed; this runs after, on its own session, and never raises. With
    # AIX_MATERIALS_ENABLED off it is a single boolean check and returns None.
    #
    # The order is deliberate: the teacher's file is safely stored and the
    # response is already decided before the new pipeline is asked to do
    # anything, so a broken pipeline cannot become a failed upload.
    shadow = await record_upload(
        owner_id=user.id, filename=filename, content=content, lesson_id=lesson_id
    )
    if shadow:
        # Written straight onto the JSON column rather than through
        # ``save_upload``: the legacy manifest format gains three optional keys
        # and loses nothing, which is what lets the flag be turned back off.
        #
        # Guarded like the rest: ``record_upload`` cannot raise, but this commit
        # is on the request's own session and could. Losing the annotation costs
        # a little observability; failing the upload over it would cost the
        # teacher their file.
        try:
            # Found by id rather than mutated through ``entry``: the refresh
            # above reloaded the column from the database, so the list the
            # route is holding is no longer the one attached to the lesson and
            # writing into it would update nothing.
            current = next(
                (
                    f
                    for f in (lesson.uploaded_files_json or [])
                    if isinstance(f, dict) and f.get("id") == entry["id"]
                ),
                None,
            )
            if current is not None:
                current.update(shadow)
                flag_modified(lesson, "uploaded_files_json")
                await session.commit()
                await session.refresh(lesson)
        except Exception as exc:  # noqa: BLE001
            await session.rollback()
            report(exc, context=f"shadow manifest annotation lesson={lesson_id}")

    return templates.TemplateResponse(
        request,
        "partials/chat_attachments.html",
        {"lesson": lesson},
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

    # Read before removing: ``delete_upload`` returns what is left, so the
    # entry's shadow ids are gone by the time it comes back.
    doomed = next(
        (
            f
            for f in (lesson.uploaded_files_json or [])
            if isinstance(f, dict) and f.get("id") == file_id
        ),
        None,
    )

    try:
        manifest = delete_upload(lesson_id, file_id, lesson.uploaded_files_json)
        lesson.uploaded_files_json = manifest
        if material_library_enabled() and doomed and doomed.get("material_id"):
            try:
                material_uuid = uuid.UUID(str(doomed["material_id"]))
            except (TypeError, ValueError):
                logger.warning(
                    "[upload-delete] invalid material id on lesson=%s file=%s",
                    lesson.id,
                    file_id,
                )
            else:
                await BindingRepository(session).set_enabled(
                    lesson_id=lesson.id,
                    material_id=material_uuid,
                    owner_id=user.id,
                    enabled=False,
                )
        await session.commit()
        await session.refresh(lesson)
    except ValueError as exc:
        return HTMLResponse(
            content=f'<p class="text-sm text-rose-700 px-2 py-1">{exc}</p>',
            status_code=404,
        )

    # Fase 1 · PR 4 — the shadow material goes with its attachment. Best effort
    # like the rest: an attachment the teacher removed must disappear from the
    # tray whether or not the shadow pipeline is reachable.
    if not material_library_enabled() and doomed and doomed.get("material_id"):
        await retire_material(material_id=doomed["material_id"], owner_id=user.id)

    return templates.TemplateResponse(
        request,
        "partials/chat_attachments.html",
        {"lesson": lesson},
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

        # #37 — shared in-flight run registry (cross-worker on Postgres,
        # in-memory on SQLite dev). Resolved once per stream.
        _registry = get_run_registry()

        def _busy_event() -> dict[str, str]:
            """Friendly 'already running elsewhere' SSE card. Emitted both for
            the up-front duplicate-attach check and when an atomic claim loses a
            cross-worker race after that check."""
            return {
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
        # the agent at a time, the rest get a friendly error and close. On
        # Postgres this check sees runs owned by *other* workers too.
        if await _registry.is_active(lesson.id):
            yield _busy_event()
            yield terminal_marker
            return

        # ── Phase 1: ask for the pedagogical intent, and ask it first ────
        # Triggered when the teacher submits their first query on a draft
        # lesson and no intent is known yet. The card is shown immediately;
        # the teacher picks → POST /run with intent_confirmed=1 → phase 2
        # runs the whole pipeline once.
        #
        # Until 2026-09-21 this branch ran the Planner first, to show its
        # card as a preview. It cost 14 s of a run the teacher then had to
        # wait through again — phase 2 re-plans from scratch — and the two
        # plans differed (6 search queries against 5, different pedagogical
        # concepts), so the analysis shown before the choice was not the
        # analysis the lesson was built on. The intent options are a fixed
        # list; they never came from that plan. Nothing was saved from it.
        _profile_j = lesson.educational_profile_json or {}
        if _profile_j.get("__ask_intent__") or _profile_j.get("__planner_only__"):
            # Atomically claim the lesson. ``None`` → another (live) connection
            # or worker won the race between our is_active() check and here.
            _token = await _registry.claim(lesson.id, owner_id=lesson.owner_id)
            if _token is None:
                yield _busy_event()
                yield terminal_marker
                return
            try:
                from aix.api.schemas.educational_profile import (
                    PEDAGOGICAL_INTENT_OPTIONS as _INTENT_OPTS,
                )

                # Pop the flag so a reconnect doesn't re-trigger phase 1
                _updated_profile = dict(_profile_j)
                _updated_profile.pop("__ask_intent__", None)
                _updated_profile.pop("__planner_only__", None)
                lesson.educational_profile_json = _updated_profile

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
                await _registry.release(lesson.id, _token)
                yield terminal_marker
            return

        # The library's keys, read once: the media panel marks what is saved.
        saved_keys = await saved_keys_safe(session, lesson.owner_id)

        # The slot, the claim, the heartbeat and the run itself: see
        # ``aix.webui.agent.turns.drive_turn``. What is left here is the HTML.
        try:
            async for event in drive_turn(lesson, session, is_disconnected=request.is_disconnected):
                if isinstance(event, TurnRefused):
                    yield {
                        "event": "error",
                        "data": _render_partial(
                            request,
                            "partials/chat_error.html",
                            {"lesson": lesson, "error": event.message},
                        ),
                    }
                    continue
                sse_message = _stream_event_to_sse(request, lesson, event, saved_keys=saved_keys)
                if sse_message is not None:
                    logger.info(
                        "[stream] → SSE lesson_id=%s kind=%s event=%s",
                        lesson.id,
                        event.kind,
                        sse_message["event"],
                    )
                    yield sse_message
        finally:
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
    session: AsyncSession = Depends(get_async_session),
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
    # The text being written is the lesson: only its owner reads it. This route
    # checked the login alone, so any teacher who had a lesson's id could follow
    # someone else's lesson as it was written (2026-10-16).
    owned = await session.scalar(
        select(Lesson.id).where(Lesson.id == lesson_id, Lesson.owner_id == user.id)
    )
    if owned is None:
        return Response(status_code=404)

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
        request,
        "partials/profile_sidebar.html",
        {"lesson": lesson, **_label_dicts()},
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
        request,
        "partials/profile_sidebar_edit.html",
        {
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
            request,
            "partials/profile_sidebar_edit.html",
            {
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
        request,
        "partials/profile_sidebar.html",
        {"lesson": lesson, **_label_dicts()},
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
    """Download the lesson: Markdown, plain text, PDF, Word or the outline."""
    if user is None:
        raise HTTPException(status_code=401)

    result = await session.execute(
        select(Lesson).where(Lesson.id == lesson_id, Lesson.owner_id == user.id)
    )
    lesson = result.scalar_one_or_none()
    if lesson is None:
        raise HTTPException(status_code=404, detail="Lezione non trovata")
    # The body is ``aix.export.service``: the API's session export is the same file.
    try:
        exported = await render_lesson_export(session, lesson, format)
    except ExportRefused as exc:
        raise HTTPException(status_code=exc.status, detail=exc.message) from None
    return Response(
        content=exported.content,
        media_type=exported.media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{exported.filename}"',
            AI_GENERATED_HEADER: AI_GENERATED_HEADER_VALUE,
        },
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
        request,
        "pages/lesson_print.html",
        {
            "lesson": lesson,
            # A printed lesson is the copy that leaves the screen: it names the
            # documents it used, not «materiale allegato».
            "lesson_plan_html": _markdown_to_html(
                lesson.lesson_plan_md or "",
                await labels_for_lesson_safe(session, lesson),
            ),
            # EU AI Act Art. 50 (#21) — visible notice on the print/PDF artifact.
            "ai_disclosure_short": DISCLOSURE_SHORT,
            "ai_disclosure_long": DISCLOSURE_LONG,
        },
        headers={AI_GENERATED_HEADER: AI_GENERATED_HEADER_VALUE},
    )


# ----------------------------------------------------------------------------
# SSE event translation (P2 phase 2)
# ----------------------------------------------------------------------------


def _stream_event_to_sse(
    request: Request,
    lesson: Lesson,
    event: Any,  # StreamEvent — typed Any to avoid a circular import
    saved_keys: Optional[set[str]] = None,
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
                "saved_keys": saved_keys or set(),
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

    # ── Light turns ───────────────────────────────────────────────────
    # The pending card carries ``id="lesson-card-loading"``: when the stream
    # closes, chat_pane.html swaps it for /card-fragment, which renders the
    # saved answer -- the same hand-over a full run uses. A hand-off or a
    # failure removes it instead, so nothing is fetched.
    if event.kind == "light_pending":
        return {
            "event": "card",
            "data": _render_partial(
                request,
                "partials/chat_light_card.html",
                {"lesson": lesson, "pending": True, "turn": (event.payload or {}).get("turn")},
            ),
        }

    if event.kind in ("light_done", "light_escalate", "light_error"):
        oob_input = _render_partial(
            request, "partials/chat_input.html", {"lesson": lesson, "_oob": True}
        )
        if event.kind == "light_done":
            return {"event": "final", "data": oob_input}
        if event.kind == "light_escalate":
            card = _render_partial(
                request,
                "partials/chat_light_escalate.html",
                {"lesson": lesson, "query": (event.payload or {}).get("query", "")},
            )
        else:
            card = _render_partial(
                request,
                "partials/chat_error.html",
                {"lesson": lesson, "error": event.error or "Errore sconosciuto"},
            )
        return {"event": "error", "data": _replace_light_pending(card) + oob_input}

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
        request,
        "partials/saved_profiles.html",
        {"saved_profiles": saved_profiles},
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
        request,
        "partials/saved_profiles.html",
        {"saved_profiles": saved_profiles},
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


def _replace_light_pending(card_html: str) -> str:
    """``card_html`` in place of the light pending card, and the spinner gone."""
    return (
        '<div id="light-turn-end" hx-swap-oob="outerHTML:#lesson-card-loading">'
        f"{card_html}</div>"
        '<div id="chat-spinner" hx-swap-oob="outerHTML"></div>'
    )


def _markdown_to_html(text: str, labels: Optional[dict[str, str]] = None) -> str:
    """Render ``text`` as HTML using our pinned extension set, then sanitise it.

    The sanitiser is here and not at the four call sites because this is the
    only function that produces the HTML those sites mark ``| safe``. The
    Markdown it renders is written by an LLM, from a corpus, from the teacher's
    typing and from text extracted out of files they upload — none of which is
    ours — and python-markdown passes raw HTML straight through.
    """
    # §8.6 — the citation markers are an internal address, resolved here so a raw
    # ``[[material:…]]`` can never reach a teacher from any of the call sites.
    # ``labels`` names the documents where the caller had a session to look them
    # up in; without it the neutral label is used, as before. U3: the same pass
    # appends the list of sources, built from the markers rather than asked of
    # the model. Text already resolved (the export route) has no markers left and
    # so gains nothing here — the block is not appended twice.
    return sanitize_html(
        md.markdown(
            with_sources(text or "", labels), extensions=_MARKDOWN_EXTENSIONS, output_format="html5"
        )
    )


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
