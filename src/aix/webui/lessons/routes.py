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

import logging
import uuid
from pathlib import Path
from typing import Any, Optional

import markdown as md
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from aix.api.schemas.educational_profile import (
    CLASS_FEATURE_LABELS,
    DISABILITY_LABELS,
    FORNITURE_MOBILITY_LABELS,
    GRADE_LABELS,
    OWN_DEVICE_LABELS,
    STUDENT_ATTR_LABELS,
)
from aix.webui.agent import run_agent_stream
from aix.webui.auth.dependencies import optional_current_user
from aix.webui.auth.models import User
from aix.webui.db import get_async_session
from aix.webui.lessons.models import Lesson
from aix.webui.lessons.schemas import form_to_profile_dict
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


def _label_dicts() -> dict[str, dict[str, str]]:
    """Bundle of Italian labels passed into every lesson template."""
    return {
        "GRADE_LABELS": GRADE_LABELS,
        "DISABILITY_LABELS": DISABILITY_LABELS,
        "CLASS_FEATURE_LABELS": CLASS_FEATURE_LABELS,
        "STUDENT_ATTR_LABELS": STUDENT_ATTR_LABELS,
        "FORNITURE_MOBILITY_LABELS": FORNITURE_MOBILITY_LABELS,
        "OWN_DEVICE_LABELS": OWN_DEVICE_LABELS,
    }


def _bounce_to_login(target_path: str) -> RedirectResponse:
    """Redirect anonymous users to /auth/login with ?next= so they come back."""
    return RedirectResponse(
        f"/auth/login?next={target_path}", status_code=303
    )


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
    """List all lessons for the current user, newest first."""
    if user is None:
        return _bounce_to_login("/webui/lessons")

    result = await session.execute(
        select(Lesson)
        .where(Lesson.owner_id == user.id)
        .order_by(Lesson.created_at.desc())
    )
    lessons = result.scalars().all()

    return templates.TemplateResponse(
        "pages/lesson_list.html",
        {"request": request, "user": user, "lessons": lessons},
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

    logger.info(
        "[card-fragment] rendering lesson_id=%s status=%s plan_len=%s",
        lesson_id, lesson.status, len(lesson.lesson_plan_md or ""),
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
) -> Response:
    """Show the EducationalProfile form. Anon users are bounced to login."""
    if user is None:
        return _bounce_to_login("/webui/lesson/new")

    return templates.TemplateResponse(
        "pages/lesson_new.html",
        {
            "request": request,
            "title": "Nuova lezione · AixLearning",
            "phase": "P3 — Chat workspace + allegati",
            "user": user,
            "form_errors": None,
            "form_values": {},
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
        return templates.TemplateResponse(
            "pages/lesson_new.html",
            {
                "request": request,
                "title": "Nuova lezione · AixLearning",
                "phase": "P3 — Chat workspace + allegati",
                "user": user,
                "form_errors": [
                    f"{'.'.join(str(p) for p in err.get('loc', ()))}: "
                    f"{err.get('msg', 'campo non valido')}"
                    for err in exc.errors()
                ],
                "form_values": form_values_for_redisplay,
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
        lesson.id, user.id, lesson.domain, lesson.title,
    )

    return RedirectResponse(
        url=f"/webui/lesson/{lesson.id}", status_code=303
    )


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
    meta_for_replay = (
        {"approved": True, "revision_count": 0}
        if lesson.status == "complete"
        else {}
    )

    return templates.TemplateResponse(
        "pages/lesson_show.html",
        {
            "request": request,
            "title": (lesson.title or "Lezione") + " · AixLearning",
            "phase": "P3 — Chat workspace + allegati",
            "user": user,
            "lesson": lesson,
            "lesson_plan_html": lesson_plan_html,
            "meta": meta_for_replay,
            "media": None,  # not persisted yet — empty placeholder on reload
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

    The actual graph execution happens inside the SSE generator; this
    endpoint just persists the (optional) new ``query``, then returns the
    chat conversation partial which embeds the SSE pane and immediately
    starts streaming planner / retriever / writer / critic cards.

    Form data:
        query   Optional[str] — the teacher's first-turn query, or a
                                rerun-without-edit ping (no body) which
                                reuses the persisted ``teacher_query``.

    Returns ``partials/chat_conversation.html`` so htmx swaps the running
    pane (input → spinner + SSE stream) into ``#chat-conversation``.
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

    # Capture the user's first-turn query if present. The chat-input form
    # always posts a non-empty ``query`` (client-side ``required`` +
    # ``minlength=3``); the regenerate/retry buttons post no body and we
    # reuse the persisted ``teacher_query``.
    form = await request.form()
    query_raw = form.get("query")
    new_query = (
        query_raw.strip()
        if isinstance(query_raw, str) and query_raw.strip()
        else None
    )

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

    # Flip status optimistically so the chat conversation partial we
    # return immediately renders the live SSE pane (rather than the input
    # form). The service layer's setup phase commits this again — that's
    # idempotent and cheaper than a second roundtrip.
    lesson.status = "running"
    lesson.error_message = None
    lesson.lesson_plan_md = None
    await session.commit()
    await session.refresh(lesson)

    logger.info(
        "▶️  Lesson run accepted: id=%s owner=%s has_new_query=%s uploads=%s",
        lesson.id, user.id, bool(new_query),
        len(lesson.uploaded_files_json or []),
    )

    return templates.TemplateResponse(
        "partials/chat_conversation.html",
        {
            "request": request,
            "lesson": lesson,
            "user": user,
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
    content = await up.read()  # type: ignore[union-attr]
    filename = getattr(up, "filename", None) or "upload"

    try:
        manifest, _entry = save_upload(
            lesson_id, filename, content, lesson.uploaded_files_json
        )
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
            lid = str(lesson.id)
            yield {
                "event": "final",
                "data": (
                    f'<div id="lesson-card-loading" class="flex items-start gap-3">'
                    f'<div class="flex-shrink-0 w-9 h-9 rounded-full bg-slate-800 text-white'
                    f' flex items-center justify-center ring-2 ring-white shadow-sm">'
                    f'<wa-icon name="book-open" style="font-size:1rem;"></wa-icon></div>'
                    f'<div class="flex-1 min-w-0"><div class="rounded-xl border border-slate-200'
                    f' bg-white shadow-sm px-4 py-3 text-sm text-slate-500 animate-pulse">'
                    f'Caricamento lezione…</div></div></div>'
                ),
            }
            yield terminal_marker
            return

        if lesson.status == "error":
            yield {
                "event": "error",
                "data": _render_partial(
                    request, "partials/chat_error.html",
                    {"lesson": lesson, "error": lesson.error_message or "Errore sconosciuto"},
                ),
            }
            yield terminal_marker
            return

        if lesson.status == "draft":
            yield {
                "event": "error",
                "data": _render_partial(
                    request, "partials/chat_error.html",
                    {
                        "lesson": lesson,
                        "error": (
                            "Esecuzione non avviata. Invia prima un messaggio "
                            "dalla chat."
                        ),
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
                    request, "partials/chat_error.html",
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
                        lesson.id, event.kind, sse_message["event"],
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
        lesson.id, user.id, lesson.domain, lesson.title,
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
        text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)       # images
        text = re.sub(r"#{1,6}\s*", "", text)                   # headings
        text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)   # bold/italic
        text = re.sub(r"`{1,3}[^`]*`{1,3}", lambda m: m.group().strip("`"), text)  # code
        text = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", text)     # underscores
        text = re.sub(r"^\s*[-*+]\s+", "• ", text, flags=re.MULTILINE)  # bullets
        return Response(
            content=text,
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{slug}.txt"'},
        )

    raise HTTPException(status_code=400, detail="Formato non supportato. Usa ?format=md o ?format=txt")


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
                request, "partials/chat_planner_card.html",
                {"payload": event.payload, "lesson": lesson},
            ),
        }

    if event.kind == "retriever":
        # Two root elements in one payload: the chat card (no OOB → lands
        # in #chat-cards via beforeend) AND the media panel (OOB → htmx
        # extracts and replaces #media-panel separately).
        retriever_html = _render_partial(
            request, "partials/chat_retriever_card.html",
            {"payload": event.payload},
        )
        media_html = _render_partial(
            request, "partials/media_panel.html",
            {"media": (event.payload or {}).get("media") or {}, "oob": True},
        )
        return {
            "event": "card",
            "data": retriever_html + " " + media_html,
        }

    if event.kind == "writer_pending":
        return {
            "event": "card",
            "data": _render_partial(
                request, "partials/chat_writer_pending.html",
                {"payload": event.payload},
            ),
        }

    if event.kind == "writer":
        # The writer card's wrapper carries ``hx-swap-oob="outerHTML"`` and
        # an id matching the pending card — htmx replaces that pending
        # card in place rather than appending a new one.
        return {
            "event": "card",
            "data": _render_partial(
                request, "partials/chat_writer_card.html",
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
                request, "partials/chat_critic_card.html",
                {"payload": event.payload},
            ),
        }

    if event.kind == "done":
        # Send a small placeholder; chat_pane.html's htmx:sseClose handler
        # fetches the full lesson card via GET /card-fragment (no SSE size limit).
        return {
            "event": "final",
            "data": (
                f'<div id="lesson-card-loading" class="flex items-start gap-3">'
                f'<div class="flex-shrink-0 w-9 h-9 rounded-full bg-slate-800 text-white'
                f' flex items-center justify-center ring-2 ring-white shadow-sm">'
                f'<wa-icon name="book-open" style="font-size:1rem;"></wa-icon></div>'
                f'<div class="flex-1 min-w-0"><div class="rounded-xl border border-slate-200'
                f' bg-white shadow-sm px-4 py-3 text-sm text-slate-500 animate-pulse">'
                f'Caricamento lezione finalizzata…</div></div></div>'
            ),
        }

    if event.kind == "error":
        return {
            "event": "error",
            "data": _render_partial(
                request, "partials/chat_error.html",
                {"lesson": lesson, "error": event.error or "Errore sconosciuto"},
            ),
        }

    # Unknown kind — log and skip rather than 500-ing the stream.
    logger.warning("[stream] unhandled StreamEvent.kind=%r", event.kind)
    return None


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
    rendered = templates.get_template(template_name).render(
        {"request": request, **ctx}
    )
    # Collapse the raw HTML to a single line. This loses indentation but
    # preserves text node whitespace via the elements' own block layout.
    return " ".join(rendered.split())
