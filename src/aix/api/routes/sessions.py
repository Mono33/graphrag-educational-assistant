"""Conversations with the agent through the API (2026-10-14).

A session is what the WebUI calls a lesson: the conversation, the current
lesson, the documents it may search. The API keeps it, so a chatbot sends only
the new message and the agent behaves as in the WebUI:

* the first turn writes a lesson (the whole chain: planner, Knowledge Graph,
  the user's documents, writer, citation check, then the critic -- or, with
  AIX_JUDGE_ENABLED, the judge's targeted review in its place);
* later turns answer in a few seconds from the lesson, the conversation and the
  documents (``kind: "answer"``), unless the request needs a new lesson -- then
  the turn comes back as ``kind: "handoff"`` and the client decides, as the
  WebUI's card lets the teacher decide;
* ``mode: "lesson"`` and the refinement chips rewrite the lesson.

The decisions are the WebUI's own (``aix.webui.agent.turns``): the same checks,
the same refusal codes, the same messages written to the same tables.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any, Optional, Union

import anyio
from fastapi import APIRouter, Depends, Header, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from aix.api import idempotency
from aix.api.explain import (
    cited_documents,
    critic_info,
    judge_info,
    lesson_structure,
    planner_info,
    retriever_info,
    run_meta,
)
from aix.api.principals import current_owner
from aix.api.schemas.sessions import (
    AnswerMeta,
    SessionCreate,
    SessionMaterial,
    SessionMaterials,
    SessionMedia,
    SessionOut,
    SessionSummary,
    TurnOut,
    TurnRequest,
    TurnResult,
)
from aix.api.token_stream import DraftEnd, TextDelta, with_text
from aix.core.ai_marking import AI_GENERATED_HEADER, AI_GENERATED_HEADER_VALUE
from aix.export.service import ExportRefused, render_lesson_export
from aix.webui.agent.service import StreamEvent, stored_cards
from aix.webui.agent.turns import TurnAccepted, TurnRefused, accept_turn, drive_turn
from aix.webui.auth.models import User
from aix.webui.db import get_async_session
from aix.webui.lessons.live_media import live_media_for_lesson
from aix.webui.lessons.media_snapshots import (
    latest_display_snapshot_safe,
    latest_user_turn,
    sanitize_curated_media,
    save_live_snapshot,
    snapshot_for_turn_safe,
)
from aix.webui.lessons.models import Lesson, LessonMessage
from aix.webui.materials import library
from aix.webui.materials.citations import labels_for_lesson_safe, resolve_for_lesson

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["sessions"])

#: The phases whose payloads a turn's result carries, as the WebUI's cards show them.
PHASES = ("planner", "retriever", "critic", "judge")


def _error(status_code: int, code: str, message: str, retry_after: Optional[int] = None):
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
        headers={"Retry-After": str(retry_after)} if retry_after else None,
    )


def _uuid(raw: str, code: str, message: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(raw))
    except ValueError:
        raise _error(404, code, message) from None


async def _lesson(session: AsyncSession, owner: User, session_id: str) -> Lesson:
    lesson_id = _uuid(session_id, "unknown_session", "Sessione non trovata.")
    lesson = (
        await session.execute(
            select(Lesson).where(Lesson.id == lesson_id, Lesson.owner_id == owner.id)
        )
    ).scalar_one_or_none()
    if lesson is None:
        raise _error(404, "unknown_session", "Sessione non trovata.")
    return lesson


async def _bind(session: AsyncSession, lesson: Lesson, owner: User, ids: list[str]) -> None:
    if not ids:
        return
    material_ids = [_uuid(i, "unknown_material", "Documento non trovato.") for i in ids]
    try:
        await library.bind_materials(
            session, lesson_id=lesson.id, owner_id=owner.id, material_ids=material_ids
        )
    except library.UnknownMaterial as exc:
        raise _error(404, exc.code, "Documento non trovato.") from exc
    except library.LibraryUnavailable as exc:
        raise _error(503, exc.code, str(exc)) from exc


def _summary(lesson: Lesson) -> dict[str, Any]:
    return {
        "session_id": str(lesson.id),
        "title": lesson.title,
        "domain": lesson.domain,
        "status": lesson.status,
        "created_at": lesson.created_at,
        "updated_at": lesson.updated_at,
    }


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=SessionOut,
    status_code=status.HTTP_201_CREATED,
    summary="Start a session",
)
async def create_session(
    payload: SessionCreate,
    response: Response,
    owner: User = Depends(current_owner),
    session: AsyncSession = Depends(get_async_session),
    idempotency_key: Optional[str] = Header(default=None, alias=idempotency.HEADER),
) -> SessionOut:
    claim = await idempotency.begin(
        session,
        owner_id=owner.id,
        scope="session",
        key=idempotency_key,
        fingerprint=idempotency.request_hash(payload.model_dump(mode="json")),
    )
    if isinstance(claim, idempotency.Replay):
        # The session as it is now, not as it was: it may have had turns since.
        response.headers[idempotency.REPLAYED_HEADER] = "true"
        return await _session_out(
            session, await _lesson(session, owner, claim.body["session_id"]), owner
        )
    try:
        out = await _create_session(session, owner, payload)
    except Exception:
        await idempotency.abandon(session, claim)
        raise
    await idempotency.finish(session, claim, 201, {"session_id": out.session_id})
    return out


async def _create_session(session: AsyncSession, owner: User, payload: SessionCreate) -> SessionOut:
    profile = (
        payload.educational_profile.model_dump(mode="json", exclude_none=True)
        if payload.educational_profile is not None
        else {}
    )
    lesson = Lesson(
        owner_id=owner.id,
        title=(payload.title or "").strip() or None,
        domain=payload.domain,
        educational_profile_json=profile,
        status="draft",
    )
    session.add(lesson)
    await session.commit()
    await session.refresh(lesson)
    await _bind(session, lesson, owner, payload.material_ids)
    logger.info("[api.sessions] owner=%s session=%s created", owner.id, lesson.id)
    return await _session_out(session, lesson, owner)


@router.get("", response_model=list[SessionSummary], summary="The caller's sessions")
async def list_sessions(
    owner: User = Depends(current_owner),
    session: AsyncSession = Depends(get_async_session),
) -> list[SessionSummary]:
    lessons = (
        (
            await session.execute(
                select(Lesson).where(Lesson.owner_id == owner.id).order_by(Lesson.updated_at.desc())
            )
        )
        .scalars()
        .all()
    )
    return [SessionSummary(**_summary(lesson)) for lesson in lessons]


@router.get(
    "/{session_id}", response_model=SessionOut, summary="A session: lesson, turns, documents"
)
async def get_session(
    session_id: str,
    response: Response,
    owner: User = Depends(current_owner),
    session: AsyncSession = Depends(get_async_session),
) -> SessionOut:
    lesson = await _lesson(session, owner, session_id)
    response.headers[AI_GENERATED_HEADER] = AI_GENERATED_HEADER_VALUE
    return await _session_out(session, lesson, owner)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a session")
async def delete_session(
    session_id: str,
    owner: User = Depends(current_owner),
    session: AsyncSession = Depends(get_async_session),
) -> None:
    lesson = await _lesson(session, owner, session_id)
    if lesson.status == "running":
        raise _error(409, "already_running", "Una generazione è in corso: riprova appena finisce.")
    await session.delete(lesson)
    await session.commit()


def _media(
    live: Optional[dict[str, Any]], turn_index: Optional[int], curated: dict[str, Any]
) -> SessionMedia:
    return SessionMedia.model_validate(
        {**(live or {}), "turn_index": turn_index, "curated": curated}
    )


@router.get(
    "/{session_id}/media",
    response_model=SessionMedia,
    summary="The resources shown with a lesson: curated and found live",
    description=(
        "What the WebUI shows with a lesson: ``curated``, the Knowledge Graph's resources "
        "(the lesson result's ``retriever.media``), and the papers, Wikipedia pages and "
        "videos found live on the lesson's subject (the right panel). Without "
        "``turn_index``, the latest lesson's: the live search runs the first time and is "
        "replayed afterwards. With ``turn_index`` (a lesson turn of ``GET /sessions/{id}``), "
        "what that lesson kept, with no new search. Best effort: empty when nothing was "
        "found or the live search is off."
    ),
)
async def session_media(
    session_id: str,
    turn_index: Optional[int] = None,
    owner: User = Depends(current_owner),
    session: AsyncSession = Depends(get_async_session),
) -> SessionMedia:
    lesson = await _lesson(session, owner, session_id)
    if turn_index is not None:
        # An earlier lesson: what it kept, never a new search.
        kept = await snapshot_for_turn_safe(
            session, lesson_id=lesson.id, owner_id=owner.id, turn_index=turn_index
        )
        if kept is None:
            return SessionMedia()
        return _media(kept.live_json, kept.turn_index, sanitize_curated_media(kept.curated_json))
    # Louis's rule for the WebUI panel: what was shown is replayed, a provider is
    # asked once. A session has no panel to load during the run, so the first
    # call searches and remembers, and every later call replays.
    saved = await latest_display_snapshot_safe(session, lesson_id=lesson.id, owner_id=owner.id)
    curated = sanitize_curated_media(saved.curated_json) if saved is not None else {}
    kept_turn = saved.turn_index if saved is not None else None
    if saved is not None and saved.live_fetched:
        return _media(saved.live_json, kept_turn, curated)
    panel = await live_media_for_lesson(lesson)
    if panel is None:
        return _media(None, kept_turn, curated)
    turn_index = kept_turn if kept_turn is not None else await latest_user_turn(session, lesson.id)
    try:
        async with session.begin_nested():
            await save_live_snapshot(
                session,
                lesson_id=lesson.id,
                owner_id=owner.id,
                turn_index=turn_index,
                media=panel,
            )
        await session.commit()
    except Exception:  # noqa: BLE001 — the answer matters more than remembering it
        await session.rollback()
        logger.warning("[api.sessions] media snapshot not saved for session=%s", lesson.id)
    return _media(panel, turn_index, curated)


@router.get(
    "/{session_id}/export",
    summary="Download the session's lesson: md, txt, pdf, docx or outline",
    description=(
        "The same files as the WebUI's download buttons: citations resolved to the "
        "documents' names, the AI notice in every format. ``outline`` is the timed outline "
        "as plain text."
    ),
    responses={
        200: {
            "content": {
                "application/pdf": {},
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": {},
                "text/markdown": {},
                "text/plain": {},
            }
        }
    },
)
async def export_session(
    session_id: str,
    format: str = "md",
    owner: User = Depends(current_owner),
    session: AsyncSession = Depends(get_async_session),
) -> Response:
    lesson = await _lesson(session, owner, session_id)
    try:
        exported = await render_lesson_export(session, lesson, format)
    except ExportRefused as exc:
        raise _error(exc.status, exc.code, exc.message) from None
    return Response(
        content=exported.content,
        media_type=exported.media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{exported.filename}"',
            AI_GENERATED_HEADER: AI_GENERATED_HEADER_VALUE,
        },
    )


@router.post(
    "/{session_id}/materials",
    response_model=SessionOut,
    summary="Let the session's turns search these documents",
)
async def bind_session_materials(
    session_id: str,
    payload: SessionMaterials,
    owner: User = Depends(current_owner),
    session: AsyncSession = Depends(get_async_session),
) -> SessionOut:
    lesson = await _lesson(session, owner, session_id)
    await _bind(session, lesson, owner, payload.material_ids)
    return await _session_out(session, lesson, owner)


async def _session_out(session: AsyncSession, lesson: Lesson, owner: User) -> SessionOut:
    rows = (
        (
            await session.execute(
                select(LessonMessage)
                .where(LessonMessage.lesson_id == lesson.id)
                .order_by(LessonMessage.turn_index, LessonMessage.created_at)
            )
        )
        .scalars()
        .all()
    )
    turns = []
    for row in rows:
        kind = (
            "message" if row.role == "user" else "answer" if row.agent_kind == "tutor" else "lesson"
        )
        text = row.content_md or ""
        meta = dict(row.meta_json or {})
        # The cards are returned typed, below; the graph digest is the light
        # turns' working material, not something to show.
        meta.pop("cards", None)
        meta.pop("kg_digest", None)
        cards = stored_cards(row.meta_json) if kind == "lesson" else {}
        turns.append(
            TurnOut(
                turn_index=row.turn_index,
                role="user" if row.role == "user" else "assistant",
                kind=kind,
                markdown=text
                if row.role == "user"
                else await resolve_for_lesson(text, session, lesson),
                created_at=row.created_at,
                meta=meta,
                planner=planner_info(cards["planner"]) if "planner" in cards else None,
                retriever=retriever_info(cards["retriever"]) if "retriever" in cards else None,
                critic=critic_info(cards["critic"]) if "critic" in cards else None,
                judge=judge_info(cards["judge"]) if "judge" in cards else None,
            )
        )
    materials = [
        SessionMaterial(material_id=m.material_id, name=m.name, status=m.status)
        for m in await library.list_materials(session, owner_id=owner.id)
        if str(lesson.id) in m.sessions
    ]
    labels = await labels_for_lesson_safe(session, lesson)
    return SessionOut(
        **_summary(lesson),
        lesson_structure=(
            lesson_structure(lesson.lesson_plan_md, lesson_id=str(lesson.id), title=lesson.title)
            if lesson.lesson_plan_md
            else None
        ),
        cited_documents=cited_documents(lesson.lesson_plan_md or "", labels),
        educational_profile={
            k: v
            for k, v in (lesson.educational_profile_json or {}).items()
            if not k.startswith("__")
        },
        lesson_markdown=(
            await resolve_for_lesson(lesson.lesson_plan_md, session, lesson)
            if lesson.lesson_plan_md
            else None
        ),
        error=lesson.error_message if lesson.status == "error" else None,
        materials=materials,
        turns=turns,
    )


# ---------------------------------------------------------------------------
# Turns
# ---------------------------------------------------------------------------


def _form(request: TurnRequest, lesson: Lesson) -> dict[str, str]:
    """The WebUI form a turn request stands for (see ``aix.webui.agent.turns``)."""
    message = (request.message or "").strip()
    if not message and request.refinement is None:
        raise _error(422, "empty_message", "Scrivi un messaggio, oppure scegli una rifinitura.")
    form: dict[str, str] = {}
    if message:
        form["query"] = message
    if request.mode == "lesson":
        form["full_run"] = "1"
    elif request.mode == "answer":
        # A turn in progress first: during it the lesson is «running», and a
        # client retrying an answer was told «no_lesson_yet» (replay, 2026-10-20).
        if lesson.status == "running":
            raise _error(
                409, "already_running", "Una generazione è già in corso. Attendi il termine."
            )
        if lesson.status not in ("complete", "error") or not (lesson.lesson_plan_md or "").strip():
            raise _error(
                409, "no_lesson_yet", "Non c'è ancora una lezione: il primo turno la scrive."
            )
        form["light_answer"] = "1"
    if request.refinement is not None:
        form["refinement_code"] = request.refinement.code
        if request.refinement.detail:
            form["refinement_detail"] = request.refinement.detail
    if request.search_graph is not None:
        form["graph_search_choice"] = "1"
        form["graph_search"] = "on" if request.search_graph else ""
    return form


async def _turn_events(
    session: AsyncSession, lesson: Lesson, owner: User, request: TurnRequest
) -> AsyncIterator[Union[StreamEvent, TurnResult]]:
    """Accept, drive, and end with the turn's :class:`TurnResult` (or raise)."""
    await _bind(session, lesson, owner, request.material_ids)
    outcome = await accept_turn(
        session, lesson, _form(request, lesson), rate_key=f"api:{owner.id}", ask_intent=False
    )
    if isinstance(outcome, TurnRefused):
        raise _error(outcome.status_code, outcome.code, outcome.message, outcome.retry_after)
    assert isinstance(outcome, TurnAccepted)

    phases: dict[str, Any] = {}
    result: Optional[TurnResult] = None
    async for event in drive_turn(lesson, session):
        if isinstance(event, TurnRefused):
            raise _error(event.status_code, event.code, event.message)
        if event.kind in PHASES:
            phases[event.kind] = event.payload or {}
        if event.kind in ("error", "light_error"):
            raise _error(502, "generation_failed", event.error or "La generazione non è riuscita.")
        if event.kind == "done":
            raw = event.lesson_plan_md or ""
            result = TurnResult(
                session_id=str(lesson.id),
                turn_index=outcome.turn_index,
                kind="lesson",
                markdown=await resolve_for_lesson(raw, session, lesson),
                cited_documents=cited_documents(raw, await labels_for_lesson_safe(session, lesson)),
                structure=lesson_structure(raw, lesson_id=str(lesson.id), title=lesson.title),
                planner=planner_info(phases.get("planner")) if "planner" in phases else None,
                retriever=retriever_info(phases.get("retriever"))
                if "retriever" in phases
                else None,
                critic=critic_info(phases.get("critic")) if "critic" in phases else None,
                judge=judge_info(phases.get("judge")) if "judge" in phases else None,
                lesson_meta=run_meta(event.meta),
            )
        elif event.kind == "light_done":
            answer = await session.scalar(
                select(LessonMessage.content_md)
                .where(
                    LessonMessage.lesson_id == lesson.id,
                    LessonMessage.turn_index == outcome.turn_index,
                    LessonMessage.role == "assistant",
                )
                .order_by(LessonMessage.created_at.desc())
                .limit(1)
            )
            result = TurnResult(
                session_id=str(lesson.id),
                turn_index=outcome.turn_index,
                kind="answer",
                markdown=await resolve_for_lesson(answer or "", session, lesson),
                cited_documents=cited_documents(
                    answer or "", await labels_for_lesson_safe(session, lesson)
                ),
                answer_meta=AnswerMeta.model_validate(event.meta or {}),
            )
        elif event.kind == "light_escalate":
            result = TurnResult(
                session_id=str(lesson.id),
                kind="handoff",
                handoff_query=(event.payload or {}).get("query"),
            )
        yield event
    if result is None:
        raise _error(502, "generation_failed", "La generazione si è interrotta senza un risultato.")
    yield result


@router.post(
    "/{session_id}/turns",
    response_model=TurnResult,
    summary="Send a message and wait for the turn's result",
    description=(
        "Waits until the turn ends: a few seconds for an answer, about a minute for a lesson. "
        "For progress as it happens use ``/turns/stream``."
    ),
)
async def send_turn(
    session_id: str,
    payload: TurnRequest,
    response: Response,
    owner: User = Depends(current_owner),
    session: AsyncSession = Depends(get_async_session),
    idempotency_key: Optional[str] = Header(default=None, alias=idempotency.HEADER),
) -> TurnResult:
    lesson = await _lesson(session, owner, session_id)
    response.headers[AI_GENERATED_HEADER] = AI_GENERATED_HEADER_VALUE
    claim = await _claim_turn(session, owner, lesson, payload, idempotency_key)
    if isinstance(claim, idempotency.Replay):
        response.headers[idempotency.REPLAYED_HEADER] = "true"
        return TurnResult.model_validate(claim.body)
    result: Optional[TurnResult] = None
    try:
        async for item in _turn_events(session, lesson, owner, payload):
            if isinstance(item, TurnResult):
                result = item
    except Exception:
        await idempotency.abandon(session, claim)
        raise
    assert result is not None
    await idempotency.finish(session, claim, 200, result.model_dump(mode="json"))
    return result


async def _claim_turn(
    session: AsyncSession,
    owner: User,
    lesson: Lesson,
    payload: TurnRequest,
    key: Optional[str],
) -> Optional[Union[idempotency.Replay, idempotency.Claim]]:
    """A turn's key: the same for ``/turns`` and ``/turns/stream``, so a client may
    switch between the two on a retry and still get the first answer."""
    return await idempotency.begin(
        session,
        owner_id=owner.id,
        scope=f"turn:{lesson.id}",
        key=key,
        fingerprint=idempotency.request_hash(payload.model_dump(mode="json")),
    )


@router.post(
    "/{session_id}/turns/stream",
    summary="Send a message and follow the turn as Server-Sent Events",
    description=(
        "Each ``data:`` frame is JSON. Phase events carry ``kind`` and ``data``: planner, "
        "retriever, writer_pending, writer, then either critic_pending and critic (judge "
        "off) or judge_pending and judge (judge on, AIX_JUDGE_ENABLED: the targeted review, "
        "typically 5-15 s after the text is written; ``data`` is the ``JudgeInfo`` of the "
        "result); for a fast answer light_pending and light_done. While the text is "
        "written, ``kind: text`` frames carry the next piece (``text``) and the ``draft`` it "
        "belongs to; ``kind: draft_end`` closes a draft, and a higher ``draft`` means the "
        "lesson is being written again (a Critic revision, the attached documents left "
        "unused, a split of the class that does not add up): start the text over. The text "
        "is a preview: the last frame is ``kind: result``, the body ``/turns`` returns under "
        "``result``, whose ``markdown`` replaces it -- with the judge on, it is the corrected "
        "lesson. A refusal or a failure is a frame ``kind: error`` with ``code`` and "
        "``message``."
    ),
)
async def stream_turn(
    session_id: str,
    payload: TurnRequest,
    owner: User = Depends(current_owner),
    session: AsyncSession = Depends(get_async_session),
    idempotency_key: Optional[str] = Header(default=None, alias=idempotency.HEADER),
) -> EventSourceResponse:
    lesson = await _lesson(session, owner, session_id)
    headers = {AI_GENERATED_HEADER: AI_GENERATED_HEADER_VALUE}
    # Claimed before the stream opens, so "in progress" and "reused" are real
    # HTTP statuses a client's retry logic can see.
    claim = await _claim_turn(session, owner, lesson, payload, idempotency_key)
    if isinstance(claim, idempotency.Replay):
        body = {"kind": "result", "result": claim.body}

        async def replay():
            yield {"event": "result", "data": json.dumps(body, ensure_ascii=False)}

        return EventSourceResponse(
            replay(), headers={**headers, idempotency.REPLAYED_HEADER: "true"}
        )

    async def frames():
        finished = False
        try:
            async for item in with_text(
                _turn_events(session, lesson, owner, payload), str(lesson.id)
            ):
                if isinstance(item, TextDelta):
                    body = {"kind": "text", "text": item.text, "draft": item.draft}
                elif isinstance(item, DraftEnd):
                    body = {"kind": "draft_end", "draft": item.draft}
                elif isinstance(item, TurnResult):
                    body = {"kind": "result", "result": item.model_dump(mode="json")}
                    await idempotency.finish(session, claim, 200, body["result"])
                    finished = True
                else:
                    body = {
                        "kind": item.kind,
                        "data": item.payload or {},
                        "meta": item.meta or {},
                    }
                yield {"event": body["kind"], "data": json.dumps(body, ensure_ascii=False)}
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
            body = {"kind": "error", "status": exc.status_code, **detail}
            yield {"event": "error", "data": json.dumps(body, ensure_ascii=False)}
        finally:
            if not finished:
                # Failed, refused, or the client went away: the key is free for a
                # retry. Shielded, because a disconnect cancels this generator.
                with anyio.CancelScope(shield=True):
                    await idempotency.abandon(session, claim)

    return EventSourceResponse(frames(), headers=headers)
