"""
Public Agent API routes (CORE 2 #7).

Exposes the multi-agent lesson-planner pipeline as a documented JSON+SSE
contract at ``/api/v1/agent/*``. This is the surface any *non-browser*
consumer hits — Postman / curl, the AixLearning embed (CORE 6), the
future mobile app, and any partner integration.

Routes
------

``POST /api/v1/agent/run``
    Synchronous JSON. Drains the entire agent pipeline and returns a
    single ``AgentRunResponse`` with the final lesson plan + planner /
    retriever / critic explainability. This is the route teachers /
    integrators will exercise from Swagger UI's "Try it out" the same
    way they exercise ``/api/v1/context``.

``POST /api/v1/agent/stream``
    Server-Sent Events. Yields the same agent phases as the webui chat
    workspace, but each event is JSON (not HTML) so non-browser clients
    can ``switch`` on ``kind`` and update their UI incrementally.

Both routes:
    * Reuse ``aix.webui.agent.service.stream_agent_events`` — a DB-less
      sibling of the webui's ``run_agent_stream``. Zero new agent code.
    * Authenticate via fastapi-users' ``current_active_user`` — accepts
      either the webui cookie OR ``Authorization: Bearer <jwt>``.
    * Are strictly additive: they do NOT touch the existing
      ``/api/v1/context``, ``/webui/*``, or ``/auth/*`` surfaces.

See: docs/product/ClickUp_Agentic_GraphRAG_Update.md → Subtask 7.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import AsyncIterator, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, status
from sse_starlette.sse import EventSourceResponse

from aix.api.schemas import (
    AgentRunRequest,
    AgentRunResponse,
    AgentRunMeta,
    CriticScores,
    MediaCounts,
    PlannerInfo,
    RetrieverInfo,
)
from aix.webui.auth import current_active_user
from aix.webui.auth.models import User
from aix.webui.agent.service import StreamEvent, stream_agent_events

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/agent", tags=["agent"])


# ---------------------------------------------------------------------------
# Swagger UI "Try it out" examples (CORE 2 #7).
#
# Mirrors the dropdown UX used by /api/v1/context (see routes/context.py):
# FastAPI converts this dict to OpenAPI 3.1 ``examples`` keyed by name on
# the request body MediaType, which Swagger UI renders as a real picker.
# The schema-level ``json_schema_extra={"example": ...}`` we keep on
# ``AgentRunRequest`` shows up in the "Schema" tab; this dict drives the
# "Try it out" pre-fill.
#
# Two named shapes — same idiom as /context:
#   * minimal — fastest path to a working call (only mandatory fields)
#   * rich    — every optional CORE 1 #2.5 EducationalProfile field
#               populated, so a teacher reading the example once knows
#               the entire surface
# ---------------------------------------------------------------------------
_AGENT_REQUEST_OPENAPI_EXAMPLES: dict = {
    "minimal": {
        "summary": "Minimal — only mandatory fields",
        "description": (
            "Smallest valid call. Omitting ``educational_profile`` makes the "
            "agent fall back to its generic prompts — fine for a quick smoke "
            "test from /docs but the lesson plan won't specialize against a "
            "specific class."
        ),
        "value": {
            "query": "Crea una lezione sull'attenzione",
            "domain": "neuro",
        },
    },
    "rich": {
        "summary": "Rich — with full EducationalProfile (CORE 1 #2.5)",
        "description": (
            "Recommended shape for production calls. Includes every "
            "optional EducationalProfile field — class title, BES, class "
            "features, student attributes, classroom resources, time "
            "budget, subject area, specific topic — so the planner / "
            "writer can specialize prompts against the real class. "
            "``teacher_provided_context`` may carry up to ~48k chars of "
            "extra material extracted from teacher uploads (PDF/TXT/MD)."
        ),
        "value": {
            "query": (
                "Crea una lezione di 45 minuti sulla fotosintesi "
                "clorofilliana adattata a una classe con 2 studenti DSA"
            ),
            "domain": "neuro",
            "language": "it",
            "max_revisions": 1,
            "educational_profile": {
                "group": {
                    "title": "3A Liceo Scientifico",
                    "students_number": 25,
                    "grade": "SECONDARIA_II_GRADO",
                    "disabilities": ["ADHD", "DSA"],
                    "class_features": ["MOTIVATA"],
                    "student_attributes": [
                        "PUNTI_DI_ECCELLENZA",
                        "PUNTI_DI_CADUTA",
                    ],
                },
                "classroom": {
                    "title": "Aula 101",
                    "forniture_mobility": "PARTIALLY",
                    "has_lim": True,
                    "has_wifi": True,
                    "has_suite": True,
                    "pc_station": False,
                    "own_device": "BES",
                },
                "time_available_minutes": 45,
                "subject_area": "Scienze",
                "specific_topic": "Fotosintesi",
            },
            "teacher_provided_context": (
                "Estratto dal manuale (opzionale): 'La fotosintesi è il "
                "processo con cui le piante convertono energia luminosa "
                "in energia chimica…'"
            ),
        },
    },
}


# ---------------------------------------------------------------------------
# Internal helpers — translate StreamEvent ↔ public Pydantic models
# ---------------------------------------------------------------------------


def _serialise_event(event: StreamEvent) -> dict:
    """
    Map an internal ``StreamEvent`` to the public SSE event JSON shape.

    Renamed: ``payload`` → ``data`` so the public contract reads more
    naturally for non-browser consumers (``event.data.intent_label``).
    The internal name is kept as-is to avoid a webui rename.
    """
    return {
        "kind": event.kind,
        "data": event.payload or {},
        "lesson_plan_md": event.lesson_plan_md,
        "error": event.error,
    }


def _meta_to_pydantic(meta: dict) -> AgentRunMeta:
    """Coerce the internal meta dict into the published response model."""
    media_counts = meta.get("media_counts") or {}
    scores_dict = meta.get("scores") or None
    return AgentRunMeta(
        duration_seconds=float(meta.get("duration_seconds", 0.0)),
        approved=bool(meta.get("approved", False)),
        revision_count=int(meta.get("revision_count", 0)),
        scores=CriticScores.model_validate(scores_dict) if isinstance(scores_dict, dict) else None,
        nodes_count=int(meta.get("nodes_count", 0)),
        recommendations_count=int(meta.get("recommendations_count", 0)),
        media_counts=MediaCounts(
            videos=int(media_counts.get("videos", 0)),
            articles=int(media_counts.get("articles", 0)),
            oer=int(media_counts.get("oer", 0)),
        ),
        search_queries_count=int(meta.get("search_queries_count", 0)),
    )


def _planner_payload_to_pydantic(payload: dict) -> PlannerInfo:
    return PlannerInfo(
        intent=payload.get("intent"),
        intent_label=payload.get("intent_label"),
        scope=payload.get("scope"),
        scope_label=payload.get("scope_label"),
        scope_confidence=payload.get("scope_confidence"),
        key_concepts=list(payload.get("key_concepts") or []),
        search_queries=list(payload.get("search_queries") or []),
        lesson_type=payload.get("lesson_type"),
        target_grade=payload.get("target_grade"),
    )


def _retriever_payload_to_pydantic(payload: dict) -> RetrieverInfo:
    media_counts = payload.get("media_counts") or {}
    return RetrieverInfo(
        nodes_count=int(payload.get("nodes_count", 0)),
        relationships_count=int(payload.get("relationships_count", 0)),
        recommendations_count=int(payload.get("recommendations_count", 0)),
        media_counts=MediaCounts(
            videos=int(media_counts.get("videos", 0)),
            articles=int(media_counts.get("articles", 0)),
            oer=int(media_counts.get("oer", 0)),
        ),
        media=dict(payload.get("media") or {}),
        top_concepts=list(payload.get("top_concepts") or []),
        retrieval_confidence=payload.get("retrieval_confidence"),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/run",
    response_model=AgentRunResponse,
    summary="Run the agent and return the final lesson plan (sync JSON)",
    description=(
        "Drives the full **Planner → Retriever → Writer → Critic** pipeline "
        "and returns a single JSON response with the final Markdown lesson "
        "plan plus planner / retriever / critic explainability. Typical "
        "run: 60–120 seconds. For incremental UI updates use "
        "``POST /api/v1/agent/stream`` instead."
    ),
    responses={
        200: {"description": "Successful agent run."},
        401: {"description": "Missing or invalid authentication."},
        422: {"description": "Validation error on the request body."},
        502: {
            "description": (
                "Agent pipeline reported a runtime error (LLM failure, "
                "Knowledge Graph unreachable, etc.)."
            ),
        },
    },
)
async def run_agent(
    payload: AgentRunRequest = Body(
        ...,
        openapi_examples=_AGENT_REQUEST_OPENAPI_EXAMPLES,
    ),
    user: User = Depends(current_active_user),
) -> AgentRunResponse:
    """
    Drain ``stream_agent_events`` to completion and assemble the final
    ``AgentRunResponse``.

    Implementation note: we collect the planner / retriever payloads as
    they fly past so they can be returned alongside the final lesson
    plan. This matches the webui contract — the chat workspace shows
    the same explainability cards — without forcing every API consumer
    to handle SSE.
    """
    session_id = payload.session_id or str(uuid.uuid4())
    profile_dict = (
        payload.educational_profile.model_dump(mode="json", exclude_none=False)
        if payload.educational_profile is not None
        else None
    )

    planner_info: Optional[PlannerInfo] = None
    retriever_info: Optional[RetrieverInfo] = None
    final_lesson_plan: str = ""
    final_meta: Optional[AgentRunMeta] = None
    error_message: Optional[str] = None

    async for event in stream_agent_events(
        query=payload.query,
        domain=payload.domain,
        language=payload.language,
        session_id=session_id,
        educational_profile=profile_dict,
        teacher_provided_context=payload.teacher_provided_context,
        max_revisions=payload.max_revisions,
    ):
        if event.kind == "planner":
            planner_info = _planner_payload_to_pydantic(event.payload or {})
        elif event.kind == "retriever":
            retriever_info = _retriever_payload_to_pydantic(event.payload or {})
        elif event.kind == "done":
            final_lesson_plan = event.lesson_plan_md or ""
            final_meta = _meta_to_pydantic(event.meta or {})
        elif event.kind == "error":
            error_message = event.error or "Unknown agent error"
            # Don't break — the stream is finite; we'll fall through and
            # raise after the loop terminates so we always release any
            # outstanding state.

    if error_message is not None or final_meta is None:
        # 502 (Bad Gateway) communicates "the agent pipeline failed", which
        # is more accurate than a 500 (we, the route, didn't crash) and
        # tells partner clients the request is potentially retryable.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=error_message or "Agent finished without a lesson plan",
        )

    logger.info(
        "[api.agent] /run complete user=%s session=%s duration=%.1fs",
        user.email, session_id, final_meta.duration_seconds,
    )

    return AgentRunResponse(
        lesson_plan_md=final_lesson_plan,
        meta=final_meta,
        planner=planner_info,
        retriever=retriever_info,
    )


@router.post(
    "/stream",
    summary="Run the agent and stream phases as SSE-encoded JSON events",
    description=(
        "Server-Sent Events stream of the same phases the webui chat "
        "workspace renders, but each event body is JSON (not HTML). "
        "Useful for non-browser clients that want incremental UI updates. "
        "Note: Swagger UI's *Try it out* renders the stream as a single "
        "blob — for live event-by-event inspection use ``curl -N`` or "
        "Postman / Bruno (both ship SSE viewers)."
    ),
    response_class=EventSourceResponse,
    responses={
        200: {
            "description": (
                "Server-Sent Events stream. Each ``data:`` frame is a JSON "
                "object matching one variant of ``AgentStreamEvent`` — "
                "discriminated by the ``kind`` field."
            ),
            "content": {"text/event-stream": {}},
        },
        401: {"description": "Missing or invalid authentication."},
        422: {"description": "Validation error on the request body."},
    },
)
async def stream_agent(
    payload: AgentRunRequest = Body(
        ...,
        openapi_examples=_AGENT_REQUEST_OPENAPI_EXAMPLES,
    ),
    user: User = Depends(current_active_user),
) -> EventSourceResponse:
    """
    Wrap ``stream_agent_events`` in an SSE response that JSON-encodes
    each event. Mirrors the event taxonomy used by the webui htmx pane,
    minus the HTML rendering.
    """
    session_id = payload.session_id or str(uuid.uuid4())
    profile_dict = (
        payload.educational_profile.model_dump(mode="json", exclude_none=False)
        if payload.educational_profile is not None
        else None
    )

    logger.info(
        "[api.agent] /stream open user=%s session=%s domain=%s",
        user.email, session_id, payload.domain,
    )

    async def _publisher() -> AsyncIterator[dict]:
        # ``EventSourceResponse`` accepts a generator yielding either a
        # ``dict`` (with ``event`` / ``data`` / ``id`` keys) or a string.
        # We always yield a dict so the consumer sees per-event ``event:``
        # lines, useful for clients that filter on event type.
        async for event in stream_agent_events(
            query=payload.query,
            domain=payload.domain,
            language=payload.language,
            session_id=session_id,
            educational_profile=profile_dict,
            teacher_provided_context=payload.teacher_provided_context,
            max_revisions=payload.max_revisions,
        ):
            body = _serialise_event(event)
            yield {
                "event": event.kind,
                "data": json.dumps(body, ensure_ascii=False, default=str),
            }

    # ``ping=15`` keeps proxies / load balancers from killing idle
    # connections during the slow Writer call (60-90s LLM round-trip).
    return EventSourceResponse(_publisher(), ping=15)
