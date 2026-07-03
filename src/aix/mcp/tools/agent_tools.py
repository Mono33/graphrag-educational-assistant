"""
Agent MCP tool — Phase 4 of CORE 5 #20.

Exposes the full Aix multi-agent lesson-planner pipeline (Planner →
Retriever → Writer → Critic) as a single MCP tool: ``agent.run_lesson_plan``.

This is the keystone of #20: phases 1-3 expose read-only KG / media / schema
data, but only this tool lets external MCP clients (Claude Desktop, Cursor
IDE, Lovable, partner LangGraph systems) actually *run* the agent and
receive a fully-rendered lesson plan back.

Design notes
------------
* **Same engine as the public HTTP API.** The tool body wraps
  ``aix.webui.agent.service.stream_agent_events`` — the DB-less helper
  that backs ``POST /api/v1/agent/run`` (CORE 2 #7). MCP clients and
  HTTP clients hit the *same* code path; the only difference is the
  transport.
* **Same response shape.** We return the same ``AgentRunResponse``
  Pydantic model the HTTP API publishes. This means an MCP-aware client
  written against ``/api/v1/agent/run`` schemas can be ported to MCP
  with zero contract churn.
* **MCP-native progress.** While the agent runs we call
  ``ctx.report_progress(progress, total, message)`` after each phase
  (planner, retriever, writer attempts, critic, done). FastMCP routes
  this to the client's progress channel when one is connected and
  silently no-ops otherwise — so the same tool works for clients that
  don't support progress notifications.
* **Errors are structured, not silent.** Agent-layer failures (LLM
  empty body, KG unreachable, planner JSON parse error) surface as a
  raised exception, which FastMCP renders as ``isError=true`` on the
  tool response. This matches the HTTP API's 502 contract — clients
  treat both the same.

The 3 ``_*_to_pydantic`` payload mappers below are inlined from
``aix.api.routes.agent`` (~30 LOC) so the MCP layer doesn't depend on
the FastAPI route package. Both copies are kept in sync via the public
``StreamEvent`` taxonomy documented in ``service.py``.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Literal, Optional

from fastmcp import Context, FastMCP

from aix.api.schemas.agent import (
    AgentRunMeta,
    AgentRunResponse,
    CriticScores,
    MediaCounts,
    PlannerInfo,
    RetrieverInfo,
)
from aix.api.schemas.educational_profile import EducationalProfile
from aix.core.concurrency import AtCapacity, run_slot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers — translate StreamEvent payload dicts → public Pydantic.
# Inlined from aix.api.routes.agent (kept in sync via the StreamEvent
# taxonomy in aix.webui.agent.service.StreamEvent's docstring).
# ---------------------------------------------------------------------------


def _meta_to_pydantic(meta: dict[str, Any]) -> AgentRunMeta:
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


def _planner_payload_to_pydantic(payload: dict[str, Any]) -> PlannerInfo:
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


def _retriever_payload_to_pydantic(payload: dict[str, Any]) -> RetrieverInfo:
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
# Progress mapping — how StreamEvent.kind translates to a (progress, total,
# message) tuple. We use a 6-step skeleton so the client UI shows a clean
# percentage bar (1/6 ... 6/6). Critic revisions reuse the writer/critic
# slots in-place — total never grows past 6 to avoid the bar going backwards.
# ---------------------------------------------------------------------------
_PROGRESS_TOTAL: float = 6.0
_PROGRESS_STAGES: dict[str, tuple[float, str]] = {
    "planner": (1.0, "Pianificazione lezione…"),
    "retriever": (2.0, "Recupero contesto dal Knowledge Graph…"),
    "writer_pending": (3.0, "Scrittura della lezione in corso…"),
    "writer": (4.0, "Lezione redatta — in attesa del Critic"),
    "critic": (5.0, "Critic ha valutato il draft"),
    "done": (6.0, "Lezione finalizzata"),
    "error": (6.0, "Esecuzione interrotta da un errore"),
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(mcp: FastMCP) -> None:
    """Register ``agent.run_lesson_plan`` onto the shared FastMCP instance."""

    @mcp.tool(
        name="agent.run_lesson_plan",
        description=(
            "Run the full Aix multi-agent lesson-planner pipeline (Planner → "
            "Retriever → Writer → Critic) and return a structured lesson "
            "plan. Same engine and same response shape as POST "
            "/api/v1/agent/run — for MCP clients, this is THE tool that "
            "produces a finished lesson from a teacher's free-text query. "
            "Typical run: 60-120s. Reports MCP progress notifications "
            "after each phase (clients without progress support see them "
            "silently dropped). Pass ``educational_profile`` for "
            "differentiated-instruction outputs (CORE 1 #2.5 schema)."
        ),
        tags={"agent", "lesson", "graphrag", "production"},
    )
    async def agent_run_lesson_plan(
        query: str,
        domain: Literal["neuro", "udl"] = "neuro",
        language: Literal["it", "en"] = "it",
        session_id: Optional[str] = None,
        educational_profile: Optional[EducationalProfile] = None,
        teacher_provided_context: Optional[str] = None,
        max_revisions: Optional[int] = None,
        ctx: Optional[Context] = None,
    ) -> AgentRunResponse:
        """Run the agent pipeline end-to-end.

        Args:
            query: Teacher query in natural language. Same shape the
                webui chat input accepts.
            domain: Knowledge graph domain — 'neuro' (neuroscience-of-
                learning) or 'udl' (Universal Design for Learning).
            language: Output language for the lesson plan and reasoning.
            session_id: Optional client-side correlation id (for log
                tracing / Langfuse). A UUID4 is generated if omitted.
            educational_profile: CORE 1 #2.5 per-request class context
                (group, classroom, time, subject, topic). When provided
                the planner / writer specialise prompts; when omitted
                the agent falls back to generic prompts.
            teacher_provided_context: Plain text (≤ 48k chars) extracted
                from teacher uploads (PDF/TXT/MD). The Writer treats it
                as additional context. NOT ingested into the KG.
            max_revisions: Cap on critic revision loops (0..4). ``None``
                defers to the pipeline default (currently 2). Setting 0
                disables critic revisions for fast smoke runs.
            ctx: Injected by FastMCP — used for progress notifications.

        Returns:
            ``AgentRunResponse`` — final Markdown lesson plan plus
            planner / retriever / critic explainability fields. Same
            shape as ``POST /api/v1/agent/run``.

        Raises:
            RuntimeError: When the agent pipeline reports an error
                (LLM failure, KG unreachable, empty final state). MCP
                surfaces this as ``isError=true`` to the client.
        """
        if not query or len(query.strip()) < 3:
            raise ValueError("`query` must be at least 3 characters")
        if max_revisions is not None and not 0 <= max_revisions <= 4:
            raise ValueError("`max_revisions` must be in the range 0..4")

        effective_session_id = session_id or str(uuid.uuid4())
        profile_dict: Optional[dict[str, Any]] = (
            educational_profile.model_dump(mode="json", exclude_none=False)
            if educational_profile is not None
            else None
        )

        # Lazy import — keeps the MCP package importable in environments
        # where the heavy agent stack isn't installed (e.g. partial CI).
        from aix.webui.agent.service import stream_agent_events

        planner_info: Optional[PlannerInfo] = None
        retriever_info: Optional[RetrieverInfo] = None
        final_lesson_plan: str = ""
        final_meta: Optional[AgentRunMeta] = None
        error_message: Optional[str] = None

        async def _emit_progress(kind: str) -> None:
            if ctx is None:
                return
            stage = _PROGRESS_STAGES.get(kind)
            if stage is None:
                return
            progress, message = stage
            try:
                await ctx.report_progress(
                    progress=progress,
                    total=_PROGRESS_TOTAL,
                    message=message,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("[agent.run_lesson_plan] report_progress failed: %s", exc)

        logger.info(
            "[mcp.agent] /run_lesson_plan starting session=%s domain=%s "
            "query=%r profile=%s teacher_ctx_chars=%s",
            effective_session_id,
            domain,
            query[:80],
            "yes" if profile_dict else "no",
            len(teacher_provided_context or ""),
        )

        # CORE 6 #31/#34 — MCP shares the same global generation cap as the
        # HTTP API and the webui. Acquire a run slot (or shed) so MCP clients
        # can't bypass the limit. On capacity we raise — FastMCP renders it as
        # isError=true, matching the existing failure contract above.
        try:
            async with run_slot(label=f"mcp:{effective_session_id}"):
                async for event in stream_agent_events(
                    query=query,
                    domain=domain,
                    language=language,
                    session_id=effective_session_id,
                    educational_profile=profile_dict,
                    teacher_provided_context=teacher_provided_context,
                    max_revisions=max_revisions,
                ):
                    await _emit_progress(event.kind)

                    if event.kind == "planner":
                        planner_info = _planner_payload_to_pydantic(event.payload or {})
                    elif event.kind == "retriever":
                        retriever_info = _retriever_payload_to_pydantic(event.payload or {})
                    elif event.kind == "done":
                        final_lesson_plan = event.lesson_plan_md or ""
                        final_meta = _meta_to_pydantic(event.meta or {})
                    elif event.kind == "error":
                        error_message = event.error or "Unknown agent error"
                        # Fall through — the stream is finite; we raise after
                        # the loop terminates so any in-flight state releases.

                if error_message is not None or final_meta is None:
                    detail = error_message or "Agent finished without a lesson plan"
                    logger.warning(
                        "[mcp.agent] /run_lesson_plan FAILED session=%s detail=%r",
                        effective_session_id,
                        detail[:200],
                    )
                    raise RuntimeError(detail)

                logger.info(
                    "[mcp.agent] /run_lesson_plan complete session=%s duration=%.1fs "
                    "approved=%s revisions=%s",
                    effective_session_id,
                    final_meta.duration_seconds,
                    final_meta.approved,
                    final_meta.revision_count,
                )

                return AgentRunResponse(
                    lesson_plan_md=final_lesson_plan,
                    meta=final_meta,
                    planner=planner_info,
                    retriever=retriever_info,
                )
        except AtCapacity as exc:
            logger.warning(
                "[mcp.agent] /run_lesson_plan at capacity — shedding session=%s",
                effective_session_id,
            )
            raise RuntimeError(
                "Il sistema è momentaneamente occupato: troppe generazioni in corso. "
                "Riprova tra qualche istante."
            ) from exc

    _ = agent_run_lesson_plan  # silence "unused" lint — FastMCP registers it
