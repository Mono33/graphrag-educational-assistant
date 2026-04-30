"""
Agent streaming service — the single seam between ``aix.webui`` and the
LangGraph agent (CORE 2 #6.6 P2).

Why a service module instead of calling the orchestrator from the route:
    1. The route layer should only translate HTTP ↔ events. All knowledge of
       LangGraph's state shape, node names, and revision-loop quirks lives
       here so we can swap the engine (or move it behind an RPC) without
       touching ``lessons/routes.py`` or any template.
    2. ``AgentOrchestrator.create_lesson_plan`` is *atomic* — it ``ainvoke``s
       the compiled graph and returns the final result. P2 needs phase-level
       progress, which means we have to drop down to the same compiled graph
       and use ``astream(..., stream_mode="updates")`` ourselves.
    3. Persisting the run lifecycle (``Lesson.status``, ``lesson_plan_md``,
       ``error_message``, ``teacher_query``) is something the agent doesn't
       know about — it's a webui concern. Doing it here means the route
       handler stays trivial.

Event taxonomy (P2 phase 2 — chat workspace):
    The chat UI is a stack of cards, one per agent step, accumulated in
    order. The service emits a ``StreamEvent`` with a ``kind`` that maps
    1:1 to a Jinja2 partial in ``templates/partials/chat_*.html``:

        kind             card                                emitted when
        ─────────────    ────────────────────────────────    ────────────────────
        "planner"        chat_planner_card.html              after `plan` node
        "retriever"      chat_retriever_card.html            after `retrieve`
                         (also drives the right-side          (single event,
                         media panel via OOB swap)            two render targets)
        "writer_pending" chat_writer_pending.html            right before write
                                                              starts (synthetic;
                                                              writer is the slow
                                                              call so the chat
                                                              must not look stuck)
        "writer"         chat_writer_card.html               after `write` node
                                                              (replaces the
                                                              matching pending
                                                              card via OOB)
        "critic"         chat_critic_card.html               after `critique`
                                                              node
        "done"           chat_lesson_card.html (+ summary)   end of run
        "error"          chat_error.html                     on exception

    LLM token streaming is *not* enabled inside the agent's nodes today
    (planner / writer / critic call ``chat.completions.create`` non-
    streamingly), so phase-level granularity is the realistic ceiling
    without modifying the agents themselves. P2 phase 3 will switch the
    writer agent to ``stream_mode="messages"`` so writer tokens land in
    the writer card live.

Reentrancy / concurrency:
    The compiled graph and its underlying agents are module-level singletons
    inside ``aix.agent.graph.nodes``. That's fine for a single user dev box
    but worth re-evaluating in CORE 6 deploy. The route layer enforces
    one in-flight run per ``lesson.id`` via ``_ACTIVE_RUNS``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Optional

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical phase metadata (must match the node names registered in
# ``build_lesson_planner_graph`` — see aix/agent/graph/lesson_planner_graph.py).
# Italian labels are user-facing; keep them in sync with the templates.
#
# We keep ``PHASE_ORDER`` / ``PHASE_LABELS`` exported for routes.py and any
# legacy partial that still wants a generic tracker. The chat workspace
# itself doesn't render a separate tracker — the per-agent cards *are* the
# tracker.
# ---------------------------------------------------------------------------

PHASE_ORDER: tuple[str, ...] = ("plan", "retrieve", "write", "critique")

PHASE_LABELS: dict[str, str] = {
    "plan": "Pianificazione della lezione",
    "retrieve": "Recupero contesto dal Knowledge Graph",
    "write": "Scrittura della lezione",
    "critique": "Revisione e valutazione qualità",
}


# Italian labels for the planner's enum outputs. Keep these here (not in
# the template) so the template only needs `{{ payload.intent_label }}`.
_INTENT_LABELS: dict[str, str] = {
    "lesson_creation": "Creazione lezione",
    "activity_design": "Progettazione attività",
    "definition": "Definizione",
    "comparison": "Confronto",
    "explanation": "Spiegazione",
    "recommendation": "Raccomandazione",
    "list": "Elenco",
    "unknown": "Sconosciuto",
}

# Scope status → (label, wa-tag variant). The variant choice is intentional:
# in_scope is "success" (green) because the KG can speak to the topic;
# partial / out_of_scope are "warning" (amber) because the lesson will be
# composed primarily from external sources (Wikipedia, OER).
_SCOPE_LABELS: dict[str, tuple[str, str]] = {
    "in_scope":      ("Nel Knowledge Graph",     "success"),
    "partial_scope": ("Parzialmente nel KG",     "warning"),
    "out_of_scope":  ("Fuori dal KG",            "warning"),
    "unknown":       ("Scope sconosciuto",       "neutral"),
}


# ---------------------------------------------------------------------------
# Event model — what the route layer / SSE rendering consumes
# ---------------------------------------------------------------------------

@dataclass
class StreamEvent:
    """
    Normalized event yielded by ``run_agent_stream``.

    Field usage by ``kind``:

        kind=="planner"
            payload = { intent, intent_label, scope, scope_label,
                        scope_variant, scope_confidence,
                        key_concepts: [...], search_queries: [...] }

        kind=="retriever"
            payload = { nodes_count, relationships_count,
                        recommendations_count, media_counts: {videos, articles, oer},
                        media: <full curated_media dict for the side panel>,
                        top_concepts: [...], retrieval_confidence }

        kind=="writer_pending"
            payload = { revision, is_revision, feedback }
            ``revision`` is a 1-based counter so the UI can render unique
            DOM ids per attempt (writer-card-rev1, -rev2, ...). The first
            attempt has is_revision=False; subsequent attempts (entered
            from a critic non-approval) have is_revision=True and may
            carry the critic's ``revision_instructions`` text.

        kind=="writer"
            payload = { revision }
            lesson_plan_md = <draft markdown for THIS revision>
            The route renders this *as* the chat-side writer card and uses
            ``revision`` to OOB-replace the matching writer-pending card.

        kind=="critic"
            payload = { approved, revision_count, max_revisions, score,
                        score_pct, critique, revision_instructions }

        kind=="done"
            lesson_plan_md = <full final markdown>
            meta = { duration_seconds, approved, revision_count,
                     scores, nodes_count, recommendations_count,
                     media_counts, search_queries_count }

        kind=="error"
            error = "<short message, ≤ 480 chars>"
    """

    kind: str
    payload: Dict[str, Any] = field(default_factory=dict)
    lesson_plan_md: Optional[str] = None
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers — query synthesis + payload builders
# ---------------------------------------------------------------------------

_ITALIAN_INDICATORS = {
    "come", "cosa", "quali", "che", "per", "con", "gli", "delle", "nella",
    "posso", "sono", "può", "hanno", "studenti", "lezione", "crea", "crea",
    "obiettivi", "classe", "metodologie", "strategie", "apprendimento",
    "una", "del", "dei", "dal", "nel", "sul", "agli", "agli",
}


def _detect_language(query: str) -> str:
    """Detect response language from teacher query. Returns 'it' or 'en'."""
    words = set(query.lower().split())
    return "it" if words & _ITALIAN_INDICATORS else "en"


def _query_from_lesson(lesson: Any) -> str:
    """
    Build the natural-language teacher query the agent expects when the
    user did *not* supply a free-text query in the lesson form.

    The agent expects "Crea una lezione su X per Y". We synthesize a
    sensible default from the profile fields. The agent then specializes
    via the ``educational_profile`` we also pass through (CORE 1 #2.5).
    """
    profile = lesson.educational_profile_json or {}
    topic = profile.get("specific_topic") or lesson.title or "argomento generale"
    subject = profile.get("subject_area")
    grade = (profile.get("group") or {}).get("grade")

    parts = [f"Crea una lezione su {topic}"]
    if subject:
        parts.append(f"per {subject}")
    if grade:
        # Grade is a code like "PRIMARY_4" — the prompt is friendlier with
        # the raw code than nothing; the educational_profile dict carries
        # the structured value too.
        parts.append(f"(livello: {grade})")

    return " ".join(parts)


def _extract_lesson_plan_md(final_state: Dict[str, Any]) -> str:
    """
    The lesson plan ends up under ``final_lesson_plan`` after a successful
    run, or ``lesson_plan_draft`` if the critic loop bailed out before
    approving. Empty string if neither is present (degenerate case — the
    route layer will surface this as an error).
    """
    return (
        final_state.get("final_lesson_plan")
        or final_state.get("lesson_plan_draft")
        or ""
    )


def _count_media(media: Optional[Dict[str, Any]]) -> Dict[str, int]:
    """
    Reduce the ``curated_media`` dict to a flat tallied summary suitable
    for the retriever card and the final run summary.

    Source shape (see ``retriever_agent.py``):
        curated_media = {
            "videos":         [ {title, url, ...}, ... ],
            "resources":      [ {title, url, ...}, ... ],
            "citations":      [ {title, authors, year, ...}, ... ],
            "open_textbooks": [ {title, source, chapter, ...}, ... ],
            "images":         [ ... ],   # not surfaced in the UI yet
        }

    We bucket these into the three sidebar groups the user actually sees:
        videos    → "Video curati"
        articles  → "Articoli scientifici"   (citations from Semantic Scholar)
        oer       → "Risorse OER & Manuali"  (resources + open_textbooks)
    """
    if not isinstance(media, dict):
        return {"videos": 0, "articles": 0, "oer": 0}

    def _len_of(key: str) -> int:
        value = media.get(key)
        return len(value) if isinstance(value, (list, dict)) else 0

    return {
        "videos":   _len_of("videos"),
        "articles": _len_of("citations"),
        "oer":      _len_of("resources") + _len_of("open_textbooks"),
    }


def _teacher_upload_context(lesson: Any) -> Optional[str]:
    """
    Join the text excerpts of the teacher's uploaded files into a single
    plain-text block for ``AgentState.teacher_provided_context``.

    This is *only* used by the Writer prompt (CORE 2 #6.6 P3). The Planner
    and Retriever stay GraphRAG-only — uploads are not ingested into the
    domain Knowledge Graph in this scope.
    """
    raw = getattr(lesson, "uploaded_files_json", None)
    if not raw or not isinstance(raw, list):
        return None
    parts: list[str] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        ex = (item.get("text_excerpt") or "").strip()
        if not ex:
            continue
        title = (item.get("filename") or "file").strip()
        parts.append(f"### {title}\n{ex}")
    if not parts:
        return None
    joined = "\n\n".join(parts)
    return joined[:48000] if len(joined) > 48000 else joined


def _build_planner_payload(state: Dict[str, Any]) -> Dict[str, Any]:
    """Shape the post-plan state into the planner card's context."""
    intent = state.get("query_intent") or "unknown"
    scope = state.get("scope_status") or "unknown"
    scope_label, scope_variant = _SCOPE_LABELS.get(scope, (scope, "neutral"))
    return {
        "intent": intent,
        "intent_label": _INTENT_LABELS.get(intent, intent),
        "scope": scope,
        "scope_label": scope_label,
        "scope_variant": scope_variant,
        "scope_confidence": state.get("scope_confidence"),
        "key_concepts": list(state.get("key_concepts") or [])[:8],
        "search_queries": list(state.get("search_queries") or [])[:8],
        "lesson_type": state.get("lesson_type"),
        "target_grade": state.get("target_grade"),
    }


def _build_retriever_payload(state: Dict[str, Any]) -> Dict[str, Any]:
    """Shape the post-retrieve state into the retriever card's context."""
    nodes = state.get("retrieved_nodes") or []
    rels = state.get("retrieved_relationships") or []
    recs = state.get("recommendations") or []
    media = state.get("curated_media") or {}

    # Best-effort top-N concept titles. Different code paths populate
    # different keys (``title`` vs ``name`` vs ``id``); we walk them all.
    top_concepts: list[str] = []
    for n in nodes[:5]:
        if not isinstance(n, dict):
            continue
        title = n.get("title") or n.get("name") or n.get("label") or n.get("id")
        if title:
            top_concepts.append(str(title))

    return {
        "nodes_count": len(nodes),
        "relationships_count": len(rels),
        "recommendations_count": len(recs),
        "media_counts": _count_media(media),
        "media": media,  # full payload for the right sidebar
        "top_concepts": top_concepts,
        "retrieval_confidence": state.get("retrieval_confidence"),
    }


def _build_critic_payload(state: Dict[str, Any]) -> Dict[str, Any]:
    """Shape the post-critique state into the critic card's context."""
    score = state.get("critique_score")
    score_pct: Optional[int] = None
    if isinstance(score, (int, float)):
        raw = float(score)
        # The critic agent's ``average_score`` is on a 1–5 scale (see
        # ``CritiqueResult.average_score`` and ``critic_prompt.py``).
        # Convert to 0–100% for display, clamped.
        if raw <= 5.0:
            score_pct = min(100, max(0, int(round(raw / 5.0 * 100))))
        else:
            score_pct = min(100, int(round(raw)))

    return {
        "approved": bool(state.get("approved", False)),
        "revision_count": int(state.get("revision_count", 0)),
        "max_revisions": int(state.get("max_revisions", 2)),
        "score": score,
        "score_pct": score_pct,
        "critique": (state.get("critique") or "").strip(),
        "revision_instructions": (state.get("revision_instructions") or "").strip(),
    }


def _extract_meta(final_state: Dict[str, Any]) -> Dict[str, Any]:
    """Run summary used by the final lesson card and the run-complete banner."""
    media = final_state.get("curated_media") or {}
    return {
        "approved": bool(final_state.get("approved", False)),
        "revision_count": int(final_state.get("revision_count", 0)),
        "scores": (final_state.get("final_metadata") or {}).get("scores"),
        "nodes_count": len(final_state.get("retrieved_nodes") or []),
        "recommendations_count": len(final_state.get("recommendations") or []),
        "media_counts": _count_media(media),
        "search_queries_count": len(final_state.get("search_queries") or []),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_agent_stream(
    lesson: Any,
    session: AsyncSession,
) -> AsyncIterator[StreamEvent]:
    """
    Run the LangGraph lesson-planner pipeline for ``lesson`` and yield
    chat-card events as the graph progresses.

    Lifecycle side-effects (persisted via ``session``):
        on entry         : lesson.status = "running",
                           lesson.teacher_query persisted (if not already)
        on success       : lesson.status = "complete",
                           lesson_plan_md = <markdown>
        on exception     : lesson.status = "error",
                           error_message = <short msg>

    Yield order for a 0-revision run (happy path):
        planner → retriever → writer_pending → writer → critic → done

    Yield order for a 1-revision run:
        planner → retriever
                → writer_pending(rev=1)  → writer(rev=1)  → critic
                → writer_pending(rev=2)  → writer(rev=2)  → critic
                → done

    The route layer maps each event 1:1 to an SSE message; the partial
    template named ``partials/chat_<kind>_card.html`` (or ``chat_writer_*``)
    renders the card. The ``writer_pending`` event lets the UI show a
    "Sto scrivendo…" placeholder during the (slow, non-streaming) writer
    call; the matching ``writer`` event then replaces that placeholder
    in-place via an OOB swap on the unique ``writer-card-rev{N}`` id.
    """
    started_at = time.monotonic()
    final_state: Dict[str, Any] = {}
    write_revision_idx = 0  # 1-based counter once first writer_pending fires

    # ── Setup phase. Anything that can blow up here (import errors, OpenAI
    # client init, missing env vars, …) MUST be caught so we never leave the
    # lesson in ``status="running"`` with no streaming events. Without this
    # guard the chat appears frozen for the user (P3 regression).
    try:
        # Lazy imports: keep the heavy LangGraph + OpenAI stack out of the
        # import path of cold modules (test collection, etc.). The
        # orchestrator already does ``load_dotenv()`` at import — fine to
        # incur once per process.
        from aix.agent.graph.state import create_initial_state
        from aix.agent.orchestrator import AgentOrchestrator

        profile_dict = lesson.educational_profile_json or None

        # Use the persisted teacher_query if the user supplied one in the
        # form; otherwise synthesize from the profile and *write it back*
        # so the user's first chat bubble survives reloads.
        if getattr(lesson, "teacher_query", None) and lesson.teacher_query.strip():
            query = lesson.teacher_query.strip()
        else:
            query = _query_from_lesson(lesson)
            lesson.teacher_query = query

        # Domain comes from the form ("neuro" / "udl" — captured at submit
        # time in P1). Language is inferred from the teacher's query so that
        # English queries get English lessons.
        orchestrator = AgentOrchestrator(
            domain=lesson.domain or "neuro",
            language=_detect_language(query),
        )
        pipeline = orchestrator._get_pipeline()  # noqa: SLF001 — intentional seam
        graph = pipeline._get_graph()  # noqa: SLF001

        teacher_ctx = _teacher_upload_context(lesson)

        initial_state = create_initial_state(
            query=query,
            domain=pipeline.domain,
            language=pipeline.language,
            session_id=str(lesson.id),
            max_revisions=pipeline.max_revisions,
            educational_profile=profile_dict,
            teacher_provided_context=teacher_ctx,
        )

        logger.info(
            "[webui.agent] starting run lesson_id=%s domain=%s query=%r "
            "uploads=%s",
            lesson.id, lesson.domain, query[:80],
            len(getattr(lesson, "uploaded_files_json", None) or []),
        )

        # ── Mark RUNNING ─────────────────────────────────────────────
        lesson.status = "running"
        lesson.error_message = None
        await session.commit()
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "[webui.agent] setup FAILED lesson_id=%s", lesson.id,
        )
        msg = str(exc) or exc.__class__.__name__
        short_msg = msg[:480] + ("…" if len(msg) > 480 else "")
        try:
            lesson.status = "error"
            lesson.error_message = short_msg
            await session.commit()
        except Exception:  # noqa: BLE001
            logger.exception(
                "[webui.agent] failed to persist setup error for lesson_id=%s",
                lesson.id,
            )
        yield StreamEvent(kind="error", error=short_msg)
        return

    try:
        async for chunk in graph.astream(initial_state, stream_mode="updates"):
            # ``chunk`` is ``{node_name: partial_state_update}``. In normal
            # operation a chunk has exactly one key — the node that just
            # finished. We tolerate the multi-key shape defensively.
            for node_name, state_diff in chunk.items():
                if node_name not in PHASE_LABELS:
                    # START / END pseudo-nodes, or any future internal node
                    # we don't have a label for — silently ignore.
                    continue
                if isinstance(state_diff, dict):
                    final_state.update(state_diff)

                if node_name == "plan":
                    yield StreamEvent(
                        kind="planner",
                        payload=_build_planner_payload(final_state),
                    )

                elif node_name == "retrieve":
                    yield StreamEvent(
                        kind="retriever",
                        payload=_build_retriever_payload(final_state),
                    )
                    # Writer is about to start. Emit a synthetic pending
                    # card NOW so the chat doesn't look frozen for the
                    # 60-90s the writer call typically takes.
                    write_revision_idx += 1
                    yield StreamEvent(
                        kind="writer_pending",
                        payload={
                            "revision": write_revision_idx,
                            "is_revision": False,
                            "feedback": "",
                        },
                    )

                elif node_name == "write":
                    yield StreamEvent(
                        kind="writer",
                        payload={"revision": write_revision_idx},
                        lesson_plan_md=final_state.get("lesson_plan_draft") or "",
                    )

                elif node_name == "critique":
                    yield StreamEvent(
                        kind="critic",
                        payload=_build_critic_payload(final_state),
                    )
                    # Will the graph loop back into write? Mirror the
                    # routing in ``build_lesson_planner_graph``: revise
                    # while not approved AND under max_revisions.
                    approved = final_state.get("approved", False)
                    rev_count = int(final_state.get("revision_count", 0))
                    max_rev = int(final_state.get("max_revisions", 2))
                    if (not approved) and rev_count < max_rev:
                        write_revision_idx += 1
                        yield StreamEvent(
                            kind="writer_pending",
                            payload={
                                "revision": write_revision_idx,
                                "is_revision": True,
                                "feedback": (final_state.get("revision_instructions") or "").strip(),
                            },
                        )

        # ── Run finished cleanly ─────────────────────────────────────────
        elapsed = time.monotonic() - started_at
        lesson_plan_md = _extract_lesson_plan_md(final_state)
        meta = _extract_meta(final_state)
        meta["duration_seconds"] = round(elapsed, 1)

        # If the writer never produced anything, the run technically didn't
        # crash but it's still useless to the user — surface as error.
        if not lesson_plan_md.strip():
            raise RuntimeError(
                "L'agente ha terminato senza produrre una lezione "
                "(stato finale vuoto)."
            )

        lesson.status = "complete"
        lesson.lesson_plan_md = lesson_plan_md
        await session.commit()

        logger.info(
            "[webui.agent] run complete lesson_id=%s duration=%.1fs "
            "approved=%s revisions=%s",
            lesson.id, elapsed, meta["approved"], meta["revision_count"],
        )

        yield StreamEvent(
            kind="done",
            lesson_plan_md=lesson_plan_md,
            meta=meta,
        )

    except Exception as exc:  # noqa: BLE001 — we *do* want every failure
        logger.exception(
            "[webui.agent] run FAILED lesson_id=%s after %.1fs",
            lesson.id, time.monotonic() - started_at,
        )
        msg = str(exc) or exc.__class__.__name__
        # Truncate to the model column width (500). Hard slice so a giant
        # traceback string doesn't blow the row.
        short_msg = msg[:480] + ("…" if len(msg) > 480 else "")

        try:
            lesson.status = "error"
            lesson.error_message = short_msg
            await session.commit()
        except Exception:  # noqa: BLE001
            # Don't let a secondary commit failure mask the original error.
            logger.exception(
                "[webui.agent] failed to persist error state for lesson_id=%s",
                lesson.id,
            )

        yield StreamEvent(kind="error", error=short_msg)


# ---------------------------------------------------------------------------
# Public DB-less helper — used by the JSON+SSE API at /api/v1/agent/*
#
# This is the same engine ``run_agent_stream`` drives, with two changes:
#   1. No ``Lesson`` row, no ``AsyncSession``, no DB writes.
#   2. Inputs come from the validated Pydantic request, not a persisted row.
#
# Why a separate helper instead of refactoring ``run_agent_stream``:
#   - ``run_agent_stream`` is on the hot path of the webui chat workspace
#     and has been smoke-tested through P2 → P3. Touching it would put that
#     surface at risk for the sake of code dedup.
#   - Both helpers share the *real* logic — the LangGraph ``astream`` loop
#     and the ``_build_*_payload`` shapers — by direct call. The only
#     duplication is the ~25-line dispatch loop, which is the cheapest
#     possible price for backward compatibility.
#   - When CORE 6 deploy is ready, both can be collapsed into a single
#     ``aix.agent.streaming`` module that lives under the agent layer
#     proper. That refactor is tracked under #7's "future improvements"
#     bullet, not in scope here.
#
# CORE 2 #7 — see docs/product/ClickUp_Agentic_GraphRAG_Update.md.
# ---------------------------------------------------------------------------


async def stream_agent_events(
    *,
    query: str,
    domain: str = "neuro",
    language: str = "it",
    session_id: Optional[str] = None,
    educational_profile: Optional[Dict[str, Any]] = None,
    teacher_provided_context: Optional[str] = None,
    max_revisions: Optional[int] = None,
) -> AsyncIterator[StreamEvent]:
    """
    Run the LangGraph lesson-planner pipeline for an *ad-hoc* request and
    yield ``StreamEvent`` objects in the same order ``run_agent_stream``
    does.

    This helper has no DB persistence: callers are responsible for any
    state they want to keep. It exists to back the public JSON+SSE API
    where the request is one-shot and the contract is the *event stream*
    itself, not a persisted ``Lesson`` row.

    Parameters mirror :func:`aix.agent.graph.state.create_initial_state`,
    which is the single source of truth for the agent's input shape.

    Yields the same ``kind`` taxonomy as :func:`run_agent_stream`:

        planner → retriever → writer_pending → writer → critic → done

    On any setup or runtime exception, yields exactly one ``error`` event
    and returns. The caller never sees an exception cross the generator
    boundary; failures are domain data.
    """
    started_at = time.monotonic()
    final_state: Dict[str, Any] = {}
    write_revision_idx = 0

    # ── Setup. Mirrors the guard in run_agent_stream so a missing API
    # key / import error / etc. surfaces as a clean ``error`` event
    # instead of a 500 from the route layer.
    try:
        from aix.agent.graph.state import create_initial_state
        from aix.agent.orchestrator import AgentOrchestrator

        orchestrator = AgentOrchestrator(domain=domain, language=language)
        pipeline = orchestrator._get_pipeline()  # noqa: SLF001 — same seam as webui
        graph = pipeline._get_graph()  # noqa: SLF001

        effective_max_revisions = (
            max_revisions if max_revisions is not None else pipeline.max_revisions
        )

        initial_state = create_initial_state(
            query=query,
            domain=pipeline.domain,
            language=pipeline.language,
            session_id=session_id,
            max_revisions=effective_max_revisions,
            educational_profile=educational_profile,
            teacher_provided_context=teacher_provided_context,
        )

        logger.info(
            "[api.agent] starting run session_id=%s domain=%s query=%r "
            "max_revisions=%s profile=%s teacher_ctx_chars=%s",
            session_id, domain, query[:80], effective_max_revisions,
            "yes" if educational_profile else "no",
            len(teacher_provided_context or ""),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[api.agent] setup FAILED session_id=%s", session_id)
        msg = str(exc) or exc.__class__.__name__
        short_msg = msg[:480] + ("…" if len(msg) > 480 else "")
        yield StreamEvent(kind="error", error=short_msg)
        return

    try:
        async for chunk in graph.astream(initial_state, stream_mode="updates"):
            for node_name, state_diff in chunk.items():
                if node_name not in PHASE_LABELS:
                    continue
                if isinstance(state_diff, dict):
                    final_state.update(state_diff)

                if node_name == "plan":
                    yield StreamEvent(
                        kind="planner",
                        payload=_build_planner_payload(final_state),
                    )

                elif node_name == "retrieve":
                    yield StreamEvent(
                        kind="retriever",
                        payload=_build_retriever_payload(final_state),
                    )
                    write_revision_idx += 1
                    yield StreamEvent(
                        kind="writer_pending",
                        payload={
                            "revision": write_revision_idx,
                            "is_revision": False,
                            "feedback": "",
                        },
                    )

                elif node_name == "write":
                    yield StreamEvent(
                        kind="writer",
                        payload={"revision": write_revision_idx},
                        lesson_plan_md=final_state.get("lesson_plan_draft") or "",
                    )

                elif node_name == "critique":
                    yield StreamEvent(
                        kind="critic",
                        payload=_build_critic_payload(final_state),
                    )
                    approved = final_state.get("approved", False)
                    rev_count = int(final_state.get("revision_count", 0))
                    max_rev = int(final_state.get("max_revisions", 2))
                    if (not approved) and rev_count < max_rev:
                        write_revision_idx += 1
                        yield StreamEvent(
                            kind="writer_pending",
                            payload={
                                "revision": write_revision_idx,
                                "is_revision": True,
                                "feedback": (final_state.get("revision_instructions") or "").strip(),
                            },
                        )

        # ── Done ────────────────────────────────────────────────────────
        elapsed = time.monotonic() - started_at
        lesson_plan_md = _extract_lesson_plan_md(final_state)
        meta = _extract_meta(final_state)
        meta["duration_seconds"] = round(elapsed, 1)

        if not lesson_plan_md.strip():
            raise RuntimeError(
                "L'agente ha terminato senza produrre una lezione "
                "(stato finale vuoto)."
            )

        logger.info(
            "[api.agent] run complete session_id=%s duration=%.1fs "
            "approved=%s revisions=%s",
            session_id, elapsed, meta["approved"], meta["revision_count"],
        )

        yield StreamEvent(
            kind="done",
            lesson_plan_md=lesson_plan_md,
            meta=meta,
        )

    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "[api.agent] run FAILED session_id=%s after %.1fs",
            session_id, time.monotonic() - started_at,
        )
        msg = str(exc) or exc.__class__.__name__
        short_msg = msg[:480] + ("…" if len(msg) > 480 else "")
        yield StreamEvent(kind="error", error=short_msg)
