"""
LangGraph Node Implementations

Each node wraps an agent and handles state updates.
Nodes are the building blocks of the LangGraph state machine.
"""

import logging
import os
import re
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from aix.agent.agents.critic_agent import CriticAgent
from aix.agent.agents.planner_agent import PlannerAgent
from aix.agent.agents.retriever_agent import RetrieverAgent
from aix.agent.agents.writer_agent import WriterAgent
from aix.agent.graph import write_stream
from aix.agent.graph.state import AgentState

# CORE 2 #12b.3 — duration-mention sniffer used by plan_node to decide
# whether the *current* user turn explicitly mentions a duration. When it
# does, the Planner-extracted duration wins (teacher just said it). When
# it doesn't, the educational profile's ``time_available_minutes`` is
# authoritative and overrides any duration the Planner may have extracted
# from the conversation history. Word-boundary anchored to avoid matching
# stray digits inside concept names ("UDL 7.2" must not trigger).
_DURATION_MENTION_RE = re.compile(
    r"\b\d+\s*(?:min(?:ut[oi]|utes?)?|h(?:rs?)?|hours?|or[ae])\b",
    re.IGNORECASE,
)


def _current_turn_mentions_duration(state: AgentState) -> bool:
    """True iff the teacher's CURRENT turn (not the history) mentions a duration.

    Falls back to ``teacher_query`` when ``raw_user_turn`` is missing — this
    keeps legacy callers (single-turn, no history) byte-identical to pre-#12b.3,
    since on first turn ``teacher_query`` IS the raw turn.
    """
    raw = state.get("raw_user_turn")
    if raw is None:
        raw = state.get("teacher_query") or ""
    return bool(_DURATION_MENTION_RE.search(raw))


# CORE 2 #9 — Corrective RAG. Imported eagerly so the module import
# fails fast if the file is malformed; the actual agent is only
# *instantiated* (and the LLM ever called) when the corrective-RAG
# topology is wired in by ``build_lesson_planner_graph_async``, which
# itself is gated by ``AIX_CORRECTIVE_RAG_ENABLED``.
from aix.agent.agents.retrieval_grader_agent import RetrievalGraderAgent


def _sanitize(obj: Any) -> Any:
    """Recursively convert numpy scalar/array types to plain Python types.

    LangGraph's SQLite checkpointer uses msgpack which cannot serialize
    numpy.float64, numpy.int64, etc.  These values come from Neo4j node
    properties (node2vec embeddings, sklearn similarity scores stored as
    numpy scalars).  We normalise them here, once, before the retriever
    output enters AgentState — keeping the rest of the pipeline clean.
    """
    try:
        import numpy as np

        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass  # numpy not installed — nothing to sanitize

    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Transient LLM failures
# ---------------------------------------------------------------------------
#
# The LLM nodes catch everything and turn it into an error state, which is the
# right default — a bad prompt or a malformed response should end the run
# cleanly rather than crash it. But it also means nothing ever propagates, so
# LangGraph's RetryPolicy would never fire: it only sees exceptions that leave
# the node.
#
# These four are the ones worth another attempt, and they are re-raised so the
# node-level RetryPolicy in ``lesson_planner_graph`` can act on them. The same
# tuple is the policy's ``retry_on``, so the two cannot drift apart.
#
# The outer retry rebuilds the prompt rather than replaying the same HTTP
# request, which is why it earns its place on top of the client's own retry
# (see ``OpenAIConfig`` for how the total budget is split).
TRANSIENT_LLM_ERRORS: tuple[type[Exception], ...] = (
    RateLimitError,  # 429 — provider throttling
    APITimeoutError,  # the request outlived AIX_LLM_TIMEOUT_S
    APIConnectionError,  # DNS / TLS / socket
    InternalServerError,  # 5xx from the provider
)

# Initialize agents (lazy loading).
#
# The planner/writer/critic/grader are domain-agnostic and hold no per-call
# state on ``self`` (only an immutable ``model`` + a concurrency-safe async
# client), so a single shared instance is safe across concurrent runs.
#
# The retriever is domain-specific, so it is cached PER DOMAIN (see #38).
_planner: PlannerAgent = None
_retrievers: dict[str, RetrieverAgent] = {}
_writer: WriterAgent = None
_critic: CriticAgent = None
_retrieval_grader: RetrievalGraderAgent = None


def get_planner() -> PlannerAgent:
    global _planner
    if _planner is None:
        _planner = PlannerAgent()
    return _planner


def get_retriever(domain: str) -> RetrieverAgent:
    """Return the cached :class:`RetrieverAgent` for ``domain``, building it once.

    #38 — previously this was a single module-level slot that got *swapped*
    whenever the requested domain changed. Once #31 allows several pipelines
    in flight at once, two runs in different domains (e.g. ``neuro`` + ``udl``)
    would clobber each other's retriever mid-run → wrong-domain retrieval (a
    correctness bug). Caching one warmed instance per domain removes the race
    and also avoids re-instantiating the heavy ``GraphRAGTool`` (Node2Vec +
    embeddings) on every domain switch.

    Safe to share across concurrent same-domain runs: ``RetrieverAgent`` keeps
    no per-call state on ``self`` (only domain-scoped, lazily-built tools).
    The get-or-create has no ``await`` between the read and the write, so it is
    atomic on the single-threaded event loop — no lock required. Bounded by the
    number of domains (currently 2), so memory is trivially scalable.
    """
    retriever = _retrievers.get(domain)
    if retriever is None:
        retriever = RetrieverAgent(domain=domain)
        _retrievers[domain] = retriever
    return retriever


def get_writer() -> WriterAgent:
    global _writer
    if _writer is None:
        _writer = WriterAgent()
    return _writer


def get_critic() -> CriticAgent:
    global _critic
    if _critic is None:
        _critic = CriticAgent()
    return _critic


def get_retrieval_grader() -> RetrievalGraderAgent:
    """Lazy singleton for the corrective-RAG grader (CORE 2 #9).
    Instantiated only when ``grade_retrieval_node`` actually runs, so
    the LLM client/object is never created when the feature flag is off."""
    global _retrieval_grader
    if _retrieval_grader is None:
        _retrieval_grader = RetrievalGraderAgent()
    return _retrieval_grader


# CORE 2 #9 — Corrective RAG configuration knobs.
# Defaults are tuned for the "feature off" scenario: even when
# ``AIX_CORRECTIVE_RAG_ENABLED=true`` flips the topology, the loop has a
# hard ceiling of 2 attempts so a flaky LLM cannot run the pipeline in
# circles. Both knobs are env-driven so ops can adjust without a redeploy.
def _corrective_rag_max_attempts() -> int:
    raw = os.getenv("AIX_CORRECTIVE_RAG_MAX_ATTEMPTS", "2")
    try:
        v = int(raw)
        return max(1, min(v, 4))  # clamp to a sane range
    except ValueError:
        return 2


def _corrective_rag_enabled() -> bool:
    return (os.getenv("AIX_CORRECTIVE_RAG_ENABLED") or "false").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


async def plan_node(state: AgentState) -> dict[str, Any]:
    """
    PLANNER NODE

    Analyzes the teacher query, detects intent, and creates a retrieval plan.

    Input: teacher_query, domain, language
    Output: plan, query_intent, lesson_type, key_concepts, search_queries,
            language (possibly overridden by LLM-driven detection — Point (a))
    """
    logger.info("[Node: Plan] Starting planning phase...")

    planner = get_planner()
    seed_language = state.get("language", "it")

    try:
        # T6: on a follow-up, the planner is also asked whether this turn needs
        # a new search. It cannot answer that from the query alone — "traduci
        # la lezione in inglese" only means something next to the lesson it
        # refers to — so it gets the thread and the corpus with it. Only when
        # the feature is on, so a run with the flag off sends the same prompt
        # it always sent.
        thread_context: dict[str, Any] = {}
        if _conditional_retrieval_enabled() and (state.get("conversation_history") or []):
            thread_context = {
                "conversation_history": state.get("conversation_history"),
                "existing_corpus": sorted(_corpus_names(state)),
            }

        plan = await planner.plan(
            query=state["teacher_query"],
            domain=state.get("domain", "neuro"),
            language=seed_language,
            pedagogical_intent=state.get("pedagogical_intent"),
            # Q1 · 1.1: the classroom constraints reach the Planner as well as the
            # Writer, so retrieval is not spent on activities the room cannot run.
            educational_profile=state.get("educational_profile"),
            # The attachments, so concepts and queries come from the document
            # rather than from the words "documento allegato".
            teacher_materials=state.get("teacher_provided_context"),
            **thread_context,
        )

        # Point (a): the Planner LLM is the canonical L1 language detector.
        # The service layer pre-seeds state["language"] with a statistical
        # L2 (lingua-py) or L3 (default "it") guess so the Planner has a
        # reasonable starting hint, but the Planner sees the actual query
        # text and can correct mistakes (e.g., a follow-up "adatta per
        # studenti con DSA" that the L2 detector might mis-identify because
        # of the all-caps acronym + short length). When the Planner is
        # confident, we OVERRIDE state["language"] for the rest of the
        # pipeline so writer + critic both reply in the correct language.
        effective_language = seed_language
        language_overridden = False
        if plan.has_confident_language and plan.response_language != seed_language:
            effective_language = plan.response_language
            language_overridden = True
            logger.info(
                "[Node: Plan] 🌐 Language override: seed=%r → planner=%r "
                "(confidence=%s) — writer/critic will reply in %r",
                seed_language,
                plan.response_language,
                plan.language_confidence,
                plan.response_language,
            )
        elif plan.response_language:
            logger.info(
                "[Node: Plan] Language confirmed: %r (planner_conf=%s, seed=%r)",
                effective_language,
                plan.language_confidence,
                seed_language,
            )
        else:
            # Planner didn't emit response_language (legacy prompt / older
            # model output) — keep the seed silently.
            logger.debug(
                "[Node: Plan] Planner did not emit response_language; keeping seed language %r",
                seed_language,
            )

        # Reconcile time_constraints — CORE 2 #12b.1 → #12b.3.
        #
        # Three precedence rules, in order:
        #   1. The teacher's CURRENT turn explicitly mentions a duration
        #      ("ora rifalla in 30 minuti") → that wins, even if it differs
        #      from the profile. Detected via ``_current_turn_mentions_duration``
        #      reading ``raw_user_turn`` (populated by the service layer
        #      since #12b.3) — falls back to ``teacher_query`` when the
        #      caller didn't populate the raw turn (single-turn / legacy).
        #   2. The educational profile has ``time_available_minutes`` set
        #      AND the current turn does NOT mention a duration → the
        #      profile is authoritative. This overrides any Planner-
        #      extracted duration that came from the conversation history.
        #      Without this branch (the pre-#12b.3 behaviour), a previous
        #      turn's "45 min" would silently outrank the teacher's just-
        #      updated 60 min sidebar setting.
        #   3. Legacy fallback for callers that don't pass an educational
        #      profile: keep whatever the Planner extracted (or None).
        ep = state.get("educational_profile") or {}
        profile_duration = ep.get("time_available_minutes") or ep.get("lesson_duration")
        current_turn_has_duration = _current_turn_mentions_duration(state)

        if profile_duration and not current_turn_has_duration:
            inferred = f"{profile_duration} minutes"
            if plan.time_constraints and plan.time_constraints != inferred:
                logger.info(
                    "[Node: Plan] Overriding planner-extracted time_constraints=%r "
                    "with profile-authoritative %s (current turn has no duration; "
                    "profile.time_available_minutes=%s).",
                    plan.time_constraints,
                    inferred,
                    profile_duration,
                )
            elif not plan.time_constraints:
                logger.info(
                    "[Node: Plan] time_constraints filled from profile "
                    "(time_available_minutes=%s): %s",
                    profile_duration,
                    inferred,
                )
            plan.time_constraints = inferred
        elif profile_duration and current_turn_has_duration and plan.time_constraints:
            logger.info(
                "[Node: Plan] Current turn mentions a duration; honouring "
                "planner-extracted time_constraints=%r over profile "
                "time_available_minutes=%s.",
                plan.time_constraints,
                profile_duration,
            )

        # Enhanced logging with scope status
        scope_emoji = {"in_scope": "✅", "partial_scope": "⚠️", "out_of_scope": "❌"}.get(
            plan.scope_status, "❓"
        )
        logger.info(
            f"[Node: Plan] Detected intent: {plan.query_intent} (confidence: {plan.intent_confidence}), "
            f"scope: {scope_emoji} {plan.scope_status} ({plan.scope_confidence:.0%})"
        )

        updates: dict[str, Any] = {
            "plan": {
                "query_intent": plan.query_intent,
                "intent_confidence": plan.intent_confidence,
                "lesson_type": plan.lesson_type,
                "key_concepts": plan.key_concepts,
                "search_queries": plan.search_queries,
                "target_grade": plan.target_grade,
                "special_needs": plan.special_needs,
                "time_constraints": plan.time_constraints,
                "reasoning": plan.reasoning,
                # NEW Phase A: Scope detection fields
                "scope_status": plan.scope_status,
                "scope_confidence": plan.scope_confidence,
                "subject_concepts": plan.subject_concepts,
                "pedagogy_concepts": plan.pedagogy_concepts,
                # Point (a): Surface the planner's language verdict in the
                # plan dict so the chat-card (Planner Agent UI) and any
                # downstream observability can see what the LLM decided.
                "response_language": plan.response_language,
                "language_confidence": plan.language_confidence,
                "language_overridden": language_overridden,
                "retrieval_reason": plan.retrieval_reason,
            },
            "query_intent": plan.query_intent,
            "lesson_type": plan.lesson_type,
            "target_grade": plan.target_grade,
            "key_concepts": plan.key_concepts,
            "search_queries": plan.search_queries,
            # NEW Phase A: Scope detection
            "scope_status": plan.scope_status,
            "scope_confidence": plan.scope_confidence,
            "subject_concepts": plan.subject_concepts,
            "pedagogy_concepts": plan.pedagogy_concepts,
            # None on a first turn and whenever the planner was not asked —
            # which is exactly when should_retrieve falls back to concepts.
            "requires_new_retrieval": plan.requires_new_retrieval,
            "current_step": "plan_complete",
        }
        # Only emit a language update when we're actually changing it —
        # avoids a no-op state mutation and keeps LangGraph's checkpoint
        # diffs minimal on the common (seed-confirmed) path.
        if language_overridden:
            updates["language"] = effective_language
        return updates

    except TRANSIENT_LLM_ERRORS:
        # Re-raised so the node's RetryPolicy sees it. If the budget is
        # spent it escapes to the caller, which reports a failed run —
        # better than the empty 'done' an error state used to produce.
        raise
    except Exception as e:
        logger.error(f"[Node: Plan] Error: {e}")
        return {"error": str(e), "current_step": "error"}


async def retrieve_node(state: AgentState) -> dict[str, Any]:
    """
    RETRIEVER NODE

    Executes GraphRAG searches based on the plan.

    Input: plan, search_queries, domain
    Output: graphrag_results, retrieved_nodes, recommendations
    """
    # Early exit if previous node failed
    if state.get("error"):
        logger.warning("[Node: Retrieve] Skipping - previous node failed")
        return {"current_step": "error"}

    logger.info("[Node: Retrieve] Starting retrieval phase...")

    domain = state.get("domain", "neuro")
    retriever = get_retriever(domain)

    try:
        # Reconstruct the plan
        from aix.agent.agents.planner_agent import RetrievalPlan

        plan_data = state.get("plan") or {}
        plan = RetrievalPlan(
            query_intent=plan_data.get("query_intent", "lesson_creation"),
            key_concepts=plan_data.get("key_concepts", []),
            search_queries=plan_data.get("search_queries", [state["teacher_query"]]),
            lesson_type=plan_data.get("lesson_type"),
            target_grade=plan_data.get("target_grade"),
            special_needs=plan_data.get("special_needs"),
            time_constraints=plan_data.get("time_constraints"),
            intent_confidence=plan_data.get("intent_confidence", "MEDIUM"),
            # NEW Phase A: Scope detection fields
            scope_status=plan_data.get("scope_status", "in_scope"),
            scope_confidence=plan_data.get("scope_confidence", 1.0),
            subject_concepts=plan_data.get("subject_concepts"),
            pedagogy_concepts=plan_data.get("pedagogy_concepts"),
        )

        # Enrich plan.search_queries with educational profile terms
        ep = state.get("educational_profile") or {}
        profile_terms: list = []
        if ep.get("specific_topic"):
            profile_terms.append(ep["specific_topic"])
        if ep.get("subject_area") and ep["subject_area"] != ep.get("specific_topic"):
            profile_terms.append(ep["subject_area"])
        for d in (ep.get("group") or {}).get("disabilities") or []:
            if d:
                profile_terms.append(d)
        if profile_terms:
            existing_lower = {q.lower() for q in plan.search_queries}
            for pt in profile_terms:
                if pt.lower() not in existing_lower:
                    plan.search_queries.append(pt)
                    existing_lower.add(pt.lower())
            logger.info(
                "[Node: Retrieve] Profile enrichment added %d terms: %s",
                len(profile_terms),
                profile_terms,
            )

        result = await retriever.retrieve(plan)

        # Log hybrid retrieval if applicable
        if result.is_hybrid:
            logger.info("[Node: Retrieve] ⚠️ HYBRID mode: KG pedagogy + external resources")

        # Sanitize before storing in AgentState — msgpack (LangGraph checkpointer)
        # cannot serialize numpy scalar types that Neo4j / sklearn may return.
        return {
            "graphrag_results": _sanitize(
                [
                    {"nodes": r.nodes, "relationships": r.relationships}
                    for r in result.search_results
                ]
            ),
            "retrieved_nodes": _sanitize(result.nodes),
            "retrieved_relationships": _sanitize(result.relationships),
            "recommendations": _sanitize(result.recommendations),
            "retrieval_confidence": result.confidence,
            # NEW Phase 1: Curated media from sidecar JSON
            "curated_media": _sanitize(result.curated_media) if result.curated_media else None,
            # NEW Phase A: External resources for out-of-scope queries
            "external_resources": _sanitize(result.external_resources)
            if result.external_resources
            else None,
            "current_step": "retrieve_complete",
        }

    except Exception as e:
        logger.error(f"[Node: Retrieve] Error: {e}")
        return {"error": str(e), "current_step": "error"}


async def retrieve_private_node(state: AgentState) -> dict[str, Any]:
    """PRIVATE MATERIAL NODE — the teacher's own documents (§8.1, §8.5).

    Runs beside ``retrieve`` in the same superstep, which is why
    ``current_step`` and ``error`` carry reducers (§8.9).

    It searches nothing on its own: the WebUI registers a search function for
    this session (see ``material_source``), because the rows belong to the WebUI
    and the authorization is the teacher's. An API or CLI run registers none, so
    this node finds nothing and the lesson is written exactly as before.

    A failure here is never the lesson's failure. If the private index is down
    or the embedding provider refuses, the run continues on the Knowledge Graph
    alone — a lesson without the handout is worth more than no lesson.
    """
    from aix.agent.graph import material_source

    search = material_source.get(state.get("session_id"))
    if search is None:
        return {"current_step": "retrieve_complete"}

    # The subject concepts sharpen the query: "amplia la lezione" retrieves
    # little on its own, "amplia la lezione · guerra fredda · Yalta" retrieves
    # the pages the teacher means.
    concepts = " · ".join((state.get("subject_concepts") or [])[:5])
    query = f"{state.get('teacher_query') or ''} {concepts}".strip()

    try:
        items = await search(query)
    except Exception as exc:  # noqa: BLE001 — the Knowledge Graph alone still writes a lesson
        # The reason goes in the warning itself. The first real test logged only
        # "search failed", with the cause at DEBUG, and a broken import looked
        # exactly like a provider outage until someone read the ingestion job's
        # traceback instead.
        logger.warning(
            "[Node: PrivateMaterial] search failed (%s: %s), continuing without it",
            type(exc).__name__,
            exc,
        )
        logger.debug("[Node: PrivateMaterial] failure detail", exc_info=True)
        return {"current_step": "retrieve_complete"}

    logger.info("[Node: PrivateMaterial] %d passage(s) from the teacher's material", len(items))
    return {"material_context_items": items, "current_step": "retrieve_complete"}


async def write_node(state: AgentState) -> dict[str, Any]:
    """
    WRITER NODE

    Generates or revises the lesson plan.

    Input: teacher_query, plan, retrieved_nodes, recommendations
    Output: lesson_plan_draft
    """
    # Early exit if previous node failed
    if state.get("error"):
        logger.warning("[Node: Write] Skipping - previous node failed")
        return {"current_step": "error"}

    revision_count = state.get("revision_count", 0)

    if revision_count > 0:
        logger.info(f"[Node: Write] Revising (iteration {revision_count})...")
    else:
        logger.info("[Node: Write] Starting initial writing...")

    writer = get_writer()

    try:
        plan_data = state.get("plan") or {}
        query_intent = plan_data.get("query_intent", state.get("query_intent", "lesson_creation"))

        # Check if this is a revision
        if revision_count > 0 and state.get("lesson_plan_draft"):
            # Revision mode - pass intent for consistent formatting
            lesson_plan = await writer.revise(
                current_draft=state["lesson_plan_draft"],
                critique=state.get("critique", ""),
                revision_instructions=state.get("revision_instructions", ""),
                language=state.get("language", "it"),
                intent=query_intent,
                # Same field the initial write() gets. Omitting it here meant a
                # Critic-triggered revision silently dropped whatever the
                # teacher had uploaded.
                teacher_provided_context=state.get("teacher_provided_context"),
                material_context_items=state.get("material_context_items"),
            )
        else:
            # Initial writing
            from aix.agent.agents.planner_agent import RetrievalPlan
            from aix.agent.agents.retriever_agent import RetrievalResult

            plan = RetrievalPlan(
                query_intent=query_intent,
                key_concepts=plan_data.get("key_concepts", []),
                search_queries=plan_data.get("search_queries", []),
                lesson_type=plan_data.get("lesson_type"),
                target_grade=plan_data.get("target_grade"),
                special_needs=plan_data.get("special_needs"),
                time_constraints=plan_data.get("time_constraints"),
                intent_confidence=plan_data.get("intent_confidence", "MEDIUM"),
                # NEW Phase A: Scope detection fields
                scope_status=plan_data.get("scope_status", "in_scope"),
                scope_confidence=plan_data.get("scope_confidence", 1.0),
                subject_concepts=plan_data.get("subject_concepts"),
                pedagogy_concepts=plan_data.get("pedagogy_concepts"),
            )

            retrieval_result = RetrievalResult(
                nodes=(state.get("retrieved_nodes") or []),
                relationships=(state.get("retrieved_relationships") or []),
                recommendations=(state.get("recommendations") or []),
                confidence=state.get("retrieval_confidence", "MEDIUM"),
            )

            # Look up the SSE token bus for this session (registered by
            # service.py before graph.astream() starts). Present only for
            # webUI runs; None for API/MCP callers.
            _session_id = state.get("session_id")
            _token_bus = write_stream.get_bus(_session_id)

            try:
                lesson_plan = await writer.write(
                    teacher_query=state["teacher_query"],
                    plan=plan,
                    retrieval_result=retrieval_result,
                    language=state.get("language", "it"),
                    # NEW Phase 2: Pass curated media if available
                    curated_media=state.get("curated_media"),
                    # NEW Phase A: Pass external resources for hybrid mode
                    external_resources=state.get("external_resources"),
                    # NEW Phase B: Pass domain for extensions
                    domain=state.get("domain", "neuro"),
                    teacher_provided_context=state.get("teacher_provided_context"),
                    # Fase 2 · PR 6 — the retrieved passages. When there are
                    # any, the Writer uses them *instead* of the pasted
                    # document: that is the whole point of chunking it.
                    material_context_items=state.get("material_context_items"),
                    educational_profile=state.get("educational_profile"),
                    pedagogical_intent=state.get("pedagogical_intent"),
                    refinement_instruction=state.get("refinement_instruction"),
                    # CORE 2 #9 — Corrective RAG. None when the feature flag is
                    # off, so this is byte-identical to pre-#9 in default mode.
                    retrieval_warning=state.get("retrieval_warning"),
                    retrieval_grade_reason=state.get("retrieval_grade_reason"),
                    token_bus=_token_bus,
                )
            finally:
                # Always send the sentinel so the drain task in service.py exits
                # cleanly — even if write() raised an exception.
                if _token_bus is not None:
                    _token_bus.put_nowait(None)

        # An empty draft is a failed turn, and it has to say so. In a thread the
        # checkpointer still holds the previous turn's ``final_lesson_plan``, so
        # a silent empty write ends the run "complete" and saves the OLD lesson
        # as the answer to the NEW request — which is what happened on
        # 2026-09-16: the teacher asked to extend the lesson with a second
        # document and got the same lesson back, with no error.
        if not (lesson_plan or "").strip():
            logger.error("[Node: Write] the model returned no text (token budget exhausted?)")
            return {
                "error": (
                    "Il modello non ha prodotto testo: il budget di token per la scrittura "
                    "è stato consumato dal ragionamento. Aumenta AIX_WRITER_MAX_TOKENS "
                    "(consigliato 8000) e riprova."
                ),
                "current_step": "error",
            }

        return {"lesson_plan_draft": lesson_plan, "current_step": "write_complete"}

    except TRANSIENT_LLM_ERRORS:
        # Re-raised so the node's RetryPolicy sees it. If the budget is
        # spent it escapes to the caller, which reports a failed run —
        # better than the empty 'done' an error state used to produce.
        raise
    except Exception as e:
        logger.error(f"[Node: Write] Error: {e}")
        return {"error": str(e), "current_step": "error"}


async def critique_node(state: AgentState) -> dict[str, Any]:
    """
    CRITIC NODE

    Reviews the content and decides to approve or request revision.
    Adapts evaluation criteria based on query intent.

    Input: lesson_plan_draft, teacher_query, retrieved_nodes, query_intent
    Output: critique, approved, revision_instructions
    """
    # Early exit if previous node failed
    if state.get("error"):
        logger.warning("[Node: Critique] Skipping - previous node failed")
        return {"current_step": "error", "approved": False}

    # Check if we have content to critique
    if not state.get("lesson_plan_draft"):
        logger.warning("[Node: Critique] No content to review")
        return {"error": "No content generated", "current_step": "error", "approved": False}

    # Get query intent for appropriate evaluation
    plan_data = state.get("plan") or {}
    query_intent = plan_data.get("query_intent", state.get("query_intent", "lesson_creation"))

    logger.info(f"[Node: Critique] Reviewing content (intent: {query_intent})...")

    critic = get_critic()

    try:
        from aix.agent.agents.retriever_agent import RetrievalResult

        retrieval_result = RetrievalResult(
            nodes=(state.get("retrieved_nodes") or []),
            relationships=(state.get("retrieved_relationships") or []),
            recommendations=(state.get("recommendations") or []),
            confidence=state.get("retrieval_confidence", "MEDIUM"),
        )

        result = await critic.critique(
            lesson_plan=state["lesson_plan_draft"],
            teacher_query=state["teacher_query"],
            retrieval_result=retrieval_result,
            revision_count=state.get("revision_count", 0),
            max_revisions=state.get("max_revisions", 1),
            domain=state.get("domain", "neuro"),
            language=state.get("language", "it"),
            query_intent=query_intent,
        )

        updates = {
            "critique": result.summary,
            "critique_score": result.average_score,
            "approved": result.approved,
            "current_step": "critique_complete",
        }

        if result.approved:
            updates["final_lesson_plan"] = state["lesson_plan_draft"]
            updates["final_metadata"] = {
                "scores": result.scores,
                "strengths": result.strengths,
                "revision_count": state.get("revision_count", 0),
            }
        else:
            updates["revision_instructions"] = result.revision_instructions
            updates["revision_count"] = state.get("revision_count", 0) + 1

        return updates

    except TRANSIENT_LLM_ERRORS:
        # Re-raised so the node's RetryPolicy sees it. If the budget is
        # spent it escapes to the caller, which reports a failed run —
        # better than the empty 'done' an error state used to produce.
        raise
    except Exception as e:
        logger.error(f"[Node: Critique] Error: {e}")
        return {"error": str(e), "current_step": "error"}


async def validate_node(state: AgentState) -> dict[str, Any]:
    """Check the draft against the classroom. Pure Python, no model call.

    Runs between the Writer and the Critic so that an infeasible lesson is
    caught before anything scores it. The Critic used to be the only judge, and
    it rated pedagogical quality and practical feasibility with the same 1–5
    rubric — which is how a lesson that four fifths of a class could not do came
    back at 91%.
    """
    from aix.agent.validation import validate_lesson, validator_enabled

    draft = state.get("lesson_plan_draft") or ""
    if not validator_enabled() or not draft:
        return {"validation_passed": True, "current_step": "write_complete"}

    report = validate_lesson(draft, state.get("educational_profile"))
    updates: dict[str, Any] = {
        "validation_violations": report.as_dicts(),
        "validation_passed": report.passed,
        "current_step": "write_complete",
    }

    if report.blocking:
        # The remediation goes through ``refinement_instruction``, which
        # ``write_node`` already forwards to the Writer. A retry therefore needs
        # no new mechanism — and it carries only what is blocking, so the one
        # thing that must change is not buried under the fifteen that must not.
        updates["refinement_instruction"] = report.remediation_prompt()
        updates["validation_attempts"] = (state.get("validation_attempts") or 0) + 1
        logger.info(
            "[Node: Validate] %d blocking violation(s): %s",
            len(report.blocking),
            ", ".join(sorted({v.rule for v in report.blocking})),
        )
    else:
        logger.info(
            "[Node: Validate] feasible (%d warning(s))",
            len(report.warnings),
        )
    return updates


def should_retry_after_validation(state: AgentState) -> str:
    """Conditional edge after the validator: rewrite, or hand over to the Critic.

    Returns ``"rewrite"`` or ``"critique"``. The budget is the validator's own,
    not the Critic's: spending a revision on an infeasible draft would leave the
    teacher with fewer chances at a *better* lesson, which is a different thing.
    """
    from aix.agent.validation import max_retries

    if state.get("error"):
        return "critique"
    if state.get("validation_passed", True):
        return "critique"

    # ``validation_attempts`` counts *failed validations*, and the node has
    # already incremented it for this one. So after the first failure it reads 1
    # while zero rewrites have happened yet: the rewrites done so far are
    # ``attempts - 1``, and a budget of N rewrites is spent when attempts > N.
    # Comparing with ``>=`` would quietly grant one fewer rewrite than the
    # setting promises.
    attempts = state.get("validation_attempts") or 0
    if attempts > max_retries():
        # Out of retries. The lesson goes on to the Critic with its violations
        # recorded: the teacher is better served seeing them than waiting for a
        # rewrite that the last two did not achieve.
        logger.info("[Router] validator budget spent (%d) — continuing to critique", attempts)
        return "critique"

    logger.info("[Router] lesson not feasible — targeted rewrite %d", attempts)
    return "rewrite"


def should_continue_to_revision(state: AgentState) -> str:
    """
    Conditional edge: Decide whether to revise or finish.

    Returns:
        "revise" - Go back to writer
        "finish" - End the pipeline
        "error" - Handle error
    """
    if state.get("error"):
        return "error"

    if state.get("approved", False):
        logger.info("[Router] Lesson plan approved, finishing...")
        return "finish"

    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 1)

    if revision_count >= max_revisions:
        logger.info("[Router] Max revisions reached, finishing...")
        return "finish"

    logger.info(f"[Router] Revision requested ({revision_count}/{max_revisions})")
    return "revise"


# ---------------------------------------------------------------------------
# T6 — Conditional retrieval: reuse the corpus when the turn adds nothing new
# ---------------------------------------------------------------------------
#
# Most follow-ups do not need a second Knowledge Graph search. "riscrivila più
# corta" and "adatta per ADHD" re-use the same evidence; only a turn that
# introduces a concept the corpus does not cover needs to go back to Neo4j.
#
# The decision is the planner's, not a manual toggle, because the planner
# already extracts ``key_concepts`` for the turn — comparing those against what
# the thread holds is free. A toggle would put the burden on the teacher to
# predict whether their own question introduces a new concept.
#
# This depends on the input schema from T4: without it the previous turn's
# retrieval is nulled before the graph starts, so there would be nothing to
# reuse. Gated by ``AIX_CONDITIONAL_RETRIEVAL_ENABLED``; with the flag off the
# ``plan → retrieve`` edge stays unconditional and behaviour is unchanged.


def _conditional_retrieval_enabled() -> bool:
    return (os.getenv("AIX_CONDITIONAL_RETRIEVAL_ENABLED") or "false").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _private_rag_enabled() -> bool:
    """The same flag the WebUI reads, read here without importing that layer."""
    return (os.getenv("AIX_PRIVATE_RAG_ENABLED") or "false").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _corpus_names(state: AgentState) -> set[str]:
    """Lower-cased names of everything the thread already retrieved."""
    names: set[str] = set()
    for item in (state.get("retrieved_nodes") or []) + (state.get("recommendations") or []):
        if not isinstance(item, dict):
            continue
        label = item.get("name") or item.get("title")
        if label:
            names.add(str(label).strip().lower())
    return names


def _corpus_covers_this_turn(state: AgentState) -> bool:
    """True when every concept the planner extracted is already in the corpus.

    Matching is substring in both directions rather than equality: the planner
    writes free text ("memoria di lavoro") while the graph stores node names
    ("Memoria di Lavoro"), and Italian inflection makes exact comparison miss
    far more often than a loose match over-matches. A false negative here only
    costs a retrieval that was not strictly needed; a false positive answers
    from evidence that does not cover the question, so the bias is deliberate.
    """
    concepts = [c.strip().lower() for c in (state.get("key_concepts") or []) if c and c.strip()]
    if not concepts:
        # The planner named nothing to look for, so there is nothing to add.
        return True

    corpus = _corpus_names(state)
    if not corpus:
        return False

    return all(any(c in name or name in c for name in corpus) for c in concepts)


def should_retrieve(state: AgentState) -> str:
    """Conditional edge after the planner: search the graph, or reuse it.

    Returns ``"retrieve"`` or ``"reuse"``. Criteria, in precedence order:

    1. an upstream error — keep today's path, where every remaining node
       no-ops its way to the end;
    2. the feature flag, off by default;
    3. the teacher's explicit override, which wins in both directions;
    4. the first turn of a thread, which always searches;
    5. an empty corpus, which leaves nothing to reuse;
    6. the planner's own verdict, given the thread and the corpus it holds;
    7. failing that, whether the planner's concepts are already covered.

    Six is new; seven used to be the whole of it. Comparing concepts never
    once chose reuse on a real thread: the planner was called with the
    bare query and no thread, so asked to translate an existing lesson it read
    the turn as a fresh request and emitted the concepts of a lesson nobody had
    asked for — generic pedagogy in English, against a corpus of subject nodes
    in Italian, which cannot match. The design always said the decision was the
    planner's; it just was not being given anything to decide with. Now it is,
    and the concept comparison stays only as the fallback for a turn where the
    planner did not answer.
    """
    if state.get("error"):
        return "retrieve"

    if not _conditional_retrieval_enabled():
        return "retrieve"

    override = state.get("force_retrieval")
    if override is True:
        logger.info("[Router] retrieval forced by the teacher → retrieve")
        return "retrieve"
    if override is False:
        logger.info("[Router] retrieval waived by the teacher → reuse")
        return "reuse"

    if not (state.get("conversation_history") or []):
        return "retrieve"

    if not state.get("retrieved_nodes"):
        logger.info("[Router] no corpus on this thread yet → retrieve")
        return "retrieve"

    verdict = state.get("requires_new_retrieval")
    if verdict is not None:
        reason = (state.get("plan") or {}).get("retrieval_reason") or "no reason given"
        logger.info(
            "[Router] planner says requires_new_retrieval=%s (%s) → %s",
            verdict,
            reason,
            "retrieve" if verdict else "reuse",
        )
        return "retrieve" if verdict else "reuse"

    if _corpus_covers_this_turn(state):
        logger.info(
            "[Router] turn adds no new concepts (%s) → reusing the existing corpus",
            state.get("key_concepts"),
        )
        return "reuse"

    logger.info("[Router] turn introduces new concepts (%s) → retrieve", state.get("key_concepts"))
    return "retrieve"


async def reuse_retrieval_node(state: AgentState) -> dict[str, Any]:
    """Mark the turn as reusing the thread's corpus, without touching Neo4j.

    A node rather than a bare edge so the decision is visible: it shows up in
    ``astream`` updates like every other stage, which is what lets the UI tell
    the teacher the answer came from evidence already gathered.
    """
    logger.info(
        "[Node: ReuseRetrieval] Skipping the Knowledge Graph search; "
        "reusing %d node(s) from an earlier turn.",
        len(state.get("retrieved_nodes") or []),
    )
    return {"retrieval_reused": True, "current_step": "retrieve_complete"}


# ---------------------------------------------------------------------------
# CORE 2 #9 — Corrective RAG: grade_retrieval_node + should_retry_retrieval
# ---------------------------------------------------------------------------
#
# Topology (when AIX_CORRECTIVE_RAG_ENABLED=true):
#
#     plan → retrieve → grade_retrieval ─[relevant or attempts==max]→ write
#                              │
#                              └─[ambiguous|irrelevant & attempts<max]→ retrieve
#
# When the flag is OFF (default), this node is NOT added to the workflow at
# all (see lesson_planner_graph._build_workflow), so the pre-#9 edge
# ``retrieve → write`` is preserved bit-for-bit. **No** code below runs in
# that mode; this section only matters once the flag flips on.


async def grade_retrieval_node(state: AgentState) -> dict[str, Any]:
    """
    GRADE-RETRIEVAL NODE  (CORE 2 #9 — Corrective RAG)

    Calls the cheap :class:`RetrievalGraderAgent` to decide whether the
    just-completed retrieval pass is good enough to feed the Writer.

    Reads:
        retrieved_nodes, recommendations, key_concepts, search_queries,
        retrieval_attempts, retrieval_rewritten_query (set by a prior
        retry pass on this same turn).
    Writes:
        retrieval_grade, retrieval_grade_reason, retrieval_rewritten_query,
        retrieval_attempts, retrieval_warning, plan.search_queries (when
        a rewrite is applied).

    On any error the node falls back to ``grade=relevant`` so the loop
    cannot block the writer — the worst case under failure equals the
    pre-#9 (no-grading) behaviour.
    """
    # Honour upstream errors — never grade a broken state.
    if state.get("error"):
        logger.warning("[Node: GradeRetrieval] Skipping - previous node failed")
        return {"current_step": "error"}

    attempts = int(state.get("retrieval_attempts") or 0) + 1
    max_attempts = _corrective_rag_max_attempts()
    logger.info(
        "[Node: GradeRetrieval] Grading retrieval attempt %d/%d",
        attempts,
        max_attempts,
    )

    grader = get_retrieval_grader()
    plan_data = state.get("plan") or {}

    try:
        grader_result = await grader.grade(
            query=state.get("teacher_query", ""),
            key_concepts=plan_data.get("key_concepts") or state.get("key_concepts"),
            search_queries=plan_data.get("search_queries") or state.get("search_queries"),
            retrieved_nodes=state.get("retrieved_nodes") or [],
            recommendations=state.get("recommendations") or [],
        )
    except Exception as e:  # noqa: BLE001 — defense in depth; agent already catches most
        logger.error(
            "[Node: GradeRetrieval] Unexpected grader failure (%s); "
            "defaulting to grade=relevant to preserve pre-#9 behaviour.",
            e,
        )
        return {
            "retrieval_grade": "relevant",
            "retrieval_grade_reason": f"Grader exception: {e.__class__.__name__}",
            "retrieval_attempts": attempts,
            "current_step": "grade_complete",
        }

    updates: dict[str, Any] = {
        "retrieval_grade": grader_result.grade,
        "retrieval_grade_reason": grader_result.reason,
        "retrieval_attempts": attempts,
        "current_step": "grade_complete",
    }

    # If the grader recommends a rewrite AND we still have budget, mutate
    # the plan's search_queries so the next ``retrieve`` pass picks them
    # up. We *prepend* the rewrite to keep the original queries as a
    # safety net (so the second pass never returns *less* than the first).
    can_retry = grader_result.needs_retry and attempts < max_attempts
    if can_retry and grader_result.rewritten_query:
        rewritten = grader_result.rewritten_query.strip()
        existing = list(plan_data.get("search_queries") or [])
        if rewritten and rewritten.lower() not in {q.lower() for q in existing}:
            new_queries = [rewritten] + existing
            new_plan = dict(plan_data)
            new_plan["search_queries"] = new_queries
            updates["plan"] = new_plan
            updates["search_queries"] = new_queries
            updates["retrieval_rewritten_query"] = rewritten
            logger.info(
                "[Node: GradeRetrieval] Rewriting search_queries with %r (attempt %d→%d)",
                rewritten,
                attempts,
                attempts + 1,
            )
        else:
            logger.info(
                "[Node: GradeRetrieval] Rewrite suggestion %r is a duplicate — "
                "skipping rewrite and continuing with current queries.",
                rewritten,
            )

    # When attempts are exhausted with a non-relevant grade, mark a warning
    # for the Writer so the lesson can carry a short caveat. This is
    # additive — Writer reads ``retrieval_warning`` only when the field
    # exists in state; pre-#9 callers set it to None.
    if grader_result.needs_retry and attempts >= max_attempts:
        updates["retrieval_warning"] = True
        logger.warning(
            "[Node: GradeRetrieval] grade=%s after max attempts (%d); flagging "
            "retrieval_warning for Writer to carry a low-confidence caveat.",
            grader_result.grade,
            max_attempts,
        )
    else:
        # Explicit None on the relevant path so the writer doesn't see a
        # stale warning from a checkpoint of an earlier turn.
        updates["retrieval_warning"] = (
            False if grader_result.grade == "relevant" else (updates.get("retrieval_warning"))
        )

    return updates


def should_retry_retrieval(state: AgentState) -> str:
    """
    Conditional edge after :func:`grade_retrieval_node` (CORE 2 #9).

    Routes:
      * ``"retry"``  — re-enter the retriever with the (possibly rewritten)
                       search_queries; bounded by ``max_attempts``.
      * ``"continue"`` — proceed to the writer with whatever we have.
      * ``"error"``  — propagate upstream failures.
    """
    if state.get("error"):
        return "error"

    grade = state.get("retrieval_grade", "relevant")
    attempts = int(state.get("retrieval_attempts") or 0)
    max_attempts = _corrective_rag_max_attempts()

    if grade == "relevant":
        logger.info("[Router] retrieval grade=relevant → writer")
        return "continue"

    if attempts >= max_attempts:
        logger.info(
            "[Router] retrieval grade=%s but max_attempts=%d reached → writer "
            "(retrieval_warning=True)",
            grade,
            max_attempts,
        )
        return "continue"

    logger.info(
        "[Router] retrieval grade=%s (attempts=%d/%d) → retry retrieve",
        grade,
        attempts,
        max_attempts,
    )
    return "retry"
