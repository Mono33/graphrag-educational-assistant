"""
LangGraph Node Implementations

Each node wraps an agent and handles state updates.
Nodes are the building blocks of the LangGraph state machine.
"""

import logging
import os
import re
from typing import Dict, Any

from aix.agent.graph.state import AgentState
from aix.agent.agents.planner_agent import PlannerAgent
from aix.agent.agents.retriever_agent import RetrieverAgent
from aix.agent.agents.writer_agent import WriterAgent
from aix.agent.agents.critic_agent import CriticAgent

# CORE 2 #12b.3 — duration-mention sniffer used by plan_node to decide
# whether the *current* user turn explicitly mentions a duration. When it
# does, the Planner-extracted duration wins (teacher just said it). When
# it doesn't, the educational profile's ``time_available_minutes`` is
# authoritative and overrides any duration the Planner may have extracted
# from the conversation history. Word-boundary anchored to avoid matching
# stray digits inside concept names ("UDL 7.2" must not trigger).
_DURATION_MENTION_RE = re.compile(
    r"\b\d+\s*(?:min(?:ut[oi]|utes?)?|h(?:rs?)?|hours?|ore?)\b",
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

# Initialize agents (lazy loading)
_planner: PlannerAgent = None
_retriever: RetrieverAgent = None
_writer: WriterAgent = None
_critic: CriticAgent = None
_retrieval_grader: RetrievalGraderAgent = None


def get_planner() -> PlannerAgent:
    global _planner
    if _planner is None:
        _planner = PlannerAgent()
    return _planner


def get_retriever(domain: str) -> RetrieverAgent:
    global _retriever
    if _retriever is None or _retriever.domain != domain:
        _retriever = RetrieverAgent(domain=domain)
    return _retriever


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
        "1", "true", "yes", "on"
    )


async def plan_node(state: AgentState) -> Dict[str, Any]:
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
        plan = await planner.plan(
            query=state["teacher_query"],
            domain=state.get("domain", "neuro"),
            language=seed_language,
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
                seed_language, plan.response_language,
                plan.language_confidence, plan.response_language,
            )
        elif plan.response_language:
            logger.info(
                "[Node: Plan] Language confirmed: %r (planner_conf=%s, seed=%r)",
                effective_language, plan.language_confidence, seed_language,
            )
        else:
            # Planner didn't emit response_language (legacy prompt / older
            # model output) — keep the seed silently.
            logger.debug(
                "[Node: Plan] Planner did not emit response_language; "
                "keeping seed language %r",
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
                    plan.time_constraints, inferred, profile_duration,
                )
            elif not plan.time_constraints:
                logger.info(
                    "[Node: Plan] time_constraints filled from profile "
                    "(time_available_minutes=%s): %s",
                    profile_duration, inferred,
                )
            plan.time_constraints = inferred
        elif profile_duration and current_turn_has_duration and plan.time_constraints:
            logger.info(
                "[Node: Plan] Current turn mentions a duration; honouring "
                "planner-extracted time_constraints=%r over profile "
                "time_available_minutes=%s.",
                plan.time_constraints, profile_duration,
            )

        # Enhanced logging with scope status
        scope_emoji = {"in_scope": "✅", "partial_scope": "⚠️", "out_of_scope": "❌"}.get(plan.scope_status, "❓")
        logger.info(
            f"[Node: Plan] Detected intent: {plan.query_intent} (confidence: {plan.intent_confidence}), "
            f"scope: {scope_emoji} {plan.scope_status} ({plan.scope_confidence:.0%})"
        )
        
        updates: Dict[str, Any] = {
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
            "current_step": "plan_complete",
        }
        # Only emit a language update when we're actually changing it —
        # avoids a no-op state mutation and keeps LangGraph's checkpoint
        # diffs minimal on the common (seed-confirmed) path.
        if language_overridden:
            updates["language"] = effective_language
        return updates
        
    except Exception as e:
        logger.error(f"[Node: Plan] Error: {e}")
        return {
            "error": str(e),
            "current_step": "error"
        }


async def retrieve_node(state: AgentState) -> Dict[str, Any]:
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
        
        plan_data = state.get("plan", {})
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
            pedagogy_concepts=plan_data.get("pedagogy_concepts")
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
                len(profile_terms), profile_terms
            )

        result = await retriever.retrieve(plan)

        # Log hybrid retrieval if applicable
        if result.is_hybrid:
            logger.info(f"[Node: Retrieve] ⚠️ HYBRID mode: KG pedagogy + external resources")

        # Sanitize before storing in AgentState — msgpack (LangGraph checkpointer)
        # cannot serialize numpy scalar types that Neo4j / sklearn may return.
        return {
            "graphrag_results": _sanitize([
                {"nodes": r.nodes, "relationships": r.relationships}
                for r in result.search_results
            ]),
            "retrieved_nodes": _sanitize(result.nodes),
            "retrieved_relationships": _sanitize(result.relationships),
            "recommendations": _sanitize(result.recommendations),
            "retrieval_confidence": result.confidence,
            # NEW Phase 1: Curated media from sidecar JSON
            "curated_media": _sanitize(result.curated_media) if result.curated_media else None,
            # NEW Phase A: External resources for out-of-scope queries
            "external_resources": _sanitize(result.external_resources) if result.external_resources else None,
            "current_step": "retrieve_complete"
        }
        
    except Exception as e:
        logger.error(f"[Node: Retrieve] Error: {e}")
        return {
            "error": str(e),
            "current_step": "error"
        }


async def write_node(state: AgentState) -> Dict[str, Any]:
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
        plan_data = state.get("plan", {})
        query_intent = plan_data.get("query_intent", state.get("query_intent", "lesson_creation"))
        
        # Check if this is a revision
        if revision_count > 0 and state.get("lesson_plan_draft"):
            # Revision mode - pass intent for consistent formatting
            lesson_plan = await writer.revise(
                current_draft=state["lesson_plan_draft"],
                critique=state.get("critique", ""),
                revision_instructions=state.get("revision_instructions", ""),
                language=state.get("language", "it"),
                intent=query_intent
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
                pedagogy_concepts=plan_data.get("pedagogy_concepts")
            )
            
            retrieval_result = RetrievalResult(
                nodes=state.get("retrieved_nodes", []),
                relationships=state.get("retrieved_relationships", []),
                recommendations=state.get("recommendations", []),
                confidence=state.get("retrieval_confidence", "MEDIUM")
            )
            
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
                educational_profile=state.get("educational_profile"),
                # CORE 2 #9 — Corrective RAG. None when the feature flag is
                # off, so this is byte-identical to pre-#9 in default mode.
                retrieval_warning=state.get("retrieval_warning"),
                retrieval_grade_reason=state.get("retrieval_grade_reason"),
            )
        
        return {
            "lesson_plan_draft": lesson_plan,
            "current_step": "write_complete"
        }
        
    except Exception as e:
        logger.error(f"[Node: Write] Error: {e}")
        return {
            "error": str(e),
            "current_step": "error"
        }


async def critique_node(state: AgentState) -> Dict[str, Any]:
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
        return {
            "error": "No content generated",
            "current_step": "error",
            "approved": False
        }
    
    # Get query intent for appropriate evaluation
    plan_data = state.get("plan", {})
    query_intent = plan_data.get("query_intent", state.get("query_intent", "lesson_creation"))
    
    logger.info(f"[Node: Critique] Reviewing content (intent: {query_intent})...")
    
    critic = get_critic()
    
    try:
        from aix.agent.agents.retriever_agent import RetrievalResult
        
        retrieval_result = RetrievalResult(
            nodes=state.get("retrieved_nodes", []),
            relationships=state.get("retrieved_relationships", []),
            recommendations=state.get("recommendations", []),
            confidence=state.get("retrieval_confidence", "MEDIUM")
        )
        
        result = await critic.critique(
            lesson_plan=state["lesson_plan_draft"],
            teacher_query=state["teacher_query"],
            retrieval_result=retrieval_result,
            revision_count=state.get("revision_count", 0),
            max_revisions=state.get("max_revisions", 2),
            domain=state.get("domain", "neuro"),
            language=state.get("language", "it"),
            query_intent=query_intent
        )
        
        updates = {
            "critique": result.summary,
            "critique_score": result.average_score,
            "approved": result.approved,
            "current_step": "critique_complete"
        }
        
        if result.approved:
            updates["final_lesson_plan"] = state["lesson_plan_draft"]
            updates["final_metadata"] = {
                "scores": result.scores,
                "strengths": result.strengths,
                "revision_count": state.get("revision_count", 0)
            }
        else:
            updates["revision_instructions"] = result.revision_instructions
            updates["revision_count"] = state.get("revision_count", 0) + 1
        
        return updates
        
    except Exception as e:
        logger.error(f"[Node: Critique] Error: {e}")
        return {
            "error": str(e),
            "current_step": "error"
        }


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
    max_revisions = state.get("max_revisions", 2)
    
    if revision_count >= max_revisions:
        logger.info("[Router] Max revisions reached, finishing...")
        return "finish"
    
    logger.info(f"[Router] Revision requested ({revision_count}/{max_revisions})")
    return "revise"


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

async def grade_retrieval_node(state: AgentState) -> Dict[str, Any]:
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
        attempts, max_attempts,
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
            "defaulting to grade=relevant to preserve pre-#9 behaviour.", e,
        )
        return {
            "retrieval_grade": "relevant",
            "retrieval_grade_reason": f"Grader exception: {e.__class__.__name__}",
            "retrieval_attempts": attempts,
            "current_step": "grade_complete",
        }

    updates: Dict[str, Any] = {
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
                rewritten, attempts, attempts + 1,
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
            grader_result.grade, max_attempts,
        )
    else:
        # Explicit None on the relevant path so the writer doesn't see a
        # stale warning from a checkpoint of an earlier turn.
        updates["retrieval_warning"] = False if grader_result.grade == "relevant" else (
            updates.get("retrieval_warning")
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
            "(retrieval_warning=True)", grade, max_attempts,
        )
        return "continue"

    logger.info(
        "[Router] retrieval grade=%s (attempts=%d/%d) → retry retrieve",
        grade, attempts, max_attempts,
    )
    return "retry"

