"""
LangGraph Node Implementations

Each node wraps an agent and handles state updates.
Nodes are the building blocks of the LangGraph state machine.
"""

import logging
from typing import Dict, Any

from aix.agent.graph.state import AgentState
from aix.agent.agents.planner_agent import PlannerAgent
from aix.agent.agents.retriever_agent import RetrieverAgent
from aix.agent.agents.writer_agent import WriterAgent
from aix.agent.agents.critic_agent import CriticAgent

logger = logging.getLogger(__name__)

# Initialize agents (lazy loading)
_planner: PlannerAgent = None
_retriever: RetrieverAgent = None
_writer: WriterAgent = None
_critic: CriticAgent = None


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
        
        return {
            "graphrag_results": [
                {"nodes": r.nodes, "relationships": r.relationships}
                for r in result.search_results
            ],
            "retrieved_nodes": result.nodes,
            "retrieved_relationships": result.relationships,
            "recommendations": result.recommendations,
            "retrieval_confidence": result.confidence,
            # NEW Phase 1: Curated media from sidecar JSON
            "curated_media": result.curated_media if result.curated_media else None,
            # NEW Phase A: External resources for out-of-scope queries
            "external_resources": result.external_resources if result.external_resources else None,
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

