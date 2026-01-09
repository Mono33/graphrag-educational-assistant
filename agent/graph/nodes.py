"""
LangGraph Node Implementations

Each node wraps an agent and handles state updates.
Nodes are the building blocks of the LangGraph state machine.
"""

import logging
from typing import Dict, Any

from agent.graph.state import AgentState
from agent.agents.planner_agent import PlannerAgent
from agent.agents.retriever_agent import RetrieverAgent
from agent.agents.writer_agent import WriterAgent
from agent.agents.critic_agent import CriticAgent

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
    Output: plan, query_intent, lesson_type, key_concepts, search_queries
    """
    logger.info("[Node: Plan] Starting planning phase...")
    
    planner = get_planner()
    
    try:
        plan = await planner.plan(
            query=state["teacher_query"],
            domain=state.get("domain", "neuro"),
            language=state.get("language", "it")
        )
        
        logger.info(f"[Node: Plan] Detected intent: {plan.query_intent} (confidence: {plan.intent_confidence})")
        
        return {
            "plan": {
                "query_intent": plan.query_intent,
                "intent_confidence": plan.intent_confidence,
                "lesson_type": plan.lesson_type,
                "key_concepts": plan.key_concepts,
                "search_queries": plan.search_queries,
                "target_grade": plan.target_grade,
                "special_needs": plan.special_needs,
                "time_constraints": plan.time_constraints,
                "reasoning": plan.reasoning
            },
            "query_intent": plan.query_intent,
            "lesson_type": plan.lesson_type,
            "target_grade": plan.target_grade,
            "key_concepts": plan.key_concepts,
            "search_queries": plan.search_queries,
            "current_step": "plan_complete"
        }
        
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
        from agent.agents.planner_agent import RetrievalPlan
        
        plan_data = state.get("plan", {})
        plan = RetrievalPlan(
            query_intent=plan_data.get("query_intent", "lesson_creation"),
            key_concepts=plan_data.get("key_concepts", []),
            search_queries=plan_data.get("search_queries", [state["teacher_query"]]),
            lesson_type=plan_data.get("lesson_type"),
            target_grade=plan_data.get("target_grade"),
            special_needs=plan_data.get("special_needs"),
            time_constraints=plan_data.get("time_constraints"),
            intent_confidence=plan_data.get("intent_confidence", "MEDIUM")
        )
        
        result = await retriever.retrieve(plan)
        
        return {
            "graphrag_results": [
                {"nodes": r.nodes, "relationships": r.relationships}
                for r in result.search_results
            ],
            "retrieved_nodes": result.nodes,
            "retrieved_relationships": result.relationships,
            "recommendations": result.recommendations,
            "retrieval_confidence": result.confidence,
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
            from agent.agents.planner_agent import RetrievalPlan
            from agent.agents.retriever_agent import RetrievalResult
            
            plan = RetrievalPlan(
                query_intent=query_intent,
                key_concepts=plan_data.get("key_concepts", []),
                search_queries=plan_data.get("search_queries", []),
                lesson_type=plan_data.get("lesson_type"),
                target_grade=plan_data.get("target_grade"),
                special_needs=plan_data.get("special_needs"),
                time_constraints=plan_data.get("time_constraints"),
                intent_confidence=plan_data.get("intent_confidence", "MEDIUM")
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
                language=state.get("language", "it")
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
        from agent.agents.retriever_agent import RetrievalResult
        
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

