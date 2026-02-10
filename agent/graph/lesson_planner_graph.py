"""
Lesson Planner Graph

LangGraph state machine that orchestrates the multi-agent pipeline.
This is the main graph that connects all agents together.
"""

import logging
from typing import Optional

from langgraph.graph import StateGraph, END

from agent.graph.state import AgentState, create_initial_state
from agent.graph.nodes import (
    plan_node,
    retrieve_node,
    write_node,
    critique_node,
    should_continue_to_revision
)

logger = logging.getLogger(__name__)


def build_lesson_planner_graph() -> StateGraph:
    """
    Build the lesson planner state machine.
    
    Pipeline:
        START → Plan → Retrieve → Write → Critique → [Revise/END]
                                           ↑    ↓
                                           └────┘ (revision loop)
    
    Returns:
        Compiled LangGraph StateGraph
    """
    logger.info("[LessonPlannerGraph] Building graph...")
    
    # Create the graph with AgentState
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("plan", plan_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("write", write_node)
    workflow.add_node("critique", critique_node)
    
    # Set entry point
    workflow.set_entry_point("plan")
    
    # Add edges (sequential flow)
    workflow.add_edge("plan", "retrieve")
    workflow.add_edge("retrieve", "write")
    workflow.add_edge("write", "critique")
    
    # Add conditional edge for revision loop
    workflow.add_conditional_edges(
        "critique",
        should_continue_to_revision,
        {
            "revise": "write",    # Go back to writer
            "finish": END,        # End the pipeline
            "error": END          # End on error
        }
    )
    
    # Compile the graph
    compiled = workflow.compile()
    
    logger.info("[LessonPlannerGraph] Graph compiled successfully")
    
    return compiled


class LessonPlannerPipeline:
    """
    High-level interface for the lesson planner pipeline.
    
    Usage:
        pipeline = LessonPlannerPipeline(domain="neuro")
        result = await pipeline.run("Crea una lezione sulla motivazione")
    """
    
    def __init__(
        self,
        domain: str = "neuro",
        language: str = "it",
        max_revisions: int = 2
    ):
        """
        Initialize the lesson planner pipeline.
        
        Args:
            domain: Knowledge domain ("neuro" or "udl")
            language: Output language ("it" or "en")
            max_revisions: Maximum revision cycles
        """
        self.domain = domain
        self.language = language
        self.max_revisions = max_revisions
        self._graph = None
    
    def _get_graph(self) -> StateGraph:
        """Lazy initialization of the graph"""
        if self._graph is None:
            self._graph = build_lesson_planner_graph()
        return self._graph
    
    async def run(
        self,
        query: str,
        session_id: Optional[str] = None
    ) -> dict:
        """
        Run the lesson planner pipeline.
        
        Args:
            query: Teacher's natural language query
            session_id: Optional session ID for persistence
            
        Returns:
            Dictionary with lesson plan and metadata
        """
        logger.info(f"[Pipeline] Starting for query: {query[:50]}...")
        
        # Create initial state
        initial_state = create_initial_state(
            query=query,
            domain=self.domain,
            language=self.language,
            session_id=session_id,
            max_revisions=self.max_revisions
        )
        
        # Get compiled graph
        graph = self._get_graph()
        
        # Run the pipeline
        try:
            final_state = await graph.ainvoke(initial_state)
            
            # Extract results
            result = {
                "success": not bool(final_state.get("error")),
                "lesson_plan": final_state.get("final_lesson_plan") or final_state.get("lesson_plan_draft"),
                "approved": final_state.get("approved", False),
                "revision_count": final_state.get("revision_count", 0),
                "critique": final_state.get("critique"),
                "scores": final_state.get("final_metadata", {}).get("scores"),
                "sources": {
                    "nodes_count": len(final_state.get("retrieved_nodes", [])),
                    "recommendations_count": len(final_state.get("recommendations", []))
                },
                "error": final_state.get("error"),
                # Phase 3: Add query_intent and key_concepts for upsell buttons
                "query_intent": final_state.get("query_intent", "lesson_creation"),
                "key_concepts": final_state.get("key_concepts", []),
                # Phase 3 (Media): Pass curated media for enhancement buttons
                "curated_media": final_state.get("curated_media"),
                # Phase A: Scope detection for hybrid mode
                "scope_status": final_state.get("scope_status", "in_scope"),
                "scope_confidence": final_state.get("scope_confidence", 1.0),
                "external_resources": final_state.get("external_resources")
            }
            
            logger.info(
                f"[Pipeline] Complete. "
                f"Approved: {result['approved']}, "
                f"Revisions: {result['revision_count']}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"[Pipeline] Failed: {e}")
            return {
                "success": False,
                "lesson_plan": None,
                "error": str(e)
            }
    
    def run_sync(self, query: str, session_id: Optional[str] = None) -> dict:
        """Synchronous version of run()"""
        import asyncio
        return asyncio.run(self.run(query, session_id))

