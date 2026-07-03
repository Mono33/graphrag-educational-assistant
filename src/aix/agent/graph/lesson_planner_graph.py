"""
Lesson Planner Graph

LangGraph state machine that orchestrates the multi-agent pipeline.
This is the main graph that connects all agents together.

Compilation modes
-----------------
``build_lesson_planner_graph()`` (sync) compiles WITHOUT a checkpointer —
preserved for backward compatibility with any caller that doesn't need
multi-turn memory (legacy CLI test harness, smoke scripts, etc.).

``build_lesson_planner_graph_async()`` (async) compiles WITH the
``AsyncSqliteSaver`` singleton when available, falling back to the sync
path on graceful degradation. Anything that wants multi-turn behaviour
(webui SSE, public JSON+SSE API, MCP server) should use this. Callers
must then pass ``config={"configurable": {"thread_id": str(...)}}`` on
every ``ainvoke`` / ``astream`` — see ``aix.agent.graph.checkpointer.thread_config``.

Both paths return the *same* runnable shape; the only difference is the
presence of a checkpointer attached to the compiled graph.
"""

import logging
import os
from typing import Any, Optional

from langgraph.graph import END, StateGraph

from aix.agent.graph.nodes import (
    _corrective_rag_enabled,
    critique_node,
    # CORE 2 #9 — Corrective RAG (Retrieval Grading). Imported eagerly so
    # an import error fails fast even when the feature flag is off; the
    # nodes are only added to the topology when the flag is on.
    grade_retrieval_node,
    plan_node,
    retrieve_node,
    should_continue_to_revision,
    should_retry_retrieval,
    write_node,
)
from aix.agent.graph.state import AgentState, create_initial_state

logger = logging.getLogger(__name__)


def _build_workflow() -> StateGraph:
    """
    Construct the agent ``StateGraph`` (nodes + edges) without compiling.

    Factored out so the sync and async ``build_*`` entry points can share
    the topology and only differ in the ``compile(checkpointer=...)`` call.

    Topologies
    ----------
    * **Default** (``AIX_CORRECTIVE_RAG_ENABLED`` unset/false) — byte-
      identical to pre-#9::

          plan → retrieve → write → critique → [revise|finish|error]

    * **Corrective-RAG mode** (``AIX_CORRECTIVE_RAG_ENABLED=true`` —
      CORE 2 #9)::

          plan → retrieve → grade_retrieval ─[continue]→ write → critique → [revise|finish|error]
                                   │
                                   └─[retry]→ retrieve

      The retry edge re-enters the retriever with the grader's rewritten
      query (when one was emitted), bounded by
      ``AIX_CORRECTIVE_RAG_MAX_ATTEMPTS`` (default 2). After max attempts
      the loop unconditionally falls through to the writer with
      ``retrieval_warning=True`` so the lesson carries a low-confidence
      caveat instead of pretending nothing happened.
    """
    workflow = StateGraph(AgentState)

    # Always-on agent nodes (pre-#9 topology)
    workflow.add_node("plan", plan_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("write", write_node)
    workflow.add_node("critique", critique_node)

    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "retrieve")

    if _corrective_rag_enabled():
        # CORE 2 #9 — Corrective RAG topology
        workflow.add_node("grade_retrieval", grade_retrieval_node)
        workflow.add_edge("retrieve", "grade_retrieval")
        workflow.add_conditional_edges(
            "grade_retrieval",
            should_retry_retrieval,
            {
                "retry": "retrieve",
                "continue": "write",
                "error": END,
            },
        )
        logger.info(
            "[LessonPlannerGraph] Corrective RAG (#9) ENABLED — "
            "topology includes grade_retrieval node with retry loop."
        )
    else:
        # Pre-#9 default topology — direct retrieve → write edge.
        workflow.add_edge("retrieve", "write")
        logger.debug(
            "[LessonPlannerGraph] Corrective RAG disabled (default) — "
            "using direct retrieve → write edge."
        )

    workflow.add_edge("write", "critique")

    # Add conditional edge for revision loop
    workflow.add_conditional_edges(
        "critique",
        should_continue_to_revision,
        {
            "revise": "write",
            "finish": END,
            "error": END,
        },  # Go back to writer  # End the pipeline  # End on error
    )
    return workflow


def build_lesson_planner_graph() -> Any:
    """
    Build the lesson planner state machine *without* a checkpointer.

    Preserved for backward compatibility with sync callers (CLI test
    harness, smoke scripts). Returns a compiled graph that can be
    invoked WITHOUT a ``thread_id`` config — every invocation is a
    fresh, ephemeral run.

    For multi-turn memory, use :func:`build_lesson_planner_graph_async`.

    Pipeline:
        START → Plan → Retrieve → Write → Critique → [Revise/END]
                                           ↑    ↓
                                           └────┘ (revision loop)

    Returns:
        Compiled LangGraph runnable (no checkpointer attached).
    """
    logger.info("[LessonPlannerGraph] Building graph (no checkpointer)...")
    compiled = _build_workflow().compile()
    logger.info("[LessonPlannerGraph] Graph compiled successfully (no checkpointer)")
    return compiled


async def build_lesson_planner_graph_async() -> Any:
    """
    Build the lesson planner state machine *with* a checkpointer when
    available (CORE 2 #10.2).

    Looks up the process-singleton checkpointer (``AsyncPostgresSaver`` when
    ``LANGGRAPH_DATABASE_URL`` is a Postgres URL, else ``AsyncSqliteSaver``,
    or ``None`` if the relevant backend package is missing), and compiles the
    graph with it attached. Falls back to the no-checkpointer path on
    graceful degradation.

    Callers MUST pass ``config={"configurable": {"thread_id": str(...)}}``
    on every ``ainvoke`` / ``astream`` when the returned graph has a
    checkpointer attached, otherwise LangGraph raises ``ValueError``.
    Use :func:`aix.agent.graph.checkpointer.thread_config` as the
    canonical builder for that dict.

    Returns:
        Compiled LangGraph runnable (checkpointer attached when possible).
    """
    # Local import — avoids a hard module-level dependency when the
    # checkpointer module is unavailable in degraded environments.
    from aix.agent.graph.checkpointer import get_checkpointer

    saver = await get_checkpointer()
    workflow = _build_workflow()

    if saver is not None:
        # Report the ACTUAL backend (AsyncPostgresSaver / AsyncSqliteSaver),
        # not a hardcoded name — the saver is chosen at runtime from the env.
        logger.info(
            "[LessonPlannerGraph] Building graph with %s checkpointer...",
            type(saver).__name__,
        )
        compiled = workflow.compile(checkpointer=saver)
        logger.info("[LessonPlannerGraph] Graph compiled successfully (multi-turn memory: ENABLED)")
    else:
        logger.info("[LessonPlannerGraph] Building graph without checkpointer (degraded)...")
        compiled = workflow.compile()
        logger.info(
            "[LessonPlannerGraph] Graph compiled successfully (multi-turn memory: DISABLED)"
        )
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
        max_revisions: int = int(os.getenv("AIX_MAX_REVISIONS", "1")),
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
        self._graph: Optional[Any] = None

    def _get_graph(self) -> Any:
        """
        Lazy initialisation of the compiled graph (sync, no checkpointer).

        Kept for backward compatibility with legacy callers that don't
        need multi-turn memory. The webui SSE service and the public
        JSON+SSE API both use ``_get_graph_async()`` instead so they
        benefit from CORE 2 #10's multi-turn checkpointer.
        """
        if self._graph is None:
            self._graph = build_lesson_planner_graph()
        return self._graph

    async def _get_graph_async(self) -> Any:
        """
        Lazy initialisation of the compiled graph WITH the LangGraph
        checkpointer attached when available (CORE 2 #10.2).

        Caches on ``self._graph`` like the sync sibling — first caller
        wins; subsequent callers receive the same compiled instance.
        Calling this on a pipeline whose ``self._graph`` was already
        populated by ``_get_graph()`` will re-use the no-checkpointer
        graph; in practice this only happens in tests.
        """
        if self._graph is None:
            self._graph = await build_lesson_planner_graph_async()
        return self._graph

    async def run(
        self,
        query: str,
        session_id: Optional[str] = None,
        educational_profile: Optional[dict[str, Any]] = None,
    ) -> dict:
        """
        Run the lesson planner pipeline.

        Args:
            query: Teacher's natural language query
            session_id: Optional session ID for persistence
            educational_profile: Optional class/classroom context (CORE 1 #2.5).
                Passed through `AgentState` to prompts and ranking. Pass either
                a `dict` (already serialized) or a Pydantic
                `EducationalProfile` (it will be normalized to dict here).

        Returns:
            Dictionary with lesson plan and metadata
        """
        logger.info(f"[Pipeline] Starting for query: {query[:50]}...")

        # Normalize Pydantic models to dict so the LangGraph state stays
        # JSON-serializable across nodes / checkpoints.
        profile_dict: Optional[dict[str, Any]] = None
        if educational_profile is not None:
            if hasattr(educational_profile, "model_dump"):
                profile_dict = educational_profile.model_dump(exclude_none=True)
            elif hasattr(educational_profile, "dict"):
                profile_dict = educational_profile.dict(exclude_none=True)  # type: ignore[attr-defined]
            else:
                profile_dict = educational_profile

        # Create initial state
        initial_state = create_initial_state(
            query=query,
            domain=self.domain,
            language=self.language,
            session_id=session_id,
            max_revisions=self.max_revisions,
            educational_profile=profile_dict,
        )

        # Get compiled graph (with checkpointer when available — CORE 2 #10.2).
        # The async path is the canonical one; the sync ``_get_graph()`` is
        # kept as a backward-compat seam for legacy callers only.
        graph = await self._get_graph_async()

        # When a checkpointer is attached we MUST pass a thread_id config,
        # otherwise LangGraph raises. Generate a per-invocation UUID when
        # the caller didn't supply ``session_id`` so ephemeral runs still
        # work (no cross-thread state — each run is its own thread).
        import uuid as _uuid

        from aix.agent.graph.checkpointer import thread_config

        effective_thread_id = session_id or f"ephemeral-{_uuid.uuid4()}"
        run_config = thread_config(effective_thread_id)

        # Run the pipeline
        try:
            final_state = await graph.ainvoke(initial_state, config=run_config)

            # Extract results
            result = {
                "success": not bool(final_state.get("error")),
                "lesson_plan": final_state.get("final_lesson_plan")
                or final_state.get("lesson_plan_draft"),
                "approved": final_state.get("approved", False),
                "revision_count": final_state.get("revision_count", 0),
                "critique": final_state.get("critique"),
                "scores": final_state.get("final_metadata", {}).get("scores"),
                "sources": {
                    "nodes_count": len(final_state.get("retrieved_nodes", [])),
                    "recommendations_count": len(final_state.get("recommendations", [])),
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
                "external_resources": final_state.get("external_resources"),
            }

            logger.info(
                f"[Pipeline] Complete. "
                f"Approved: {result['approved']}, "
                f"Revisions: {result['revision_count']}"
            )

            return result

        except Exception as e:
            logger.error(f"[Pipeline] Failed: {e}")
            return {"success": False, "lesson_plan": None, "error": str(e)}

    def run_sync(self, query: str, session_id: Optional[str] = None) -> dict:
        """Synchronous version of run()"""
        import asyncio

        return asyncio.run(self.run(query, session_id))
