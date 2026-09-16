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

import asyncio
import logging
import os
from typing import Any, Optional

from langgraph.graph import END, StateGraph
from langgraph.types import RetryPolicy

from aix.agent.graph.nodes import (
    TRANSIENT_LLM_ERRORS,
    _conditional_retrieval_enabled,
    _corrective_rag_enabled,
    _private_rag_enabled,
    critique_node,
    # CORE 2 #9 — Corrective RAG (Retrieval Grading). Imported eagerly so
    # an import error fails fast even when the feature flag is off; the
    # nodes are only added to the topology when the flag is on.
    grade_retrieval_node,
    plan_node,
    retrieve_node,
    retrieve_private_node,
    reuse_retrieval_node,
    should_continue_to_revision,
    should_retrieve,
    should_retry_after_validation,
    should_retry_retrieval,
    validate_node,
    write_node,
)
from aix.agent.graph.state import AgentInput, AgentState, create_initial_state

logger = logging.getLogger(__name__)


# Outer half of the LLM retry budget; the inner half lives on the OpenAI client
# (see ``OpenAIConfig``). Two attempts here on top of two there bounds a node at
# four calls — where stacking the SDK's default three under a three-attempt
# policy would have allowed nine, each able to run to the read timeout.
#
# It earns a place above the client's retry because it re-enters the node: the
# prompt is rebuilt and the response re-parsed, rather than the same HTTP
# request being replayed.
#
# ``retry_on`` is the same tuple the nodes re-raise, so the two cannot disagree
# about what counts as transient. ``grade_retrieval`` is deliberately excluded:
# it is an optional quality gate that already degrades to ``grade=relevant``
# rather than failing a run, and retrying it would spend latency on a node
# designed never to matter that much.
LLM_RETRY_POLICY = RetryPolicy(
    retry_on=TRANSIENT_LLM_ERRORS,
    max_attempts=2,
    initial_interval=1.0,
    backoff_factor=2.0,
    max_interval=10.0,
    jitter=True,
)


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
    # ``input_schema`` is what makes a turn's retrieval survive into the next
    # one: LangGraph merges only the AgentInput keys and drops the rest, so the
    # checkpointed retrieval corpus is no longer overwritten with nulls on
    # every follow-up. See the partition note in ``state.py``.
    workflow = StateGraph(AgentState, input_schema=AgentInput)

    # Always-on agent nodes (pre-#9 topology)
    workflow.add_node("plan", plan_node, retry_policy=LLM_RETRY_POLICY)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("write", write_node, retry_policy=LLM_RETRY_POLICY)
    # Q1 · 1.3 — deterministic feasibility, between the Writer and the Critic.
    # Always in the topology, unlike the flagged nodes above: the node itself
    # no-ops when AIX_VALIDATOR_ENABLED is off, so the graph has one shape
    # instead of two and the environment cannot change what is wired to what.
    # No retry policy: it makes no network call, so there is nothing to retry.
    workflow.add_node("validate", validate_node)
    workflow.add_node("critique", critique_node, retry_policy=LLM_RETRY_POLICY)

    workflow.set_entry_point("plan")

    if _conditional_retrieval_enabled():
        # T6 — a follow-up that adds no new concepts reuses the corpus the
        # thread already holds instead of searching Neo4j again:
        #
        #     plan ─┬─[new concepts | first turn]→ retrieve ─→ write
        #           └─[nothing new]──────────────→ reuse ────→ write
        #
        # ``reuse`` is a node rather than a bare edge so the decision shows up
        # in the stream like every other stage. Requires the input schema from
        # the previous commit: without it the corpus is nulled before the graph
        # starts, so there is never anything to reuse.
        workflow.add_node("reuse", reuse_retrieval_node)
        workflow.add_conditional_edges(
            "plan",
            should_retrieve,
            {"retrieve": "retrieve", "reuse": "reuse"},
        )
        workflow.add_edge("reuse", "write")
        logger.info(
            "[LessonPlannerGraph] Conditional retrieval ENABLED — "
            "follow-up turns may reuse the thread's corpus."
        )
    else:
        # Default: unconditional, byte-identical to the pre-T6 topology.
        workflow.add_edge("plan", "retrieve")

    if _private_rag_enabled():
        # Fase 2 · PR 6 — the second retriever, beside the first:
        #
        #     plan ─┬─→ retrieve ─────────→ write
        #           └─→ retrieve_private ──┘
        #
        # Both branches write ``current_step``, which is what the reducers in
        # ``state.py`` were added for (§8.9). ``write`` waits for both: it is a
        # join, so a lesson is never written from half its evidence.
        workflow.add_node("retrieve_private", retrieve_private_node)
        workflow.add_edge("plan", "retrieve_private")
        workflow.add_edge("retrieve_private", "write")
        logger.info(
            "[LessonPlannerGraph] Private material RAG ENABLED — "
            "the teacher's own documents are retrieved beside the Knowledge Graph."
        )

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

    # write → validate → critique, with a targeted rewrite when the lesson is
    # not feasible:
    #
    #     write ──→ validate ─┬─[feasible | budget spent]→ critique
    #       ↑                 │
    #       └──[blocking]─────┘
    #
    # The rewrite carries only the remediation, through the existing
    # ``refinement_instruction`` channel, and spends the validator's own budget
    # rather than the Critic's.
    workflow.add_edge("write", "validate")
    workflow.add_conditional_edges(
        "validate",
        should_retry_after_validation,
        {"rewrite": "write", "critique": "critique"},
    )

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


# ---------------------------------------------------------------------------
# Process-wide compiled-graph singleton
# ---------------------------------------------------------------------------
#
# The graph depends only on the topology and the checkpointer — never on
# ``domain`` / ``language`` / ``max_revisions``, which travel in AgentState.
# So one compiled instance serves every request in the process.
#
# Before this existed, ``LessonPlannerPipeline`` cached on ``self._graph`` while
# callers built a fresh ``AgentOrchestrator`` (and therefore a fresh Pipeline)
# per run: the cache was always empty and ``compile()`` ran for every lesson.
#
# Invalidation is by saver identity rather than an explicit reset hook. The
# compiled graph closes over the checkpointer it was built with, so if
# ``close_checkpointer()`` tears that saver down and a later call initialises a
# new one, a cached graph would be holding a dead saver. Comparing identity on
# each call makes that self-healing without the checkpointer module needing to
# know graphs exist.

_COMPILED_GRAPH: Optional[Any] = None
_COMPILED_WITH_SAVER: Optional[Any] = None
_GRAPH_LOCK: asyncio.Lock = asyncio.Lock()


async def get_compiled_graph() -> Any:
    """Return the process-wide compiled graph, building it on first use.

    Concurrent first-callers are serialised on ``_GRAPH_LOCK`` and the winner's
    instance is shared, mirroring ``checkpointer.get_checkpointer()``.

    Recompiles only when the live checkpointer is not the one the cached graph
    was compiled against — normally never, and in tests whenever
    ``close_checkpointer()`` has run in between.
    """
    global _COMPILED_GRAPH, _COMPILED_WITH_SAVER

    from aix.agent.graph.checkpointer import get_checkpointer

    saver = await get_checkpointer()

    # Fast path: `is` rather than `==` — we care whether it is the same live
    # object, not whether two savers look alike.
    if _COMPILED_GRAPH is not None and _COMPILED_WITH_SAVER is saver:
        return _COMPILED_GRAPH

    async with _GRAPH_LOCK:
        # Re-check inside the lock; another coroutine may have won the race.
        if _COMPILED_GRAPH is not None and _COMPILED_WITH_SAVER is saver:
            return _COMPILED_GRAPH

        workflow = _build_workflow()
        if saver is not None:
            logger.info(
                "[LessonPlannerGraph] Compiling process graph with %s checkpointer...",
                type(saver).__name__,
            )
            compiled = workflow.compile(checkpointer=saver)
        else:
            logger.info(
                "[LessonPlannerGraph] Compiling process graph without checkpointer (degraded)..."
            )
            compiled = workflow.compile()

        _COMPILED_GRAPH = compiled
        _COMPILED_WITH_SAVER = saver
        logger.info("[LessonPlannerGraph] Graph compiled successfully (once per process)")
        return compiled


def reset_compiled_graph() -> None:
    """Drop the cached graph so the next :func:`get_compiled_graph` rebuilds.

    Needed by tests that change the topology through
    ``AIX_CORRECTIVE_RAG_ENABLED``: the flag is read at build time, so a graph
    cached under the old value would otherwise survive the change.
    """
    global _COMPILED_GRAPH, _COMPILED_WITH_SAVER
    _COMPILED_GRAPH = None
    _COMPILED_WITH_SAVER = None


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

    async def get_graph(self) -> Any:
        """Return the compiled graph with the checkpointer attached.

        Public because streaming callers need the graph itself rather than
        :meth:`run`, which is atomic: the WebUI and the public API both drive
        ``astream`` so they can emit per-node events. They used to reach through
        ``_get_graph_async`` for it.

        Delegates to the process-wide singleton, so the per-instance cache this
        used to keep is gone — it never helped anyway, because callers build a
        fresh Pipeline per run.
        """
        return await get_compiled_graph()

    async def _get_graph_async(self) -> Any:
        """Deprecated alias for :meth:`get_graph`, kept for existing callers."""
        return await self.get_graph()

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
        graph = await self.get_graph()

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
