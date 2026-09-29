"""
Agent State Definition

Defines the shared state that flows through the multi-agent pipeline.
Each agent can read and update this state.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Optional, TypedDict

# ---------------------------------------------------------------------------
# Reducers
# ---------------------------------------------------------------------------
#
# Every node returns ``current_step``, and every error path returns ``error``.
# Two nodes in the same superstep therefore both write these keys — which
# LangGraph rejects with ``InvalidUpdateError: can receive only one value per
# step`` unless the key declares a reducer. The corrective-RAG retry loop is
# sequential today, but the moment a second retriever runs alongside the
# Knowledge Graph one (the PrivateMaterialRetriever), these are the two keys
# that would fail first.
#
# Both reducers are order-independent, per LINEE-GUIDA-langgraph.md §1: the
# runtime does not apply concurrent updates in declaration order, so a
# last-write-wins reducer would silently drop data in an unpredictable way.

_STEP_RANK: dict[str, int] = {
    "start": 0,
    "plan_complete": 1,
    "retrieve_complete": 2,
    "grade_complete": 3,
    "write_complete": 4,
    "critique_complete": 5,
    "error": 99,
}


def merge_current_step(current: Optional[str], update: Optional[str]) -> Optional[str]:
    """Keep the furthest stage the pipeline has reached.

    ``"start"`` is the one value that moves backwards: it marks a new run, and
    only the per-run input sets it, so it always wins. Otherwise the update
    with the higher rank wins, and equal ranks fall back to the greater name —
    making the result identical whichever order the runtime applies the two
    updates in.

    Note this is a *progress marker only*: nothing in the codebase reads it, so
    the revision loop no longer walking it back from ``critique_complete`` to
    ``write_complete`` changes no behaviour.
    """
    if update == "start":
        return "start"
    if update is None:
        return current
    if current is None:
        return update
    return max(
        (_STEP_RANK.get(current, 0), current),
        (_STEP_RANK.get(update, 0), update),
    )[1]


def merge_error(current: Optional[str], update: Optional[str]) -> Optional[str]:
    """Preserve every failure, in an order-independent way.

    ``None`` clears — that is the per-run reset coming from the input schema,
    and it is never concurrent with a node's write because the input is applied
    before any node runs. Nodes only ever write a non-``None`` error.

    Two branches failing in the same superstep are joined in sorted order, so
    neither the surviving message nor its ordering depends on which coroutine
    happened to finish first. Downstream readers only test truthiness.
    """
    if update is None:
        return None
    if current is None or current == update:
        return update
    return " | ".join(sorted({current, update}))


def merge_material_items(
    current: Optional[list[dict[str, Any]]], update: Optional[list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Join the private retriever's passages, in an order-independent way (§8.9).

    Concatenation rather than last-write-wins: the private retriever and the
    Knowledge Graph one run in the same superstep, and a reducer that kept only
    one write would drop a whole branch without raising anything.

    The order comes from the payload — the score — and not from which coroutine
    finished first, so the prompt is the same whichever branch lands first.
    Duplicates are dropped by *text*: the same passage retrieved twice is one
    passage, and it should not get two shares of the context budget.

    Not by citation, which is what this did until 2026-09-21. A citation
    anchors on the page and the first block of the passage, and a PDF page is
    usually one block — so every chunk of page 6 carries the same citation. In
    the first real run that silently threw away two of the eight passages the
    retriever had found, and on a page-heavy document it would have thrown away
    five. The citation identifies *where to look*; only the text identifies
    *what was said*.
    """
    if update is None:
        return list(current or [])

    merged = list(current or []) + list(update)
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in merged:
        key = str(item.get("text") or item.get("citation") or "").strip()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    return sorted(unique, key=lambda i: float(i.get("score") or 0.0), reverse=True)


class LessonPlanType(Enum):
    """Types of lesson plans that can be generated"""

    FULL_LESSON = "full_lesson"
    ACTIVITY = "activity"
    ASSESSMENT = "assessment"
    UNIT_PLAN = "unit_plan"


class QueryIntent(Enum):
    """Types of queries the Agent can handle"""

    LESSON_CREATION = "lesson_creation"  # Full lesson plan request
    ACTIVITY_DESIGN = "activity_design"  # Single activity request
    DEFINITION = "definition"  # "What is X?" questions
    COMPARISON = "comparison"  # "Compare X and Y" questions
    EXPLANATION = "explanation"  # "How does X work?" questions
    RECOMMENDATION = "recommendation"  # "What strategies for...?" questions
    LIST = "list"  # "List the types of..." questions


class ScopeStatus(Enum):
    """Scope status for query topics relative to Knowledge Graph"""

    IN_SCOPE = "in_scope"  # Topic found in KG (e.g., metacognition, attention)
    PARTIAL_SCOPE = "partial_scope"  # Pedagogy in KG, but subject topic outside
    OUT_OF_SCOPE = "out_of_scope"  # Topic completely outside KG domain
    UNKNOWN = "unknown"  # Could not determine scope


class AgentState(TypedDict, total=False):
    """
    Shared state passed between all agents in the pipeline.

    This TypedDict defines the complete state that flows through:
    Planner → Retriever → Writer → Critic → (Output or back to Writer)
    """

    # ========================================
    # INPUT (set at start)
    # ========================================
    teacher_query: str  # Original query from teacher
    domain: str  # Knowledge domain ("neuro" or "udl")
    language: str  # Response language ("it" or "en")
    session_id: Optional[str]  # For conversation persistence
    educational_profile: Optional[
        dict[str, Any]
    ]  # CORE 1 #2.5 — per-request class/classroom context (group, classroom, time, subject)
    pedagogical_intent: Optional[str]  # Teacher's lesson goal: "{code}" or "{code}: {detail}"
    refinement_instruction: Optional[str]  # SAM refinement: instruction from guided panel on re-run
    # WebUI #6.6 P3 — joined text from teacher file uploads. Used by the Writer
    # only as additional context; never sent to the GraphRAG / KG retriever.
    teacher_provided_context: Optional[str]
    # CORE 2 #10.3 — multi-turn conversation history for follow-up turns.
    # List of {"role": "user"|"assistant", "content": <markdown>} dicts in
    # chronological order, EXCLUDING the current turn's user message
    # (which is already in `teacher_query`). Empty/None on the first turn —
    # writer prompt renders identically to the pre-#10 single-turn behaviour.
    # On follow-up turns, the writer is shown the prior exchange so it
    # can adapt its output (e.g., "now adapt for ADHD"). Populated by the
    # service layer from the persisted `lesson_message` table.
    # Will be summary-buffered in #10.4 once threads exceed N tokens.
    conversation_history: Optional[list[dict[str, str]]]
    # CORE 2 #10.4 — summary of the oldest turns when the window is
    # exceeded. None on short threads. Populated by the service layer
    # before invoking the graph. Writer prompt prepends this before
    # `conversation_history` so the agent has continuity even when older
    # turns are summarised.
    conversation_summary: Optional[str]
    # CORE 2 #12b.3 — the *current turn's* raw user text, BEFORE the service
    # layer prepended the conversation history / summary. ``teacher_query``
    # carries the augmented blob (so every agent sees prior context); this
    # field carries just what the teacher actually typed in this turn so
    # nodes that need to honour user-vs-history precedence can do so. None
    # on legacy callers that don't populate it — nodes must fall back to
    # ``teacher_query`` in that case to stay backward-compatible.
    raw_user_turn: Optional[str]
    # Teacher's explicit override of the retrieval decision (T6). ``True``
    # forces a fresh Knowledge Graph search, ``False`` forces reuse of what the
    # thread already holds, ``None`` leaves the choice to the planner. Kept as a
    # state field rather than an env knob so a UI control can set it per turn
    # once the interaction is designed — the graph needs no further change.
    force_retrieval: Optional[bool]

    # ========================================
    # PLANNER OUTPUT
    # ========================================
    plan: Optional[dict[str, Any]]  # Retrieval plan created by planner
    query_intent: Optional[str]  # Detected intent (lesson_creation, definition, etc.)
    lesson_type: Optional[str]  # Type of lesson to generate (if lesson_creation)
    target_grade: Optional[str]  # Target grade level (if detected)
    key_concepts: Optional[list[str]]  # Key concepts to search for
    search_queries: Optional[list[str]]  # Queries to run on GraphRAG
    # T6 — the planner's own answer to "does this turn need a new search?".
    # ``None`` on a first turn, and whenever the follow-up context was not put
    # in front of the planner at all (flag off, or no history): that is the
    # signal that tells the router to fall back to comparing concepts.
    requires_new_retrieval: Optional[bool]

    # NEW: Scope Detection (Phase A - Out-of-domain handling)
    scope_status: Optional[str]  # "in_scope", "partial_scope", "out_of_scope"
    scope_confidence: Optional[float]  # 0.0-1.0 confidence in scope detection
    subject_concepts: Optional[list[str]]  # Subject-specific concepts (may be out of scope)
    pedagogy_concepts: Optional[list[str]]  # Pedagogical concepts (always from KG)

    # ========================================
    # RETRIEVER OUTPUT
    # ========================================
    graphrag_results: Optional[list[dict[str, Any]]]  # Results from GraphRAG searches
    retrieved_nodes: Optional[list[dict[str, Any]]]  # All retrieved nodes
    retrieved_relationships: Optional[list[dict[str, Any]]]  # All relationships
    recommendations: Optional[list[dict[str, Any]]]  # Educational recommendations
    retrieval_confidence: Optional[str]  # Confidence from GraphRAG
    # NEW Phase 1: Curated media from sidecar JSON (optional, backward compatible)
    curated_media: Optional[dict[str, Any]]  # Videos, resources, citations

    # NEW Phase A: External resources for out-of-scope queries
    external_resources: Optional[dict[str, Any]]  # Wikipedia, OER, Semantic Scholar results
    #: True when the search came back empty *and* the graph database did not
    #: answer. Distinguishes «your question is outside the Knowledge Graph»
    #: from «the Knowledge Graph was down», which look identical otherwise.
    kg_unreachable: Optional[bool]

    #: Which registered set of the teacher's documents this turn may search.
    #:
    #: Separate from ``session_id`` on purpose. ``session_id`` is LangGraph's
    #: thread id *and* it is chosen by the caller on the API, so using it to
    #: key private material means two callers who pick the same string share
    #: each other's documents. The WebUI has no such exposure — it uses the
    #: lesson id — but the API must hand out a value the request cannot set.
    #:
    #: ``retrieve_private_node`` prefers this and falls back to ``session_id``,
    #: so a caller that registers the old way keeps working.
    material_scope_id: Optional[str]

    # Fase 2 · PR 6 — passages of the teacher's own documents (§8.2). Kept apart
    # from the Knowledge Graph results all the way to the prompt: one is shared
    # pedagogy, the other is one school's handout, and the Writer has to be able
    # to tell them apart. Every item is ``instruction_trust="data_only"``.
    material_context_items: Annotated[list[dict[str, Any]], merge_material_items]
    # The maps of those documents (``aix.webui.materials.document_map``): what
    # each one holds, part by part. Orientation for the Writer, never cited.
    # Written only by ``retrieve_private_node``, so it needs no reducer.
    material_maps: list[dict[str, Any]]
    # T6 — True when this turn skipped the retriever and reused the corpus a
    # previous turn fetched. Per-run, so it resets. Downstream readers use it to
    # say so in the UI; the Critic will need it too, to know it is grounding
    # against evidence it did not just see fetched.
    retrieval_reused: Optional[bool]

    # ========================================
    # CORE 2 #9 — Corrective RAG (Retrieval Grading)
    # ========================================
    # All fields below are populated ONLY when AIX_CORRECTIVE_RAG_ENABLED=true
    # AND the grade_retrieval_node is in the compiled graph topology. When the
    # feature flag is off (default), these stay None and the rest of the
    # pipeline behaves byte-identically to pre-#9. Writer / Critic read them
    # additively — a None value is interpreted as "no grading was performed".
    retrieval_grade: Optional[str]  # "relevant" | "ambiguous" | "irrelevant"
    retrieval_grade_reason: Optional[str]  # 1-line LLM rationale (for traces)
    retrieval_attempts: Optional[int]  # Number of retrieval passes done so far (1-N)
    retrieval_rewritten_query: Optional[str]  # Grader's suggested query rewrite, if any
    retrieval_warning: Optional[bool]  # True iff all attempts ended non-relevant

    # ========================================
    # WRITER OUTPUT
    # ========================================
    lesson_plan_draft: Optional[str]  # Generated lesson plan
    lesson_plan_structured: Optional[dict[str, Any]]  # Structured version
    sources_cited: Optional[list[str]]  # Sources used in lesson plan

    # ========================================
    # CRITIC OUTPUT
    # ========================================
    critique: Optional[str]  # Critic's feedback
    critique_score: Optional[float]  # Quality score (0-1)
    approved: bool  # Whether critic approved
    revision_instructions: Optional[str]  # What to fix if not approved

    # ========================================
    # Q1 · 1.3 — VALIDATOR OUTPUT (deterministic, no LLM)
    # ========================================
    # Written by ``validate_node`` between the Writer and the Critic. The Critic
    # judges whether a lesson is good; these say whether it is possible, which is
    # a different question and must not depend on a model's mood.
    validation_violations: Optional[list[dict[str, Any]]]  # Serialised Violations
    validation_passed: Optional[bool]  # False iff something blocking is open
    validation_attempts: Optional[int]  # Targeted rewrites spent, separate from
    # ``revision_count`` so a validator retry never eats the Critic's budget

    # ========================================
    # METADATA
    # ========================================
    revision_count: int  # Number of revision cycles
    max_revisions: int  # Maximum allowed revisions
    # Reducers: every node writes both of these, so they are the two keys that
    # break first when two nodes share a superstep. See the module header.
    current_step: Annotated[str, merge_current_step]  # Furthest stage reached
    error: Annotated[Optional[str], merge_error]  # Error message if any

    # ========================================
    # FINAL OUTPUT
    # ========================================
    final_lesson_plan: Optional[str]  # Approved final lesson plan
    final_metadata: Optional[dict[str, Any]]  # Final metadata


# ---------------------------------------------------------------------------
# Input schema — what a turn supplies, and therefore what a turn resets
# ---------------------------------------------------------------------------
#
# The graph is compiled as ``StateGraph(AgentState, input_schema=AgentInput)``.
# LangGraph merges only the keys declared here into the checkpointed state and
# silently drops the rest, so the partition below decides, for every key, one
# question: does it survive from one turn to the next on the same thread_id?
#
# Before this existed the answer was "nothing survives": ``create_initial_state``
# spelled out all ~46 keys with the outputs set to ``None``, and that dict was
# handed straight to ``ainvoke``. The checkpointer dutifully stored each turn's
# state and the next turn overwrote it with nulls, which is why multi-turn
# worked only through ``conversation_history`` — recomputed by the service layer
# from the database on every turn.
#
# What survives is deliberately narrow: the retrieval corpus, and nothing else.
# That is what lets a follow-up turn reuse the Knowledge Graph results instead
# of paying for a second search. Everything else must reset, and two are load
# bearing: ``revision_count`` would arrive at the next turn already at the
# ceiling and disable the revision loop, and a stale ``error`` would short
# circuit every node before it ran.

PERSISTED_ACROSS_TURNS: frozenset[str] = frozenset(
    {
        "graphrag_results",
        "retrieved_nodes",
        "retrieved_relationships",
        "recommendations",
        "retrieval_confidence",
        "curated_media",
        "external_resources",
    }
)


class AgentInput(TypedDict, total=False):
    """The per-turn slice of :class:`AgentState`.

    Every key is documented on ``AgentState``; this class only records which
    side of the persist/reset line each one falls on. ``AgentInput`` and
    ``PERSISTED_ACROSS_TURNS`` must together cover ``AgentState`` exactly —
    ``test_state_partition.py`` fails if a new key is added to the state
    without being classified here.
    """

    # Supplied by the caller
    teacher_query: str
    domain: str
    language: str
    session_id: Optional[str]
    educational_profile: Optional[dict[str, Any]]
    pedagogical_intent: Optional[str]
    refinement_instruction: Optional[str]
    teacher_provided_context: Optional[str]
    conversation_history: Optional[list[dict[str, str]]]
    conversation_summary: Optional[str]
    raw_user_turn: Optional[str]
    force_retrieval: Optional[bool]

    # Whether the graph database answered is a fact about *this* turn's search.
    # Carried over, a Tuesday outage would still be apologising on Wednesday.
    kg_unreachable: Optional[bool]

    # Per turn, and load-bearing: an ephemeral scope belongs to one request and
    # its index is gone when that request ends. Carried over, turn two would
    # look up a registration that no longer exists — or, worse, one that a
    # different request has since taken.
    material_scope_id: Optional[str]

    # The retrieval-reuse verdict is per-turn: turn 3 must decide for itself,
    # not inherit turn 2's answer.
    retrieval_reused: Optional[bool]

    # Recomputed by the planner every turn — reset so a failed plan cannot
    # leave the previous turn's concepts in place unnoticed.
    plan: Optional[dict[str, Any]]
    query_intent: Optional[str]
    lesson_type: Optional[str]
    target_grade: Optional[str]
    key_concepts: Optional[list[str]]
    search_queries: Optional[list[str]]
    requires_new_retrieval: Optional[bool]
    scope_status: Optional[str]
    scope_confidence: Optional[float]
    subject_concepts: Optional[list[str]]
    pedagogy_concepts: Optional[list[str]]

    # Corrective-RAG grading is per-turn: a retry budget must not carry over.
    retrieval_grade: Optional[str]
    retrieval_grade_reason: Optional[str]
    retrieval_attempts: Optional[int]
    retrieval_rewritten_query: Optional[str]
    retrieval_warning: Optional[bool]

    # Validator output is per-turn: a violation belongs to the draft that
    # caused it, and turn 3 must not inherit turn 2's verdict.
    validation_violations: Optional[list[dict[str, Any]]]
    validation_passed: Optional[bool]
    validation_attempts: Optional[int]

    # Writer / critic output. The draft resets so a new turn starts from the
    # writer rather than the reviser (write_node branches on
    # ``revision_count > 0 and lesson_plan_draft``).
    lesson_plan_draft: Optional[str]
    lesson_plan_structured: Optional[dict[str, Any]]
    sources_cited: Optional[list[str]]
    critique: Optional[str]
    critique_score: Optional[float]
    approved: bool
    revision_instructions: Optional[str]

    # Run metadata. ``revision_count`` reset is load bearing: see above.
    revision_count: int
    max_revisions: int
    current_step: str
    error: Optional[str]
    # Fase 2 · PR 6 — reset each turn rather than persisted: which passages of a
    # document a lesson needs depends on what this turn asks for, and carrying
    # the previous turn's passages would answer the new question with the old
    # evidence.
    material_context_items: list[dict[str, Any]]
    material_maps: list[dict[str, Any]]

    final_lesson_plan: Optional[str]
    final_metadata: Optional[dict[str, Any]]


def create_initial_state(
    query: str,
    domain: str = "neuro",
    language: str = "it",
    session_id: Optional[str] = None,
    max_revisions: int = 2,
    educational_profile: Optional[dict[str, Any]] = None,
    teacher_provided_context: Optional[str] = None,
    conversation_history: Optional[list[dict[str, str]]] = None,
    conversation_summary: Optional[str] = None,
    raw_user_turn: Optional[str] = None,
    pedagogical_intent: Optional[str] = None,
    refinement_instruction: Optional[str] = None,
    force_retrieval: Optional[bool] = None,
    material_scope_id: Optional[str] = None,
) -> AgentInput:
    """
    Create initial state for a new lesson planning request.

    Args:
        query: Teacher's natural language query
        domain: Knowledge domain ("neuro" or "udl")
        language: Response language ("it" for Italian, "en" for English)
        session_id: Optional session ID for persistence
        max_revisions: Maximum revision cycles allowed
        educational_profile: Optional per-request class/classroom context
            (CORE 1 #2.5). When provided, prompts and ranking can specialize
            against grade level, BES, classroom resources, etc. Backward
            compatible — omitting it preserves the original generic behavior.
        teacher_provided_context: Optional plain-text concatenation of files
            uploaded by the teacher in the WebUI chat (CORE 2 #6.6 P3). Passed
            into the Writer prompt only; the Planner / Retriever stay KG-only,
            and nothing here is ingested into the shared Knowledge Graph.
        conversation_history: Optional list of prior turns for multi-turn
            follow-up (CORE 2 #10.3). Each item is
            ``{"role": "user"|"assistant", "content": <markdown>}``. None or
            empty list → first turn (preserves single-turn behaviour). The
            list excludes the current turn's user query (which is in
            ``query`` / ``teacher_query``). The Writer prompt renders this
            as a "Conversation history" section only when non-empty.
        conversation_summary: Optional summary of the oldest turns when the
            thread exceeds the windowing threshold (CORE 2 #10.4). Prepended
            to ``conversation_history`` in the Writer prompt to preserve
            continuity. None on short threads.
        raw_user_turn: Optional raw text of the *current* turn only — i.e.
            ``query`` *before* the service layer prepended the conversation
            history / summary. Used by ``plan_node`` (CORE 2 #12b.3) so the
            Planner-extracted duration is only honoured when the teacher
            actually mentions one in *this* turn (otherwise the educational
            profile's ``time_available_minutes`` wins). When None, plan_node
            falls back to ``query`` — which keeps single-turn behaviour
            identical to pre-#12b.3 since on the first turn the augmenter
            short-circuits and ``query == raw_user_turn``.

    Returns:
        Initialized AgentState
    """
    return AgentInput(
        # Input
        teacher_query=query,
        domain=domain,
        language=language,
        session_id=session_id,
        # No fallback to ``session_id``, deliberately. That would mean an API
        # caller who sends the id of a lesson registered by the WebUI in this
        # same process reads that teacher's documents — the very leak the
        # separate key exists to close, in a narrower form. Whoever registers a
        # search function passes the key it was registered under; everyone else
        # gets no private retrieval, which is the safe answer.
        material_scope_id=material_scope_id,
        educational_profile=educational_profile,
        pedagogical_intent=pedagogical_intent,
        refinement_instruction=refinement_instruction,
        teacher_provided_context=teacher_provided_context,
        conversation_history=conversation_history,
        conversation_summary=conversation_summary,
        raw_user_turn=raw_user_turn,
        force_retrieval=force_retrieval,
        # Emitted explicitly so the verdict resets: a key that is merely
        # declared in AgentInput but absent from the payload is not
        # written at all, so the checkpointed value would survive.
        retrieval_reused=None,
        # Planner (empty)
        plan=None,
        query_intent=None,
        lesson_type=None,
        target_grade=None,
        key_concepts=None,
        search_queries=None,
        requires_new_retrieval=None,
        # NEW Phase A: Scope detection
        scope_status=None,
        scope_confidence=None,
        subject_concepts=None,
        pedagogy_concepts=None,
        # The retriever block is deliberately absent: those keys are the only
        # ones that persist across turns (PERSISTED_ACROSS_TURNS), so listing
        # them here as None would be the very reset we are removing. They are
        # not in AgentInput, so LangGraph would drop them anyway.
        #
        # CORE 2 #9 — Corrective RAG (gated by AIX_CORRECTIVE_RAG_ENABLED).
        # Initialised to None so the Writer/Critic see the same shape they
        # always have when the feature is off.
        retrieval_grade=None,
        retrieval_grade_reason=None,
        retrieval_attempts=None,
        retrieval_rewritten_query=None,
        retrieval_warning=None,
        # Fase 2 · PR 6 — retrieved fresh each turn, so it starts empty rather
        # than persisting like the Knowledge Graph corpus above: what the lesson
        # needs from a document depends on what this turn asks.
        material_context_items=[],
        material_maps=[],
        # Writer (empty)
        lesson_plan_draft=None,
        lesson_plan_structured=None,
        sources_cited=None,
        # Validator (empty) — a verdict belongs to the draft that earned it
        validation_violations=None,
        validation_passed=None,
        validation_attempts=None,
        # Critic (empty)
        critique=None,
        critique_score=None,
        approved=False,
        revision_instructions=None,
        # Metadata
        revision_count=0,
        max_revisions=max_revisions,
        current_step="start",
        error=None,
        # Final (empty)
        final_lesson_plan=None,
        final_metadata=None,
    )


@dataclass
class RetrievalPlan:
    """Structured plan for what to retrieve from GraphRAG"""

    lesson_type: str
    key_concepts: list[str]
    search_queries: list[str]
    target_grade: Optional[str] = None
    special_needs: Optional[list[str]] = None
    time_constraints: Optional[str] = None


@dataclass
class LessonPlanStructure:
    """Structured lesson plan output"""

    title: str
    grade_level: str
    duration: str
    learning_objectives: list[str]
    materials: list[str]
    introduction: str
    main_activities: list[dict[str, str]]
    assessment: str
    differentiation: dict[str, list[str]]
    sources: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
