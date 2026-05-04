"""
Agent State Definition

Defines the shared state that flows through the multi-agent pipeline.
Each agent can read and update this state.
"""

from typing import TypedDict, Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class LessonPlanType(Enum):
    """Types of lesson plans that can be generated"""
    FULL_LESSON = "full_lesson"
    ACTIVITY = "activity"
    ASSESSMENT = "assessment"
    UNIT_PLAN = "unit_plan"


class QueryIntent(Enum):
    """Types of queries the Agent can handle"""
    LESSON_CREATION = "lesson_creation"      # Full lesson plan request
    ACTIVITY_DESIGN = "activity_design"      # Single activity request
    DEFINITION = "definition"                 # "What is X?" questions
    COMPARISON = "comparison"                 # "Compare X and Y" questions
    EXPLANATION = "explanation"               # "How does X work?" questions
    RECOMMENDATION = "recommendation"         # "What strategies for...?" questions
    LIST = "list"                             # "List the types of..." questions


class ScopeStatus(Enum):
    """Scope status for query topics relative to Knowledge Graph"""
    IN_SCOPE = "in_scope"              # Topic found in KG (e.g., metacognition, attention)
    PARTIAL_SCOPE = "partial_scope"    # Pedagogy in KG, but subject topic outside
    OUT_OF_SCOPE = "out_of_scope"      # Topic completely outside KG domain
    UNKNOWN = "unknown"                # Could not determine scope


class AgentState(TypedDict, total=False):
    """
    Shared state passed between all agents in the pipeline.

    This TypedDict defines the complete state that flows through:
    Planner → Retriever → Writer → Critic → (Output or back to Writer)
    """

    # ========================================
    # INPUT (set at start)
    # ========================================
    teacher_query: str              # Original query from teacher
    domain: str                     # Knowledge domain ("neuro" or "udl")
    language: str                   # Response language ("it" or "en")
    session_id: Optional[str]       # For conversation persistence
    educational_profile: Optional[Dict[str, Any]]  # CORE 1 #2.5 — per-request class/classroom context (group, classroom, time, subject)
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
    conversation_history: Optional[List[Dict[str, str]]]
    # CORE 2 #10.4 — summary of the oldest turns when the window is
    # exceeded. None on short threads. Populated by the service layer
    # before invoking the graph. Writer prompt prepends this before
    # `conversation_history` so the agent has continuity even when older
    # turns are summarised.
    conversation_summary: Optional[str]

    # ========================================
    # PLANNER OUTPUT
    # ========================================
    plan: Optional[Dict[str, Any]]  # Retrieval plan created by planner
    query_intent: Optional[str]     # Detected intent (lesson_creation, definition, etc.)
    lesson_type: Optional[str]      # Type of lesson to generate (if lesson_creation)
    target_grade: Optional[str]     # Target grade level (if detected)
    key_concepts: Optional[List[str]]  # Key concepts to search for
    search_queries: Optional[List[str]]  # Queries to run on GraphRAG

    # NEW: Scope Detection (Phase A - Out-of-domain handling)
    scope_status: Optional[str]     # "in_scope", "partial_scope", "out_of_scope"
    scope_confidence: Optional[float]  # 0.0-1.0 confidence in scope detection
    subject_concepts: Optional[List[str]]  # Subject-specific concepts (may be out of scope)
    pedagogy_concepts: Optional[List[str]]  # Pedagogical concepts (always from KG)

    # ========================================
    # RETRIEVER OUTPUT
    # ========================================
    graphrag_results: Optional[List[Dict[str, Any]]]  # Results from GraphRAG searches
    retrieved_nodes: Optional[List[Dict[str, Any]]]   # All retrieved nodes
    retrieved_relationships: Optional[List[Dict[str, Any]]]  # All relationships
    recommendations: Optional[List[Dict[str, Any]]]   # Educational recommendations
    retrieval_confidence: Optional[str]  # Confidence from GraphRAG
    # NEW Phase 1: Curated media from sidecar JSON (optional, backward compatible)
    curated_media: Optional[Dict[str, Any]]  # Videos, resources, citations

    # NEW Phase A: External resources for out-of-scope queries
    external_resources: Optional[Dict[str, Any]]  # Wikipedia, OER, Semantic Scholar results

    # ========================================
    # WRITER OUTPUT
    # ========================================
    lesson_plan_draft: Optional[str]  # Generated lesson plan
    lesson_plan_structured: Optional[Dict[str, Any]]  # Structured version
    sources_cited: Optional[List[str]]  # Sources used in lesson plan

    # ========================================
    # CRITIC OUTPUT
    # ========================================
    critique: Optional[str]          # Critic's feedback
    critique_score: Optional[float]  # Quality score (0-1)
    approved: bool                   # Whether critic approved
    revision_instructions: Optional[str]  # What to fix if not approved

    # ========================================
    # METADATA
    # ========================================
    revision_count: int              # Number of revision cycles
    max_revisions: int               # Maximum allowed revisions
    current_step: str                # Current step in pipeline
    error: Optional[str]             # Error message if any

    # ========================================
    # FINAL OUTPUT
    # ========================================
    final_lesson_plan: Optional[str]  # Approved final lesson plan
    final_metadata: Optional[Dict[str, Any]]  # Final metadata


def create_initial_state(
    query: str,
    domain: str = "neuro",
    language: str = "it",
    session_id: Optional[str] = None,
    max_revisions: int = 2,
    educational_profile: Optional[Dict[str, Any]] = None,
    teacher_provided_context: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    conversation_summary: Optional[str] = None,
) -> AgentState:
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

    Returns:
        Initialized AgentState
    """
    return AgentState(
        # Input
        teacher_query=query,
        domain=domain,
        language=language,
        session_id=session_id,
        educational_profile=educational_profile,
        teacher_provided_context=teacher_provided_context,
        conversation_history=conversation_history,
        conversation_summary=conversation_summary,

        # Planner (empty)
        plan=None,
        query_intent=None,
        lesson_type=None,
        target_grade=None,
        key_concepts=None,
        search_queries=None,
        # NEW Phase A: Scope detection
        scope_status=None,
        scope_confidence=None,
        subject_concepts=None,
        pedagogy_concepts=None,

        # Retriever (empty)
        graphrag_results=None,
        retrieved_nodes=None,
        retrieved_relationships=None,
        recommendations=None,
        retrieval_confidence=None,
        curated_media=None,  # NEW Phase 1
        external_resources=None,  # NEW Phase A

        # Writer (empty)
        lesson_plan_draft=None,
        lesson_plan_structured=None,
        sources_cited=None,

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
    key_concepts: List[str]
    search_queries: List[str]
    target_grade: Optional[str] = None
    special_needs: Optional[List[str]] = None
    time_constraints: Optional[str] = None


@dataclass
class LessonPlanStructure:
    """Structured lesson plan output"""
    title: str
    grade_level: str
    duration: str
    learning_objectives: List[str]
    materials: List[str]
    introduction: str
    main_activities: List[Dict[str, str]]
    assessment: str
    differentiation: Dict[str, List[str]]
    sources: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
