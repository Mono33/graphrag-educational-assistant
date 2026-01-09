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
    
    # ========================================
    # PLANNER OUTPUT
    # ========================================
    plan: Optional[Dict[str, Any]]  # Retrieval plan created by planner
    query_intent: Optional[str]     # Detected intent (lesson_creation, definition, etc.)
    lesson_type: Optional[str]      # Type of lesson to generate (if lesson_creation)
    target_grade: Optional[str]     # Target grade level (if detected)
    key_concepts: Optional[List[str]]  # Key concepts to search for
    search_queries: Optional[List[str]]  # Queries to run on GraphRAG
    
    # ========================================
    # RETRIEVER OUTPUT
    # ========================================
    graphrag_results: Optional[List[Dict[str, Any]]]  # Results from GraphRAG searches
    retrieved_nodes: Optional[List[Dict[str, Any]]]   # All retrieved nodes
    retrieved_relationships: Optional[List[Dict[str, Any]]]  # All relationships
    recommendations: Optional[List[Dict[str, Any]]]   # Educational recommendations
    retrieval_confidence: Optional[str]  # Confidence from GraphRAG
    
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
    max_revisions: int = 2
) -> AgentState:
    """
    Create initial state for a new lesson planning request.
    
    Args:
        query: Teacher's natural language query
        domain: Knowledge domain ("neuro" or "udl")
        language: Response language ("it" for Italian, "en" for English)
        session_id: Optional session ID for persistence
        max_revisions: Maximum revision cycles allowed
        
    Returns:
        Initialized AgentState
    """
    return AgentState(
        # Input
        teacher_query=query,
        domain=domain,
        language=language,
        session_id=session_id,
        
        # Planner (empty)
        plan=None,
        query_intent=None,
        lesson_type=None,
        target_grade=None,
        key_concepts=None,
        search_queries=None,
        
        # Retriever (empty)
        graphrag_results=None,
        retrieved_nodes=None,
        retrieved_relationships=None,
        recommendations=None,
        retrieval_confidence=None,
        
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
        final_metadata=None
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

