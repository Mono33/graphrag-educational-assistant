"""
Pydantic models for GraphRAG API request/response validation
These models define the contract between AI Team and DEV Team
"""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

# Educational Profile is defined in a sibling module so the same shape can be
# reused by Agent mode (#7) and the future AgentRequest. Imported eagerly so
# the field below is available at module-import time. See CORE 1 #2.5.
from aix.api.schemas.educational_profile import EducationalProfile


class DomainType(str, Enum):
    """Available knowledge domains"""

    NEURO = "neuro"
    UDL = "udl"
    ALL = "all"


class LanguageType(str, Enum):
    """Supported languages"""

    ITALIAN = "it"
    ENGLISH = "en"


class ConfidenceLevel(str, Enum):
    """Confidence levels for recommendations"""

    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


# ============================================================================
# REQUEST MODELS
# ============================================================================


class ContextRequest(BaseModel):
    """
    Request model for getting GraphRAG context

    This is what DEV team sends to the API
    """

    query: str = Field(
        ...,
        description="The educational query in Italian or English",
        example="Quali strategie per studenti con ADHD?",
    )
    domain: DomainType = Field(default=DomainType.NEURO, description="Knowledge domain to search")
    language: LanguageType = Field(default=LanguageType.ITALIAN, description="Query language")
    include_raw_nodes: bool = Field(default=False, description="Include raw node data from graph")
    include_explainability: bool = Field(
        default=True,
        description="Include explainability data: per-methodology provenance + response-level KG summary + concept graph for visualization",
    )
    max_methodologies: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Maximum number of methodologies to return (split ~50/50 primary/supporting)",
    )
    educational_profile: Optional[EducationalProfile] = Field(
        default=None,
        description=(
            "Optional per-request educational context (CORE 1 #2.5). "
            "When provided, downstream prompt enrichment and methodology "
            "ranking can specialize against the class profile (BES, grade, "
            "classroom environment). Backward compatible: omitting this "
            "field preserves the original generic behavior."
        ),
    )

    class Config:
        # Canonical schema example — shows the full shape including the optional
        # educational_profile (CORE 1 #2.5) so DEV can see every available field.
        # Multiple named call examples (Minimal vs Rich) are exposed via
        # FastAPI's `Body(..., openapi_examples=...)` on the route handler — that
        # is the OpenAPI 3.1 + Swagger UI canonical way to render a dropdown.
        json_schema_extra = {
            "example": {
                "query": "Crea una lezione sulla fotosintesi",
                "domain": "neuro",
                "language": "it",
                "include_raw_nodes": False,
                "max_methodologies": 10,
                "educational_profile": {
                    "group": {
                        "title": "3A Liceo Scientifico",
                        "students_number": 25,
                        "grade": "SECONDARIA_II_GRADO",
                        "disabilities": ["ADHD", "DSA"],
                        "class_features": ["MOTIVATA"],
                        "student_attributes": [
                            "PUNTI_DI_ECCELLENZA",
                            "PUNTI_DI_CADUTA",
                        ],
                    },
                    "classroom": {
                        "title": "Aula 101",
                        "forniture_mobility": "PARTIALLY",
                        "has_lim": True,
                        "has_wifi": True,
                        "has_suite": True,
                        "pc_station": False,
                        "own_device": "BES",
                    },
                    "time_available_minutes": 60,
                    "subject_area": "Scienze",
                    "specific_topic": "Fotosintesi",
                },
            },
        }


# ============================================================================
# RESPONSE MODELS
# ============================================================================


class MethodologyInfo(BaseModel):
    """Information about a recommended methodology"""

    name: str = Field(..., description="Methodology name")
    category: str = Field(..., description="Methodology category")
    relevance_score: float = Field(..., ge=0, le=1, description="Relevance score 0-1")
    evidence_type: str = Field(..., description="Type of evidence supporting this")
    implementation_guidance: str = Field(..., description="How to implement")
    classroom_applications: list[str] = Field(default=[], description="Practical applications")
    special_considerations: list[str] = Field(
        default=[], description="Special needs considerations"
    )
    confidence: ConfidenceLevel = Field(..., description="Confidence level")
    # Explainability fields (populated when include_explainability=True)
    explainability_name: Optional[str] = Field(
        None,
        description="Italian UI label for the source type (e.g. 'Raccomandazione diretta dal Knowledge Graph')",
    )
    explainability_phrase: Optional[str] = Field(
        None,
        description="Italian human-readable sentence explaining how this methodology was retrieved — render directly as a badge or tooltip",
    )
    explainability: Optional["MethodologyExplainability"] = Field(
        None, description="Full provenance data for advanced UI rendering"
    )


class QueryInfo(BaseModel):
    """Information about query processing"""

    original_query: str = Field(..., description="Original query as received")
    translated_query: Optional[str] = Field(
        None, description="Translated query (if translation occurred)"
    )
    detected_language: str = Field(..., description="Detected query language")
    cypher_query: str = Field(..., description="Generated Cypher query")


class ContextData(BaseModel):
    """
    Structured educational context - ready for prompt injection

    DEV team can use these fields directly in their Jinja2 templates
    """

    educational_context_type: str = Field(
        ..., description="Type of educational context (e.g., 'special_needs', 'general')"
    )
    student_profile: str = Field(..., description="Formatted student profile string")
    primary_methodologies: list[MethodologyInfo] = Field(
        default=[], description="Primary recommended methodologies"
    )
    supporting_methodologies: list[MethodologyInfo] = Field(
        default=[], description="Supporting/alternative methodologies"
    )
    evidence_summary: str = Field(..., description="Summary of evidence supporting recommendations")
    implementation_priority: list[str] = Field(
        default=[], description="Ordered list of implementation priorities"
    )
    confidence_level: ConfidenceLevel = Field(
        ..., description="Overall confidence in recommendations"
    )
    fallback_strategies: list[str] = Field(
        default=[], description="Fallback strategies if primary methods don't work"
    )


class DomainPromptContext(BaseModel):
    """
    Domain-specific prompt context for production integration

    Provides the DEV team with everything needed to align their
    production prompt with GraphRAG domain expertise.

    This is designed to be scalable: each domain (neuro, udl, etc.)
    provides its own system_prompt and response_template through the
    domain config system.
    """

    domain: str = Field(..., description="Domain identifier (e.g., 'neuro', 'udl')")
    domain_display_name: str = Field(
        ..., description="Human-readable domain name (e.g., 'Neuro (Neuroscience)')"
    )
    system_prompt: str = Field(
        ..., description="Rich domain system prompt (RUOLO, TAG-CLOUD, PRINCIPI, META-REGOLE)"
    )
    response_template: str = Field(
        ...,
        description="Domain-specific response structure (e.g., I Do/We Do/You Do lesson schema)",
    )
    kg_context_formatted: str = Field(
        ...,
        description="KG data formatted in domain-specific structure, ready for prompt injection",
    )


# ============================================================================
# EXPLAINABILITY MODELS
# ============================================================================


class GraphPath(BaseModel):
    """Represents a single hop in the knowledge graph: source -[rel]-> target"""

    source_node: str = Field(..., description="Name of the source node")
    source_label: str = Field(..., description="Primary Neo4j label of the source node")
    relationship: str = Field(..., description="Relationship type (e.g. MITIGATED_BY, SUGGESTS)")
    target_node: str = Field(..., description="Name of the target node")
    target_label: str = Field(..., description="Primary Neo4j label of the target node")


class ScoringBreakdown(BaseModel):
    """Score components used to rank this node"""

    base_score: float = Field(
        ...,
        description="Base score by retrieval source (graph=1.0, structural=0.8, vector=0.6, semantic=0.5)",
    )
    semantic_score: Optional[float] = Field(
        None, description="Embedding similarity score 0-1 (null if not computed)"
    )
    vector_similarity: Optional[float] = Field(
        None, description="Node2Vec similarity score 0-1 (null if not computed)"
    )
    domain_boost: float = Field(..., description="Domain-specific multiplier applied to base score")
    final_rank_score: float = Field(
        ..., description="Final ranking score = base × domain_boost × semantic × vector"
    )


class MethodologyExplainability(BaseModel):
    """
    Provenance data for a single methodology — explains HOW and WHY it was retrieved.

    Frontend can use this to render source badges, graph path arrows, and score tooltips.
    """

    retrieval_method: str = Field(
        ...,
        description="How this node was found: 'direct_query' | 'structural_neighbor' | 'vector_neighbor' | 'semantic_search'",
    )
    hop_distance: int = Field(
        ...,
        description="Graph distance from query: 0=direct match, 1=1-hop neighbor, 2=2-hop/semantic",
    )
    graph_path: Optional[GraphPath] = Field(
        None,
        description="The graph path that led to this node (only present when hop_distance >= 1)",
    )
    scoring_breakdown: ScoringBreakdown = Field(
        ..., description="Score components for transparency"
    )
    reasoning: str = Field(..., description="Technical English sentence explaining the retrieval")


class RetrievalPhase(BaseModel):
    """Stats for a single retrieval phase"""

    nodes_found: int = Field(..., description="Number of nodes found in this phase")
    time_ms: int = Field(..., description="Time taken in milliseconds")


class KGStats(BaseModel):
    """Aggregate statistics about the knowledge graph retrieval"""

    total_nodes_retrieved: int
    total_relationships: int
    direct_hits: int = Field(..., description="Nodes found by direct Cypher query (hop_distance=0)")
    structural_neighbors: int = Field(
        ..., description="Nodes found by graph neighbor expansion (hop_distance=1)"
    )
    semantic_matches: int = Field(
        ..., description="Nodes found by semantic/vector similarity (hop_distance=2)"
    )
    label_distribution: dict[str, int] = Field(
        default={},
        description="Count of nodes per Neo4j label — use this to render concept tags in the UI",
    )


class ExplainabilitySummary(BaseModel):
    """
    Response-level explainability data.

    Frontend use:
    - label_distribution → render colored concept tag chips
    - graph_coverage → display as summary sentence in reliability panel
    - retrieval_phases → show retrieval breakdown (graph vs semantic)
    """

    embedding_mode: str = Field(
        ..., description="Embedding mode used: 'hybrid_semantic' | 'node2vec' | 'openai_only'"
    )
    retrieval_phases: dict[str, RetrievalPhase] = Field(
        default={},
        description="Timing and node count per retrieval phase: graph_traversal, semantic_search, fusion_ranking",
    )
    knowledge_graph_stats: KGStats = Field(..., description="Aggregate KG retrieval statistics")
    graph_coverage: str = Field(
        ..., description="Human-readable Italian sentence summarising graph coverage"
    )


class ConceptGraphNode(BaseModel):
    """A node in the concept graph for visualization"""

    id: str = Field(..., description="Unique node identifier (label:name)")
    label: str = Field(..., description="Primary Neo4j label")
    score: float = Field(..., description="Normalized rank score 0-1")
    hop_distance: int = Field(..., description="Graph distance from query")


class ConceptGraphEdge(BaseModel):
    """A directed edge in the concept graph"""

    source: str = Field(..., description="Source node name")
    target: str = Field(..., description="Target node name")
    relation: str = Field(..., description="Relationship type")


class ConceptGraph(BaseModel):
    """
    Nodes and edges extracted from the knowledge graph retrieval.

    Feed directly into D3.js / vis.js / sigma.js for graph visualization.
    Nodes capped at 20, edges at 30 to keep rendering performant.
    """

    nodes: list[ConceptGraphNode] = []
    edges: list[ConceptGraphEdge] = []


class RawNode(BaseModel):
    """Raw node data from knowledge graph"""

    id: Optional[str] = None
    name: str
    labels: list[str] = []
    category: Optional[str] = None
    description: Optional[str] = None
    properties: dict[str, Any] = {}


class MetricsInfo(BaseModel):
    """Metrics about the retrieval"""

    total_nodes: int = Field(..., description="Total nodes retrieved")
    total_relationships: int = Field(..., description="Total relationships found")
    kg_data_available: bool = Field(
        default=True,
        description="Whether the Knowledge Graph contained relevant data for this query. "
        "When false, kg_context_formatted contains guidance for the LLM to use "
        "system_prompt principles instead. DEV team can optionally use this flag "
        "to skip GraphRAG calls on follow-up messages.",
    )
    context_relevance: Optional[float] = Field(None, description="Context relevance score")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")


class ContextResponse(BaseModel):
    """
    Full response from GraphRAG API

    This is what DEV team receives and uses in their prompts
    """

    success: bool = Field(..., description="Whether the request was successful")

    query_info: QueryInfo = Field(..., description="Query processing information")

    context: ContextData = Field(..., description="Structured context for prompt injection")

    raw_nodes: Optional[list[RawNode]] = Field(
        None, description="Raw nodes (only if include_raw_nodes=True)"
    )

    metrics: MetricsInfo = Field(..., description="Retrieval metrics")

    # Pre-formatted prompt section (optional convenience)
    formatted_prompt_section: Optional[str] = Field(
        None, description="Pre-formatted text ready to inject into prompts (domain-aware)"
    )

    # Domain-specific prompt context for production integration (Option B)
    domain_prompt_context: Optional[DomainPromptContext] = Field(
        None,
        description="Domain-specific system prompt, response template, and formatted KG context for production integration",
    )

    # Explainability (populated when include_explainability=True)
    explainability_summary: Optional["ExplainabilitySummary"] = Field(
        None,
        description="Response-level KG stats + Italian summary sentence. Use label_distribution for concept tag chips.",
    )
    concept_graph: Optional["ConceptGraph"] = Field(
        None,
        description="Nodes and edges for graph visualization (D3.js / vis.js ready). Max 20 nodes, 30 edges.",
    )
    context_warning: Optional[str] = Field(
        None, description="Optional warning about retrieval quality"
    )
    error: Optional[str] = Field(None, description="Error message if success=False")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "query_info": {
                    "original_query": "Quali strategie per studenti con ADHD?",
                    "translated_query": "What strategies for students with ADHD?",
                    "detected_language": "it",
                    "cypher_query": "MATCH (n:TeachingPractices {domain: 'neuro'})...",
                },
                "context": {
                    "educational_context_type": "special_needs",
                    "student_profile": "Bisogni primari: ADHD | Contesto: inclusivo",
                    "primary_methodologies": [],
                    "supporting_methodologies": [],
                    "evidence_summary": "Based on 15 educational concepts...",
                    "implementation_priority": ["Start with structured environment"],
                    "confidence_level": "high",
                    "fallback_strategies": [],
                },
                "metrics": {"total_nodes": 15, "total_relationships": 29},
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(..., description="Service status")
    neo4j_connected: bool = Field(..., description="Neo4j connection status")
    version: str = Field(..., description="API version")
    domain_configs_loaded: list[str] = Field(..., description="Loaded domain configurations")
