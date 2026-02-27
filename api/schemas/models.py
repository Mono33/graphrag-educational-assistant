"""
Pydantic models for GraphRAG API request/response validation
These models define the contract between AI Team and DEV Team
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


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
        example="Quali strategie per studenti con ADHD?"
    )
    domain: DomainType = Field(
        default=DomainType.NEURO,
        description="Knowledge domain to search"
    )
    language: LanguageType = Field(
        default=LanguageType.ITALIAN,
        description="Query language"
    )
    include_raw_nodes: bool = Field(
        default=False,
        description="Include raw node data from graph"
    )
    max_methodologies: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Maximum number of methodologies to return (split ~50/50 primary/supporting)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Quali strategie per studenti con ADHD?",
                "domain": "neuro",
                "language": "it",
                "include_raw_nodes": False,
                "max_methodologies": 10
            }
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
    classroom_applications: List[str] = Field(default=[], description="Practical applications")
    special_considerations: List[str] = Field(default=[], description="Special needs considerations")
    confidence: ConfidenceLevel = Field(..., description="Confidence level")


class QueryInfo(BaseModel):
    """Information about query processing"""
    original_query: str = Field(..., description="Original query as received")
    translated_query: Optional[str] = Field(None, description="Translated query (if translation occurred)")
    detected_language: str = Field(..., description="Detected query language")
    cypher_query: str = Field(..., description="Generated Cypher query")


class ContextData(BaseModel):
    """
    Structured educational context - ready for prompt injection
    
    DEV team can use these fields directly in their Jinja2 templates
    """
    educational_context_type: str = Field(
        ..., 
        description="Type of educational context (e.g., 'special_needs', 'general')"
    )
    student_profile: str = Field(
        ..., 
        description="Formatted student profile string"
    )
    primary_methodologies: List[MethodologyInfo] = Field(
        default=[], 
        description="Primary recommended methodologies"
    )
    supporting_methodologies: List[MethodologyInfo] = Field(
        default=[], 
        description="Supporting/alternative methodologies"
    )
    evidence_summary: str = Field(
        ..., 
        description="Summary of evidence supporting recommendations"
    )
    implementation_priority: List[str] = Field(
        default=[], 
        description="Ordered list of implementation priorities"
    )
    confidence_level: ConfidenceLevel = Field(
        ..., 
        description="Overall confidence in recommendations"
    )
    fallback_strategies: List[str] = Field(
        default=[], 
        description="Fallback strategies if primary methods don't work"
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
    domain: str = Field(
        ..., 
        description="Domain identifier (e.g., 'neuro', 'udl')"
    )
    domain_display_name: str = Field(
        ..., 
        description="Human-readable domain name (e.g., 'Neuro (Neuroscience)')"
    )
    system_prompt: str = Field(
        ..., 
        description="Rich domain system prompt (RUOLO, TAG-CLOUD, PRINCIPI, META-REGOLE)"
    )
    response_template: str = Field(
        ..., 
        description="Domain-specific response structure (e.g., I Do/We Do/You Do lesson schema)"
    )
    kg_context_formatted: str = Field(
        ..., 
        description="KG data formatted in domain-specific structure, ready for prompt injection"
    )


class RawNode(BaseModel):
    """Raw node data from knowledge graph"""
    id: Optional[str] = None
    name: str
    labels: List[str] = []
    category: Optional[str] = None
    description: Optional[str] = None
    properties: Dict[str, Any] = {}


class MetricsInfo(BaseModel):
    """Metrics about the retrieval"""
    total_nodes: int = Field(..., description="Total nodes retrieved")
    total_relationships: int = Field(..., description="Total relationships found")
    kg_data_available: bool = Field(
        default=True,
        description="Whether the Knowledge Graph contained relevant data for this query. "
                    "When false, kg_context_formatted contains guidance for the LLM to use "
                    "system_prompt principles instead. DEV team can optionally use this flag "
                    "to skip GraphRAG calls on follow-up messages."
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
    
    raw_nodes: Optional[List[RawNode]] = Field(
        None, 
        description="Raw nodes (only if include_raw_nodes=True)"
    )
    
    metrics: MetricsInfo = Field(..., description="Retrieval metrics")
    
    # Pre-formatted prompt section (optional convenience)
    formatted_prompt_section: Optional[str] = Field(
        None,
        description="Pre-formatted text ready to inject into prompts (domain-aware)"
    )
    
    # Domain-specific prompt context for production integration (Option B)
    domain_prompt_context: Optional[DomainPromptContext] = Field(
        None,
        description="Domain-specific system prompt, response template, and formatted KG context for production integration"
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
                    "cypher_query": "MATCH (n:TeachingPractices {domain: 'neuro'})..."
                },
                "context": {
                    "educational_context_type": "special_needs",
                    "student_profile": "Bisogni primari: ADHD | Contesto: inclusivo",
                    "primary_methodologies": [],
                    "supporting_methodologies": [],
                    "evidence_summary": "Based on 15 educational concepts...",
                    "implementation_priority": ["Start with structured environment"],
                    "confidence_level": "high",
                    "fallback_strategies": []
                },
                "metrics": {
                    "total_nodes": 15,
                    "total_relationships": 29
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    neo4j_connected: bool = Field(..., description="Neo4j connection status")
    version: str = Field(..., description="API version")
    domain_configs_loaded: List[str] = Field(..., description="Loaded domain configurations")

