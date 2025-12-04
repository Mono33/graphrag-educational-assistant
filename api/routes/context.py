"""
Context Route - Main endpoint for getting GraphRAG educational context

This is the core endpoint that DEV team will call to get knowledge graph context
"""

import logging
import time
from typing import Optional
from fastapi import APIRouter, HTTPException

# Import schemas
from api.schemas import (
    ContextRequest,
    ContextResponse,
    QueryInfo,
    ContextData,
    MethodologyInfo,
    RawNode,
    MetricsInfo,
    ConfidenceLevel
)

# Import GraphRAG components - use the enhanced wrapper that handles everything
from graph_retriever import EnhancedMultilingualText2Cypher
from context_builder import ConfidenceLevel as CBConfidenceLevel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/context", tags=["context"])


def _map_confidence(cb_confidence: CBConfidenceLevel) -> ConfidenceLevel:
    """Map context_builder ConfidenceLevel to API ConfidenceLevel"""
    mapping = {
        CBConfidenceLevel.VERY_HIGH: ConfidenceLevel.VERY_HIGH,
        CBConfidenceLevel.HIGH: ConfidenceLevel.HIGH,
        CBConfidenceLevel.MEDIUM: ConfidenceLevel.MEDIUM,
        CBConfidenceLevel.LOW: ConfidenceLevel.LOW,
        CBConfidenceLevel.VERY_LOW: ConfidenceLevel.VERY_LOW,
    }
    return mapping.get(cb_confidence, ConfidenceLevel.MEDIUM)


def _format_student_profile(profile) -> str:
    """Format student profile for prompt"""
    parts = []
    if profile.primary_needs:
        parts.append(f"Bisogni primari: {', '.join(profile.primary_needs)}")
    if profile.secondary_needs:
        parts.append(f"Bisogni secondari: {', '.join(profile.secondary_needs)}")
    parts.append(f"Contesto: {profile.educational_context}")
    if profile.grade_level:
        parts.append(f"Livello scolastico: {profile.grade_level}")
    if profile.subject_area:
        parts.append(f"Materia: {profile.subject_area}")
    return " | ".join(parts) if parts else "Profilo generale"


def _format_methodologies(methodologies: list, max_count: int = 5) -> list[MethodologyInfo]:
    """Convert context_builder methodologies to API format"""
    result = []
    for method in methodologies[:max_count]:
        result.append(MethodologyInfo(
            name=method.name,
            category=method.category,
            relevance_score=method.relevance_score,
            evidence_type=method.evidence_type,
            implementation_guidance=method.implementation_guidance,
            classroom_applications=method.classroom_applications[:3] if method.classroom_applications else [],
            special_considerations=method.special_considerations[:3] if method.special_considerations else [],
            confidence=_map_confidence(method.confidence)
        ))
    return result


def _format_prompt_section(context: ContextData, language: str = "it") -> str:
    """
    Generate a pre-formatted prompt section that DEV team can directly inject
    
    This is a convenience feature - DEV team can use this directly or build their own
    """
    if language == "it":
        lines = [
            "## CONTESTO DAL KNOWLEDGE GRAPH NEUROSCIENTIFICO",
            "",
            f"**Contesto Educativo:** {context.educational_context_type}",
            f"**Profilo Studente:** {context.student_profile}",
            "",
            "**Metodologie Raccomandate:**"
        ]
        
        for i, method in enumerate(context.primary_methodologies, 1):
            lines.append(f"{i}. **{method.name}** ({method.category})")
            lines.append(f"   - Rilevanza: {method.relevance_score:.2f}")
            lines.append(f"   - Implementazione: {method.implementation_guidance}")
            if method.classroom_applications:
                lines.append(f"   - Applicazioni: {', '.join(method.classroom_applications)}")
        
        if context.supporting_methodologies:
            lines.append("")
            lines.append("**Metodologie di Supporto:**")
            for method in context.supporting_methodologies[:2]:
                lines.append(f"- {method.name} ({method.category})")
        
        lines.append("")
        lines.append(f"**Evidenza:** {context.evidence_summary}")
        lines.append("")
        lines.append("**Priorità di Implementazione:**")
        for priority in context.implementation_priority:
            lines.append(f"- {priority}")
        
        lines.append("")
        lines.append(f"**Livello di Confidenza:** {context.confidence_level.value.upper()}")
        
        if context.fallback_strategies:
            lines.append("")
            lines.append("**Strategie Alternative:**")
            for strategy in context.fallback_strategies:
                lines.append(f"- {strategy}")
        
        return "\n".join(lines)
    
    else:  # English
        lines = [
            "## CONTEXT FROM NEUROSCIENCE KNOWLEDGE GRAPH",
            "",
            f"**Educational Context:** {context.educational_context_type}",
            f"**Student Profile:** {context.student_profile}",
            "",
            "**Recommended Methodologies:**"
        ]
        
        for i, method in enumerate(context.primary_methodologies, 1):
            lines.append(f"{i}. **{method.name}** ({method.category})")
            lines.append(f"   - Relevance: {method.relevance_score:.2f}")
            lines.append(f"   - Implementation: {method.implementation_guidance}")
        
        lines.append("")
        lines.append(f"**Evidence:** {context.evidence_summary}")
        lines.append(f"**Confidence Level:** {context.confidence_level.value.upper()}")
        
        return "\n".join(lines)


@router.post("", response_model=ContextResponse)
async def get_context(request: ContextRequest) -> ContextResponse:
    """
    Get educational context from GraphRAG knowledge graph
    
    This endpoint:
    1. Translates the query (if needed)
    2. Generates a Cypher query
    3. Retrieves relevant nodes from Neo4j
    4. Builds structured educational context
    5. Returns everything in a format ready for prompt injection
    
    DEV team can use the returned data directly in their Jinja2 templates.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing context request: domain={request.domain}, query={request.query[:50]}...")
        
        domain = request.domain.value
        
        # Use the enhanced processor that wraps everything
        # use_vectors=True enables Node2Vec semantic search (same as Streamlit default)
        processor = EnhancedMultilingualText2Cypher(
            use_vectors=True,
            domain=domain
        )
        
        # Process the query through the full pipeline
        result = await processor.process_query_with_retrieval(
            query=request.query,
            domain=domain
        )
        
        # Extract data from result (correct structure from process_query_with_retrieval)
        educational_context = result.get('educational_context_obj')
        
        # Cypher query is inside cypher_result
        cypher_result = result.get('cypher_result', {})
        cypher_query = cypher_result.get('cypher_query', '')
        
        original_query = result.get('original_query', request.query)
        enhanced_query = cypher_result.get('enhanced_query')
        detected_language = cypher_result.get('detected_language', 'it')
        
        # Nodes and relationships are in retrieval_result object
        retrieval_result = result.get('retrieval_result')
        if retrieval_result:
            nodes = retrieval_result.nodes if hasattr(retrieval_result, 'nodes') else []
            relationships = retrieval_result.triples if hasattr(retrieval_result, 'triples') else []
        else:
            nodes = []
            relationships = []
        
        # Get counts from educational_context.metadata (same source as Streamlit)
        # This includes data after P1 filter expansion and context building
        if educational_context and hasattr(educational_context, 'metadata'):
            edu_metadata = educational_context.metadata
            total_nodes_count = edu_metadata.get('total_nodes', len(nodes))
            total_relationships_count = edu_metadata.get('total_triples', len(relationships))
            logger.info(f"[API] Using educational_context metadata: nodes={total_nodes_count}, relationships={total_relationships_count}")
        else:
            # Fallback to retrieval_result metadata
            metadata = retrieval_result.metadata if retrieval_result and hasattr(retrieval_result, 'metadata') else {}
            total_nodes_count = metadata.get('total_nodes', len(nodes))
            total_relationships_count = metadata.get('total_triples', len(relationships))
            logger.info(f"[API] Using retrieval_result metadata: nodes={total_nodes_count}, relationships={total_relationships_count}")
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # Handle case where educational_context wasn't built
        if not educational_context:
            return ContextResponse(
                success=False,
                query_info=QueryInfo(
                    original_query=request.query,
                    translated_query=enhanced_query,
                    detected_language=detected_language,
                    cypher_query=cypher_query
                ),
                context=ContextData(
                    educational_context_type="unknown",
                    student_profile="",
                    primary_methodologies=[],
                    supporting_methodologies=[],
                    evidence_summary="No context could be built for this query",
                    implementation_priority=[],
                    confidence_level=ConfidenceLevel.VERY_LOW,
                    fallback_strategies=["Try rephrasing your question"]
                ),
                metrics=MetricsInfo(
                    total_nodes=total_nodes_count,
                    total_relationships=total_relationships_count,
                    processing_time_ms=processing_time
                ),
                error="Could not build educational context"
            )
        
        # Build context data from educational_context
        context_data = ContextData(
            educational_context_type=educational_context.student_profile.educational_context,
            student_profile=_format_student_profile(educational_context.student_profile),
            primary_methodologies=_format_methodologies(
                educational_context.primary_methodologies, 
                request.max_methodologies
            ),
            supporting_methodologies=_format_methodologies(
                educational_context.supporting_methodologies,
                max(2, request.max_methodologies // 2)
            ),
            evidence_summary=educational_context.evidence_summary,
            implementation_priority=educational_context.implementation_priority,
            confidence_level=_map_confidence(educational_context.confidence_assessment),
            fallback_strategies=educational_context.fallback_strategies
        )
        
        # Build raw nodes if requested
        raw_nodes = None
        if request.include_raw_nodes:
            raw_nodes = [
                RawNode(
                    id=str(node.get("id", "")),
                    name=node.get("name", "Unknown"),
                    labels=node.get("labels", []),
                    category=node.get("category"),
                    description=node.get("description"),
                    properties={k: v for k, v in node.items() if k not in ["id", "name", "labels", "category", "description"]}
                )
                for node in nodes[:20]  # Limit to 20 nodes
            ]
        
        # Build response
        response = ContextResponse(
            success=True,
            query_info=QueryInfo(
                original_query=original_query,
                translated_query=enhanced_query if enhanced_query and enhanced_query != original_query else None,
                detected_language=detected_language,
                cypher_query=cypher_query
            ),
            context=context_data,
            raw_nodes=raw_nodes,
            metrics=MetricsInfo(
                total_nodes=total_nodes_count,
                total_relationships=total_relationships_count,
                processing_time_ms=processing_time
            ),
            formatted_prompt_section=_format_prompt_section(context_data, detected_language)
        )
        
        logger.info(f"Context generated successfully in {processing_time}ms")
        return response
        
    except Exception as e:
        logger.error(f"Error processing context request: {e}", exc_info=True)
        processing_time = int((time.time() - start_time) * 1000)
        
        return ContextResponse(
            success=False,
            query_info=QueryInfo(
                original_query=request.query,
                translated_query=None,
                detected_language="unknown",
                cypher_query=""
            ),
            context=ContextData(
                educational_context_type="error",
                student_profile="",
                primary_methodologies=[],
                supporting_methodologies=[],
                evidence_summary="",
                implementation_priority=[],
                confidence_level=ConfidenceLevel.VERY_LOW,
                fallback_strategies=["Please try again or contact support"]
            ),
            metrics=MetricsInfo(
                total_nodes=0,
                total_relationships=0,
                processing_time_ms=processing_time
            ),
            error=str(e)
        )


@router.get("/domains")
async def list_domains():
    """List available knowledge domains"""
    return {
        "domains": [
            {
                "id": "neuro",
                "name": "Neuroscienze dell'Apprendimento",
                "description": "Knowledge graph focused on neuroscience of learning"
            },
            {
                "id": "udl",
                "name": "Universal Design for Learning",
                "description": "Knowledge graph focused on UDL principles and methodologies"
            },
            {
                "id": "all",
                "name": "All Domains",
                "description": "Search across all available knowledge domains"
            }
        ]
    }
