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
    DomainPromptContext,
    MethodologyInfo,
    RawNode,
    MetricsInfo,
    ConfidenceLevel,
    ExplainabilityDetail,
    ExplainabilitySummary,
    GraphPathInfo,
    ScoringBreakdown,
    RetrievalPhaseInfo,
    KnowledgeGraphStats,
)

# Import GraphRAG components - use the enhanced wrapper that handles everything
from graph_retriever import EnhancedMultilingualText2Cypher
from context_builder import ConfidenceLevel as CBConfidenceLevel

# Import domain configuration system for scalable domain-aware formatting
from domains import get_domain_config

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


def _build_explainability(method) -> Optional[ExplainabilityDetail]:
    """Build per-methodology explainability from raw node metadata."""
    meta = getattr(method, 'raw_node_metadata', None)
    if not meta:
        return None

    retrieval_method = meta.get('retrieval_stage', 'unknown')
    hop_distance = meta.get('hop_distance', 0)
    rel_type = meta.get('rel_type', '')
    source_name = meta.get('source_node_name', '')
    source_label = meta.get('source_node_label', '')
    labels = meta.get('labels', [])
    target_label = labels[0] if labels else ''
    rank_score = meta.get('rank_score', 0.0)
    base_score = meta.get('base_score', 0.5)
    domain_boost = meta.get('domain_boost', 1.0)
    semantic_score = meta.get('semantic_score')
    vector_similarity = meta.get('vector_similarity')

    graph_path = None
    if source_name and rel_type:
        graph_path = GraphPathInfo(
            source_node=source_name,
            source_label=source_label,
            relationship=rel_type,
            target_node=method.name,
            target_label=target_label
        )

    scoring = ScoringBreakdown(
        base_score=round(base_score, 3),
        semantic_score=round(semantic_score, 3) if semantic_score is not None else None,
        vector_similarity=round(vector_similarity, 3) if vector_similarity is not None else None,
        domain_boost=round(domain_boost, 3),
        final_rank_score=round(rank_score, 3)
    )

    reasoning = _build_reasoning_text(
        method.name, retrieval_method, hop_distance,
        source_name, rel_type, semantic_score, vector_similarity
    )

    return ExplainabilityDetail(
        retrieval_method=retrieval_method,
        hop_distance=hop_distance,
        graph_path=graph_path,
        scoring_breakdown=scoring,
        reasoning=reasoning
    )


def _build_reasoning_text(
    name: str, retrieval_method: str, hop: int,
    source_name: str, rel_type: str,
    semantic_score: Optional[float], vector_similarity: Optional[float]
) -> str:
    """Generate a human-readable explanation of how a methodology was found."""
    if retrieval_method == 'direct_query' and rel_type and source_name:
        return f"Found via direct KG relationship: {source_name} -[{rel_type}]-> {name} (0 hops)"
    if retrieval_method == 'direct_query':
        return f"Direct match from Cypher query (0 hops)"
    if retrieval_method == 'structural_neighbor':
        path = f"{source_name} -[{rel_type}]-> {name}" if source_name and rel_type else name
        return f"Found as structural graph neighbor: {path} (1 hop)"
    if retrieval_method == 'vector_neighbor':
        sim = f", similarity={vector_similarity:.2f}" if vector_similarity else ""
        return f"Found via Node2Vec embedding similarity{sim} (2 hops)"
    if retrieval_method == 'semantic_search':
        sim = f", similarity={semantic_score:.2f}" if semantic_score else ""
        return f"Found via semantic embedding search{sim} (2 hops)"
    if retrieval_method == 'keyword_semantic':
        return f"Found via keyword matching in node descriptions (2 hops)"
    return f"Included from domain knowledge base"


_CONFIDENCE_IT = {
    "very_high": "molto alta",
    "high": "alta",
    "medium": "media",
    "low": "bassa",
    "very_low": "molto bassa",
}

_RETRIEVAL_NAME_IT = {
    "direct_query": "Raccomandazione diretta dal Knowledge Graph",
    "structural_neighbor": "Strategia collegata nel Knowledge Graph",
    "vector_neighbor": "Strategia simile individuata dall'AI",
    "semantic_search": "Suggerimento basato su similarità semantica",
    "keyword_semantic": "Trovato per corrispondenza tematica",
}


def _build_explainability_name(method) -> Optional[str]:
    """Italian teacher-friendly label derived from retrieval method."""
    meta = getattr(method, 'raw_node_metadata', None)
    if not meta:
        return "Basato su conoscenza pedagogica generale"
    stage = meta.get('retrieval_stage', 'unknown')
    return _RETRIEVAL_NAME_IT.get(stage, "Basato su conoscenza pedagogica generale")


_REL_VERBS_IT = {
    'SUGGESTS': 'suggerisce',
    'MITIGATED_BY': 'mitiga',
    'INFLUENCES': 'influenza',
    'RELATED_TO': 'è collegato a',
    'PART_OF': 'fa parte di',
    'NO_SUGGESTS': 'sconsiglia',
}


def _build_explainability_phrase(method, confidence_value: str) -> Optional[str]:
    """Italian teacher-facing sentence explaining pedagogical relevance.
    
    Node names are wrapped in guillemets («…») to visually separate
    English KG labels from the surrounding Italian text.  Raw Neo4j
    relationship types (e.g. MITIGATED_BY) are replaced by their
    Italian verb equivalents and never shown to teachers.
    """
    meta = getattr(method, 'raw_node_metadata', None)
    conf_it = _CONFIDENCE_IT.get(confidence_value, confidence_value)

    if not meta:
        return (
            f"Questa raccomandazione si basa su principi pedagogici generali. "
            f"Confidenza: {conf_it}."
        )

    stage = meta.get('retrieval_stage', 'unknown')
    rel_type = meta.get('rel_type', '')
    source_name = meta.get('source_node_name', '')

    if stage == 'direct_query' and source_name and rel_type:
        verb = _REL_VERBS_IT.get(rel_type, 'è collegato a')
        return (
            f"Questa strategia è collegata a «{source_name}» che {verb} "
            f"«{method.name}» nel Knowledge Graph. Confidenza: {conf_it}."
        )
    if stage == 'direct_query':
        return (
            f"Strategia individuata direttamente dalla query nel Knowledge Graph. "
            f"Confidenza: {conf_it}."
        )
    if stage == 'structural_neighbor' and source_name and rel_type:
        verb = _REL_VERBS_IT.get(rel_type, 'è collegato a')
        return (
            f"Questa strategia è collegata a «{source_name}» che {verb} "
            f"«{method.name}» nel Knowledge Graph. Confidenza: {conf_it}."
        )
    if stage == 'structural_neighbor':
        return (
            f"Strategia trovata tramite connessione nel Knowledge Graph. "
            f"Confidenza: {conf_it}."
        )
    if stage in ('vector_neighbor', 'semantic_search'):
        return (
            f"Strategia individuata dall'intelligenza artificiale per similarità "
            f"con i concetti della tua domanda. Confidenza: {conf_it}."
        )
    if stage == 'keyword_semantic':
        return (
            f"Strategia trovata per corrispondenza tematica con la tua domanda. "
            f"Confidenza: {conf_it}."
        )
    return (
        f"Questa raccomandazione si basa su principi pedagogici generali. "
        f"Confidenza: {conf_it}."
    )


def _format_methodologies(methodologies: list, max_count: int = 5) -> list[MethodologyInfo]:
    """Convert context_builder methodologies to API format with explainability"""
    result = []
    for method in methodologies[:max_count]:
        confidence = _map_confidence(method.confidence)
        result.append(MethodologyInfo(
            name=method.name,
            category=method.category,
            relevance_score=method.relevance_score,
            evidence_type=method.evidence_type,
            implementation_guidance=method.implementation_guidance,
            classroom_applications=method.classroom_applications[:3] if method.classroom_applications else [],
            special_considerations=method.special_considerations[:3] if method.special_considerations else [],
            confidence=confidence,
            explainability_name=_build_explainability_name(method),
            explainability_phrase=_build_explainability_phrase(method, confidence.value),
            explainability=_build_explainability(method)
        ))
    return result


def _build_context_warning(
    kg_data_available: bool,
    overall_confidence: str,
    methodologies_count: int
) -> Optional[str]:
    """Return an Italian warning when the KG lacks specific data for the query.
    
    Returns None when the KG returned solid, relevant results.
    """
    if not kg_data_available or methodologies_count == 0:
        return (
            "Attenzione: il Knowledge Graph non contiene dati specifici per questa richiesta. "
            "Le raccomandazioni si basano su principi pedagogici generali. "
            "Per risultati più mirati, prova a specificare il tipo di studenti o le difficoltà specifiche."
        )
    if overall_confidence in ('very_low', 'low'):
        return (
            "Nota: i risultati hanno una confidenza limitata. "
            "Il Knowledge Graph ha trovato pochi dati pertinenti per questa domanda. "
            "Prova a riformulare la richiesta in modo più specifico."
        )
    return None


def _get_domain_title(domain: str, language: str = "it") -> str:
    """Get domain-aware title for the KG context section.
    
    Scalable: add new domains here when they are registered.
    """
    domain_titles = {
        "it": {
            "neuro": "NEUROSCIENTIFICO",
            "udl": "UDL (UNIVERSAL DESIGN FOR LEARNING)",
            "all": "EDUCATIVO MULTI-DOMINIO"
        },
        "en": {
            "neuro": "NEUROSCIENCE",
            "udl": "UDL (UNIVERSAL DESIGN FOR LEARNING)",
            "all": "MULTI-DOMAIN EDUCATIONAL"
        }
    }
    # Normalize language code (handle both "it"/"italian" and "en"/"english")
    lang_key = "it" if language in ("it", "italian") else "en"
    lang_titles = domain_titles.get(lang_key, domain_titles["it"])
    return lang_titles.get(domain, domain.upper())


def _format_prompt_section(context: ContextData, language: str = "it", domain: str = "neuro") -> str:
    """
    Generate a domain-aware pre-formatted prompt section for direct injection.
    
    DEV team can use this directly in their prompts or build their own
    from the structured `context` data.
    
    When no KG data is found (0 methodologies), generates a transparent
    fallback message that instructs the LLM to use system_prompt principles
    instead of faking data. This is dynamic and works for any out-of-scope topic.
    """
    domain_title = _get_domain_title(domain, language)
    has_data = bool(context.primary_methodologies)
    
    if language in ("it", "italian"):
        lines = [
            f"## CONTESTO DAL KNOWLEDGE GRAPH {domain_title}",
            "",
            f"**Contesto Educativo:** {context.educational_context_type}",
            f"**Profilo Studente:** {context.student_profile}",
        ]
        
        if not has_data:
            if domain == "udl":
                lines.extend([
                    "",
                    "### Nota sul Knowledge Graph",
                    "",
                    "Per questa richiesta specifica, il Knowledge Graph non contiene dati "
                    "direttamente correlati al contenuto disciplinare richiesto.",
                    "Tuttavia, i principi UDL e le strategie inclusive presenti "
                    "nel system prompt sono pienamente applicabili per rispondere a questa domanda.",
                    "",
                    "### Come procedere:",
                    "- Utilizza i **PRINCIPI UDL** (Coinvolgimento, Rappresentazione, Azione ed Espressione) "
                    "per ottimizzare la risposta",
                    "- Considera la **variabilità degli apprendenti** e le loro esigenze specifiche",
                    "- Integra strategie **inclusive, differenziate e accessibili** nel contenuto "
                    "disciplinare specifico",
                    "- Il contenuto disciplinare specifico va basato sulle tue conoscenze generali",
                    "",
                    "### Livello di Confidenza: BASATO_SU_PRINCIPI_GENERALI",
                    "I principi UDL sono affidabili; il contenuto disciplinare "
                    "specifico non è verificato dal Knowledge Graph.",
                ])
            else:
                lines.extend([
                    "",
                    "### Nota sul Knowledge Graph",
                    "",
                    "Per questa richiesta specifica, il Knowledge Graph non contiene dati "
                    "direttamente correlati al contenuto disciplinare richiesto.",
                    "Tuttavia, i principi neuroscientifici e le strategie didattiche presenti "
                    "nel system prompt sono pienamente applicabili per rispondere a questa domanda.",
                    "",
                    "### Come procedere:",
                    "- Utilizza i **PRINCIPI NEUROSCIENTIFICI** (sezioni A-F del system prompt) "
                    "per ottimizzare la risposta",
                    "- Applica il **modello I Do – We Do – You Do** alla struttura della lezione (se pertinente)",
                    "- Integra strategie **attentive, metacognitive e motivazionali** nel contenuto "
                    "disciplinare specifico",
                    "- Il contenuto disciplinare specifico va basato sulle tue conoscenze generali",
                    "",
                    "### Livello di Confidenza: BASATO_SU_PRINCIPI_GENERALI",
                    "I principi neurodidattici sono affidabili; il contenuto disciplinare "
                    "specifico non è verificato dal Knowledge Graph.",
                ])
            return "\n".join(lines)
        
        lines.extend([
            "",
            "### Metodologie Raccomandate dal Knowledge Graph"
        ])
        
        for i, method in enumerate(context.primary_methodologies, 1):
            lines.append(f"\n{i}. **{method.name}** ({method.category})")
            lines.append(f"   - Rilevanza: {method.relevance_score:.2f}")
            lines.append(f"   - Confidenza: {method.confidence.value}")
            lines.append(f"   - Implementazione: {method.implementation_guidance}")
            if method.classroom_applications:
                lines.append(f"   - Applicazioni in classe:")
                for app in method.classroom_applications:
                    lines.append(f"     - {app}")
            if method.special_considerations:
                lines.append(f"   - Considerazioni speciali:")
                for consideration in method.special_considerations:
                    lines.append(f"     - {consideration}")
        
        if context.supporting_methodologies:
            lines.append("")
            lines.append("### Metodologie di Supporto")
            for method in context.supporting_methodologies:
                lines.append(f"- **{method.name}** ({method.category}) - Rilevanza: {method.relevance_score:.2f}")
        
        lines.append("")
        lines.append("### Evidenza e Basi Teoriche")
        lines.append(context.evidence_summary)
        
        lines.append("")
        lines.append("### Priorità di Implementazione")
        for i, priority in enumerate(context.implementation_priority, 1):
            lines.append(f"{i}. {priority}")
        
        lines.append("")
        lines.append(f"### Livello di Confidenza: {context.confidence_level.value.upper()}")
        
        if context.confidence_level.value in ['low', 'very_low']:
            lines.append("")
            lines.append("**Nota**: Il livello di confidenza è basso. Si consiglia di consultare uno specialista per raccomandazioni personalizzate.")
        
        if context.fallback_strategies:
            lines.append("")
            lines.append("### Strategie Alternative")
            for strategy in context.fallback_strategies:
                lines.append(f"- {strategy}")
        
        return "\n".join(lines)
    
    else:  # English
        lines = [
            f"## CONTEXT FROM {domain_title} KNOWLEDGE GRAPH",
            "",
            f"**Educational Context:** {context.educational_context_type}",
            f"**Student Profile:** {context.student_profile}",
        ]
        
        if not has_data:
            if domain == "udl":
                lines.extend([
                    "",
                    "### Knowledge Graph Note",
                    "",
                    "For this specific request, the Knowledge Graph does not contain data "
                    "directly related to the requested subject content.",
                    "However, the UDL principles and inclusive strategies in the "
                    "system prompt are fully applicable to answer this question.",
                    "",
                    "### How to proceed:",
                    "- Use the **UDL PRINCIPLES** (Engagement, Representation, Action & Expression) "
                    "to optimize the response",
                    "- Consider **learner variability** and their specific needs",
                    "- Integrate **inclusive, differentiated, and accessible strategies** into "
                    "the specific subject content",
                    "- The specific subject content should be based on your general knowledge",
                    "",
                    "### Confidence Level: BASED_ON_GENERAL_PRINCIPLES",
                    "The UDL principles are reliable; the specific subject content "
                    "is not verified by the Knowledge Graph.",
                ])
            else:
                lines.extend([
                    "",
                    "### Knowledge Graph Note",
                    "",
                    "For this specific request, the Knowledge Graph does not contain data "
                    "directly related to the requested subject content.",
                    "However, the neuroscience principles and teaching strategies in the "
                    "system prompt are fully applicable to answer this question.",
                    "",
                    "### How to proceed:",
                    "- Use the **NEUROSCIENCE PRINCIPLES** (sections A-F from system prompt) "
                    "to optimize the response",
                    "- Apply the **I Do – We Do – You Do model** to the lesson structure (if relevant)",
                    "- Integrate **attention, metacognitive, and motivational strategies** into "
                    "the specific subject content",
                    "- The specific subject content should be based on your general knowledge",
                    "",
                    "### Confidence Level: BASED_ON_GENERAL_PRINCIPLES",
                    "The neurodidactic principles are reliable; the specific subject content "
                    "is not verified by the Knowledge Graph.",
                ])
            return "\n".join(lines)
        
        lines.extend([
            "",
            "### Recommended Methodologies from Knowledge Graph"
        ])
        
        for i, method in enumerate(context.primary_methodologies, 1):
            lines.append(f"\n{i}. **{method.name}** ({method.category})")
            lines.append(f"   - Relevance: {method.relevance_score:.2f}")
            lines.append(f"   - Confidence: {method.confidence.value}")
            lines.append(f"   - Implementation: {method.implementation_guidance}")
            if method.classroom_applications:
                lines.append(f"   - Applications:")
                for app in method.classroom_applications:
                    lines.append(f"     - {app}")
        
        lines.append("")
        lines.append("### Evidence and Theoretical Basis")
        lines.append(context.evidence_summary)
        
        lines.append("")
        lines.append(f"### Confidence Level: {context.confidence_level.value.upper()}")
        
        return "\n".join(lines)


def _build_domain_prompt_context(
    context: ContextData,
    domain: str,
    language: str = "it"
) -> DomainPromptContext:
    """
    Build domain-specific prompt context for production integration (Option B).
    
    Returns the domain's system prompt, response template, and KG data
    formatted in a domain-specific structure.
    
    Scalable: automatically uses whichever domain config is registered.
    When a new domain is added (e.g., UDL, Math), it just needs to implement
    get_system_prompt() and get_response_template() in its domain config.
    """
    domain_config = get_domain_config(domain)
    
    if domain_config:
        system_prompt = domain_config.get_system_prompt()
        response_template = domain_config.get_response_template()
        display_name = domain_config.display_name
    else:
        # Fallback for "all" domain or unregistered domains
        system_prompt = "Sei un esperto consulente educativo italiano."
        response_template = "Genera una risposta pedagogica strutturata e pratica."
        display_name = domain.upper()
    
    # Build KG context formatted for this domain
    kg_context_formatted = _format_prompt_section(context, language, domain)
    
    return DomainPromptContext(
        domain=domain,
        domain_display_name=display_name,
        system_prompt=system_prompt,
        response_template=response_template,
        kg_context_formatted=kg_context_formatted
    )


def _build_explainability_summary(
    retrieval_result,
    context_data: ContextData,
    embedding_mode: str = "node2vec"
) -> ExplainabilitySummary:
    """Build top-level retrieval explainability from retrieval metadata."""
    metadata = {}
    if retrieval_result and hasattr(retrieval_result, 'metadata'):
        metadata = retrieval_result.metadata or {}
    
    timings = metadata.get('timings', {})
    
    retrieval_phases = {
        'graph_traversal': RetrievalPhaseInfo(
            nodes_found=metadata.get('graph_count', 0),
            time_ms=int(timings.get('graph_traversal', 0) * 1000) if timings.get('graph_traversal') else None
        ),
        'semantic_search': RetrievalPhaseInfo(
            nodes_found=metadata.get('semantic_count', 0),
            time_ms=int(timings.get('semantic_search', 0) * 1000) if timings.get('semantic_search') else None
        ),
        'fusion_ranking': RetrievalPhaseInfo(
            nodes_found=metadata.get('total_nodes', 0),
            time_ms=int(timings.get('fusion', 0) * 1000) if timings.get('fusion') else None
        ),
    }

    direct_hits = 0
    structural_neighbors = 0
    semantic_matches = 0
    label_distribution: dict = {}

    all_methods = list(context_data.primary_methodologies) + list(context_data.supporting_methodologies)
    for m in all_methods:
        ex = m.explainability
        if not ex:
            continue
        if ex.hop_distance == 0:
            direct_hits += 1
        elif ex.hop_distance == 1:
            structural_neighbors += 1
        else:
            semantic_matches += 1

    facets = metadata.get('facets', {}) if metadata else {}
    if isinstance(facets, dict):
        label_distribution = facets.get('label_counts', {})

    if not label_distribution and retrieval_result and hasattr(retrieval_result, 'facets'):
        rf = retrieval_result.facets or {}
        label_distribution = rf.get('label_counts', {})

    kg_stats = KnowledgeGraphStats(
        total_nodes_retrieved=metadata.get('total_nodes', 0),
        total_relationships=metadata.get('total_triples', 0),
        direct_hits=direct_hits,
        structural_neighbors=structural_neighbors,
        semantic_matches=semantic_matches,
        label_distribution=label_distribution
    )

    total = len(all_methods)
    parts = []
    if total:
        parts.append(
            f"This response used {kg_stats.total_nodes_retrieved} concepts from the Knowledge Graph, "
            f"producing {total} methodology recommendations."
        )
        if direct_hits:
            parts.append(f"{direct_hits} found through direct KG relationships (high confidence).")
        if structural_neighbors:
            parts.append(f"{structural_neighbors} through graph neighbor expansion.")
        if semantic_matches:
            parts.append(f"{semantic_matches} through semantic/vector similarity.")
    else:
        parts.append("No methodology recommendations were produced from the Knowledge Graph for this query.")

    return ExplainabilitySummary(
        embedding_mode=embedding_mode,
        retrieval_phases=retrieval_phases,
        knowledge_graph_stats=kg_stats,
        graph_coverage=" ".join(parts)
    )


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
            domain=domain,
            max_methodologies=request.max_methodologies
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
                    kg_data_available=False,
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
        
        # Build domain-aware prompt context (Option B for production integration)
        domain_prompt_ctx = _build_domain_prompt_context(
            context_data, domain, detected_language
        )
        
        # Determine if KG had relevant data (0 methodologies = out of scope)
        kg_data_available = bool(context_data.primary_methodologies)
        
        if not kg_data_available:
            logger.info("[API] No KG data found for this query — transparent fallback applied")
        
        # Build explainability summary
        try:
            from config import config as app_cfg
            embedding_mode = app_cfg.embedding.mode
        except Exception:
            embedding_mode = "node2vec"
        
        explain_summary = _build_explainability_summary(
            retrieval_result, context_data, embedding_mode
        )
        
        overall_conf = (
            context_data.confidence_level.value
            if hasattr(context_data, 'confidence_level') and context_data.confidence_level
            else 'medium'
        )
        ctx_warning = _build_context_warning(
            kg_data_available,
            overall_conf,
            len(context_data.primary_methodologies) + len(context_data.supporting_methodologies)
        )
        
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
                kg_data_available=kg_data_available,
                processing_time_ms=processing_time
            ),
            formatted_prompt_section=_format_prompt_section(context_data, detected_language, domain),
            domain_prompt_context=domain_prompt_ctx,
            explainability_summary=explain_summary,
            context_warning=ctx_warning
        )
        
        logger.info(f"Context generated successfully in {processing_time}ms")
        return response
        
    except Exception as e:
        logger.error(f"Error processing context request: {e}", exc_info=True)
        try:
            import sentry_sdk
            sentry_sdk.capture_exception(e)
        except ImportError:
            pass
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
                kg_data_available=False,
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
