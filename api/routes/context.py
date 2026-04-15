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
    # Explainability
    GraphPath,
    ScoringBreakdown,
    MethodologyExplainability,
    RetrievalPhase,
    KGStats,
    ExplainabilitySummary,
    ConceptGraphNode,
    ConceptGraphEdge,
    ConceptGraph,
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


# ---------------------------------------------------------------------------
# Explainability helpers
# ---------------------------------------------------------------------------

# Italian translations for confidence levels
_CONFIDENCE_IT = {
    "very_high": "molto alta",
    "high": "alta",
    "medium": "media",
    "low": "bassa",
    "very_low": "molto bassa",
}

# Italian translations for common relationship types
_REL_TYPE_IT = {
    "SUGGESTS": "suggerisce",
    "MITIGATED_BY": "mitiga",
    "RELATED_TO": "è correlato a",
    "SUPPORTS": "supporta",
    "INHIBITS": "inibisce",
    "ENABLES": "abilita",
    "REQUIRES": "richiede",
    "ENHANCES": "potenzia",
    "NO_SUGGESTS": "sconsiglia",
    "CAUSES": "causa",
    "REDUCES": "riduce",
    "IMPROVES": "migliora",
    "ASSOCIATED_WITH": "è associato a",
}

# Base scores by retrieval source
_BASE_SCORES = {"graph": 1.0, "structural": 0.8, "vector": 0.6, "semantic": 0.5}


def _build_methodology_explainability(
    node: dict, confidence_str: str
) -> tuple[str, str, MethodologyExplainability]:
    """
    Build per-methodology explainability from a raw graph node dict.

    Returns (explainability_name, explainability_phrase, MethodologyExplainability).

    explainability_name  — short Italian UI label (e.g. badge title)
    explainability_phrase — full Italian sentence ready to render in the UI as a tooltip or card text

    The phrase is generated dynamically from the node's provenance data:
      hop_distance == 0  → direct Cypher query match
      hop_distance == 1  → structural graph neighbor (shows graph path)
      hop_distance >= 2  → semantic / vector similarity match
    """
    hop = node.get("hop_distance", 0)
    retrieval_stage = node.get("retrieval_stage", "direct_query")
    source_node_data = node.get("source_node") or {}
    rel_type = node.get("rel_type") or ""
    rank_score = float(node.get("rank_score", 1.0))
    semantic_score = node.get("semantic_score")
    vector_similarity = node.get("vector_similarity")
    source = node.get("source", "graph")
    node_name = node.get("name", "")
    node_labels: list = node.get("labels") or []

    confidence_it = _CONFIDENCE_IT.get(confidence_str, "media")

    # Build graph path (only for structural neighbors with a real source node)
    graph_path: GraphPath | None = None
    if hop >= 1 and source_node_data and rel_type:
        src_labels = source_node_data.get("labels") or []
        graph_path = GraphPath(
            source_node=source_node_data.get("name", ""),
            source_label=src_labels[0] if src_labels else "",
            relationship=rel_type,
            target_node=node_name,
            target_label=node_labels[0] if node_labels else "",
        )

    # base_score is fixed at 0.5 per the reference JSON spec
    # Approximate domain_boost: rank_score / (semantic * vector) with safe division
    denom = (semantic_score if semantic_score else 1.0) * (vector_similarity if vector_similarity else 1.0)
    domain_boost = round(rank_score / denom, 2) if denom else 1.0

    scoring = ScoringBreakdown(
        base_score=0.5,
        semantic_score=round(semantic_score, 3) if semantic_score is not None else None,
        vector_similarity=round(vector_similarity, 3) if vector_similarity is not None else None,
        domain_boost=domain_boost,
        final_rank_score=round(rank_score, 3),
    )

    # Generate dynamic Italian name + phrase based on hop distance
    if hop == 0:
        name = "Raccomandazione diretta dal Knowledge Graph"
        phrase = (
            f"Strategia individuata direttamente dalla query nel Knowledge Graph. "
            f"Confidenza: {confidence_it}."
        )
        reasoning = f"Direct match from Cypher query (0 hops)"

    elif hop == 1 and graph_path:
        rel_it = _REL_TYPE_IT.get(rel_type, rel_type.lower().replace("_", " "))
        name = "Strategia collegata nel Knowledge Graph"
        phrase = (
            f"Questa strategia è collegata a «{graph_path.source_node}» "
            f"che {rel_it} «{node_name}» nel Knowledge Graph. "
            f"Confidenza: {confidence_it}."
        )
        reasoning = (
            f"Found as structural graph neighbor: "
            f"{graph_path.source_node} -[{rel_type}]-> {node_name} (1 hop)"
        )

    elif retrieval_stage == "vector_neighbor":
        score_pct = f"{round((vector_similarity or 0) * 100)}%" if vector_similarity else ""
        name = "Strategia identificata per similarità vettoriale"
        phrase = (
            f"Strategia individuata per similarità strutturale nel grafo"
            + (f" (similarità: {score_pct})" if score_pct else "")
            + f". Confidenza: {confidence_it}."
        )
        reasoning = f"Found via Node2Vec vector similarity ({score_pct})"

    else:
        # semantic_search or keyword_semantic
        score_pct = f"{round((semantic_score or 0) * 100)}%" if semantic_score else ""
        name = "Strategia identificata per similarità semantica"
        phrase = (
            f"Strategia individuata per similarità semantica con i concetti del Knowledge Graph"
            + (f" (similarità: {score_pct})" if score_pct else "")
            + f". Confidenza: {confidence_it}."
        )
        reasoning = f"Found via semantic embedding similarity ({score_pct})"

    methodology_exp = MethodologyExplainability(
        retrieval_method=retrieval_stage,
        hop_distance=hop,
        graph_path=graph_path,
        scoring_breakdown=scoring,
        reasoning=reasoning,
    )

    return name, phrase, methodology_exp


def _build_explainability_summary(retrieval_result, nodes: list, n_methods: int = 0) -> ExplainabilitySummary:
    """
    Build the response-level explainability summary from retrieval metadata.

    total_nodes_retrieved and fusion_ranking.nodes_found use the pre-cap total
    (graph_count + semantic_count from metadata) to match the reference JSON format.
    direct_hits / structural_neighbors / semantic_matches use the post-cap nodes list
    since those reflect the actual provenance of visible methodologies.
    graph_coverage is in English per the approved reference format.
    label_distribution drives the concept tag chips in the frontend.
    """
    metadata = retrieval_result.metadata if hasattr(retrieval_result, "metadata") else {}
    timings = metadata.get("timings") or {}
    facets = (retrieval_result.facets if hasattr(retrieval_result, "facets") else None) or {}
    label_counts: dict = facets.get("label_counts") or {}

    # Provenance counts from the post-cap nodes (actual visible methodologies)
    direct_hits = sum(1 for n in nodes if n.get("hop_distance", 0) == 0)
    structural = sum(1 for n in nodes if n.get("hop_distance", 0) == 1)
    semantic = sum(1 for n in nodes if (n.get("retrieval_stage") or "") in ("semantic_search", "keyword_semantic", "vector_neighbor"))

    # Pre-cap totals from metadata (how many the KG found before capping)
    graph_count = metadata.get("graph_count", direct_hits + structural)
    semantic_count = metadata.get("semantic_count", semantic)
    pre_cap_total = graph_count + semantic_count
    embedding_mode = metadata.get("embedding_mode", "hybrid_semantic")

    # English summary sentence — use graph_count (pre-cap) for the "via graph" count
    # so that CASE 4 structural nodes (hop=1) are still counted as graph-sourced,
    # not conflated with direct_hits (hop=0) which may be zero after the hop fix.
    graph_sourced = graph_count  # all nodes found via Neo4j traversal (pre-cap)
    graph_coverage = (
        f"This response used {pre_cap_total} concepts from the Knowledge Graph, "
        f"producing {n_methods} methodology recommendations. "
        f"{graph_sourced} found through graph traversal ({structural} via relationships, "
        f"{direct_hits} direct matches)."
    )
    if semantic:
        graph_coverage += f" {semantic} through semantic similarity."

    def _ms(key: str) -> int:
        val = timings.get(key, 0)
        return int(val * 1000) if val < 100 else int(val)  # handle both seconds and ms

    return ExplainabilitySummary(
        embedding_mode=embedding_mode,
        retrieval_phases={
            "graph_traversal": RetrievalPhase(nodes_found=graph_count, time_ms=_ms("graph_traversal")),
            "semantic_search": RetrievalPhase(nodes_found=semantic_count, time_ms=_ms("semantic_search")),
            "fusion_ranking": RetrievalPhase(nodes_found=pre_cap_total, time_ms=_ms("fusion")),
        },
        knowledge_graph_stats=KGStats(
            total_nodes_retrieved=pre_cap_total,
            total_relationships=len(retrieval_result.triples) if hasattr(retrieval_result, "triples") else 0,
            direct_hits=direct_hits,
            structural_neighbors=structural,
            semantic_matches=semantic,
            label_distribution=label_counts,
        ),
        graph_coverage=graph_coverage,
    )


def _build_concept_graph(nodes: list, triples: list, max_nodes: int = 20, max_edges: int = 30) -> ConceptGraph:
    """
    Build a lightweight graph structure for visualization.

    Primary nodes: top-scored nodes (capped at max_nodes).
    Context nodes: endpoints referenced in triples but not in the primary set —
      added automatically so their edges are not orphaned.
    Edges include all real Neo4j relationships where at least one endpoint is a
      primary node (not just when both are), ensuring CASE 4 Cypher relationships
      (e.g. ASSOCIATES_TO, MITIGATED_BY from barrier queries) are visible.
    """
    # Sort nodes by rank_score desc, cap at max_nodes
    sorted_nodes = sorted(nodes, key=lambda n: n.get("rank_score", 0), reverse=True)[:max_nodes]
    node_names = {n.get("name", "") for n in sorted_nodes}

    # Normalize rank_score to 0-1 for visualization
    max_score = max((n.get("rank_score", 1.0) for n in sorted_nodes), default=1.0) or 1.0

    graph_nodes = []
    for n in sorted_nodes:
        labels = n.get("labels") or []
        graph_nodes.append(
            ConceptGraphNode(
                id=f"{labels[0]}:{n.get('name', '')}" if labels else n.get("name", ""),
                label=labels[0] if labels else "Unknown",
                score=round(n.get("rank_score", 0) / max_score, 3),
                hop_distance=n.get("hop_distance", 0),
            )
        )

    # Edges: real Neo4j relationships (skip synthetic ones).
    # Include edges where AT LEAST ONE endpoint is a primary node.
    # When the other endpoint is missing, add it as a context node (score=0).
    _SYNTHETIC_RELATIONS = {"VECTOR_SIMILAR", "SEMANTIC_SIMILAR", "EMBEDDING_SIMILAR"}
    context_node_names: set = set()  # names already added as context nodes
    graph_edges = []
    seen_edges: set = set()

    for triple in triples:
        if len(triple) != 3:
            continue
        src, rel, tgt = triple
        if rel in _SYNTHETIC_RELATIONS:
            continue
        src_known = src in node_names
        tgt_known = tgt in node_names
        if not src_known and not tgt_known:
            continue  # neither endpoint is relevant — skip

        key = (src, rel, tgt)
        if key in seen_edges:
            continue
        seen_edges.add(key)

        # Add a context node for the missing endpoint (capped at max_nodes budget).
        # If the context node cannot be added (cap reached), skip the edge too — no dangling edges.
        total_nodes = len(graph_nodes) + len(context_node_names)
        if not src_known and src not in context_node_names:
            if total_nodes >= max_nodes:
                continue  # can't add context node → skip edge to avoid dangling reference
            context_node_names.add(src)
            graph_nodes.append(ConceptGraphNode(id=src, label="Context", score=0.0, hop_distance=2))
        if not tgt_known and tgt not in context_node_names:
            total_nodes = len(graph_nodes) + len(context_node_names)
            if total_nodes >= max_nodes:
                continue  # can't add context node → skip edge to avoid dangling reference
            context_node_names.add(tgt)
            graph_nodes.append(ConceptGraphNode(id=tgt, label="Context", score=0.0, hop_distance=2))

        graph_edges.append(ConceptGraphEdge(source=src, target=tgt, relation=rel))
        if len(graph_edges) >= max_edges:
            break

    return ConceptGraph(nodes=graph_nodes, edges=graph_edges)


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
        
        # -----------------------------------------------------------------------
        # Explainability enrichment (opt-in via include_explainability=True)
        # -----------------------------------------------------------------------
        explainability_summary = None
        concept_graph = None
        context_warning = None

        if request.include_explainability and retrieval_result:
            # Build node lookup: name → raw node dict (with hop_distance, rank_score, etc.)
            node_lookup: dict = {n.get("name", ""): n for n in nodes if n.get("name")}

            # Enrich each MethodologyInfo with explainability fields
            for methodology in (
                context_data.primary_methodologies + context_data.supporting_methodologies
            ):
                node = node_lookup.get(methodology.name)
                if node:
                    exp_name, exp_phrase, exp_data = _build_methodology_explainability(
                        node, methodology.confidence.value
                    )
                    methodology.explainability_name = exp_name
                    methodology.explainability_phrase = exp_phrase
                    methodology.explainability = exp_data
                else:
                    # Node not found in lookup (e.g. fallback methodology from domain config)
                    confidence_it = _CONFIDENCE_IT.get(methodology.confidence.value, "media")
                    methodology.explainability_name = "Raccomandazione dal Knowledge Graph"
                    methodology.explainability_phrase = (
                        f"Metodologia estratta dal Knowledge Graph. "
                        f"Confidenza: {confidence_it}."
                    )
                    methodology.explainability = MethodologyExplainability(
                        retrieval_method="domain_knowledge",
                        hop_distance=0,
                        graph_path=None,
                        scoring_breakdown=ScoringBreakdown(
                            base_score=0.5,
                            semantic_score=None,
                            vector_similarity=None,
                            domain_boost=1.0,
                            final_rank_score=0.5,
                        ),
                        reasoning="Retrieved via domain knowledge base (node not in direct retrieval result)",
                    )

            # Response-level summary
            n_methods = len(context_data.primary_methodologies) + len(context_data.supporting_methodologies)
            explainability_summary = _build_explainability_summary(retrieval_result, nodes, n_methods)

            # Concept graph for visualization
            concept_graph = _build_concept_graph(
                nodes,
                retrieval_result.triples if hasattr(retrieval_result, "triples") else [],
            )

            # Surface a warning only when the graph truly returned nothing useful
            kg_stats = explainability_summary.knowledge_graph_stats
            if kg_stats.total_nodes_retrieved == 0 or (
                kg_stats.semantic_matches > 0
                and kg_stats.direct_hits == 0
                and kg_stats.structural_neighbors == 0
            ):
                context_warning = (
                    "Nessuna corrispondenza diretta trovata nel grafo. "
                    "Le raccomandazioni si basano su similarità semantica."
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
            explainability_summary=explainability_summary,
            concept_graph=concept_graph,
            context_warning=context_warning,
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
