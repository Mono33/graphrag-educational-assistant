#!/usr/bin/env python3
"""
Educational Context Builder for GraphRAG
Transforms raw graph retrieval results into structured educational context
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re

# Import domain configuration system
from aix.domains import get_domain_config

logger = logging.getLogger(__name__)

class ConfidenceLevel(Enum):
    """Confidence levels for educational recommendations"""
    VERY_HIGH = "VERY_HIGH"
    HIGH = "HIGH" 
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"

@dataclass
class MethodologyRecommendation:
    """A single methodology recommendation with context"""
    name: str
    category: str
    relevance_score: float
    evidence_type: str  # 'direct_relationship', 'semantic_similarity', 'domain_knowledge'
    implementation_guidance: str
    classroom_applications: List[str]
    special_considerations: List[str]
    confidence: ConfidenceLevel

@dataclass
class StudentProfile:
    """Student profile extracted from query and context"""
    primary_needs: List[str]
    secondary_needs: List[str]
    educational_context: str  # 'special_needs', 'assessment', 'general'
    grade_level: Optional[str]
    subject_area: Optional[str]

@dataclass
class EducationalContext:
    """Complete educational context for response generation"""
    student_profile: StudentProfile
    primary_methodologies: List[MethodologyRecommendation]
    supporting_methodologies: List[MethodologyRecommendation]
    evidence_summary: str
    implementation_priority: List[str]
    confidence_assessment: ConfidenceLevel
    fallback_strategies: List[str]
    metadata: Dict[str, Any]

class PedagogicalKnowledgeBase:
    """Educational domain knowledge and best practices"""
    
    def __init__(self, domain: str = "udl"):
        """Initialize knowledge base with domain-specific data
        
        Args:
            domain: Domain filter ('udl', 'neuro', 'all')
        """
        self.domain = domain
        
        
        # Load methodology categories from domain config
        domain_config = get_domain_config(domain)
        if domain_config:
            self.methodology_categories = domain_config.get_methodology_categories()
        elif domain == "all":
            udl_cfg = get_domain_config("udl")
            neuro_cfg = get_domain_config("neuro")
            udl_cats = udl_cfg.get_methodology_categories() if udl_cfg else {}
            neuro_cats = neuro_cfg.get_methodology_categories() if neuro_cfg else {}
            self.methodology_categories = {**udl_cats, **neuro_cats}
        else:
            logger.warning(f"No domain config for '{domain}', using empty methodology categories")
            self.methodology_categories = {}

        # Load special needs mapping from domain config
        if domain_config:
            self.special_needs_mapping = domain_config.get_special_needs_mapping()
        elif domain == "all":
            udl_cfg = get_domain_config("udl")
            neuro_cfg = get_domain_config("neuro")
            udl_needs = udl_cfg.get_special_needs_mapping() if udl_cfg else {}
            neuro_needs = neuro_cfg.get_special_needs_mapping() if neuro_cfg else {}
            self.special_needs_mapping = {**udl_needs, **neuro_needs}
        else:
            logger.warning(f"No domain config for '{domain}', using empty special needs mapping")
            self.special_needs_mapping = {}

        # Domain-aware fallback strategies
        if domain == "neuro":
            self.fallback_strategies = {
                'no_results': [
                    'Evidence-based neuroscience principles for education',
                    'Cognitive load management strategies',
                    'Attention and metacognitive enhancement techniques',
                    'Motivational and emotional regulation approaches'
                ],
                'low_confidence': [
                    'Consult neuroscience education specialists',
                    'Implement evidence-based instructional strategies',
                    'Use assessment-based cognitive profiling',
                    'Seek interdisciplinary collaboration'
                ]
            }
        else:
            self.fallback_strategies = {
                'no_results': [
                    'Universal Design for Learning (UDL) principles',
                    'Differentiated instruction approaches',
                    'Multi-sensory learning techniques',
                    'Collaborative learning environments'
                ],
                'low_confidence': [
                    'Consult special education specialists',
                    'Implement gradual methodology introduction',
                    'Use assessment-based adaptation',
                    'Seek peer teacher collaboration'
                ]
            }

class MethodologyRanker:
    """Ranks and prioritizes educational methodologies with dynamic balancing"""
    
    def __init__(self, knowledge_base: PedagogicalKnowledgeBase):
        self.kb = knowledge_base
    
    def rank_methodologies(self, nodes: List[Dict], query_metadata: Dict) -> List[MethodologyRecommendation]:
        """
        Rank methodologies with dynamic balancing for multi-label queries.
        
        Pipeline: deduplicate → filter → balance (if multi-label) → interleave → recommend.
        
        Balancing is triggered when:
        - Query contains comparison keywords (e.g., "differenza", "vs"), OR
        - Nodes span 2+ distinct primary label groups (e.g., Attention + Metacognition)
        
        Interleaving ensures the primary/supporting split in build_context()
        contains balanced representation from all label groups.
        """
        logger.info(f"[DEBUG] Ranking {len(nodes)} nodes for recommendations")
        
        # Step 0: Deduplicate nodes by name (same concept from multiple retrieval sources)
        deduped_nodes = self._deduplicate_nodes(nodes)
        
        # Step 1: Filter valid methodologies
        valid_nodes = []
        for i, node in enumerate(deduped_nodes):
            logger.info(f"[DEBUG] Node {i+1}: name='{node.get('name', 'N/A')}', labels={node.get('labels', [])}, keys={list(node.keys())}")
            
            if self._is_methodology(node):
                logger.info(f"[DEBUG] Node {i+1} ACCEPTED as methodology")
                valid_nodes.append(node)
            else:
                logger.info(f"[DEBUG] Node {i+1} REJECTED - not a valid methodology")
        
        # Step 2: Detect if balancing is needed (comparison OR multi-label distribution)
        query_intent = self._detect_query_intent(query_metadata.get('original_query', ''))
        
        primary_label_counts: Dict[str, int] = {}
        for node in valid_nodes:
            labels = node.get('labels', [])
            if isinstance(labels, str):
                labels = [labels]
            pl = labels[0] if labels else 'Unknown'
            primary_label_counts[pl] = primary_label_counts.get(pl, 0) + 1
        
        needs_balancing = query_intent['is_comparison'] or 2 <= len(primary_label_counts) <= 4
        
        if needs_balancing:
            reason = "comparison keywords" if query_intent['is_comparison'] else "multi-label distribution"
            logger.info(f"[Smart Ranking] Balancing triggered ({reason}): {primary_label_counts}")
            balanced_nodes = self._apply_dynamic_balancing(valid_nodes, target_size=15)
            balanced_nodes = self._interleave_nodes_by_label(balanced_nodes)
        else:
            balanced_nodes = valid_nodes
        
        # Step 3: Create recommendations
        recommendations = []
        for node in balanced_nodes:
            recommendation = self._create_recommendation(node, query_metadata)
            if recommendation:
                recommendations.append(recommendation)
        
        if needs_balancing:
            logger.info(f"[Smart Ranking] Preserving interleaved order for balanced primary/supporting split")
        else:
            recommendations.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.info(f"[DEBUG] Final recommendations: {len(recommendations)}")
        return recommendations
    
    def _deduplicate_nodes(self, nodes: List[Dict]) -> List[Dict]:
        """Merge duplicate nodes (same name) from multiple retrieval sources.
        
        The retriever returns the same concept from graph traversal, semantic search,
        and Node2Vec — each as a separate dict. This merges them, keeping the entry
        with the highest rank_score and enriching it with data from others.
        
        Dynamic: works for any node names, any domain, any retrieval source.
        """
        if not nodes:
            return []
        
        seen: Dict[str, Dict] = {}
        
        for node in nodes:
            name = node.get('name', '')
            if not name:
                continue
            
            if name not in seen:
                seen[name] = node
            else:
                existing = seen[name]
                if node.get('rank_score', 0) > existing.get('rank_score', 0):
                    # New node wins — enrich it with metadata from the loser
                    for key in ('description', 'rel_type', 'source', 'vector_similarity'):
                        if not node.get(key) and existing.get(key):
                            node[key] = existing[key]
                    seen[name] = node
                else:
                    # Existing wins — enrich it with metadata from the loser
                    for key in ('description', 'rel_type', 'source', 'vector_similarity'):
                        if not existing.get(key) and node.get(key):
                            existing[key] = node[key]
        
        deduped = list(seen.values())
        removed = len(nodes) - len(deduped)
        if removed > 0:
            logger.info(f"[Dedup] Merged {len(nodes)} → {len(deduped)} nodes ({removed} duplicates removed)")
        
        return deduped
    
    # Relationship-type names that leak through as nodes (they are edge labels, not content)
    _RELATIONSHIP_NAMES = frozenset({'SUGGESTS', 'NO_SUGGESTS', 'MITIGATED_BY', 'RELATED_TO', 'PART_OF'})

    # Negative-example node names — real KG nodes that model what NOT to do.
    # Sending these to the LLM as "recommendations" produces confusing output.
    _NEGATIVE_APPROACH_NAMES = frozenset({
        'Long Frontal Lesson', 'Long frontal reading lessons',
        'Passive Learning', 'Passive learning',
    })

    # Challenge/characteristic labels — describe student profiles, not teaching strategies.
    # A node whose ALL labels fall in this set is a symptom/profile node, not a methodology.
    # Mixed-label nodes (e.g. ['Adhd', 'CognitiveStrategy']) still pass through.
    _CHALLENGE_LABELS = frozenset({
        'Adhd', 'AutismSpectrum', 'Dyscalculia', 'Dyslexia',
        'Gifted', 'ForeignStudents', 'SensoryDisabilities', 'PhysicalDisabilities',
    })

    def _is_methodology(self, node: Dict) -> bool:
        """Check if node should be included in recommendations.

        Rejects five classes of noise that survive the P1+ retrieval filter:
        1. Neo4j system / infrastructure labels (_GraphConfig, Node, Entity)
        2. Relationship-type names accidentally stored as nodes (SUGGESTS, NO_SUGGESTS)
        3. Known negative-example nodes (Long Frontal Lesson, Passive Learning)
        4. Sentence-nodes — full sentences stored as node names (heuristic: ends
           with '.' and is longer than 60 chars). These are KG description
           fragments, not actionable methodology names.
        5. Pure challenge/characteristic nodes (Adhd, Dyslexia, etc.) — these
           describe student profiles. Their data is preserved in student_profile
           and triples; they should not appear as methodology recommendations.

        Everything else is accepted — the retrieval pipeline already validated
        relevance via Node2Vec and semantic similarity.
        """
        name = node.get('name', '')
        if not name:
            return False

        # Rule 2: relationship-type names
        if name in self._RELATIONSHIP_NAMES:
            return False

        # Rule 3: negative-example nodes
        if name in self._NEGATIVE_APPROACH_NAMES:
            return False

        # Rule 4: sentence-nodes (heuristic)
        if name.endswith('.') and len(name) > 60:
            return False

        labels = node.get('labels', [])
        if not labels:
            return True

        # Rule 1: system / infrastructure labels
        system_labels = {'_GraphConfig', 'Node', 'Entity', '__Entity__'}
        if all(label in system_labels for label in labels):
            return False

        # Rule 5: pure challenge/characteristic nodes — all labels are challenge-only.
        # Node with mixed labels like ['Adhd', 'CognitiveStrategy'] still passes.
        if all(label in self._CHALLENGE_LABELS for label in labels):
            return False

        return True
    
    def _create_recommendation(self, node: Dict, query_metadata: Dict) -> Optional[MethodologyRecommendation]:
        """Create a methodology recommendation from a node.
        
        Priority chain for each field:
        1. Hardcoded kb_info (methodology_categories dict) — most curated
        2. Node's own KG properties (description, category) — real graph data
        3. Generic fallback template — last resort
        """
        name = node.get('name', '')
        if not name:
            return None
        
        kb_info = self.kb.methodology_categories.get(name, {})
        
        relevance_score = self._calculate_relevance_score(node, query_metadata)
        evidence_type = self._determine_evidence_type(node)
        
        # --- Implementation guidance: kb_info > node description > generic ---
        node_description = node.get('description', '')
        if isinstance(node_description, float):
            node_description = ''
        implementation = kb_info.get(
            'implementation',
            node_description or f'Apply {name} with appropriate adaptations'
        )
        
        # --- Classroom applications: kb_info only (LLM handles specifics) ---
        applications = kb_info.get('applications', None)
        if not applications:
            applications = [f'Consultare il contesto specifico della classe per applicazioni pratiche di {name}']
        
        # --- Special considerations: kb_info > generic ---
        special_considerations = kb_info.get(
            'special_needs_adaptations',
            ['Adapt based on individual student needs']
        )
        
        # --- Category: kb_info > domain label map > generic ---
        category = kb_info.get('category', None)
        if not category:
            category = self._resolve_category_from_labels(node)
        
        confidence = self._calculate_confidence(relevance_score, evidence_type, kb_info)
        
        return MethodologyRecommendation(
            name=name,
            category=category,
            relevance_score=relevance_score,
            evidence_type=evidence_type,
            implementation_guidance=implementation,
            classroom_applications=applications,
            special_considerations=special_considerations,
            confidence=confidence
        )
    
    def _resolve_category_from_labels(self, node: Dict) -> str:
        """Resolve a human-readable category from node labels using domain config.
        
        Falls back to first label name or generic 'Educational Methodology'.
        """
        labels = node.get('labels', [])
        if isinstance(labels, str):
            labels = [labels]
        
        # Try domain config label→category map
        domain = self.kb.domain
        try:
            domain_config = get_domain_config(domain)
            if domain_config:
                label_map = domain_config.get_label_category_map()
                if label_map:
                    for label in labels:
                        if label in label_map:
                            return label_map[label]
        except Exception:
            pass
        
        # Fallback: use the first label as-is (still better than generic)
        if labels:
            return labels[0]
        return 'Educational Methodology'
    
    def _calculate_relevance_score(self, node: Dict, query_metadata: Dict) -> float:
        """Calculate relevance score for a methodology"""
        base_score = 0.5
        
        # Boost for semantic similarity
        if node.get('source') == 'semantic':
            semantic_score = node.get('semantic_score', 0.5)
            base_score += semantic_score * 0.3
        
        # Boost for direct graph relationships
        if node.get('source') == 'graph' or node.get('rel_type'):
            base_score += 0.4
        
        # Boost for Node2Vec vector similarity
        if 'vector_similarity' in node:
            vector_score = node.get('vector_similarity', 0.0)
            base_score += vector_score * 0.2
        
        # Context-specific boosts (domain-aware)
        educational_context = query_metadata.get('educational_context', 'general')
        
        if educational_context == 'special_needs':
            # UDL domain: boost nodes with adaptation/inclusion keywords
            if any(adaptation in str(node).lower() 
                  for adaptation in ['inclusive', 'adaptive', 'support']):
                base_score += 0.1
        
        elif educational_context == 'neuroscience':
            # Neuro domain: boost nodes with cognitive/affective keywords
            if any(keyword in str(node).lower() 
                  for keyword in ['cognitive', 'memory', 'attention', 'emotion', 'motivation']):
                base_score += 0.1
        
        return min(base_score, 1.0)
    
    def _determine_evidence_type(self, node: Dict) -> str:
        """Determine the type of evidence supporting this recommendation"""
        if node.get('rel_type'):
            return 'direct_relationship'
        elif node.get('source') == 'semantic':
            return 'semantic_similarity'
        elif node.get('vector_similarity'):
            return 'vector_similarity'
        else:
            return 'domain_knowledge'
    
    def _calculate_confidence(self, relevance_score: float, evidence_type: str, kb_info: Dict) -> ConfidenceLevel:
        """Calculate confidence level for recommendation"""
        if relevance_score >= 0.8 and evidence_type == 'direct_relationship':
            return ConfidenceLevel.VERY_HIGH
        elif relevance_score >= 0.7:
            return ConfidenceLevel.HIGH
        elif relevance_score >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif relevance_score >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    # ========================================================================
    # PHASE 2: DYNAMIC BALANCING (Scalable, No Hardcoding)
    # ========================================================================
    
    def _detect_query_intent(self, query: str) -> Dict:
        """
        Detect if query is a comparison query (e.g., "A vs B", "difference between A and B").
        
        Uses word boundary matching to avoid false positives from substrings
        (e.g., "tra" inside "mostrano", "strategia").
        
        Returns:
            {'is_comparison': bool, 'comparison_keywords': list}
        """
        query_lower = query.lower()
        
        # Strong comparison signals (multi-word or unambiguous)
        strong_keywords = [
            'difference', 'differences', 'versus', 'compare', 'comparison',
            'contrast', 'compared to', 'different from', 'distinguish',
            'differenza', 'differenze', 'confronto', 'confrontare', 'rispetto a',
            'distinguere', 'distingue',
        ]
        
        # Short/ambiguous keywords requiring word boundary matching
        boundary_keywords = [
            'vs', 'differ', 'diverso', 'diversa', 'tra',
        ]
        
        found_keywords = []
        
        for kw in strong_keywords:
            if kw in query_lower:
                found_keywords.append(kw)
        
        for kw in boundary_keywords:
            if re.search(rf'\b{re.escape(kw)}\b', query_lower):
                found_keywords.append(kw)
        
        is_comparison = len(found_keywords) > 0
        
        if is_comparison:
            logger.info(f"[Intent Detection] Comparison query detected (keywords: {found_keywords})")
        
        return {
            'is_comparison': is_comparison,
            'comparison_keywords': found_keywords
        }
    
    def _apply_dynamic_balancing(self, nodes: List[Dict], target_size: int = 15) -> List[Dict]:
        """
        Apply dynamic balancing to ensure fair representation of different label groups.
        
        SCALABLE SOLUTION:
        - NO hardcoded label pairs
        - Works for ANY labels in ANY domain
        - Auto-detects imbalanced distribution (needs balancing)
        - Distributes target_size across top labels fairly
        
        Algorithm:
        1. Analyze label distribution (count nodes per label)
        2. Check if balancing is needed (comparison query with 2+ labels)
        3. If needed → balance by distributing slots fairly
        4. If not needed → use standard ranking
        
        Args:
            nodes: List of valid methodology nodes
            target_size: Target number of nodes to return (default: 15)
            
        Returns:
            Balanced list of nodes (max target_size nodes)
        """
        if not nodes:
            return []
        
        # Step 1: Analyze label distribution
        label_distribution = self._analyze_label_distribution(nodes)
        
        # Step 2: Check if balancing is needed
        # For comparison queries, we ALWAYS want balanced representation if 2+ labels
        top_labels = sorted(label_distribution.items(), key=lambda x: x[1], reverse=True)
        
        if len(top_labels) < 2:
            # Only 1 label type, no balancing needed
            logger.info(f"[Smart Ranking] Only 1 label type, using standard ranking")
            return sorted(nodes, key=lambda n: n.get('rank_score', 0), reverse=True)[:target_size]
        
        # Step 3: Apply balancing (for ALL comparison queries with 2+ labels)
        logger.info(f"[Smart Ranking] Applying balancing for {len(top_labels)} labels")
        balanced_nodes = self._balance_by_labels(nodes, label_distribution, target_size)
        
        return balanced_nodes
    
    def _analyze_label_distribution(self, nodes: List[Dict]) -> Dict[str, int]:
        """
        Count nodes per label (works for ANY labels, not hardcoded).
        
        Returns:
            {'IntrinsicMotivation': 20, 'ExtrinsicMotivation': 5, ...}
        """
        distribution = {}
        
        for node in nodes:
            labels = node.get('labels', [])
            if isinstance(labels, str):
                labels = [labels]
            
            for label in labels:
                if label:  # Skip empty labels
                    distribution[label] = distribution.get(label, 0) + 1
        
        logger.info(f"[Smart Ranking] Label distribution: {distribution}")
        return distribution
    
    def _detect_dominant_labels(self, distribution: Dict[str, int], threshold: float = 0.7) -> List[str]:
        """
        Detect labels that dominate >70% of nodes (unbalanced).
        
        Scalable: Works for any label names, any domain.
        
        Args:
            distribution: Label counts
            threshold: Dominance threshold (default: 0.7 = 70%)
            
        Returns:
            List of dominant label names
        """
        if not distribution:
            return []
        
        total = sum(distribution.values())
        dominant = []
        
        for label, count in distribution.items():
            ratio = count / total
            if ratio > threshold:
                dominant.append(label)
                logger.info(
                    f"[Smart Ranking] Dominant label detected: {label} "
                    f"({count}/{total} = {ratio:.0%})"
                )
        
        return dominant
    
    def _balance_by_labels(
        self, 
        nodes: List[Dict], 
        distribution: Dict[str, int], 
        target_size: int = 15
    ) -> List[Dict]:
        """
        Dynamic balancing: Distribute target_size across top N labels.
        
        Scalable formula:
        - If 2 labels: 7/7/1 split (balanced)
        - If 3 labels: 5/5/5 split
        - If 4+ labels: Equal distribution
        
        NO hardcoded label names!
        
        Args:
            nodes: List of nodes to balance
            distribution: Label counts
            target_size: Target number of nodes (default: 15)
            
        Returns:
            Balanced list of nodes
        """
        # Get top labels by count
        sorted_labels = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        top_labels = [label for label, count in sorted_labels if count > 0]
        
        if len(top_labels) <= 1:
            # Only one label type, standard ranking
            logger.info(f"[Smart Ranking] Only 1 label type, using standard ranking")
            return sorted(nodes, key=lambda n: n.get('rank_score', 0), reverse=True)[:target_size]
        
        # Calculate slots per label (dynamic!)
        if len(top_labels) == 2:
            slots_per_label = [7, 7, 1]  # 7+7+1 = 15 (2 main + 1 other)
            logger.info(f"[Smart Ranking] 2 labels detected → 7/7/1 split")
        elif len(top_labels) == 3:
            slots_per_label = [5, 5, 5]  # 5+5+5 = 15
            logger.info(f"[Smart Ranking] 3 labels detected → 5/5/5 split")
        else:
            # 4+ labels: distribute equally
            base_slots = target_size // len(top_labels)
            slots_per_label = [base_slots] * len(top_labels)
            logger.info(f"[Smart Ranking] {len(top_labels)} labels detected → equal distribution")
        
        # Group nodes by label
        nodes_by_label = {label: [] for label in top_labels}
        other_nodes = []
        
        for node in nodes:
            node_labels = node.get('labels', [])
            if isinstance(node_labels, str):
                node_labels = [node_labels]
            
            matched = False
            for label in top_labels:
                if label in node_labels:
                    nodes_by_label[label].append(node)
                    matched = True
                    break
            
            if not matched:
                other_nodes.append(node)
        
        # Rank within each label group and take top N slots
        balanced_results = []
        
        for i, label in enumerate(top_labels[:len(slots_per_label)]):
            # Sort by rank_score within this label group
            label_nodes = sorted(
                nodes_by_label[label],
                key=lambda n: n.get('rank_score', 0),
                reverse=True
            )
            
            # Take top N slots for this label
            slots = slots_per_label[i] if i < len(slots_per_label) else 0
            selected = label_nodes[:slots]
            balanced_results.extend(selected)
            
            logger.info(
                f"[Smart Ranking] Label '{label}': {len(label_nodes)} nodes → selected top {len(selected)}"
            )
        
        # Fill remaining slots with other nodes (if any)
        remaining_slots = target_size - len(balanced_results)
        if remaining_slots > 0 and other_nodes:
            other_sorted = sorted(other_nodes, key=lambda n: n.get('rank_score', 0), reverse=True)
            extra = other_sorted[:remaining_slots]
            balanced_results.extend(extra)
            logger.info(f"[Smart Ranking] Added {len(extra)} 'other' nodes to fill remaining slots")
        
        # Log final distribution
        final_distribution = self._analyze_label_distribution(balanced_results)
        logger.info(
            f"[Smart Ranking] Final balanced distribution ({len(balanced_results)} nodes): "
            f"{final_distribution}"
        )
        
        return balanced_results
    
    def _interleave_nodes_by_label(self, nodes: List[Dict]) -> List[Dict]:
        """
        Round-robin interleave nodes from different primary label groups.
        
        Ensures the downstream primary/supporting split (in build_context)
        contains balanced representation from all label groups, not just
        the highest-scored one.
        
        Within each group, nodes are ordered by rank_score (best first).
        
        Example with 7 ExtrinsicMotivation + 6 IntrinsicMotivation:
          Before (score sort): [Ext, Ext, Ext, Ext, Ext, Int, Int, Int, ...]
          After  (interleave): [Ext, Int, Ext, Int, Ext, Int, Ext, Int, ...]
        
        This way, slicing [:5] for primary gives [Ext, Int, Ext, Int, Ext]
        instead of [Ext, Ext, Ext, Ext, Ext].
        """
        if not nodes:
            return []
        
        groups: Dict[str, List[Dict]] = {}
        for node in nodes:
            labels = node.get('labels', [])
            if isinstance(labels, str):
                labels = [labels]
            primary_label = labels[0] if labels else 'Unknown'
            if primary_label not in groups:
                groups[primary_label] = []
            groups[primary_label].append(node)
        
        if len(groups) <= 1:
            return nodes
        
        for key in groups:
            groups[key].sort(key=lambda n: n.get('rank_score', 0), reverse=True)
        
        result: List[Dict] = []
        group_lists = list(groups.values())
        max_len = max(len(g) for g in group_lists)
        
        for i in range(max_len):
            for group in group_lists:
                if i < len(group):
                    result.append(group[i])
        
        logger.info(
            f"[Smart Ranking] Interleaved {len(nodes)} nodes across "
            f"{len(groups)} label groups: {list(groups.keys())}"
        )
        
        return result

class EvidenceSynthesizer:
    """Synthesizes evidence from graph relationships and semantic similarities"""
    
    def synthesize_evidence(self, triples: List[Dict], nodes: List[Dict]) -> str:
        """Create evidence summary from relationships and nodes"""
        evidence_parts = []
        
        # Analyze direct relationships
        relationship_evidence = self._analyze_relationships(triples)
        if relationship_evidence:
            evidence_parts.append(f"Direct pedagogical evidence: {relationship_evidence}")
        
        # Analyze semantic similarities
        semantic_evidence = self._analyze_semantic_nodes(nodes)
        if semantic_evidence:
            evidence_parts.append(f"Semantic analysis: {semantic_evidence}")
        
        # Analyze vector similarities
        vector_evidence = self._analyze_vector_similarities(nodes)
        if vector_evidence:
            evidence_parts.append(f"Conceptual similarity: {vector_evidence}")
        
        if not evidence_parts:
            return "Recommendations based on general pedagogical principles and domain expertise."
        
        return " | ".join(evidence_parts)
    
    def _analyze_relationships(self, triples: List[Dict]) -> str:
        """Analyze direct graph relationships"""
        if not triples:
            return ""
        
        suggests_count = sum(1 for t in triples if 'SUGGESTS' in t.get('relationship', ''))
        applies_count = sum(1 for t in triples if 'APPLIES_TO' in t.get('relationship', ''))
        
        parts = []
        if suggests_count:
            parts.append(f"{suggests_count} direct methodology suggestions")
        if applies_count:
            parts.append(f"{applies_count} application contexts")
        
        return ", ".join(parts)
    
    def _analyze_semantic_nodes(self, nodes: List[Dict]) -> str:
        """Analyze semantic similarity nodes"""
        semantic_nodes = [n for n in nodes if n.get('source') == 'semantic']
        if not semantic_nodes:
            return ""
        
        return f"Found {len(semantic_nodes)} semantically related educational concepts"
    
    def _analyze_vector_similarities(self, nodes: List[Dict]) -> str:
        """Analyze vector similarity evidence"""
        vector_nodes = [n for n in nodes if 'vector_similarity' in n]
        if not vector_nodes:
            return ""
        
        avg_similarity = sum(n.get('vector_similarity', 0) for n in vector_nodes) / len(vector_nodes)
        return f"Average conceptual similarity of {avg_similarity:.2f} across {len(vector_nodes)} related concepts"

class EducationalContextBuilder:
    """Main context builder that orchestrates the transformation"""
    
    def __init__(self, domain: str = "udl"):
        """Initialize context builder with domain-specific knowledge
        
        Args:
            domain: Domain filter ('udl', 'neuro', 'all')
        """
        self.domain = domain
        self.knowledge_base = PedagogicalKnowledgeBase(domain=domain)
        self.methodology_ranker = MethodologyRanker(self.knowledge_base)
        self.evidence_synthesizer = EvidenceSynthesizer()
    
    async def build_context(
        self, 
        retrieval_result: Dict, 
        original_query: str, 
        query_metadata: Dict,
        max_methodologies: int = 10
    ) -> EducationalContext:
        """Build comprehensive educational context from retrieval results.
        
        Args:
            max_methodologies: Total methodologies to surface (split ~50/50 primary/supporting).
                              Default 10 = 5 primary + 5 supporting.
        """
        
        try:
            logger.info(f"Building context for query: {original_query[:50]}...")
            
            # Extract components
            nodes = retrieval_result.get('nodes', [])
            triples = retrieval_result.get('triples', [])
            metadata = retrieval_result.get('metadata', {})
            
            # Build student profile
            student_profile = self._build_student_profile(original_query, query_metadata, nodes)
            
            # Rank methodologies
            all_recommendations = self.methodology_ranker.rank_methodologies(nodes, query_metadata)
            
            # Dynamic split: ~50/50 primary/supporting based on max_methodologies
            max_primary = max(3, (max_methodologies + 1) // 2)  # At least 3, ceil half
            max_supporting = max(2, max_methodologies - max_primary)  # Remainder
            primary_methodologies = all_recommendations[:max_primary]
            supporting_methodologies = all_recommendations[max_primary:max_primary + max_supporting]
            
            # Synthesize evidence
            evidence_summary = self.evidence_synthesizer.synthesize_evidence(triples, nodes)
            
            # Determine implementation priority
            implementation_priority = self._determine_implementation_priority(
                primary_methodologies, student_profile
            )
            
            # Calculate overall confidence
            confidence_assessment = self._calculate_overall_confidence(
                all_recommendations, len(triples), metadata
            )
            
            # Get fallback strategies if needed (pass metadata for data gap detection)
            fallback_strategies = self._get_fallback_strategies(
                confidence_assessment, student_profile, metadata
            )
            
            context = EducationalContext(
                student_profile=student_profile,
                primary_methodologies=primary_methodologies,
                supporting_methodologies=supporting_methodologies,
                evidence_summary=evidence_summary,
                implementation_priority=implementation_priority,
                confidence_assessment=confidence_assessment,
                fallback_strategies=fallback_strategies,
                metadata={
                    'total_nodes': len(nodes),
                    'total_triples': len(triples),
                    'semantic_nodes': metadata.get('semantic_count', 0),
                    'graph_nodes': metadata.get('graph_count', 0),
                    'original_query': original_query,
                    'query_type': query_metadata.get('educational_context', 'general')
                }
            )
            
            logger.info(f"Context built successfully with {len(primary_methodologies)} primary recommendations")
            return context
            
        except Exception as e:
            logger.error(f"Error building educational context: {e}")
            return self._create_fallback_context(original_query, query_metadata)
    
    def _build_student_profile(self, query: str, metadata: Dict, nodes: List[Dict]) -> StudentProfile:
        """Build student profile from query and context"""
        
        # Extract needs from query and nodes
        primary_needs = []
        secondary_needs = []
        
        # Map Italian terms to educational needs
        query_lower = query.lower()
        for term, needs_data in self.knowledge_base.special_needs_mapping.items():
            # Handle both dict format (new domain configs) and list format (legacy)
            if isinstance(needs_data, dict):
                needs_list = needs_data.get('support_needs', [])
            else:
                needs_list = needs_data

            if any(need_keyword in query_lower for need_keyword in [
                'ipovedenti', 'ciechi', 'blind',
                'sord', 'deaf', 'uditiv',
                'disabilità fisica', 'physical',
                'cognitive', 'cognitiv',
                'adhd', 'attenzione', 'attention',
                'autis', 'spettro',
                'motivazione', 'motivation',
                'dislessia', 'dyslexia',
                'discalculia', 'dyscalculia',
                'gifted', 'plusdotazione',
                'stranieri', 'foreign'
            ]):
                if term.lower() in query_lower or any(alt in query_lower for alt in needs_list):
                    primary_needs.extend(needs_list[:1])
                    secondary_needs.extend(needs_list[1:])
        
        # Extract from node names
        for node in nodes:
            node_name = node.get('name', '').lower()
            for term, needs_data in self.knowledge_base.special_needs_mapping.items():
                if isinstance(needs_data, dict):
                    needs_list = needs_data.get('support_needs', [])
                else:
                    needs_list = needs_data
                if term.lower() in node_name:
                    if needs_list and needs_list[0] not in primary_needs:
                        primary_needs.append(needs_list[0])
        
        # Determine educational context
        educational_context = metadata.get('educational_context', 'general')
        if any(term in query_lower for term in ['special', 'disabilità', 'difficoltà', 'bisogni']):
            educational_context = 'special_needs'
        elif any(re.search(r'\b' + re.escape(term), query_lower) for term in
                 ['verific', 'valut', 'test', 'esam', 'quiz', 'interroga']):
            educational_context = 'assessment'
        
        return StudentProfile(
            primary_needs=list(set(primary_needs)),
            secondary_needs=list(set(secondary_needs)),
            educational_context=educational_context,
            grade_level=None,  # Could be extracted if available
            subject_area=None   # Could be extracted if available
        )
    
    def _determine_implementation_priority(
        self, 
        methodologies: List[MethodologyRecommendation], 
        student_profile: StudentProfile
    ) -> List[str]:
        """Determine implementation priority order"""
        
        if not methodologies:
            return ["Consult with educational specialists for personalized recommendations"]
        
        priority_order = []
        
        # High-confidence methodologies first
        high_confidence = [m for m in methodologies if m.confidence in [ConfidenceLevel.VERY_HIGH, ConfidenceLevel.HIGH]]
        if high_confidence:
            priority_order.append(f"Start with {high_confidence[0].name} (high confidence)")
        
        # Special needs considerations
        if student_profile.educational_context == 'special_needs':
            priority_order.append("Ensure accessibility accommodations are in place")
            priority_order.append("Begin with small-group implementation")
        
        # General implementation advice
        priority_order.extend([
            "Pilot with a subset of students first",
            "Gather feedback and adjust based on student response",
            "Gradually expand implementation across all relevant contexts"
        ])
        
        return priority_order
    
    def _calculate_overall_confidence(
        self, 
        recommendations: List[MethodologyRecommendation], 
        triple_count: int, 
        metadata: Dict
    ) -> ConfidenceLevel:
        """Calculate overall confidence in recommendations"""
        
        if not recommendations:
            return ConfidenceLevel.VERY_LOW
        
        # Average confidence of recommendations
        confidence_scores = {
            ConfidenceLevel.VERY_HIGH: 5,
            ConfidenceLevel.HIGH: 4,
            ConfidenceLevel.MEDIUM: 3,
            ConfidenceLevel.LOW: 2,
            ConfidenceLevel.VERY_LOW: 1
        }
        
        avg_confidence = sum(confidence_scores[r.confidence] for r in recommendations) / len(recommendations)
        
        # Boost for relationship evidence
        if triple_count > 0:
            avg_confidence += 0.5
        
        # Boost for semantic/vector evidence
        semantic_count = metadata.get('semantic_count', 0)
        if semantic_count > 5:
            avg_confidence += 0.3
        
        # Map back to confidence levels
        if avg_confidence >= 4.5:
            return ConfidenceLevel.VERY_HIGH
        elif avg_confidence >= 3.5:
            return ConfidenceLevel.HIGH
        elif avg_confidence >= 2.5:
            return ConfidenceLevel.MEDIUM
        elif avg_confidence >= 1.5:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _get_fallback_strategies(
        self, 
        confidence: ConfidenceLevel, 
        student_profile: StudentProfile,
        metadata: Dict = None
    ) -> List[str]:
        """Get fallback strategies based on confidence and context
        
        Args:
            confidence: Confidence level
            student_profile: Student profile
            metadata: Optional metadata with data gap indicators
        """
        
        fallbacks = []
        
        # Check for data gaps (Priority 4: Honest messaging)
        if metadata:
            used_fallback = metadata.get('used_fallback', False)
            has_contamination = metadata.get('has_udl_contamination', False)
            
            if used_fallback:
                if self.domain == "neuro":
                    fallbacks.append(
                        "⚠️ NOTA: Il knowledge graph di neuroscienze contiene principalmente concetti teorici. "
                        "La relazione specifica richiesta non è presente nei dati, ma vengono forniti concetti correlati. "
                        "Per espandere il grafo con più relazioni, consulta il team di neuroscienze."
                    )
                else:
                    fallbacks.append(
                        "⚠️ NOTA: I dati specifici richiesti non sono presenti nel knowledge graph. "
                        "Le raccomandazioni si basano su concetti correlati disponibili."
                    )
            
            if has_contamination and self.domain == "neuro":
                fallbacks.append(
                    "ℹ️ SUGGERIMENTO: Per strategie didattiche pratiche, seleziona il dominio 'UDL (Universal Design for Learning)' "
                    "o 'All Domains' per raccomandazioni integrate."
                )
        
        if confidence in [ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW]:
            fallbacks.extend(self.knowledge_base.fallback_strategies['low_confidence'])
        
        if not student_profile.primary_needs:
            fallbacks.extend(self.knowledge_base.fallback_strategies['no_results'])
        
        # Context-specific fallbacks
        if student_profile.educational_context == 'special_needs':
            fallbacks.extend([
                "Consult Individualized Education Program (IEP) if available",
                "Consider assistive technology options",
                "Collaborate with special education support team"
            ])
        
        return list(set(fallbacks))  # Remove duplicates
    
    def _create_fallback_context(self, query: str, metadata: Dict) -> EducationalContext:
        """Create a fallback context when primary building fails"""

        if self.domain == "neuro":
            fallback_methodologies = [
                MethodologyRecommendation(
                    name="Evidence-Based Neuroscience Principles",
                    category="Cognitive Science",
                    relevance_score=0.6,
                    evidence_type="domain_knowledge",
                    implementation_guidance="Apply neuroscience-informed teaching strategies",
                    classroom_applications=["Cognitive load management", "Attention regulation", "Metacognitive scaffolding"],
                    special_considerations=["Evidence-based instructional design", "Brain-compatible learning environments"],
                    confidence=ConfidenceLevel.MEDIUM
                )
            ]
        else:
            fallback_methodologies = [
                MethodologyRecommendation(
                    name="Universal Design for Learning",
                    category="Inclusive Pedagogy",
                    relevance_score=0.6,
                    evidence_type="domain_knowledge",
                    implementation_guidance="Apply UDL principles to make learning accessible to all students",
                    classroom_applications=["Multiple means of representation", "Multiple means of engagement", "Multiple means of expression"],
                    special_considerations=["Flexible content delivery", "Choice in learning activities", "Varied assessment methods"],
                    confidence=ConfidenceLevel.MEDIUM
                )
            ]
        
        return EducationalContext(
            student_profile=StudentProfile(
                primary_needs=["general_support"],
                secondary_needs=[],
                educational_context=metadata.get('educational_context', 'general'),
                grade_level=None,
                subject_area=None
            ),
            primary_methodologies=fallback_methodologies,
            supporting_methodologies=[],
            evidence_summary="Fallback recommendations based on general pedagogical principles",
            implementation_priority=["Apply universal design principles", "Consult with educational specialists"],
            confidence_assessment=ConfidenceLevel.LOW,
            fallback_strategies=self.knowledge_base.fallback_strategies['no_results'],
            metadata={'fallback': True, 'original_query': query}
        )

# Example usage
if __name__ == "__main__":
    import asyncio
    import logging
    import json
    from dataclasses import asdict
    
    async def test_context_builder():
        builder = EducationalContextBuilder()
        
        # Mock retrieval result
        mock_result = {
            'nodes': [
                {
                    'name': 'Cooperative Learning',
                    'labels': ['PedagogicalMethodology'],
                    'source': 'graph',
                    'rel_type': 'SUGGESTS'
                },
                {
                    'name': 'Flipped Classroom', 
                    'labels': ['PedagogicalMethodology'],
                    'source': 'semantic',
                    'semantic_score': 0.8
                }
            ],
            'triples': [
                {'relationship': 'SUGGESTS', 'source': 'Blind', 'target': 'Cooperative Learning'}
            ],
            'metadata': {'semantic_count': 1, 'graph_count': 1}
        }
        
        context = await builder.build_context(
            mock_result,
            "Il mio studente ha l'ADHD, cosa posso fare?",
            {'educational_context': 'special_needs'}
        )
        
        # Pretty-print the full context as JSON (ideal output shape)
        print("\n=== EDUCATIONAL CONTEXT (Full JSON) ===")
        print(json.dumps(asdict(context), indent=2, default=str))
        
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_context_builder())
