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
        
        # UDL-specific methodology categories
        self.udl_methodology_categories = {
            'Cooperative Learning': {
                'category': 'Collaborative Pedagogy',
                'best_for': ['social_interaction', 'peer_learning', 'inclusion'],
                'implementation': 'Organize students in diverse groups of 3-5 members',
                'applications': [
                    'Jigsaw method for complex topics',
                    'Think-Pair-Share for quick engagement',
                    'Group investigations for project work'
                ],
                'special_needs_adaptations': [
                    'Assign complementary roles based on abilities',
                    'Provide visual and verbal instructions',
                    'Use peer tutoring for support'
                ]
            },
            'Flipped Classroom': {
                'category': 'Blended Learning',
                'best_for': ['self_paced_learning', 'active_classroom_time', 'differentiation'],
                'implementation': 'Pre-recorded content at home, active learning in class',
                'applications': [
                    'Video lectures for concept introduction',
                    'Interactive activities during class time',
                    'Personalized learning paths'
                ],
                'special_needs_adaptations': [
                    'Closed captions for hearing impaired',
                    'Audio descriptions for visually impaired',
                    'Flexible pacing for cognitive disabilities'
                ]
            },
            'Project based learning': {
                'category': 'Constructivist Pedagogy',
                'best_for': ['real_world_application', 'creativity', 'problem_solving'],
                'implementation': 'Long-term projects addressing real-world problems',
                'applications': [
                    'Community service projects',
                    'Scientific investigations',
                    'Creative multimedia presentations'
                ],
                'special_needs_adaptations': [
                    'Break projects into manageable steps',
                    'Provide multiple means of expression',
                    'Offer choice in topics and formats'
                ]
            },
            'Station Rotation': {
                'category': 'Differentiated Instruction',
                'best_for': ['varied_learning_styles', 'small_group_instruction', 'skill_building'],
                'implementation': 'Multiple learning stations with different activities',
                'applications': [
                    'Skills practice stations',
                    'Technology integration stations',
                    'Teacher-led instruction station'
                ],
                'special_needs_adaptations': [
                    'Adaptive technology stations',
                    'Sensory-friendly environments',
                    'Modified task complexity'
                ]
            }
        }
        
        # Neuro-specific methodology/concept categories
        self.neuro_methodology_categories = {
            'Working Memory': {
                'category': 'Cognitive Process',
                'best_for': ['information_retention', 'task_completion', 'learning_efficiency'],
                'implementation': 'Minimize cognitive load, chunk information, use rehearsal strategies',
                'applications': [
                    'Reduce distractions during instruction',
                    'Break tasks into smaller steps',
                    'Use visual aids to support verbal information'
                ],
                'special_needs_adaptations': [
                    'Provide written instructions alongside verbal',
                    'Allow extra processing time',
                    'Use multimodal presentation'
                ]
            },
            'Attention': {
                'category': 'Cognitive Process',
                'best_for': ['focus', 'concentration', 'task_engagement'],
                'implementation': 'Manage attentional resources, minimize distractions, vary stimuli',
                'applications': [
                    'Use attention-grabbing signals',
                    'Implement focused work periods',
                    'Vary teaching methods to maintain engagement'
                ],
                'special_needs_adaptations': [
                    'Provide movement breaks',
                    'Use fidget tools appropriately',
                    'Structure environment to reduce distractions'
                ]
            },
            'Executive Functions': {
                'category': 'Cognitive Control',
                'best_for': ['planning', 'organization', 'self_regulation'],
                'implementation': 'Teach explicit strategies, provide scaffolding, model processes',
                'applications': [
                    'Use planning templates',
                    'Teach self-monitoring strategies',
                    'Implement goal-setting practices'
                ],
                'special_needs_adaptations': [
                    'Provide external organization systems',
                    'Use checklists and visual schedules',
                    'Break long-term projects into milestones'
                ]
            },
            'Emotions': {
                'category': 'Affective Process',
                'best_for': ['motivation', 'engagement', 'learning_climate'],
                'implementation': 'Create positive emotional climate, recognize emotions in learning',
                'applications': [
                    'Build growth mindset',
                    'Celebrate progress and effort',
                    'Create psychologically safe environment'
                ],
                'special_needs_adaptations': [
                    'Teach emotional regulation strategies',
                    'Provide safe spaces for emotional processing',
                    'Use mindfulness techniques'
                ]
            },
            'Motivation': {
                'category': 'Affective Process',
                'best_for': ['engagement', 'persistence', 'goal_pursuit'],
                'implementation': 'Foster intrinsic motivation, provide autonomy and choice',
                'applications': [
                    'Connect learning to student interests',
                    'Offer choices in assignments',
                    'Provide meaningful feedback'
                ],
                'special_needs_adaptations': [
                    'Use individualized reward systems',
                    'Break goals into achievable steps',
                    'Highlight personal growth'
                ]
            }
        }
        
        # Select methodology categories based on domain
        if domain == "neuro":
            self.methodology_categories = self.neuro_methodology_categories
        elif domain == "all":
            self.methodology_categories = {**self.udl_methodology_categories, **self.neuro_methodology_categories}
        else:  # default to UDL
            self.methodology_categories = self.udl_methodology_categories
        
        # UDL-specific special needs mapping
        self.udl_special_needs_mapping = {
            'Blind': ['visual_impairment', 'tactile_learning', 'audio_support'],
            'Deaf': ['hearing_impairment', 'visual_learning', 'sign_language'],
            'Physical disability': ['mobility_accommodation', 'assistive_technology', 'environmental_modification'],
            'Cognitive disability': ['cognitive_support', 'simplified_instruction', 'repetition'],
            'Adhd': ['attention_management', 'movement_breaks', 'structured_environment'],
            'Attention Deficit': ['focus_strategies', 'clear_instructions', 'minimal_distractions'],
            'Autism spectrum disorder': ['routine_structure', 'sensory_considerations', 'social_support'],
            'NoPersonalMotivation': ['engagement_strategies', 'relevance_connection', 'choice_provision']
        }
        
        # Neuro-specific concept mapping
        self.neuro_special_needs_mapping = {
            'Attention': ['focus_support', 'distraction_management', 'engagement_strategies'],
            'Working Memory': ['cognitive_load_reduction', 'chunking_strategies', 'rehearsal_support'],
            'Executive Functions': ['planning_support', 'organization_strategies', 'self_regulation_tools'],
            'Emotions': ['emotional_climate', 'motivation_enhancement', 'stress_management'],
            'Motivation': ['intrinsic_motivation', 'goal_setting', 'autonomy_support'],
            'Creativity': ['divergent_thinking', 'idea_generation', 'risk_taking_support'],
            'Critical Thinking': ['analytical_skills', 'problem_solving', 'metacognition'],
            'Memory': ['encoding_strategies', 'retrieval_practice', 'consolidation_support']
        }
        
        # Select special needs mapping based on domain
        if domain == "neuro":
            self.special_needs_mapping = self.neuro_special_needs_mapping
        elif domain == "all":
            self.special_needs_mapping = {**self.udl_special_needs_mapping, **self.neuro_special_needs_mapping}
        else:  # default to UDL
            self.special_needs_mapping = self.udl_special_needs_mapping
        
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
        Rank methodologies with dynamic balancing for comparison queries.
        
        PHASE 2 ENHANCEMENT: Scalable, no hardcoding, works for ANY labels.
        """
        # Step 1: Filter valid methodologies
        valid_nodes = []
        logger.info(f"[DEBUG] Ranking {len(nodes)} nodes for recommendations")
        
        for i, node in enumerate(nodes):
            logger.info(f"[DEBUG] Node {i+1}: name='{node.get('name', 'N/A')}', labels={node.get('labels', [])}, keys={list(node.keys())}")
            
            if self._is_methodology(node):
                logger.info(f"[DEBUG] Node {i+1} ACCEPTED as methodology")
                valid_nodes.append(node)
            else:
                logger.info(f"[DEBUG] Node {i+1} REJECTED - not a valid methodology")
        
        # Step 2: Apply dynamic balancing if needed (PHASE 2 - Scalable solution)
        query_intent = self._detect_query_intent(query_metadata.get('original_query', ''))
        
        if query_intent['is_comparison']:
            logger.info(f"[Smart Ranking] Comparison query detected, applying dynamic balancing...")
            balanced_nodes = self._apply_dynamic_balancing(valid_nodes, target_size=15)
        else:
            # Standard ranking (no balancing needed)
            balanced_nodes = valid_nodes
        
        # Step 3: Create recommendations
        recommendations = []
        for node in balanced_nodes:
            recommendation = self._create_recommendation(node, query_metadata)
            if recommendation:
                recommendations.append(recommendation)
        
        # Sort by relevance score (descending)
        recommendations.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.info(f"[DEBUG] Final recommendations: {len(recommendations)}")
        return recommendations
    
    def _is_methodology(self, node: Dict) -> bool:
        """Check if node represents a pedagogical methodology or neuroscience concept"""
        labels = node.get('labels', [])
        # UDL methodologies
        udl_labels = ['PedagogicalMethodology', 'TeachingApproach', 'LearningStrategy']
        # Neuro concepts (SYNCED with graph_retriever.py domain_boosts - Nov 2025)
        # Based on neuro_audit_report.json: 478 nodes, 195 unique labels
        neuro_labels = [
            # Core cognitive processes (most frequent)
            'Attention', 'CriticalThinking', 'ExtrinsicMotivation', 'ExecutiveFunctions',
            'IntrinsicMotivation', 'LearningOutcomes', 'TeachingPractices', 'LearningDevelopment',
            'NegativeStressDistress', 'Motivation',
            
            # Hub nodes (high connectivity)
            'CognitiveFlexibility', 'KnowledgeConstructionAttention', 'PrefrontalCortexActivation',
            'OptimalAttentionalNetworkActivation',
            
            # Authority nodes (key outcomes)
            'Creativity', 'Memory', 'MemoryEncoding', 'MemorySystems',
            
            # Critical cognitive
            'WorkingMemory', 'Metacognition', 'SelfRegulation', 'CognitiveControl', 'CognitiveProcesses',
            
            # Affective & motivational
            'EmotionalRegulation', 'EmotionalWellBeing', 'PositiveEmotions', 'NegativeEmotions',
            'AffectiveProcesses',
            
            # Mindset & growth
            'GrowthMindset', 'FixedMindset', 'Mindset',
            
            # Stress & coping
            'PositiveStressEustress', 'NegativeStressDistress', 'StressResponse',  # StressResponse = physiological stress
            'LongTermGrowth', 'LongTermDecline',
            'AdaptiveCoping', 'MaladaptiveCoping',
            
            # Social & communication
            'SocialCognition', 'SocialLearning', 'Communication',
            
            # Educational outcomes
            'LearningEngagement', 'LearningPerformance', 'EducationalSupport',
            
            # Additional important
            'HigherOrderThinking', 'LowerOrderThinking', 'ProblemSolving',
            'LongTermMemory', 'PersonalGrowth', 'Strengths', 'CognitiveStrengths',
            'ReflectiveThinking', 'Consolidation', 'MotivationalModulation',
            
            # Additional labels found during testing (Nov 2025)
            'BrainAdaptability',  # e.g., Neuroplasticity
            'Vulnerability',      # e.g., Learned Helplessness, Disengagement, Perfectionism
            'Resilience',         # exists as both category and label
            'CognitiveBias',      # e.g., Overconfidence Bias
            
            # Generic fallback
            'LearningProcess', 'Emotions', 'Concept'  # Concept = fallback for inferred labels
        ]
        return any(label in udl_labels + neuro_labels for label in labels)
    
    def _create_recommendation(self, node: Dict, query_metadata: Dict) -> Optional[MethodologyRecommendation]:
        """Create a methodology recommendation from a node"""
        name = node.get('name', '')
        if not name:
            return None
        
        # Get knowledge base info
        kb_info = self.kb.methodology_categories.get(name, {})
        
        # Calculate relevance score
        relevance_score = self._calculate_relevance_score(node, query_metadata)
        
        # Determine evidence type
        evidence_type = self._determine_evidence_type(node)
        
        # Get implementation guidance
        implementation = kb_info.get('implementation', f'Apply {name} methodology with appropriate adaptations')
        
        # Get classroom applications
        applications = kb_info.get('applications', [f'Implement {name} in classroom context'])
        
        # Get special considerations
        special_considerations = kb_info.get('special_needs_adaptations', ['Adapt based on individual student needs'])
        
        # Determine confidence
        confidence = self._calculate_confidence(relevance_score, evidence_type, kb_info)
        
        return MethodologyRecommendation(
            name=name,
            category=kb_info.get('category', 'Educational Methodology'),
            relevance_score=relevance_score,
            evidence_type=evidence_type,
            implementation_guidance=implementation,
            classroom_applications=applications,
            special_considerations=special_considerations,
            confidence=confidence
        )
    
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
        
        Scalable: Works for ANY comparison, not hardcoded for specific labels.
        
        Returns:
            {'is_comparison': bool, 'comparison_keywords': list}
        """
        query_lower = query.lower()
        
        # Comparison keywords (multilingual)
        comparison_keywords = [
            # English
            'difference', 'differences', 'vs', 'versus', 'compare', 'comparison',
            'contrast', 'compared to', 'different from', 'differ', 'distinguish',
            # Italian
            'differenza', 'differenze', 'confronto', 'confrontare', 'rispetto a',
            'diverso', 'diversa', 'distinguere', 'distingue', 'tra'
        ]
        
        found_keywords = [kw for kw in comparison_keywords if kw in query_lower]
        
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
        query_metadata: Dict
    ) -> EducationalContext:
        """Build comprehensive educational context from retrieval results"""
        
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
            
            # Split into primary and supporting
            primary_methodologies = all_recommendations[:3]  # Top 3
            supporting_methodologies = all_recommendations[3:6]  # Next 3
            
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
        for term, needs in self.knowledge_base.special_needs_mapping.items():
            if any(need_keyword in query_lower for need_keyword in [
                'ipovedenti', 'ciechi', 'blind',
                'sord', 'deaf', 'uditiv',
                'disabilità fisica', 'physical',
                'cognitive', 'cognitiv',
                'adhd', 'attenzione', 'attention',
                'autis', 'spettro',
                'motivazione', 'motivation'
            ]):
                if term.lower() in query_lower or any(alt in query_lower for alt in needs):
                    primary_needs.extend(needs[:1])  # Primary need
                    secondary_needs.extend(needs[1:])  # Secondary needs
        
        # Extract from node names
        for node in nodes:
            node_name = node.get('name', '').lower()
            for term, needs in self.knowledge_base.special_needs_mapping.items():
                if term.lower() in node_name:
                    if needs[0] not in primary_needs:
                        primary_needs.append(needs[0])
        
        # Determine educational context
        educational_context = metadata.get('educational_context', 'general')
        if any(term in query_lower for term in ['special', 'disabilità', 'difficoltà', 'bisogni']):
            educational_context = 'special_needs'
        elif any(term in query_lower for term in ['verific', 'valut', 'test', 'esam']):
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
