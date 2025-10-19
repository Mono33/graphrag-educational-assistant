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
    
    def __init__(self):
        self.methodology_categories = {
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
        
        self.special_needs_mapping = {
            'Blind': ['visual_impairment', 'tactile_learning', 'audio_support'],
            'Deaf': ['hearing_impairment', 'visual_learning', 'sign_language'],
            'Physical disability': ['mobility_accommodation', 'assistive_technology', 'environmental_modification'],
            'Cognitive disability': ['cognitive_support', 'simplified_instruction', 'repetition'],
            'Adhd': ['attention_management', 'movement_breaks', 'structured_environment'],
            'Attention Deficit': ['focus_strategies', 'clear_instructions', 'minimal_distractions'],
            'Autism spectrum disorder': ['routine_structure', 'sensory_considerations', 'social_support'],
            'NoPersonalMotivation': ['engagement_strategies', 'relevance_connection', 'choice_provision']
        }
        
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
    """Ranks and prioritizes educational methodologies"""
    
    def __init__(self, knowledge_base: PedagogicalKnowledgeBase):
        self.kb = knowledge_base
    
    def rank_methodologies(self, nodes: List[Dict], query_metadata: Dict) -> List[MethodologyRecommendation]:
        """Rank methodologies based on relevance and evidence"""
        recommendations = []
        
        for node in nodes:
            if self._is_methodology(node):
                recommendation = self._create_recommendation(node, query_metadata)
                if recommendation:
                    recommendations.append(recommendation)
        
        # Sort by relevance score (descending)
        recommendations.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return recommendations
    
    def _is_methodology(self, node: Dict) -> bool:
        """Check if node represents a pedagogical methodology"""
        labels = node.get('labels', [])
        return any(label in ['PedagogicalMethodology', 'TeachingApproach', 'LearningStrategy'] 
                  for label in labels)
    
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
        
        # Context-specific boosts
        if query_metadata.get('educational_context') == 'special_needs':
            if any(adaptation in str(node).lower() 
                  for adaptation in ['inclusive', 'adaptive', 'support']):
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
    
    def __init__(self):
        self.knowledge_base = PedagogicalKnowledgeBase()
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
            
            # Get fallback strategies if needed
            fallback_strategies = self._get_fallback_strategies(
                confidence_assessment, student_profile
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
        student_profile: StudentProfile
    ) -> List[str]:
        """Get fallback strategies based on confidence and context"""
        
        fallbacks = []
        
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
