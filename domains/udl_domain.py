#!/usr/bin/env python3
"""
UDL Domain Configuration
All UDL-specific configuration extracted from codebase

This module contains ALL UDL (Universal Design for Learning) domain logic:
- Node2Vec weights for graph embedding
- Retrieval boosts for search prioritization
- Text2Cypher examples and patterns
- Italian→English term mapping
- LLM system prompts
- Methodology categories and special needs mapping
"""

from domains.base_config import BaseDomainConfig
from typing import Dict, List, Tuple
import re


class UDLDomainConfig(BaseDomainConfig):
    """UDL (Universal Design for Learning) domain configuration"""
    
    # ============================================================
    # CORE DOMAIN IDENTITY
    # ============================================================
    
    def _get_name(self) -> str:
        return "udl"
    
    def _get_display_name(self) -> str:
        return "UDL (Universal Design for Learning)"
    
    def _get_icon(self) -> str:
        return "🎯"
    
    # ============================================================
    # NODE2VEC WEIGHTS (from train_node2vec.py lines 52-65)
    # ============================================================
    
    def get_node2vec_weights(self) -> Dict[str, float]:
        """UDL Node2Vec label weights - prioritize student needs and methodologies"""
        return {
            'StudentWithSpecialNeeds': 3.0,
            'PedagogicalMethodology': 3.0,
            'StudentCharacteristic': 2.5,
            'Context': 2.0,
            'LearningResource': 2.0,
            'Lighting': 1.5,
            'Colour': 1.5,
            'Furniture': 1.5,
            'Acoustic': 1.5,
            'InteractiveBoard': 1.8,
            'EnvironmentalBarrier': 1.2,
            'EnvironmentalSupport': 1.2
        }
    
    # ============================================================
    # RETRIEVAL BOOSTS (from graph_retriever.py)
    # ============================================================
    
    def get_retrieval_boosts(self) -> Dict[str, Dict[str, float]]:
        """UDL retrieval boosts - EXACT COPY from graph_retriever_old.py lines 59-70"""
        return {
            'label_boosts': {
                # UDL domain labels (UNTOUCHED from old code)
                'StudentWithSpecialNeeds': 2.0,
                'PedagogicalMethodology': 2.0,
                'StudentCharacteristic': 1.5,
                'Context': 1.5,
                'Lighting': 1.2,
                'Colour': 1.2,
                'Furniture': 1.2,
                'Acoustic': 1.2,
                'InteractiveBoard': 1.3,
                'EnvironmentalBarrier': 1.1,
                'EnvironmentalSupport': 1.1
            },
            'relationship_boosts': {
                'SUGGESTS': 1.3,
                'NO_SUGGESTS': 1.2,
                'SUPPORTS': 1.25,
                'FACILITATES': 1.2
            }
        }
    
    def get_similarity_threshold(self) -> float:
        """UDL similarity threshold - higher (stricter) due to structured data"""
        return 0.80
    
    # ============================================================
    # TEXT2CYPHER EXAMPLES (from text2cypher.py lines 265-301)
    # ============================================================
    
    def get_few_shot_examples(self) -> str:
        """Get UDL-specific few-shot examples for Cypher generation
        
        EXACT COPY from old text2cypher.py lines 265-301 (_get_udl_examples method)
        
        Returns:
            String of examples in "Question: ... Cypher: ..." format
        """
        examples = """
Question: "What teaching methods help students with ADHD?"
Cypher: MATCH (s:StudentWithSpecialNeeds)-[r:SUGGESTS]->(m:PedagogicalMethodology) WHERE s.name = "Adhd" OR s.name = "Attention Deficit" RETURN m.name, m.category LIMIT 10

Question: "What strategies help blind students?"
Cypher: MATCH (s:StudentWithSpecialNeeds)-[r:SUGGESTS]->(m:PedagogicalMethodology) WHERE s.name = "Blind" RETURN m.name, m.category LIMIT 10

Question: "What approaches work for deaf students?"
Cypher: MATCH (s:StudentWithSpecialNeeds)-[r:SUGGESTS]->(m:PedagogicalMethodology) WHERE s.name = "Deaf" RETURN m.name, m.category LIMIT 10

Question: "What methods help students with autism spectrum disorder?"
Cypher: MATCH (s:StudentWithSpecialNeeds)-[r:SUGGESTS]->(m:PedagogicalMethodology) WHERE s.name = "Autism spectrum disorder" RETURN m.name, m.category LIMIT 10

Question: "What pedagogical approaches should be avoided for students with no personal motivation?"
Cypher: MATCH (s:StudentWithSpecialNeeds)-[r:NO_SUGGESTS]->(m:PedagogicalMethodology) WHERE s.name = "NoPersonalMotivation" RETURN m.name, m.category LIMIT 10

Question: "How many pedagogical methodologies are in the database?"
Cypher: MATCH (m:PedagogicalMethodology) RETURN COUNT(m) as methodology_count

Question: "What lighting conditions support learning focus?"
Cypher: MATCH (l:Lighting)-[r:SUPPORTS]->(p:LearningProcess) WHERE p.name = "focus" RETURN l.name, p.name LIMIT 10

Question: "What colors facilitate attention and relaxation?"
Cypher: MATCH (c:Colour)-[r:FACILITATES]->(r:LearnerResponse) WHERE toLower(r.name) CONTAINS "attention" RETURN c.name, r.name LIMIT 10

Question: "What methodologies help students with excellence in subjects?"
Cypher: MATCH (s:StudentCharacteristic)-[r:SUGGESTS]->(m:PedagogicalMethodology) WHERE s.name = "Excellence in some or all subjects" RETURN m.name, m.category LIMIT 10

Question: "What furniture causes discomfort and reduced focus?"
Cypher: MATCH (f:Furniture)-[r:CAUSES]->(e:EnvironmentalBarrier) WHERE toLower(e.name) CONTAINS "discomfort" RETURN f.name, e.name LIMIT 10

Question: "What methodologies work for cohesive classes?"
Cypher: MATCH (c:Context)-[r:SUGGESTS]->(m:PedagogicalMethodology) WHERE c.name = "Cohesive" RETURN m.name, m.category LIMIT 10
"""
        return examples.strip()
    
    def get_cypher_patterns(self) -> str:
        """UDL-specific Cypher patterns for system prompt"""
        return """
QUERY PATTERNS (UDL - Universal Design for Learning):
1. Student needs → Methodologies:
   MATCH (s:StudentWithSpecialNeeds)-[r:SUGGESTS]->(m:PedagogicalMethodology)
   WHERE s.name = "specific_need"
   
2. Methodologies to avoid:
   MATCH (s:StudentWithSpecialNeeds)-[r:NO_SUGGESTS]->(m:PedagogicalMethodology)
   
3. Environmental factors:
   MATCH (env:Lighting|Colour|Furniture|Acoustic)-[r]->(outcome)
   
4. Learning resources:
   MATCH (lr:LearningResource)-[r:SUPPORTS]->(m:PedagogicalMethodology)
   
5. Student characteristics:
   MATCH (sc:StudentCharacteristic)-[r:SUGGESTS]->(m:PedagogicalMethodology)
   
6. Context-specific methodologies:
   MATCH (c:Context)-[r:SUGGESTS]->(m:PedagogicalMethodology)

CRITICAL RULES:
- Use EXACT node names from database (case-sensitive!)
- StudentWithSpecialNeeds examples: "Adhd", "Blind", "Deaf", "Autism spectrum disorder"
- PedagogicalMethodology is the correct label (NOT PedagogicalStrategy)
- Relationships: SUGGESTS (recommended), NO_SUGGESTS (avoid), SUPPORTS, FACILITATES
- Always add domain filter: {domain: "udl"} for production queries
"""
    
    def repair_cypher_query(self, query: str) -> str:
        """UDL-specific Cypher repair logic (from text2cypher.py lines 563-798)"""
        
        # 1) Force default pattern for StudentWithSpecialNeeds
        query = re.sub(
            r'(\(s:StudentWithSpecialNeeds\)\s*-\s*\[r:\s*SUGGESTS\s*\]->\s*\()[a-zA-Z]*:PedagogicalStrategy(\))',
            r'\1m:PedagogicalMethodology\2',
            query
        )
        query = re.sub(
            r'(\(s:StudentWithSpecialNeeds\)\s*-\s*\[r:\s*NO_SUGGESTS\s*\]->\s*\()[a-zA-Z]*:PedagogicalStrategy(\))',
            r'\1m:PedagogicalMethodology\2',
            query
        )
        
        # 2) Fix case sensitivity issues
        if 'studentwithspecialneeds' in query.lower():
            query = query.replace('studentwithspecialneeds', 'StudentWithSpecialNeeds')
            query = query.replace('Studentwithspecialneeds', 'StudentWithSpecialNeeds')
            query = query.replace('STUDENTWITHSPECIALNEEDS', 'StudentWithSpecialNeeds')
        
        if 'pedagogicalmethodology' in query.lower():
            query = query.replace('pedagogicalmethodology', 'PedagogicalMethodology')
            query = query.replace('Pedagogicalmethodology', 'PedagogicalMethodology')
            query = query.replace('PEDAGOGICALMETHODOLOGY', 'PedagogicalMethodology')
        
        if 'pedagogicalstrategy' in query.lower():
            query = query.replace('PedagogicalStrategy', 'PedagogicalMethodology')
            query = query.replace('pedagogicalstrategy', 'PedagogicalMethodology')
            query = query.replace('Pedagogicalstrategy', 'PedagogicalMethodology')
        
        # 3) Fix StudentCharacteristic
        if 'studentcharacteristic' in query.lower():
            query = query.replace('studentcharacteristic', 'StudentCharacteristic')
            query = query.replace('Studentcharacteristic', 'StudentCharacteristic')
        
        # 4) Fix LearningResource
        if 'learningresource' in query.lower():
            query = query.replace('learningresource', 'LearningResource')
            query = query.replace('Learningresource', 'LearningResource')
        
        # 5) Fix EnvironmentalBarrier and EnvironmentalSupport
        if 'environmentalbarrier' in query.lower():
            query = query.replace('environmentalbarrier', 'EnvironmentalBarrier')
            query = query.replace('Environmentalbarrier', 'EnvironmentalBarrier')
        
        if 'environmentalsupport' in query.lower():
            query = query.replace('environmentalsupport', 'EnvironmentalSupport')
            query = query.replace('Environmentalsupport', 'EnvironmentalSupport')
        
        # 6) Fix InteractiveBoard
        if 'interactiveboard' in query.lower():
            query = query.replace('interactiveboard', 'InteractiveBoard')
            query = query.replace('Interactiveboard', 'InteractiveBoard')
        
        return query
    
    # ============================================================
    # MULTILINGUAL TERMS (from multilingual_text2cypher.py lines 30-154)
    # ============================================================
    
    def get_italian_terms(self) -> Dict[str, str]:
        """UDL Italian→English term mapping for query translation"""
        return {
            # Special Educational Needs - Map to actual StudentWithSpecialNeeds nodes
            "ipovedenti": "Blind",
            "disabilità uditive": "Deaf",
            "disabilità": "Physical disability",
            "disabilità fisica": "Physical disability",
            "dislessia": "Language difficulties due to foreign origin",
            "difficoltà linguistiche": "Language difficulties due to foreign origin",
            "ADHD": "Adhd",
            "deficit di attenzione": "Attention Deficit",
            "autismo": "Autism spectrum disorder",
            "disturbi dello spettro autistico": "Autism spectrum disorder",
            "motivazione": "NoPersonalMotivation",
            "senza motivazione": "NoPersonalMotivation",
            "eccellenza": "Excellence in some or all subjects",
            "difficoltà cognitive": "Cognitive disability [mild, moderate, severe]",
            "difficoltà cognitive lievi": "Cognitive disability [mild, moderate, severe]",
            "difficoltà cognitive moderate": "Cognitive disability [mild, moderate, severe]",
            "difficoltà cognitive gravi": "Cognitive disability [mild, moderate, severe]",
            "difficoltà di lettura": "reading difficulties",
            "iperattività": "Hyperactivity Disorder",
            "disturbo oppositivo": "Oppositional Defiant Disorder - ODD",
            "plusdotazione": "Giftedness",
            "muto": "Mute or no verbal",
            
            # UDL (Universal Design for Learning)
            "UDL": "Universal Design for Learning",
            "Universal Design for Learning": "Universal Design for Learning",
            "progettazione universale": "universal design",
            "linee guida UDL": "UDL guidelines",
            "principi UDL": "UDL principles",
            "strategie UDL": "UDL strategies",
            "tecniche UDL": "UDL techniques",
            "framework UDL": "UDL framework",
            
            # Teaching and Learning Methods - Map to actual PedagogicalMethodology nodes
            "apprendimento cooperativo": "Cooperative Learning",
            "flipped classroom": "Flipped Classroom",
            "game based learning": "GameBasedLearning",
            "debate": "Debate",
            "project based learning": "Project based learning",
            "role playing": "Role Playing, Debate",
            "station rotation": "Station Rotation",
            "stem": "Stem",
            "peertopeereducation": "Peertopeereducation",
            
            # Teaching Approaches - Map to actual TeachingApproach nodes
            "lezioni frontali": "Frontal lessons",
            "lezioni frontali lunghe": "long frontal lessons",
            "supporti visivi, alternative bilingue": "Visual supports, bilingual alternatives",
            
            # Generic terms (useful for query context)
            "obiettivi diversificati": "differentiated objectives",
            "obiettivi didattici": "learning objectives",
            "classe eterogenea": "heterogeneous class",
            "classe multilingue": "multilingual class",
            "risorse visive": "visual resources",
            "risorse analogiche": "analog resources",
            "supporti visivi": "visual supports",
            "supporti linguistici": "linguistic supports",
            "supporti motori": "motor supports",
            "attività": "activities",
            "partecipazione": "participation",
            "prerequisiti": "prerequisites",
            "metodologie": "methodologies",
            "strategie": "strategies",
            "strumenti tecnologici": "technological tools",
            "unità didattica": "didactic unit",
            "sequenza didattica": "didactic sequence",
            
            # Learning contexts
            "contesto educativo": "educational context",
            "contesto scolastico": "school context",
            "ambiente di apprendimento": "learning environment",
            "ambiente inclusivo": "inclusive environment",
            
            # Pedagogical terms
            "differenziazione": "differentiation",
            "personalizzazione": "personalization",
            "individualizzazione": "individualization",
            "inclusione": "inclusion",
            "accessibilità": "accessibility",
            "barriere": "barriers",
            "facilitatori": "facilitators"
        }
    
    def get_query_context(self) -> str:
        """UDL query context for multilingual enhancement"""
        return "education and Universal Design for Learning"
    
    # ============================================================
    # LLM CHAIN PROMPTS (from llm_chain.py lines 62-64)
    # ============================================================
    
    def get_system_prompt(self) -> str:
        """UDL system prompt for LLM response generation"""
        return """Sei un esperto consulente pedagogico italiano specializzato in metodologie didattiche inclusive e differenziate, basate sui principi dell'Universal Design for Learning (UDL).

Il tuo compito è fornire raccomandazioni chiare, pratiche e pedagogicamente solide per insegnanti italiani, garantendo accessibilità e inclusione per tutti gli studenti."""
    
    # NOTE: get_response_template() uses the default from BaseDomainConfig
    # which provides the current generic 8-point formatting instructions.
    # Override here when UDL-specific response structure is needed.
    
    # ============================================================
    # CONTEXT BUILDER - METHODOLOGY CATEGORIES
    # (from context_builder.py lines 68-153)
    # ============================================================
    
    def get_methodology_categories(self) -> Dict:
        """UDL methodology categories with implementation guidance"""
        return {
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
                'best_for': ['deep_learning', 'critical_thinking', 'real_world_skills'],
                'implementation': 'Students work on extended projects addressing real problems',
                'applications': [
                    'Research projects with presentation',
                    'Design challenges',
                    'Community service projects'
                ],
                'special_needs_adaptations': [
                    'Scaffold project steps clearly',
                    'Provide multiple representation options',
                    'Allow flexible demonstration methods'
                ]
            },
            'GameBasedLearning': {
                'category': 'Engagement Strategy',
                'best_for': ['motivation', 'engagement', 'practice'],
                'implementation': 'Use educational games and gamification elements',
                'applications': [
                    'Digital educational games',
                    'Board games for concepts',
                    'Gamified challenges and rewards'
                ],
                'special_needs_adaptations': [
                    'Adjust difficulty levels',
                    'Provide multimodal interfaces',
                    'Allow collaborative play options'
                ]
            },
            'Station Rotation': {
                'category': 'Differentiated Instruction',
                'best_for': ['diverse_learners', 'multiple_activities', 'small_groups'],
                'implementation': 'Students rotate through different learning stations',
                'applications': [
                    'Technology station, teacher station, collaborative station',
                    'Different difficulty levels at each station',
                    'Varied activity types'
                ],
                'special_needs_adaptations': [
                    'Adjust station complexity',
                    'Provide clear transition signals',
                    'Allow extended time if needed'
                ]
            },
            'Debate': {
                'category': 'Critical Thinking',
                'best_for': ['argumentation', 'communication', 'perspective_taking'],
                'implementation': 'Structured discussions where students defend positions',
                'applications': [
                    'Formal debates on controversial topics',
                    'Role-playing different viewpoints',
                    'Socratic seminars'
                ],
                'special_needs_adaptations': [
                    'Provide preparation time',
                    'Allow written responses',
                    'Use visual prompts for structure'
                ]
            },
            'Stem': {
                'category': 'Integrated Learning',
                'best_for': ['problem_solving', 'hands_on_learning', 'interdisciplinary'],
                'implementation': 'Integrate science, technology, engineering, and math',
                'applications': [
                    'Engineering design challenges',
                    'Scientific investigations',
                    'Technology-enhanced projects'
                ],
                'special_needs_adaptations': [
                    'Provide multiple entry points',
                    'Use manipulatives and visual models',
                    'Allow varied presentation formats'
                ]
            },
            'Peertopeereducation': {
                'category': 'Peer Learning',
                'best_for': ['social_learning', 'peer_support', 'tutoring'],
                'implementation': 'Students teach and learn from each other',
                'applications': [
                    'Peer tutoring programs',
                    'Reciprocal teaching',
                    'Peer feedback sessions'
                ],
                'special_needs_adaptations': [
                    'Train peer tutors in disability awareness',
                    'Match students strategically',
                    'Provide structured interaction protocols'
                ]
            }
        }
    
    # ============================================================
    # CONTEXT BUILDER - SPECIAL NEEDS MAPPING
    # (from context_builder.py lines 156-208)
    # ============================================================
    
    def get_special_needs_mapping(self) -> Dict:
        """UDL special needs to methodology mapping"""
        return {
            'Adhd': {
                'primary_characteristics': ['attention_difficulties', 'hyperactivity', 'impulsivity'],
                'recommended_methodologies': ['Cooperative Learning', 'Flipped Classroom', 'GameBasedLearning'],
                'support_needs': ['structured_environment', 'frequent_breaks', 'clear_expectations'],
                'environmental_factors': ['reduced_distractions', 'movement_opportunities', 'visual_schedules']
            },
            'Autism spectrum disorder': {
                'primary_characteristics': ['social_communication_challenges', 'sensory_sensitivities', 'preference_for_routines'],
                'recommended_methodologies': ['Station Rotation', 'Project based learning', 'Structured teaching'],
                'support_needs': ['predictable_routines', 'visual_supports', 'quiet_spaces'],
                'environmental_factors': ['reduced_sensory_stimulation', 'clear_visual_organization', 'transition_warnings']
            },
            'Blind': {
                'primary_characteristics': ['visual_impairment', 'rely_on_tactile_auditory_input'],
                'recommended_methodologies': ['Cooperative Learning', 'Peertopeereducation', 'Audio-enhanced instruction'],
                'support_needs': ['tactile_materials', 'audio_descriptions', 'accessible_technology'],
                'environmental_factors': ['organized_space', 'consistent_layout', 'auditory_landmarks']
            },
            'Deaf': {
                'primary_characteristics': ['hearing_impairment', 'visual_communication_preference'],
                'recommended_methodologies': ['Visual learning strategies', 'Cooperative Learning', 'Technology-enhanced instruction'],
                'support_needs': ['visual_materials', 'captions', 'sign_language_support'],
                'environmental_factors': ['good_lighting', 'visual_accessibility', 'reduced_background_noise']
            },
            'Physical disability': {
                'primary_characteristics': ['mobility_challenges', 'may_need_assistive_devices'],
                'recommended_methodologies': ['Station Rotation', 'Flipped Classroom', 'Technology-enhanced learning'],
                'support_needs': ['accessible_materials', 'adapted_equipment', 'flexible_positioning'],
                'environmental_factors': ['accessible_space', 'appropriate_furniture', 'assistive_technology']
            },
            'Cognitive disability [mild, moderate, severe]': {
                'primary_characteristics': ['learning_difficulties', 'processing_challenges', 'need_for_repetition'],
                'recommended_methodologies': ['Cooperative Learning', 'Station Rotation', 'Scaffolded instruction'],
                'support_needs': ['simplified_materials', 'extra_time', 'multi-sensory_approaches'],
                'environmental_factors': ['structured_environment', 'visual_supports', 'minimal_distractions']
            },
            'Language difficulties due to foreign origin': {
                'primary_characteristics': ['limited_language_proficiency', 'cultural_adjustment'],
                'recommended_methodologies': ['Cooperative Learning', 'Visual learning strategies', 'Bilingual support'],
                'support_needs': ['visual_supports', 'bilingual_materials', 'language_scaffolding'],
                'environmental_factors': ['culturally_responsive_environment', 'peer_language_models', 'translation_resources']
            },
            'NoPersonalMotivation': {
                'primary_characteristics': ['lack_of_engagement', 'low_interest'],
                'recommended_methodologies': ['GameBasedLearning', 'Project based learning', 'Choice-based activities'],
                'support_needs': ['relevance_to_interests', 'autonomy_support', 'positive_relationships'],
                'environmental_factors': ['engaging_materials', 'student_choice', 'meaningful_connections']
            },
            'Excellence in some or all subjects': {
                'primary_characteristics': ['advanced_abilities', 'quick_learning', 'need_for_challenge'],
                'recommended_methodologies': ['Project based learning', 'Debate', 'Stem', 'Independent study'],
                'support_needs': ['enrichment_activities', 'acceleration_options', 'complex_challenges'],
                'environmental_factors': ['advanced_resources', 'flexible_pacing', 'peer_intellectual_interaction']
            },
            'Hyperactivity Disorder': {
                'primary_characteristics': ['excessive_movement', 'difficulty_staying_seated'],
                'recommended_methodologies': ['GameBasedLearning', 'Station Rotation', 'Movement-integrated learning'],
                'support_needs': ['movement_breaks', 'active_learning_opportunities', 'fidget_tools'],
                'environmental_factors': ['space_for_movement', 'flexible_seating', 'sensory_tools']
            },
            'Giftedness': {
                'primary_characteristics': ['exceptional_abilities', 'advanced_reasoning', 'intense_curiosity'],
                'recommended_methodologies': ['Project based learning', 'Debate', 'Stem', 'Inquiry-based learning'],
                'support_needs': ['intellectual_challenges', 'creative_opportunities', 'depth_over_breadth'],
                'environmental_factors': ['advanced_resources', 'mentorship_opportunities', 'creative_spaces']
            }
        }
    
    def get_educational_context_type(self) -> str:
        """UDL educational context type"""
        return "special_needs"  # UDL focuses on disabilities and adaptations


