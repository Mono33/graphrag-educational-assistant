#!/usr/bin/env python3
"""
Neuro Domain Configuration
All Neuroscience-specific configuration extracted from codebase

This module contains ALL Neuro (Neuroscience) domain logic:
- Node2Vec weights for cognitive concepts
- Retrieval boosts for neuroscience relationships
- Text2Cypher examples for neuro queries
- Italian→English neuroscience term mapping
- LLM system prompts for neuroscience expertise
- Cognitive process categories and concept mapping
"""

from domains.base_config import BaseDomainConfig
from typing import Dict, List, Tuple
import re


class NeuroDomainConfig(BaseDomainConfig):
    """Neuro (Neuroscience of Learning) domain configuration"""
    
    # ============================================================
    # CORE DOMAIN IDENTITY
    # ============================================================
    
    def _get_name(self) -> str:
        return "neuro"
    
    def _get_display_name(self) -> str:
        return "Neuro (Neuroscience)"
    
    def _get_icon(self) -> str:
        return "🧠"
    
    # ============================================================
    # NODE2VEC WEIGHTS (from train_node2vec.py lines 69-147)
    # ============================================================
    
    def get_node2vec_weights(self) -> Dict[str, float]:
        """Neuro Node2Vec label weights - HYBRID approach (Dec 2025)
        
        Formula: 1.0 + log10(nodes)*0.3 + log10(connectivity+1)*0.2
        Boost: +0.2 for conceptually important labels (marked with ★)
        
        Verified against actual graph data via check_new_labels.py
        """
        return {
            # =====================================================
            # TOP LABELS (high node count + connectivity)
            # =====================================================
            'Attention': 1.95,               # 23 nodes, 57 rels → 1.76 + ★boost
            'CriticalThinking': 1.85,        # 16 nodes, 26 rels → 1.65 + ★boost
            'ExtrinsicMotivation': 1.7,      # 14 nodes, 40 rels → 1.67
            'ExecutiveFunctions': 1.85,      # 12 nodes, 45 rels → 1.66 + ★boost
            'IntrinsicMotivation': 1.8,      # 11 nodes, 34 rels → 1.62 + ★boost
            'LearningOutcomes': 1.55,        # 11 nodes, 12 rels → 1.54
            'TeachingPractices': 1.8,        # 14 nodes, 22 rels → 1.62 + ★boost
            'LearningDevelopment': 1.7,      # 9 nodes, 20 rels → 1.55 + ★boost
            'NegativeStressDistress': 1.6,   # 9 nodes, 33 rels → 1.59
            'Motivation': 1.6,               # 9 nodes, 31 rels → 1.59
            
            # =====================================================
            # HUB NODES (conceptually important, few nodes but high connectivity)
            # =====================================================
            'CognitiveFlexibility': 1.45,    # 1 node, 15 rels → 1.24 + ★boost
            'KnowledgeConstructionAttention': 1.45,  # 1 node, 14 rels → 1.24 + ★boost
            'PrefrontalCortexActivation': 1.45,      # 1 node, 13 rels → 1.23 + ★boost
            'OptimalAttentionalNetworkActivation': 1.4,  # 1 node, 9 rels → 1.20 + ★boost
            
            # =====================================================
            # MEMORY & LEARNING
            # =====================================================
            'Creativity': 1.6,               # 7 nodes, 40 rels → 1.58
            'Memory': 1.5,                   # 6 nodes, 17 rels → 1.48
            'MemoryEncoding': 1.2,           # 1 node, 8 rels → 1.19
            'MemorySystems': 1.2,            # 1 node, 9 rels → 1.20
            'WorkingMemory': 1.65,           # 6 nodes, 11 rels → 1.45 + ★boost
            'LongTermMemory': 1.3,           # 2 nodes, 10 rels → 1.30
            'Consolidation': 1.1,            # 1 node, 2 rels → 1.10
            
            # =====================================================
            # COGNITIVE PROCESSES
            # =====================================================
            'Metacognition': 1.75,           # 12 nodes, 12 rels → 1.55 + ★boost
            'SelfRegulation': 1.35,          # 4 nodes, 5 rels → 1.34
            'CognitiveControl': 1.45,        # 7 nodes, 8 rels → 1.44
            'CognitiveProcesses': 1.45,      # 4 nodes, 17 rels → 1.43
            'HigherOrderThinking': 1.35,     # 3 nodes, 10 rels → 1.35
            'LowerOrderThinking': 1.3,       # 3 nodes, 5 rels → 1.30
            'ProblemSolving': 1.3,           # 3 nodes, 3 rels → 1.26
            'ReflectiveThinking': 1.1,       # 1 node, 2 rels → 1.10
            
            # =====================================================
            # AFFECTIVE & MOTIVATIONAL
            # =====================================================
            'EmotionalRegulation': 1.55,     # 8 nodes, 18 rels → 1.53
            'EmotionalWellBeing': 1.4,       # 6 nodes, 6 rels → 1.40
            'PositiveEmotions': 1.55,        # 7 nodes, 25 rels → 1.54
            'NegativeEmotions': 1.5,         # 7 nodes, 20 rels → 1.52
            'AffectiveProcesses': 1.4,       # 3 nodes, 13 rels → 1.37
            'AffectiveMotivationalModulation': 1.2,  # CORRECT label name
            'AffectiveMotivationalProcesses': 1.2,   # Additional label found
            
            # =====================================================
            # MINDSET & GROWTH
            # =====================================================
            'GrowthMindset': 1.65,           # 5 nodes, 15 rels → 1.45 + ★boost
            'FixedMindset': 1.4,             # 4 nodes, 10 rels → 1.39
            'Mindset': 1.4,                  # 2 nodes, 29 rels → 1.39
            
            # =====================================================
            # STRESS & COPING
            # =====================================================
            'PositiveStressEustress': 1.5,   # 7 nodes, 17 rels → 1.50
            'StressResponse': 1.3,           # 3 nodes, 3 rels → 1.26
            'LongTermGrowth': 1.3,           # 2 nodes, 8 rels → 1.28
            'LongTermDecline': 1.25,         # 2 nodes, 6 rels → 1.26
            'AdaptiveCoping': 1.1,           # 1 node, 1 rel → 1.06
            'MaladaptiveCoping': 1.3,        # 3 nodes, 5 rels → 1.30
            'Resilience': 1.1,               # 1 node, 1 rel → 1.06
            'Vulnerability': 1.3,            # 3 nodes, 4 rels → 1.28
            
            # =====================================================
            # SOCIAL & COMMUNICATION
            # =====================================================
            'SocialCognition': 1.5,          # 7 nodes, 17 rels → 1.50
            'SocialLearning': 1.45,          # 6 nodes, 9 rels → 1.43
            'Communication': 1.5,            # 7 nodes, 11 rels → 1.47
            
            # =====================================================
            # EDUCATIONAL OUTCOMES
            # =====================================================
            'LearningEngagement': 1.3,       # 3 nodes, 3 rels → 1.26
            'LearningPerformance': 1.55,     # 9 nodes, 19 rels → 1.55
            'EducationalSupport': 1.5,       # 7 nodes, 17 rels → 1.50
            'LearningProgress': 1.2,         # CORRECT label name (was LearningProcess)
            
            # =====================================================
            # ADDITIONAL LABELS
            # =====================================================
            'PersonalGrowth': 1.4,           # 4 nodes, 8 rels → 1.37
            'Strengths': 1.45,               # 6 nodes, 10 rels → 1.44
            'CognitiveStrengths': 1.35,      # 4 nodes, 6 rels → 1.35
            'BrainAdaptability': 1.1,        # 1 node, 2 rels → 1.10
            'CognitiveBias': 1.1,            # 1 node, 2 rels → 1.10
            
            # =====================================================
            # NEW LABELS (Dec 2025) - Q2/Q3/Q5 queries
            # =====================================================
            
            # Learning Strategies (Q3: Spaced learning, interleaving)
            'Learningstrategies': 1.55,      # 10 nodes, 14 rels → 1.54
            'LearningStrategies': 1.1,       # 1 node, 1 rel → 1.06 (case variant)
            'DistributedPracticeEffect': 1.1,  # 1 node, 1 rel → 1.06
            'LongTermLearning': 1.2,         # 2 nodes, 2 rels → 1.19
            
            # Assessment & Evaluation (Q5: Multiple-choice tests)
            'Assessment': 1.5,               # 10 nodes, 10 rels → 1.51
            'Evaluation': 1.1,               # 1 node, 2 rels → 1.10
            'KnowledgeOfCognition': 1.1,     # 1 node, 1 rel → 1.06
            
            # Brain Lateralization / Neuromyths (Q2)
            'HemisphericSpecialization': 1.1,  # 1 node, 2 rels → 1.10
            'Educationalmyths': 1.15,        # 1 node, 4 rels → 1.14
            'CognitiveNeuroscience': 1.2,    # 2 nodes, 2 rels → 1.19
            'NeurodevelopmentalLinks': 1.2,  # 2 nodes, 2 rels → 1.19
            
            # Neuroplasticity & Brain Function
            'Neuroplasticity': 1.2,          # 2 nodes, 2 rels → 1.19
            'Brainfunction': 1.2,            # 2 nodes, 2 rels → 1.19
            
            # Comorbidities & Special Needs
            'Comorbidities': 1.1,            # 1 node, 2 rels → 1.10
            'LearningStyles': 1.1            # 1 node, 1 rel → 1.06
        }
    
    # ============================================================
    # VALID METHODOLOGY LABELS (for context_builder.py filtering)
    # ============================================================
    
    def get_valid_methodology_labels(self) -> List[str]:
        """Labels that should be accepted as valid methodologies/concepts.
        
        Used by context_builder.py to filter nodes for recommendations.
        
        IMPORTANT: When adding new data with new labels, add them here!
        This is the single source of truth for valid neuro domain labels.
        
        Returns:
            List of label strings that are valid for neuro domain
        """
        return [
            # =====================================================
            # CORE COGNITIVE PROCESSES (most frequent in graph)
            # =====================================================
            'Attention', 'CriticalThinking', 'ExtrinsicMotivation', 'ExecutiveFunctions',
            'IntrinsicMotivation', 'LearningOutcomes', 'TeachingPractices', 'LearningDevelopment',
            'NegativeStressDistress', 'Motivation',
            
            # =====================================================
            # HUB NODES (high connectivity)
            # =====================================================
            'CognitiveFlexibility', 'KnowledgeConstructionAttention', 'PrefrontalCortexActivation',
            'OptimalAttentionalNetworkActivation',
            
            # =====================================================
            # AUTHORITY NODES (key outcomes)
            # =====================================================
            'Creativity', 'Memory', 'MemoryEncoding', 'MemorySystems',
            
            # =====================================================
            # CRITICAL COGNITIVE PROCESSES
            # =====================================================
            'WorkingMemory', 'Metacognition', 'SelfRegulation', 'CognitiveControl', 'CognitiveProcesses',
            
            # =====================================================
            # AFFECTIVE & MOTIVATIONAL
            # =====================================================
            'EmotionalRegulation', 'EmotionalWellBeing', 'PositiveEmotions', 'NegativeEmotions',
            'AffectiveProcesses', 'Emotions',
            
            # =====================================================
            # MINDSET & GROWTH
            # =====================================================
            'GrowthMindset', 'FixedMindset', 'Mindset',
            
            # =====================================================
            # STRESS & COPING
            # =====================================================
            'PositiveStressEustress', 'StressResponse',
            'LongTermGrowth', 'LongTermDecline',
            'AdaptiveCoping', 'MaladaptiveCoping',
            
            # =====================================================
            # SOCIAL & COMMUNICATION
            # =====================================================
            'SocialCognition', 'SocialLearning', 'Communication',
            
            # =====================================================
            # EDUCATIONAL OUTCOMES
            # =====================================================
            'LearningEngagement', 'LearningPerformance', 'EducationalSupport',
            
            # =====================================================
            # HIGHER-ORDER THINKING
            # =====================================================
            'HigherOrderThinking', 'LowerOrderThinking', 'ProblemSolving',
            'ReflectiveThinking',
            
            # =====================================================
            # MEMORY & CONSOLIDATION
            # =====================================================
            'LongTermMemory', 'Consolidation',
            
            # =====================================================
            # PERSONAL GROWTH & STRENGTHS
            # =====================================================
            'PersonalGrowth', 'Strengths', 'CognitiveStrengths',
            'AffectiveMotivationalModulation', 'AffectiveMotivationalProcesses',  # Correct label names
            
            # =====================================================
            # NEUROPLASTICITY & BRAIN
            # =====================================================
            'BrainAdaptability', 'Neuroplasticity', 'Brainfunction',  # Note: lowercase 'f'
            
            # =====================================================
            # VULNERABILITY & RESILIENCE
            # =====================================================
            'Vulnerability', 'Resilience', 'CognitiveBias',
            
            # =====================================================
            # LEARNING STRATEGIES (NEW - Dec 2025)
            # Added for spaced learning, interleaving, retrieval practice
            # =====================================================
            'LearningStrategies', 'Learningstrategies',  # Both case variants
            'DistributedPracticeEffect',  # Spaced Repetition, etc.
            'LearningProgress',  # Correct label name (not LearningProcess)
            
            # =====================================================
            # ASSESSMENT & EVALUATION (NEW - Dec 2025)
            # Added for test construction, formative assessment
            # =====================================================
            'Assessment', 'Evaluation',
            
            # =====================================================
            # NEUROSCIENCE & BRAIN FUNCTION (NEW - Dec 2025)
            # Added for brain lateralization, neuromyths
            # Note: Only labels that EXIST in the graph (verified Dec 2025)
            # =====================================================
            'CognitiveNeuroscience',  # 2 nodes
            'HemisphericSpecialization',  # 1 node
            'Educationalmyths',  # 1 node (lowercase 'm' - this is the actual label in DB)
            'NeurodevelopmentalLinks',  # 2 nodes
            
            # =====================================================
            # LONG-TERM LEARNING (NEW - Dec 2025)
            # =====================================================
            'LongTermLearning',
            
            # =====================================================
            # METACOGNITIVE KNOWLEDGE (NEW - Dec 2025)
            # =====================================================
            'KnowledgeOfCognition',
            
            # =====================================================
            # COMORBIDITIES & SPECIAL NEEDS
            # =====================================================
            'Comorbidities', 'LearningStyles',
            
            # =====================================================
            # EDUCATIONAL & CLINICAL INTERVENTIONS (Jan 2026)
            # Critical for ADHD/attention deficit queries
            # =====================================================
            'EducationalClinicalInterventions',
            'MetacognitiveControl',
            
            # =====================================================
            # ATTENTIONAL & BIOLOGICAL FACTORS (Jan 2026)
            # Relevant for attention, stress, and neurodevelopment
            # =====================================================
            'BottomUpSalience',
            'CognitiveFiltering',
            'AffectiveBiologicalConstraints',
            'AttentionalState',
            'PsychoBiologicalFactor',
            'LongTermMemoryConsolidation',
            'MotorControl',
            
            # =====================================================
            # MOTIVATIONAL MODULATION (Jan 2026)
            # =====================================================
            'MotivationalModulation',
            
            # =====================================================
            # CASE VARIANTS (for data inconsistencies)
            # =====================================================
            'Teachingpractices',  # lowercase variant of TeachingPractices
            
            # =====================================================
            # GENERIC FALLBACKS
            # =====================================================
            'Concept'  # Fallback for inferred labels
        ]
    
    # ============================================================
    # LABEL → CATEGORY MAPPING (for context_builder.py, Jan 2026)
    # ============================================================
    
    def get_label_category_map(self) -> Dict[str, str]:
        """Map Neo4j node labels to human-readable Italian category names.
        
        Used by context_builder to classify nodes properly instead of
        labeling everything as 'Educational Methodology'.
        """
        return {
            # Teaching & Learning Strategies
            'TeachingPractices': 'Strategia Didattica',
            'Teachingpractices': 'Strategia Didattica',
            'LearningStrategies': 'Strategia di Apprendimento',
            'Learningstrategies': 'Strategia di Apprendimento',
            'EducationalSupport': 'Supporto Educativo',
            'EducationalClinicalInterventions': 'Intervento Educativo',
            
            # Cognitive Processes
            'Attention': 'Processo Cognitivo – Attenzione',
            'CriticalThinking': 'Processo Cognitivo – Pensiero Critico',
            'CognitiveProcesses': 'Processo Cognitivo',
            'CognitiveFlexibility': 'Processo Cognitivo – Flessibilità',
            'CognitiveControl': 'Controllo Cognitivo',
            'CognitiveFiltering': 'Filtraggio Cognitivo',
            'HigherOrderThinking': 'Pensiero di Ordine Superiore',
            'LowerOrderThinking': 'Pensiero di Ordine Inferiore',
            'ProblemSolving': 'Problem Solving',
            'ReflectiveThinking': 'Pensiero Riflessivo',
            'Creativity': 'Creatività',
            
            # Executive Functions
            'ExecutiveFunctions': 'Funzione Esecutiva',
            'PrefrontalCortexActivation': 'Funzione Esecutiva',
            
            # Memory
            'Memory': 'Memoria',
            'MemoryEncoding': 'Codifica della Memoria',
            'MemorySystems': 'Sistemi di Memoria',
            'WorkingMemory': 'Memoria di Lavoro',
            'LongTermMemory': 'Memoria a Lungo Termine',
            'Consolidation': 'Consolidamento',
            'LongTermMemoryConsolidation': 'Consolidamento Memoria',
            
            # Metacognition
            'Metacognition': 'Metacognizione',
            'MetacognitiveControl': 'Controllo Metacognitivo',
            'KnowledgeOfCognition': 'Conoscenza della Cognizione',
            'SelfRegulation': 'Autoregolazione',
            
            # Motivation
            'Motivation': 'Motivazione',
            'IntrinsicMotivation': 'Motivazione Intrinseca',
            'ExtrinsicMotivation': 'Motivazione Estrinseca',
            'MotivationalModulation': 'Modulazione Motivazionale',
            
            # Emotions & Affect
            'Emotions': 'Processo Emotivo',
            'EmotionalRegulation': 'Regolazione Emotiva',
            'EmotionalWellBeing': 'Benessere Emotivo',
            'PositiveEmotions': 'Emozioni Positive',
            'NegativeEmotions': 'Emozioni Negative',
            'AffectiveProcesses': 'Processo Affettivo',
            'AffectiveMotivationalModulation': 'Modulazione Affettivo-Motivazionale',
            'AffectiveMotivationalProcesses': 'Processi Affettivo-Motivazionali',
            
            # Stress
            'PositiveStressEustress': 'Eustress (Stress Positivo)',
            'NegativeStressDistress': 'Distress (Stress Negativo)',
            'StressResponse': 'Risposta allo Stress',
            'AdaptiveCoping': 'Coping Adattivo',
            'MaladaptiveCoping': 'Coping Maladattivo',
            'AffectiveBiologicalConstraints': 'Vincolo Biologico-Affettivo',
            
            # Mindset
            'GrowthMindset': 'Mentalità di Crescita',
            'FixedMindset': 'Mentalità Fissa',
            'Mindset': 'Mentalità',
            
            # Attention sub-types
            'BottomUpSalience': 'Salienza Bottom-Up',
            'AttentionalState': 'Stato Attentivo',
            'KnowledgeConstructionAttention': 'Attenzione e Costruzione della Conoscenza',
            'OptimalAttentionalNetworkActivation': 'Rete Attentiva Ottimale',
            
            # Motor & Biological
            'MotorControl': 'Controllo Motorio',
            'PsychoBiologicalFactor': 'Fattore Psicobiologico',
            
            # Social & Communication
            'SocialCognition': 'Cognizione Sociale',
            'SocialLearning': 'Apprendimento Sociale',
            'Communication': 'Comunicazione',
            
            # Learning Outcomes
            'LearningEngagement': 'Coinvolgimento nell\'Apprendimento',
            'LearningPerformance': 'Prestazione di Apprendimento',
            'LearningOutcomes': 'Esiti di Apprendimento',
            'LearningDevelopment': 'Sviluppo dell\'Apprendimento',
            'LongTermLearning': 'Apprendimento a Lungo Termine',
            'LearningProgress': 'Progresso di Apprendimento',
            
            # Brain & Neuroscience
            'BrainAdaptability': 'Neuroplasticità',
            'Neuroplasticity': 'Neuroplasticità',
            'Brainfunction': 'Funzione Cerebrale',
            'CognitiveNeuroscience': 'Neuroscienza Cognitiva',
            'HemisphericSpecialization': 'Specializzazione Emisferica',
            'NeurodevelopmentalLinks': 'Collegamento Neurosviluppo',
            
            # Personal Growth
            'PersonalGrowth': 'Crescita Personale',
            'Strengths': 'Punti di Forza',
            'CognitiveStrengths': 'Punti di Forza Cognitivi',
            'LongTermGrowth': 'Crescita a Lungo Termine',
            'LongTermDecline': 'Declino a Lungo Termine',
            'Vulnerability': 'Vulnerabilità',
            'Resilience': 'Resilienza',
            'CognitiveBias': 'Bias Cognitivo',
            
            # Assessment
            'Assessment': 'Valutazione',
            'Evaluation': 'Valutazione',
            
            # Special Needs
            'Comorbidities': 'Comorbilità',
            'LearningStyles': 'Stili di Apprendimento',
            'Educationalmyths': 'Neuromiti',
            
            # Fallback
            'Concept': 'Concetto Educativo',
        }
    
    # ============================================================
    # RETRIEVAL BOOSTS (from graph_retriever.py)
    # ============================================================
    
    def get_retrieval_boosts(self) -> Dict[str, Dict[str, float]]:
        """Neuro retrieval boosts - EXACT COPY from graph_retriever_old.py lines 72-142
        
        Based on actual ingested data: 478 nodes, 195 unique labels, 111 relationship types
        """
        return {
            'label_boosts': {
                # TOP 10 MOST FREQUENT LABELS (from audit)
                'Attention': 2.2,                    # 22 nodes - most frequent, hub node (49 out + 7 in relationships)
                'CriticalThinking': 2.0,             # 15 nodes - 2nd most frequent
                'ExtrinsicMotivation': 2.0,          # 14 nodes - 3rd most frequent ✅ EQUALIZED with Intrinsic
                'ExecutiveFunctions': 2.1,           # 12 nodes - 4th most frequent, high connectivity
                'IntrinsicMotivation': 2.0,          # 11 nodes - 5th most frequent
                'LearningOutcomes': 1.8,             # 10 nodes - 6th most frequent
                'TeachingPractices': 1.8,            # 10 nodes - 6th most frequent (tied)
                'LearningDevelopment': 1.7,          # 9 nodes - 8th most frequent
                'NegativeStressDistress': 1.7,       # 9 nodes - 8th most frequent (tied), high out-degree
                'Motivation': 1.6,                   # 8 nodes - 10th most frequent
                
                # HUB NODES (high outgoing connectivity - information sources)
                'CognitiveFlexibility': 2.0,         # 1 node but 16 total connections (hub + authority)
                'KnowledgeConstructionAttention': 1.9, # 1 node but 14 connections, drives learning
                'PrefrontalCortexActivation': 1.9,   # 1 node but 13 connections, central to cognition
                'OptimalAttentionalNetworkActivation': 1.8, # 6 in + 3 out = 9 connections
                
                # AUTHORITY NODES (high incoming connectivity - learning targets)
                'Creativity': 1.8,                   # 7 nodes, 20 incoming + 19 outgoing = 39 connections
                'Memory': 1.7,                       # 1 node but 9 incoming (key outcome)
                'MemoryEncoding': 1.6,               # 7 incoming relationships
                'MemorySystems': 1.6,                # 6 incoming relationships
                
                # CRITICAL COGNITIVE PROCESSES
                'WorkingMemory': 1.7,                # 6 nodes - essential for learning
                'Metacognition': 1.6,                # 4 nodes - self-regulation
                'SelfRegulation': 1.5,               # 4 nodes - adaptive control
                'CognitiveControl': 1.6,             # 7 nodes - executive control
                'CognitiveProcesses': 1.6,           # 4 nodes, 17 incoming (key target)
                
                # AFFECTIVE & MOTIVATIONAL
                'EmotionalRegulation': 1.6,          # 8 nodes - affective process
                'EmotionalWellBeing': 1.4,           # 6 nodes - wellbeing outcomes
                'PositiveEmotions': 1.6,             # 7 nodes, 25 outgoing (drives learning)
                'NegativeEmotions': 1.5,             # 7 nodes, 20 outgoing (interferes)
                'AffectiveProcesses': 1.5,           # 3 nodes, 10 incoming + 3 outgoing
                
                # MINDSET & GROWTH
                'GrowthMindset': 1.7,                # 5 nodes, 15 outgoing
                'FixedMindset': 1.5,                 # 4 nodes, 9 outgoing + 1 incoming
                'Mindset': 1.6,                      # 2 nodes, 14 out + 6 in = 20 connections
                
                # STRESS & COPING
                'PositiveStressEustress': 1.6,       # 7 nodes, 17 outgoing
                'LongTermGrowth': 1.5,               # 2 nodes, 8 incoming
                'LongTermDecline': 1.4,              # 2 nodes, 6 incoming
                'AdaptiveCoping': 1.4,               # 1 node, 1 incoming
                'MaladaptiveCoping': 1.4,            # 3 nodes, 5 incoming
                
                # SOCIAL & COMMUNICATION
                'SocialCognition': 1.5,              # 7 nodes, 6 out + 11 in = 17 connections
                'SocialLearning': 1.4,               # 6 nodes - collaborative learning
                'Communication': 1.4,                # 7 nodes, 9 out + 2 in
                
                # EDUCATIONAL OUTCOMES
                'LearningEngagement': 1.5,           # 3 nodes, 3 incoming
                'LearningPerformance': 1.6,          # 7 nodes, 15 incoming + 1 outgoing
                'EducationalSupport': 1.5,           # 7 nodes, 17 outgoing
                
                # ADDITIONAL IMPORTANT LABELS
                'HigherOrderThinking': 1.5,          # 3 nodes, 7 in + 3 out
                'LowerOrderThinking': 1.3,           # 3 nodes, 3 in + 2 out
                'ProblemSolving': 1.4,               # 3 nodes, 3 incoming
                'LongTermMemory': 1.5,               # 2 nodes, 9 out + 1 in
                'PersonalGrowth': 1.4,               # 3 nodes, 7 incoming
                'Strengths': 1.4,                    # 6 nodes, 10 outgoing
                'CognitiveStrengths': 1.4            # 4 nodes, 6 incoming
            },
            'relationship_boosts': {
                'SUPPORTS': 1.4,
                'ENHANCES': 1.4,
                'ENHANCE': 1.4,
                'LEADS_TO': 1.3,
                'IS_LINKED_WITH': 1.2,
                'FACILITATES': 1.3,
                'PROMOTES': 1.3,
                'REQUIRES': 1.2,
                'DEPENDS_ON': 1.2
            }
        }
    
    def get_similarity_threshold(self) -> float:
        """Neuro similarity threshold - lower (broader) due to interconnected concepts"""
        return 0.70
    
    # ============================================================
    # TEXT2CYPHER EXAMPLES (from text2cypher.py lines 303-476)
    # ============================================================
    
    def get_few_shot_examples(self, domain: str = "neuro") -> str:
        """Get Neuro-specific few-shot examples for Cypher generation
        
        EXACT COPY from old text2cypher.py lines 303-398 (_get_neuro_examples method)
        
        All patterns verified against neuro_audit_report.json:
        - 478 nodes, 195 unique labels, 111 relationship types
        - Top relationships: SUPPORTS (41), ENHANCES (37), ENHANCE (29), LEADS_TO (22)
        - Focus on definition, comparison, and relationship queries
        
        Args:
            domain: Domain name to use in examples (default: "neuro")
        
        Returns:
            String of examples in "Question: ... Cypher: ..." format
        """
        # Use regular string and replace {DOMAIN} placeholder
        # EXACT COPY from text2cypher_old.py lines 312-396
        examples = """
Question: "What is intrinsic motivation?"
Cypher: MATCH (m:IntrinsicMotivation {{domain: "{domain}"}}) RETURN m, labels(m) as node_labels LIMIT 10

Question: "What is extrinsic motivation?"
Cypher: MATCH (m:ExtrinsicMotivation {{domain: "{domain}"}}) RETURN m, labels(m) as node_labels LIMIT 10

Question: "What is the difference between intrinsic and extrinsic motivation?"
Cypher: MATCH (m:IntrinsicMotivation {{domain: "{domain}"}}) RETURN "Intrinsic" as type, m, labels(m) as node_labels UNION MATCH (m:ExtrinsicMotivation {{domain: "{domain}"}}) RETURN "Extrinsic" as type, m, labels(m) as node_labels LIMIT 20

Question: "What is growth mindset?"
Cypher: MATCH (g:GrowthMindset {{domain: "{domain}"}}) RETURN g, labels(g) as node_labels LIMIT 10

Question: "What is fixed mindset?"
Cypher: MATCH (f:FixedMindset {{domain: "{domain}"}}) RETURN f, labels(f) as node_labels LIMIT 10

Question: "What is the difference between growth mindset and fixed mindset?"
Cypher: MATCH (g:GrowthMindset {{domain: "{domain}"}}) RETURN "Growth" as type, g, labels(g) as node_labels UNION MATCH (f:FixedMindset {{domain: "{domain}"}}) RETURN "Fixed" as type, f, labels(f) as node_labels LIMIT 20

Question: "What is positive stress?"
Cypher: MATCH (s:PositiveStressEustress {{domain: "{domain}"}}) RETURN s, labels(s) as node_labels LIMIT 10

Question: "What is the difference between positive stress and negative stress?"
Cypher: MATCH (s:PositiveStressEustress {{domain: "{domain}"}}) RETURN "Positive Stress" as type, s, labels(s) as node_labels UNION MATCH (s:NegativeStressDistress {{domain: "{domain}"}}) RETURN "Negative Stress" as type, s, labels(s) as node_labels LIMIT 20

Question: "What is attention?"
Cypher: MATCH (a:Attention {{domain: "{domain}"}}) RETURN a, labels(a) as node_labels LIMIT 10

Question: "What is selective attention?"
Cypher: MATCH (a:Attention {{domain: "{domain}", name: "Selective Attention"}}) RETURN a, labels(a) as node_labels

Question: "What are executive functions?"
Cypher: MATCH (e:ExecutiveFunctions {{domain: "{domain}"}}) RETURN e, labels(e) as node_labels LIMIT 10

Question: "What is working memory?"
Cypher: MATCH (w:WorkingMemory {{domain: "{domain}"}}) RETURN w, labels(w) as node_labels LIMIT 10

Question: "What is critical thinking?"
Cypher: MATCH (c:CriticalThinking {{domain: "{domain}"}}) RETURN c, labels(c) as node_labels LIMIT 10

Question: "What is metacognition?"
Cypher: MATCH (m:Metacognition {{domain: "{domain}"}}) RETURN m, labels(m) as node_labels LIMIT 10

Question: "What is emotional regulation?"
Cypher: MATCH (e:EmotionalRegulation {{domain: "{domain}"}}) RETURN e, labels(e) as node_labels LIMIT 10

Question: "How does attention support learning?"
Cypher: MATCH (a:Attention {{domain: "{domain}"}})-[r:SUPPORTS]->(o:OptimalAttentionalNetworkActivation {{domain: "{domain}"}}) RETURN a, type(r), o, labels(a) as source_labels, labels(o) as target_labels LIMIT 10

Question: "How does intrinsic motivation enhance executive functions?"
Cypher: MATCH (i:IntrinsicMotivation {{domain: "{domain}"}})-[r:ENHANCES]->(e:ExecutiveFunctions {{domain: "{domain}"}}) RETURN i, type(r), e, labels(i) as source_labels, labels(e) as target_labels LIMIT 10

Question: "What does extrinsic motivation reduce?"
Cypher: MATCH (e:ExtrinsicMotivation {{domain: "{domain}"}})-[r:REDUCES]->(target {{domain: "{domain}"}}) RETURN e, type(r), target, labels(e) as source_labels, labels(target) as target_labels LIMIT 10

Question: "What leads to learning development?"
Cypher: MATCH (source {{domain: "{domain}"}})-[r:LEADS_TO]->(l:LearningDevelopment {{domain: "{domain}"}}) RETURN source, type(r), l, labels(source) as source_labels, labels(l) as target_labels LIMIT 10

Question: "How does negative stress affect learning?"
Cypher: MATCH (n:NegativeStressDistress {{domain: "{domain}"}})-[r:UNDERMINES|LEADS_TO]->(target {{domain: "{domain}"}}) RETURN n, type(r), target, labels(n) as source_labels, labels(target) as target_labels LIMIT 10

Question: "What supports engagement?"
Cypher: MATCH (source {{domain: "{domain}"}})-[r:SUPPORTS|DRIVES]->(k:KnowledgeConstructionAttention {{domain: "{domain}"}}) RETURN source, type(r), k, labels(source) as source_labels, labels(k) as target_labels LIMIT 10

Question: "How does creativity enhance learning?"
Cypher: MATCH (c:Creativity {{domain: "{domain}"}})-[r:ENHANCE|FACILITATES]->(target {{domain: "{domain}"}}) RETURN c, type(r), target, labels(c) as source_labels, labels(target) as target_labels LIMIT 10

Question: "What are the positive emotions?"
Cypher: MATCH (p:PositiveEmotions {{domain: "{domain}"}}) RETURN p, labels(p) as node_labels LIMIT 10

Question: "How do positive emotions enhance cognition?"
Cypher: MATCH (p:PositiveEmotions {{domain: "{domain}"}})-[r:ENHANCE|ENHANCES]->(c:CognitiveProcesses {{domain: "{domain}"}}) RETURN p, type(r), c, labels(p) as source_labels, labels(c) as target_labels LIMIT 10

Question: "What is the relationship between mindset and learning?"
Cypher: MATCH (m:Mindset {{domain: "{domain}"}})-[r]->(target {{domain: "{domain}"}}) RETURN m, type(r), target, labels(m) as source_labels, labels(target) as target_labels LIMIT 10

Question: "Come posso migliorare l'attenzione degli studenti?"
Cypher: MATCH (a:Attention {{domain: "{domain}"}}) RETURN a, labels(a) as node_labels LIMIT 10

Question: "Quali sono i fattori che influenzano la memoria di lavoro?"
Cypher: MATCH (w:WorkingMemory {{domain: "{domain}"}}) RETURN w, labels(w) as node_labels LIMIT 10

Question: "Come migliorare la motivazione intrinseca?"
Cypher: MATCH (i:IntrinsicMotivation {{domain: "{domain}"}}) RETURN i, labels(i) as node_labels LIMIT 10

Question: "Is brain lateralization the same as the left-brain/right-brain myth?"
Cypher: MATCH (h:HemisphericSpecialization {{domain: "{domain}"}}) RETURN "Brain Lateralization" as type, h AS concept, labels(h) as node_labels UNION MATCH (m:EducationalMyths {{domain: "{domain}"}}) RETURN "Educational Myth" as type, m AS concept, labels(m) as node_labels LIMIT 20

Question: "What are effective assessment strategies?"
Cypher: MATCH (a:Assessment {{domain: "{domain}"}}) RETURN a, labels(a) as node_labels LIMIT 10

Question: "How to construct effective multiple-choice tests?"
Cypher: MATCH (a:Assessment {{domain: "{domain}"}}) RETURN a, labels(a) as node_labels LIMIT 10

Question: "What is the difference between evaluation and assessment?"
Cypher: MATCH (e:Evaluation {{domain: "{domain}"}}) RETURN "Evaluation" as type, e AS concept, labels(e) as node_labels UNION MATCH (a:Assessment {{domain: "{domain}"}}) RETURN "Assessment" as type, a AS concept, labels(a) as node_labels LIMIT 20

Question: "What are spaced learning and interleaving?"
Cypher: MATCH (d:DistributedPracticeEffect {{domain: "{domain}"}}) RETURN d, labels(d) as node_labels LIMIT 10

Question: "What is the difference between hemispheric specialization and brain myths?"
Cypher: MATCH (h:HemisphericSpecialization {{domain: "{domain}"}}) RETURN "Hemispheric Specialization" as type, h AS concept, labels(h) as node_labels UNION MATCH (m:CognitiveBias {{domain: "{domain}"}}) RETURN "Cognitive Bias" as type, m AS concept, labels(m) as node_labels LIMIT 20
"""
        # Replace {domain} placeholder with actual domain value
        return examples.replace("{domain}", domain).strip()
    
    def get_cypher_patterns(self) -> str:
        """Neuro-specific Cypher patterns for system prompt
        
        EXACT COPY from old text2cypher.py lines 216-223
        """
        return """
QUERY PATTERNS (NEURO):
- Concept definitions: MATCH (c:ConceptLabel) RETURN c.name, c.category
- Concept comparisons: MATCH (c1:Label1) RETURN ... UNION MATCH (c2:Label2) RETURN ...
- Concept relationships: MATCH (a:Label1)-[r:RELATIONSHIP]->(b:Label2) RETURN a.name, type(r), b.name
- Do NOT create patterns with teaching/strategy/activity labels
"""
    
    def repair_cypher_query(self, query: str) -> str:
        """Neuro-specific Cypher repair logic (from text2cypher.py lines 801-1028)"""
        
        # 0) Fix variable name conflicts (CRITICAL: relationship var used as node var)
        conflict_pattern = r'\[(\w+):([^\]]+)\]->\((\1):(\w+)'
        if re.search(conflict_pattern, query):
            old_var = re.search(conflict_pattern, query).group(1)
            
            # Replace conflicting pattern in MATCH
            query = re.sub(
                r'\[(\w+):([^\]]+)\]->\((\1):(\w+)([^)]*)\)',
                r'[rel:\2]->(target:\4\5)',
                query
            )
            
            # Fix RETURN clause
            return_pattern = rf'RETURN\s+(\w+),\s+type\({old_var}\),\s+{old_var},\s+labels\(\1\)\s+as\s+source_labels,\s+labels\({old_var}\)\s+as\s+target_labels'
            if re.search(return_pattern, query):
                query = re.sub(
                    return_pattern,
                    r'RETURN \1, type(rel), target, labels(\1) as source_labels, labels(target) as target_labels',
                    query
                )
            else:
                # Fallback: individual replacements
                query = re.sub(rf'\blabels\({old_var}\)\s+as\s+target_labels\b', 'labels(target) as target_labels', query)
                query = re.sub(rf'\btype\({old_var}\)\b', 'type(rel)', query)
                query = re.sub(rf',\s+{old_var}\s*,', ', target,', query)
                query = re.sub(rf',\s+{old_var}\s+', ', target ', query)
        
        # 1) Fix UNION queries with mismatched columns (CRITICAL FIX for Q2)
        if 'UNION' in query:
            query = self._repair_union_columns(query)
        
        # 2) Fix PascalCase for common Neuro labels
        neuro_labels = {
            'intrinsicmotivation': 'IntrinsicMotivation',
            'extrinsicmotivation': 'ExtrinsicMotivation',
            'growthmindset': 'GrowthMindset',
            'fixedmindset': 'FixedMindset',
            'workingmemory': 'WorkingMemory',
            'executivefunctions': 'ExecutiveFunctions',
            'positivestresseustress': 'PositiveStressEustress',
            'negativestressdistress': 'NegativeStressDistress',
            'criticalthinking': 'CriticalThinking',
            'emotionalregulation': 'EmotionalRegulation',
            'teachingpractices': 'TeachingPractices',
            'learningoutcomes': 'LearningOutcomes',
            'cognitiveflexibility': 'CognitiveFlexibility'
        }
        
        for lowercase, pascalcase in neuro_labels.items():
            query = re.sub(rf'\b{lowercase}\b', pascalcase, query, flags=re.IGNORECASE)
        
        # 3) Fix Assessment-related queries that use wrong patterns
        # LLM sometimes generates: MATCH (t:TeachingPractices {name: "Evaluation of Performance"})
        # Should be: MATCH (a:Assessment ...) or MATCH (e:Evaluation ...)
        query = self._repair_assessment_queries(query)
        
        return query
    
    def _repair_union_columns(self, query: str) -> str:
        """Fix UNION queries with mismatched column names.
        
        Problem: LLM generates UNION queries like:
            MATCH (h:Label1) RETURN h, labels(h) as node_labels
            UNION
            MATCH (w:Label2) RETURN w, labels(w) as node_labels
        
        Neo4j requires IDENTICAL column names, but 'h' != 'w'.
        
        Solution: Standardize all RETURN statements to use 'concept' alias:
            RETURN h AS concept, labels(h) as node_labels
            UNION
            RETURN w AS concept, labels(w) as node_labels
        
        Returns:
            Fixed query with consistent column names
        """
        parts = query.split('UNION')
        if len(parts) < 2:
            return query
        
        fixed_parts = []
        
        for i, part in enumerate(parts):
            part = part.strip()
            
            # Extract the variable name from MATCH (var:Label ...)
            match_pattern = re.search(r'MATCH\s+\((\w+):', part)
            if not match_pattern:
                fixed_parts.append(part)
                continue
            
            var_name = match_pattern.group(1)
            
            # Pattern A: Simple definition query
            # RETURN var, labels(var) as node_labels
            simple_pattern = rf'RETURN\s+{var_name}\s*,\s*labels\({var_name}\)\s+as\s+node_labels'
            if re.search(simple_pattern, part, re.IGNORECASE):
                fixed_part = re.sub(
                    simple_pattern,
                    f'RETURN {var_name} AS concept, labels({var_name}) as node_labels',
                    part,
                    flags=re.IGNORECASE
                )
                fixed_parts.append(fixed_part)
                continue
            
            # Pattern B: Definition with type column
            # RETURN "Type" as type, var, labels(var) as node_labels
            type_pattern = rf'RETURN\s+"[^"]+"\s+as\s+type\s*,\s*{var_name}\s*,\s*labels\({var_name}\)\s+as\s+node_labels'
            if re.search(type_pattern, part, re.IGNORECASE):
                fixed_part = re.sub(
                    rf'RETURN\s+("[^"]+")\s+as\s+type\s*,\s*{var_name}\s*,\s*labels\({var_name}\)\s+as\s+node_labels',
                    rf'RETURN \1 as type, {var_name} AS concept, labels({var_name}) as node_labels',
                    part,
                    flags=re.IGNORECASE
                )
                fixed_parts.append(fixed_part)
                continue
            
            # Pattern C: Relationship query
            # RETURN var1, type(r), var2, labels(var1) as source_labels, labels(var2) as target_labels
            rel_pattern = rf'RETURN\s+{var_name}\s*,\s*type\((\w+)\)\s*,\s*(\w+)\s*,'
            if re.search(rel_pattern, part, re.IGNORECASE):
                # Standardize relationship queries
                fixed_part = re.sub(
                    rf'RETURN\s+{var_name}\s*,\s*type\((\w+)\)\s*,\s*(\w+)\s*,\s*labels\({var_name}\)\s+as\s+source_labels\s*,\s*labels\(\2\)\s+as\s+target_labels',
                    rf'RETURN {var_name} AS source, type(\1) AS rel_type, \2 AS target, labels({var_name}) as source_labels, labels(\2) as target_labels',
                    part,
                    flags=re.IGNORECASE
                )
                fixed_parts.append(fixed_part)
                continue
            
            # No pattern matched, keep as-is
            fixed_parts.append(part)
        
        result = ' UNION '.join(fixed_parts)
        
        # Log if we actually changed something
        if result != query:
            import logging
            logging.getLogger(__name__).info(f"[UNION Fix] Standardized column names in {len(parts)}-way UNION query")
        
        return result
    
    def _repair_assessment_queries(self, query: str) -> str:
        """Fix Assessment-related queries that use wrong patterns.
        
        Problem: LLM generates queries like:
            MATCH (t:TeachingPractices {name: "Evaluation of Performance"})
        
        But 'Evaluation of Performance' doesn't exist - should use Assessment label.
        
        Returns:
            Fixed query using correct Assessment/Evaluation labels
        """
        # Pattern: Searching for specific assessment-related node names in TeachingPractices
        assessment_terms = [
            'Evaluation of Performance',
            'Multiple Choice',
            'Test Design',
            'Assessment Design',
            'Formative Assessment',
            'Summative Assessment',
            'Quiz Design',
            'Exam Design'
        ]
        
        for term in assessment_terms:
            # If LLM searches for these terms in wrong labels, fix it
            wrong_pattern = rf'\((\w+):TeachingPractices\s*\{{[^}}]*name:\s*"{term}"[^}}]*\}}\)'
            if re.search(wrong_pattern, query, re.IGNORECASE):
                # Replace with Assessment label search
                query = re.sub(
                    wrong_pattern,
                    r'(\1:Assessment {domain: "neuro"})',
                    query,
                    flags=re.IGNORECASE
                )
                import logging
                logging.getLogger(__name__).info(f"[Assessment Fix] Replaced TeachingPractices search with Assessment label")
        
        return query
    
    # ============================================================
    # MULTILINGUAL TERMS (from multilingual_text2cypher.py lines 192-263)
    # ============================================================
    
    def get_italian_terms(self) -> Dict[str, str]:
        """Neuro Italian→English term mapping for query translation"""
        return {
            # Neuroscience Core Concepts
            "attenzione": "attention",
            "attenzione selettiva": "selective attention",
            "attenzione divisa": "divided attention",
            "attenzione sostenuta": "sustained attention",
            
            # Memory Systems
            "memoria": "memory",
            "memoria di lavoro": "working memory",
            "memoria a lungo termine": "long term memory",
            "memoria a breve termine": "short term memory",
            "consolidamento": "consolidation",
            
            # Executive Functions
            "funzioni esecutive": "executive functions",
            "controllo esecutivo": "executive control",
            "controllo cognitivo": "cognitive control",
            "controllo inibitorio": "inhibitory control",
            "flessibilità cognitiva": "cognitive flexibility",
            
            # Thinking & Cognition
            "pensiero critico": "critical thinking",
            "creatività": "creativity",
            "pensiero divergente": "divergent thinking",
            "risoluzione problemi": "problem solving",
            "ragionamento": "reasoning",
            
            # Emotions & Motivation
            "emozioni": "emotions",
            "emozioni positive": "positive emotions",
            "emozioni negative": "negative emotions",
            "motivazione": "motivation",
            "motivazione intrinseca": "intrinsic motivation",
            "motivazione estrinseca": "extrinsic motivation",
            "regolazione emotiva": "emotional regulation",
            
            # Learning Processes
            "apprendimento": "learning",
            "carico cognitivo": "cognitive load",
            "risultati apprendimento": "learning outcomes",
            "processi cognitivi": "cognitive processes",
            "elaborazione informazioni": "information processing",
            
            # Stress & Mindset
            "stress": "stress",
            "stress positivo": "positive stress",
            "stress negativo": "negative stress",
            "mentalità crescita": "growth mindset",
            "mentalità fissa": "fixed mindset",
            "resilienza": "resilience",
            
            # Metacognition
            "metacognizione": "metacognition",
            "autoregolazione": "self regulation",
            "consapevolezza": "awareness",
            "monitoraggio": "monitoring",
            "valutazione": "evaluation",
            
            # Social & Communication
            "cognizione sociale": "social cognition",
            "apprendimento sociale": "social learning",
            "comunicazione": "communication",
            "comprensione sociale": "social understanding",
            
            # General Terms
            "cervello": "brain",
            "neurale": "neural",
            "cognitivo": "cognitive",
            "affettivo": "affective",
            "comportamentale": "behavioral"
        }
    
    def get_query_context(self) -> str:
        """Neuro query context for multilingual enhancement"""
        return "neuroscience and cognitive science"
    
    # ============================================================
    # LLM CHAIN PROMPTS (from llm_chain.py lines 53-56)
    # ============================================================
    
    def get_system_prompt(self) -> str:
        """Neuro system prompt for LLM response generation
        
        Rich prompt aligned with production AIxLearning assistant.
        Includes: Role, Tag-Cloud, Neuroscience Principles (A-F),
        Methodological Context, and Meta-Rules.
        """
        return """# RUOLO

Sei un'Esperta di Neurodidattica e progettazione didattica evidence-based.
Integra neuroscienze cognitive, psicologia dell'apprendimento, sistemi motivazionali ed emotivi e strategie didattiche efficaci per progettare risposte pedagogiche personalizzate.
Il tuo obiettivo è trasformare ogni contenuto disciplinare in un'esperienza di apprendimento cognitivamente ottimizzata, motivante e inclusiva.

# TAG-CLOUD (keywords in ordine di importanza)

- didattica e insegnamento evidence-based
- neuroscienze, psicologia cognitiva e psicologia positiva applicate all'apprendimento
- Modello "I Do, We Do, You Do"
- meditazione, tecniche di rilassamento, respirazione, mindfulness
- Zona di Sviluppo Prossimale (ZDP) e Scaffolding (Sostegno Strutturato)
- assessment e feedback formativo
- dual coding
- mentalità di crescita (growth mindset)
- cultura dell'errore
- multimodalità
- stress e apprendimento → eustress vs distress
- benessere scolastico
- auto-regolazione: da etero-regolazione → co-regolazione → autoregolazione (motivazione ed emozioni)

# PRINCIPI NEUROSCIENTIFICI APPLICATI ALL'EDUCAZIONE

A. Processi cognitivi fondamentali
   - Attenzione (selettiva, sostenuta, divisa)
   - Memoria (encoding, consolidamento, long-term memory)
   - Working memory ed executive functions
   - Pensiero critico
   - Creatività
   - Comunicazione

B. Metacognizione e autoregolazione
   - Metacognizione (planning, monitoring, evaluation, control)
   - Consapevolezza cognitiva (knowledge of cognition)
   - Autoregolazione cognitiva ed emotiva

C. Sistemi motivazionali ed emotivi
   - Motivazione intrinseca ed estrinseca
   - Emozioni positive e negative nell'apprendimento
   - Regolazione emotiva
   - Stress, distress ed eustress

D. Sistemi di credenze
   - Growth mindset vs fixed mindset
   - Mindset shift

E. Bias cognitivi e neuromiti
   - Bias cognitivi (giudizio, attribuzione, aspettative, recall)
   - Neuromiti (learning styles, lateralizzazione emisferica, 10% del cervello, ecc.)

F. Neurodiversità e inclusione
   - ADHD, Disturbo dello spettro autistico, Dislessia, Discalculia, Tourette

ATTENZIONE: Questi principi sono fondamentali e devono guidare ogni tua risposta.

# CONTESTO METODOLOGICO

Utilizzi i seguenti approcci basati sulla ricerca:
- Modello I Do – We Do – You Do
- Scaffolding e Zona di Sviluppo Prossimale (ZDP)
- Retrieval practice e spaced repetition
- Formative assessment e feedback
- Strategie metacognitive
- Peer instruction e collaborazione strutturata

Principi guida:
- Integrazione di cognizione, emozione e motivazione
- Riduzione del carico cognitivo inutile
- Valorizzazione dell'errore come risorsa
- Apprendimento attivo e riflessivo
- Adattamento alla neurodiversità

# META-REGOLE

- Personalizza sempre la risposta al contesto specifico dell'insegnante
- Evita sovraccarico cognitivo nella struttura della risposta
- Integra dimensione cognitiva, emotiva e motivazionale
- Utilizza i dati dal Knowledge Graph come fonte prioritaria
- Adotta uno stile interlocutorio, propositivo e scientificamente fondato
- Rispondi SEMPRE in italiano"""
    
    def get_response_template(self) -> str:
        """Neuro response template - structured lesson format
        
        Aligned with production AIxLearning lesson schema:
        Warm-up → I Do → We Do → You Do → Consolidation → Assessment
        """
        return """ISTRUZIONI PER LA STRUTTURA DELLA RISPOSTA:

Struttura la risposta seguendo lo schema di progettazione neurodidattica:

1. **Introduzione Empatica**
   Riconosci la domanda dell'insegnante e il contesto educativo specifico.

2. **Metodologie Principali** (basate sui dati del Knowledge Graph)
   Per ogni metodologia raccomandata, presenta:
   - **Perché è efficace**: base neuroscientifica e cognitiva
   - **Come implementarla**: passi concreti per la classe
   - **Adattamenti**: per bisogni speciali e neurodiversità (se applicabile)
   - **Esempio pratico**: un'applicazione concreta

3. **Schema Lezione** (se pertinente alla domanda)
   - **Warm-up / Gancio / Domanda guida**: Attivazione dell'attenzione e delle conoscenze pregresse
   - **I Do** (Io faccio): Spiegazione segmentata con analogie e metafore
   - **We Do** (Facciamo insieme): Pratica guidata con feedback formativo
   - **You Do** (Fai tu): Applicazione autonoma con differenziazione didattica

4. **Consolidamento**
   - Attività di chiusura e autovalutazione
   - Domande metacognitive
   - Suggerimenti per spaced repetition

5. **Basi Teoriche**
   Evidenze neuroscientifiche a supporto delle raccomandazioni.

6. **Ordine di Implementazione**
   Priorità chiare per l'insegnante.

7. **Note sulla Fiducia**
   Se la confidenza è bassa, suggerisci di consultare specialisti.

IMPORTANTE:
- Rispondi SEMPRE in italiano
- Sii concreto e pratico, non teorico
- Fornisci azioni immediate che l'insegnante può prendere
- Adatta il linguaggio al contesto scolastico italiano
- Se la confidenza è BASSA o VERY_LOW, enfatizza la necessità di supporto specialistico
- Integra sempre i principi neuroscientifici (cognizione, emozione, motivazione)"""
    
    # ============================================================
    # CONTEXT BUILDER - METHODOLOGY CATEGORIES
    # (from context_builder.py lines 132-208)
    # ============================================================
    
    def get_methodology_categories(self) -> Dict:
        """Neuro methodology categories - cognitive/affective processes"""
        return {
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
    
    # ============================================================
    # CONTEXT BUILDER - SPECIAL NEEDS MAPPING
    # (from context_builder.py lines 231-240)
    # ============================================================
    
    def get_special_needs_mapping(self) -> Dict:
        """Neuro special needs mapping - cognitive/affective concepts"""
        return {
            'Attention': ['focus_support', 'distraction_management', 'engagement_strategies'],
            'WorkingMemory': ['cognitive_load_management', 'chunking_strategies', 'rehearsal_techniques'],
            'ExecutiveFunctions': ['planning_support', 'organization_scaffolding', 'self_regulation_teaching'],
            'Motivation': ['interest_connection', 'autonomy_provision', 'meaningful_feedback'],
            'Emotions': ['emotional_climate', 'regulation_strategies', 'safe_environment'],
            'Metacognition': ['self_awareness_teaching', 'monitoring_strategies', 'reflection_opportunities'],
            'Creativity': ['divergent_thinking_encouragement', 'open_ended_tasks', 'psychological_safety'],
            'CriticalThinking': ['questioning_techniques', 'analysis_scaffolding', 'evidence_evaluation'],
            'Memory': ['encoding_strategies', 'retrieval_practice', 'spaced_repetition'],
            'GrowthMindset': ['effort_praise', 'challenge_framing', 'mistake_learning_opportunities']
        }
    
    def get_educational_context_type(self) -> str:
        """Neuro educational context type"""
        return "neuroscience"  # Neuro focuses on cognitive processes


