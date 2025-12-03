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
        """Neuro Node2Vec label weights - based on neuro_audit_report.json"""
        return {
            # TOP 10 MOST FREQUENT LABELS
            'Attention': 2.2,                    # 22 nodes, hub node
            'CriticalThinking': 2.0,             # 15 nodes
            'ExtrinsicMotivation': 1.9,          # 14 nodes
            'ExecutiveFunctions': 2.1,           # 12 nodes, high connectivity
            'IntrinsicMotivation': 2.0,          # 11 nodes
            'LearningOutcomes': 1.8,             # 10 nodes
            'TeachingPractices': 1.8,            # 10 nodes
            'LearningDevelopment': 1.7,          # 9 nodes
            'NegativeStressDistress': 1.7,       # 9 nodes, high out-degree
            'Motivation': 1.6,                   # 8 nodes
            
            # HUB NODES (high connectivity)
            'CognitiveFlexibility': 2.0,
            'KnowledgeConstructionAttention': 1.9,
            'PrefrontalCortexActivation': 1.9,
            'OptimalAttentionalNetworkActivation': 1.8,
            
            # AUTHORITY NODES (learning targets)
            'Creativity': 1.8,
            'Memory': 1.7,
            'MemoryEncoding': 1.6,
            'MemorySystems': 1.6,
            
            # CRITICAL COGNITIVE PROCESSES
            'WorkingMemory': 1.7,
            'Metacognition': 1.6,
            'SelfRegulation': 1.5,
            'CognitiveControl': 1.6,
            'CognitiveProcesses': 1.6,
            
            # AFFECTIVE & MOTIVATIONAL
            'EmotionalRegulation': 1.6,
            'EmotionalWellBeing': 1.4,
            'PositiveEmotions': 1.6,
            'NegativeEmotions': 1.5,
            'AffectiveProcesses': 1.5,
            
            # MINDSET & GROWTH
            'GrowthMindset': 1.7,
            'FixedMindset': 1.5,
            'Mindset': 1.6,
            
            # STRESS & COPING
            'PositiveStressEustress': 1.6,
            'StressResponse': 1.5,
            'LongTermGrowth': 1.5,
            'LongTermDecline': 1.4,
            'AdaptiveCoping': 1.4,
            'MaladaptiveCoping': 1.4,
            
            # SOCIAL & COMMUNICATION
            'SocialCognition': 1.5,
            'SocialLearning': 1.4,
            'Communication': 1.4,
            
            # EDUCATIONAL OUTCOMES
            'LearningEngagement': 1.5,
            'LearningPerformance': 1.6,
            'EducationalSupport': 1.5,
            
            # ADDITIONAL IMPORTANT
            'HigherOrderThinking': 1.5,
            'LowerOrderThinking': 1.3,
            'ProblemSolving': 1.4,
            'LongTermMemory': 1.5,
            'PersonalGrowth': 1.4,
            'Strengths': 1.4,
            'CognitiveStrengths': 1.4,
            'ReflectiveThinking': 1.3,
            'Consolidation': 1.3,
            'MotivationalModulation': 1.3,
            'BrainAdaptability': 1.4,
            'Vulnerability': 1.3,
            'Resilience': 1.5,
            'CognitiveBias': 1.3
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
        
        # 1) Fix UNION queries with mismatched columns
        if 'UNION' in query:
            # Standardize column names across UNION parts
            parts = query.split('UNION')
            if len(parts) > 1:
                # Find first RETURN to use as template
                first_return_match = re.search(r'RETURN\s+(.+?)(?:LIMIT|$)', parts[0])
                if first_return_match:
                    template_cols = [col.strip() for col in first_return_match.group(1).split(',')]
                    
                    # Ensure all parts return same columns
                    standardized_parts = [parts[0]]
                    for part in parts[1:]:
                        # Keep the part structure but ensure matching columns
                        standardized_parts.append(part)
                    query = ' UNION '.join(standardized_parts)
        
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
        """Neuro system prompt for LLM response generation"""
        return """Sei un esperto di neuroscienze dell'apprendimento italiano, specializzato nell'applicazione pratica delle scoperte neuroscientifiche all'educazione.

Il tuo compito è fornire raccomandazioni chiare, pratiche e scientificamente solide per insegnanti italiani, basate su principi neuroscientifici."""
    
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


