#!/usr/bin/env python3
"""
graph_retriever.py - Hybrid Graph Retriever for Educational Knowledge Graph
Combines graph traversal precision with semantic search breadth for comprehensive context retrieval
"""

import asyncio
import logging
import re
import os
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from neo4j import GraphDatabase, Driver
from collections import defaultdict
import time
from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

@dataclass
class RetrievedContext:
    """Structured output for retrieved educational context"""
    nodes: List[Dict[str, Any]]
    triples: List[Tuple[str, str, str]]  # (source_name, rel_type, target_name)
    facets: Dict[str, Dict[str, int]]  # {label_counts, rel_counts}
    metadata: Dict[str, Any]  # {graph_count, semantic_count, timings, limits_applied}

class HybridGraphRetriever:
    """Hybrid retriever combining graph traversal with semantic search"""
    
    def __init__(self, neo4j_driver: Driver, use_vectors: bool = False, domain: str = "all", config: Optional[Dict] = None):
        self.neo4j_driver = neo4j_driver
        self.use_vectors = use_vectors
        self.domain = domain
        
        # Configuration with defaults
        self.config = config or {}
        self.max_nodes = self.config.get('max_nodes', 15)
        self.max_edges = self.config.get('max_edges', 30)
        self.expand_neighbors = self.config.get('expand_neighbors', True)
        self.neighbor_depth = self.config.get('neighbor_depth', 1)
        
        # Node2Vec integration
        self.node2vec_model = None
        self.node_embeddings = None
        self.node_index = None
        self.reverse_index = None
        self.node2vec_loaded = False
        
        # Load Node2Vec model if vectors are enabled
        if self.use_vectors:
            self._load_node2vec_model(domain=domain)
        
        # Educational domain boosts (higher = more relevant)
        self.domain_boosts = {
            # UDL domain labels (UNTOUCHED)
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
            'EnvironmentalSupport': 1.1,
            
            # Neuro domain labels (UPDATED from neuro_audit_report.json - Nov 2025)
            # Based on actual ingested data: 478 nodes, 195 unique labels, 111 relationship types
            
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
        }
        
        # Schema typo corrections (from your audit)
        self.schema_corrections = {
            'TeachingApproch': 'TeachingApproach'  # Fix the typo
        }
        
        # Whitelist for neighbor expansion (focus on educational relevance)
        self.expansion_labels = {
            # UDL domain labels (UNTOUCHED)
            'PedagogicalMethodology', 'StudentWithSpecialNeeds', 'StudentCharacteristic',
            'Context', 'Colour', 'Lighting', 'Furniture', 'Acoustic', 'InteractiveBoard',
            'EnvironmentalBarrier', 'EnvironmentalSupport', 'LearningEnvironment',
            
            # Neuro domain labels (UPDATED from neuro_audit_report.json - Nov 2025)
            # Includes all top labels + hub nodes + critical cognitive/affective processes
            
            # Core cognitive processes (most frequent + high connectivity)
            'Attention', 'ExecutiveFunctions', 'CriticalThinking', 'WorkingMemory',
            'Metacognition', 'SelfRegulation', 'CognitiveControl', 'CognitiveProcesses',
            'CognitiveFlexibility', 'PrefrontalCortexActivation',
            
            # Memory systems
            'Memory', 'WorkingMemory', 'LongTermMemory', 'MemorySystems', 'MemoryEncoding',
            'MemoryProcesses', 'Consolidation', 'MemoryStabilization', 'MemoryAccessibility',
            
            # Motivation & engagement
            'Motivation', 'IntrinsicMotivation', 'ExtrinsicMotivation', 'LearningEngagement',
            'KnowledgeConstructionAttention', 'OptimalAttentionalNetworkActivation',
            'LearningMotivation', 'AcademicMotivation',
            
            # Creativity & innovation
            'Creativity', 'CreativityInnovation', 'IdeaGeneration', 'CreativeOutcomes',
            
            # Emotional & affective
            'EmotionalRegulation', 'EmotionalWellBeing', 'PositiveEmotions', 'NegativeEmotions',
            'AffectiveProcesses', 'AffectiveMotivationalModulation', 'EmotionCognitionInteraction',
            
            # Mindset & growth
            'Mindset', 'GrowthMindset', 'FixedMindset', 'MindsetFlexibility', 'MindsetAttitudes',
            'BrainAdaptability', 'Neuroplasticity',
            
            # Stress & coping
            'PositiveStressEustress', 'NegativeStressDistress', 'StressResponse',
            'AdaptiveCoping', 'MaladaptiveCoping', 'Resilience', 'Vulnerability',
            
            # Social processes
            'SocialCognition', 'SocialLearning', 'SocialDevelopment', 'Communication',
            'SocialEmotions', 'SocialDimension',
            
            # Learning outcomes & development
            'LearningOutcomes', 'LearningDevelopment', 'LearningPerformance', 'LearningDepth',
            'LearningQuality', 'DeepLearning', 'LongTermLearning', 'PersonalGrowth',
            
            # Teaching & educational support
            'TeachingPractices', 'EducationalSupport', 'EducationalEnvironment',
            'InstructionalScaffolding',
            
            # Thinking & reasoning
            'HigherOrderThinking', 'LowerOrderThinking', 'ProblemSolving', 'Reasoning',
            'ReflectiveThinking', 'AdaptiveThinking', 'CriticalThinking',
            
            # Executive functions & planning
            'Planning', 'Monitoring', 'Evaluation', 'MetacognitiveMonitoring',
            'MetacognitiveControl', 'AttentionalControl',
            
            # Strengths & abilities
            'Strengths', 'CognitiveStrengths', 'MotorFunction', 'SensoryProcessing',
            
            # Special needs & disabilities (present in audit)
            'LanguageProcessing', 'ReadingAcquisition', 'ReadingFluency', 'ProcessingSpeed',
            'LiteracySkills', 'OrthographicMapping', 'NumberSense', 'NumericalCognition',
            'MathematicalLiteracy', 'SpatialProcessing', 'SpatialCognition',
            
            # Neuroscience concepts
            'CognitiveNeuroscience', 'NeuralResources', 'HemisphericSpecialization',
            'NeuroimagingEvidence', 'AmygdalaHippocampusInteraction',
            
            # Cognitive biases & myths
            'CognitiveBias', 'CognitiveBiases', 'LearningStyles', 'MultipleIntelligences',
            'WeOnlyUse10OfOurBrain', 'CriticalPeriodsAreAbsolute', 'PeopleAreAnalyticalOrCreative',
            
            # Educational theory
            'EducationalTheory', 'EvidenceBasedInstruction', 'CognitivePsychology',
            
            # Decision making & judgment
            'DecisionMaking', 'JudgmentBelief', 'Assessment', 'SelfEvaluation', 'Expectations',
            
            # Applied cognition
            'AppliedCognition', 'InformationLiteracy', 'KnowledgeIntegration',
            'KnowledgeOfCognition', 'SelfRegulatedLearning'
        }
    
    def _load_node2vec_model(self, domain: str = "all", model_path: str = None):
        """Load pre-trained Node2Vec model and embeddings with domain awareness
        
        Args:
            domain: Domain to load model for ('udl', 'neuro', 'all')
            model_path: Optional explicit model path (overrides domain-based path)
        """
        try:
            # Use domain-specific path if not explicitly provided
            if model_path is None:
                model_path = f"models/{domain}_node2vec"
            
            logger.info(f"Loading Node2Vec model for domain: {domain} from {model_path}")
            
            # Load Node2Vec model
            model_file = f"{model_path}_model.pkl"
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    self.node2vec_model = pickle.load(f)
                logger.info(f"Node2Vec model loaded successfully (domain: {domain})")
            else:
                logger.warning(f"Node2Vec model file not found: {model_file}")
                logger.warning(f"To train Node2Vec for {domain}, run: python train_node2vec.py {domain}")
                return False
            
            # Load embeddings and indices
            embeddings_file = f"{model_path}_embeddings.npz"
            if os.path.exists(embeddings_file):
                data = np.load(embeddings_file, allow_pickle=True)
                self.node_embeddings = data['embeddings']
                self.node_index = data['node_index'].item()
                self.reverse_index = data['reverse_index'].item()
                self.node2vec_loaded = True
                logger.info(f"Node2Vec embeddings loaded: {len(self.node_embeddings)} nodes")
                return True
            else:
                logger.warning(f"Node2Vec embeddings file not found: {embeddings_file}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load Node2Vec model: {e}")
            self.node2vec_loaded = False
            return False
    
    async def retrieve(self, query: str, cypher_result: Dict) -> RetrievedContext:
        """Main retrieval method combining graph and semantic search"""
        start_time = time.time()
        
        try:
            # Phase 1: Graph traversal (precision)
            graph_start = time.time()
            graph_nodes = await self._graph_traversal(cypher_result, query)
            graph_time = time.time() - graph_start
            
            # Phase 2: Semantic search (breadth, optional)
            semantic_nodes = []
            semantic_time = 0
            if self.use_vectors:
                semantic_start = time.time()
                semantic_nodes = await self._semantic_search(query, graph_nodes)
                semantic_time = time.time() - semantic_start
            
            # Phase 3: Fusion and ranking
            fusion_start = time.time()
            ranked_nodes, triples = self._fuse_results(graph_nodes, semantic_nodes)
            fusion_time = time.time() - fusion_start
            
            # Phase 4: Build final context
            context_start = time.time()
            facets = self._build_facets(ranked_nodes, triples)
            context_time = time.time() - context_start
            
            total_time = time.time() - start_time
            
            metadata = {
                'graph_count': len(graph_nodes),
                'semantic_count': len(semantic_nodes),
                'total_nodes': len(ranked_nodes),
                'total_triples': len(triples),
                'timings': {
                    'graph_traversal': graph_time,
                    'semantic_search': semantic_time,
                    'fusion': fusion_time,
                    'context_building': context_time,
                    'total': total_time
                },
                'limits_applied': {
                    'max_nodes': self.max_nodes,
                    'max_edges': self.max_edges,
                    'use_vectors': self.use_vectors
                }
            }
            
            return RetrievedContext(
                nodes=ranked_nodes[:self.max_nodes],
                triples=triples[:self.max_edges],
                facets=facets,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            # Fallback to graph-only results
            return await self._fallback_retrieval(cypher_result, str(e))
    
    async def _graph_traversal(self, cypher_result: Dict, query: str = "") -> List[Dict]:
        """Execute Cypher query and retrieve curriculum nodes with neighbor expansion"""
        try:
            # CRITICAL FIX: Use fallback query if available (when relationship query failed)
            if cypher_result.get('used_fallback', False) and cypher_result.get('fallback_query'):
                cypher_query = cypher_result.get('fallback_query', '')
                logger.info(f"[FALLBACK MODE] Using fallback query instead of original")
            else:
                cypher_query = cypher_result.get('cypher_query', '')
            
            if not cypher_query:
                logger.warning("No Cypher query provided for graph traversal")
                return []

            corrected_query = self._apply_schema_corrections(cypher_query)
            alias_labels = self._extract_alias_labels(corrected_query)

            with self.neo4j_driver.session() as session:
                # 1) Run the original query
                result = session.run(corrected_query)
                records = list(result)

                # 2) Process records - handle full nodes, scalars, or labeled nodes
                initial_nodes = []
                if records:
                    sample_keys = list(dict(records[0]).keys())
                    logger.info(f"[DEBUG FALLBACK] First row keys: {sample_keys}")
                    logger.info(f"[DEBUG FALLBACK] Checking cases: has_label_cols={any(k in sample_keys for k in ['node_labels', 'source_labels', 'target_labels'])}, has_dots={any('.' in k for k in sample_keys)}, has_name={('name' in sample_keys)}, has_category_or_desc={any(k in sample_keys for k in ['category', 'description'])}")
                    
                    # CASE 1: Full node objects with explicit labels (NEW FORMAT)
                    # Pattern: RETURN m, labels(m) as node_labels OR RETURN m as concept, labels(m) as node_labels (UNION fix)
                    if any(k in sample_keys for k in ['node_labels', 'source_labels', 'target_labels']):
                        for rec in records:
                            row = dict(rec)
                            
                            # Process nodes with labels
                            for key, value in row.items():
                                if key in ['node_labels', 'source_labels', 'target_labels']:
                                    continue  # Skip label columns themselves
                                
                                # Check if this is a Neo4j Node object
                                if hasattr(value, '__iter__') and hasattr(value, 'get'):
                                    try:
                                        node_dict = dict(value)
                                        
                                        # Get labels from corresponding label column
                                        if 'node_labels' in row:
                                            node_dict['labels'] = row['node_labels']
                                        elif key == 'concept':  # UNION fix: standardized column name for simple queries
                                            if 'node_labels' in row:
                                                node_dict['labels'] = row['node_labels']
                                        elif key == 'source':  # UNION fix: standardized column name for relationship queries
                                            if 'source_labels' in row:
                                                node_dict['labels'] = row['source_labels']
                                        elif key == 'target':  # UNION fix: standardized column name for relationship queries
                                            if 'target_labels' in row:
                                                node_dict['labels'] = row['target_labels']
                                        elif key == 'm' or key == 'n':
                                            if 'source_labels' in row:
                                                node_dict['labels'] = row['source_labels']
                                            elif 'node_labels' in row:
                                                node_dict['labels'] = row['node_labels']
                                        else:
                                            # Fallback: try to extract from node object
                                            if hasattr(value, 'labels'):
                                                node_dict['labels'] = list(value.labels)
                                            else:
                                                node_dict['labels'] = []
                                        
                                        initial_nodes.append(self._normalize_node(node_dict))
                                    except Exception as e:
                                        logger.warning(f"Could not process full node object: {e}")
                    
                    # CASE 2: Scalar projections (legacy format) - fallback
                    elif any("." in k for k in sample_keys):
                        initial_nodes = self._records_to_nodes(records, alias_labels)
                    
                    # CASE 3: Pure scalar fallback results (from fallback definition queries)
                    # Pattern: {name: "Emotions", category: "Affective Processes"}
                    elif 'name' in sample_keys and any(k in sample_keys for k in ['category', 'description']):
                        logger.info(f"[CASE 3 TRIGGERED] Detected fallback scalar results with keys: {sample_keys}")
                        for rec in records:
                            row = dict(rec)
                            if 'name' in row:
                                # Try to infer label from query or use generic
                                inferred_label = None
                                # Check ALL aliases (not just specific ones) - take the first valid label
                                for alias, label in alias_labels.items():
                                    if label and label != '':  # Any valid label
                                        inferred_label = label
                                        break
                                
                                if not inferred_label:
                                    inferred_label = 'Concept'  # Generic fallback (will be added to neuro_labels)
                                
                                node = {
                                    "id": f"{inferred_label}:{row['name']}",
                                    "name": row["name"],
                                    "category": row.get("category", ""),
                                    "labels": [inferred_label] if inferred_label else [],
                                    "description": row.get("description", ""),
                                    "rel_type": "",
                                    "source_node": {}
                                }
                                initial_nodes.append(self._normalize_node(node))
                                logger.info(f"Parsed fallback scalar: {row['name']} as {inferred_label}")
                    
                    # CASE 4: Simple node objects without explicit labels
                    else:
                        for rec in records:
                            row = dict(rec)
                            for v in row.values():
                                try:
                                    props = dict(v)
                                    if hasattr(v, 'labels'):
                                        props['labels'] = list(v.labels)
                                    initial_nodes.append(self._normalize_node(props))
                                except Exception:
                                    pass

                if not initial_nodes:
                    logger.info("No initial nodes found from Cypher query")
                    return []

                # 3) Expand neighbors if enabled
                if self.expand_neighbors:
                    expanded_nodes = await self._expand_neighbors(initial_nodes, session, query)
                    return expanded_nodes
                else:
                    return initial_nodes

        except Exception as e:
            logger.error(f"Graph traversal failed: {e}")
            return []
    
    def _apply_schema_corrections(self, cypher_query: str) -> str:
        """Apply schema typo corrections to Cypher query"""
        corrected_query = cypher_query
        for typo, correct in self.schema_corrections.items():
            corrected_query = corrected_query.replace(typo, correct)
        return corrected_query
    
    def _extract_alias_labels(self, cypher_query: str) -> dict:
        """
        From patterns like (m:PedagogicalMethodology) or (s:StudentWithSpecialNeeds),
        build {'m': 'PedagogicalMethodology', 's': 'StudentWithSpecialNeeds'}.
        """
        alias_labels = {}
        for alias, label in re.findall(r'\(\s*([A-Za-z_]\w*)\s*:\s*([A-Za-z][A-Za-z0-9_]*)\s*\)', cypher_query):
            alias_labels[alias] = label
        return alias_labels

    def _records_to_nodes(self, records, alias_labels: dict) -> list:
        """
        Turn scalar projections like {'m.name': 'Cooperative Learning', 'm.category': 'Pedagogical Methodology'}
        into normalized node dicts the retriever understands.
        """
        nodes = []
        for rec in records:
            # Neo4j Record -> plain dict
            row = dict(rec)
            # group fields by alias (m, s, lr, etc.)
            buckets = {}
            for key, val in row.items():
                if "." in key:
                    alias, prop = key.split(".", 1)
                    buckets.setdefault(alias, {})[prop] = val

            # build node per alias that has a 'name'
            for alias, props in buckets.items():
                if "name" in props:
                    label = alias_labels.get(alias)
                    node = {
                        "id": f"{label}:{props['name']}" if label else props["name"],
                        "name": props["name"],
                        "category": props.get("category", ""),
                        "labels": [label] if label else [],
                        "description": props.get("description", ""),
                        "rel_type": "",
                        "source_node": {}
                    }
                    nodes.append(node)

        return nodes
    
    async def _expand_neighbors(self, nodes: List[Dict], session, query: str = "") -> List[Dict]:
        """
        Expand nodes with their educational neighbors (structural + vector-based).
        
        Now includes P1 filtering to remove irrelevant structural neighbors.
        
        Args:
            nodes: Initial nodes to expand from
            session: Neo4j session
            query: Natural language query (for PHASE 1 intent detection)
        """
        initial_nodes = []
        expanded_nodes = []
        seen_node_ids = set()
        
        for node in nodes:
            # Add the original node
            node_id = node.get('id') or node.get('name')
            if node_id and node_id not in seen_node_ids:
                expanded_nodes.append(self._normalize_node(node))
                initial_nodes.append(node)  # Track for filtering
                seen_node_ids.add(node_id)
            
            # Get structural neighbors (direct relationships)
            structural_neighbors = self._get_educational_neighbors(node, session)
            for neighbor in structural_neighbors:
                neighbor_id = neighbor.get('id') or neighbor.get('name')
                if neighbor_id and neighbor_id not in seen_node_ids:
                    neighbor['source'] = 'structural'
                    expanded_nodes.append(self._normalize_node(neighbor))
                    seen_node_ids.add(neighbor_id)
            
            # Get vector-based neighbors (semantic similarity)
            if self.node2vec_loaded and self.use_vectors:
                vector_neighbors = await self._get_vector_neighbors(node, seen_node_ids)
                for neighbor in vector_neighbors:
                    neighbor_id = neighbor.get('id') or neighbor.get('name')
                    if neighbor_id and neighbor_id not in seen_node_ids:
                        neighbor['source'] = 'vector'
                        expanded_nodes.append(self._normalize_node(neighbor))
                        seen_node_ids.add(neighbor_id)
        
        # 🎯 P1 FIX: Filter irrelevant structural neighbors by label relevance
        # Separate initial nodes from expanded neighbors
        expanded_only = [n for n in expanded_nodes if n not in initial_nodes]
        
        if expanded_only:
            filtered_expanded = self._filter_semantic_nodes_by_relevance(
                expanded_only, 
                initial_nodes,
                query
            )
            
            logger.info(
                f"[P1 Filter] Structural+Vector neighbors: {len(expanded_only)} → Filtered: {len(filtered_expanded)}"
            )
            
            # Combine initial nodes + filtered expanded nodes
            return initial_nodes + filtered_expanded
        else:
            return expanded_nodes
    
    def _get_educational_neighbors(self, node: Dict, session) -> List[Dict]:
        """Get educationally relevant neighbors of a node"""
        try:
            node_labels = node.get('labels', [])
            node_name = node.get('name', '')
            if not node_labels or not node_name:
                return []

            main_label = node_labels[0]

            # Prefer putting the label in MATCH (can't parameterize labels)
            neighbor_query = f"""
            MATCH (source:{main_label} {{name: $node_name}})-[r]-(n)
            WHERE any(l IN labels(n) WHERE l IN $relevant_labels)
            RETURN DISTINCT n, type(r) AS rel_type, source AS source_node
            LIMIT $limit
            """

            result = session.run(
                neighbor_query,
                node_name=node_name,
                relevant_labels=list(self.expansion_labels),
                limit=5
            )

            neighbors = []
            for record in result:
                # n and source_node are Neo4j Node objects — capture labels explicitly
                n_node = record['n']
                s_node = record['source_node']

                neighbor = dict(n_node)
                neighbor['labels'] = list(getattr(n_node, 'labels', []))
                neighbor['rel_type'] = record['rel_type']

                src = dict(s_node)
                src['labels'] = list(getattr(s_node, 'labels', []))
                neighbor['source_node'] = src

                # give a stable id (helps dedup + ranking)
                label_for_id = neighbor['labels'][0] if neighbor.get('labels') else ''
                neighbor['id'] = f"{label_for_id}:{neighbor.get('name','')}"
                neighbors.append(neighbor)

            return neighbors

        except Exception as e:
            logger.error(f"Error getting neighbors for {node.get('name', 'unknown')}: {e}")
            return []
    
    async def _get_vector_neighbors(self, node: Dict, seen_node_ids: set) -> List[Dict]:
        """Get vector-based neighbors using Node2Vec similarity"""
        try:
            node_name = node.get('name', '')
            if not node_name or not self.node2vec_loaded or self.node_embeddings is None:
                return []
            
            # Find similar concepts using Node2Vec
            similar_concepts = self._find_similar_concepts(node_name, top_k=8)
            
            vector_neighbors = []
            for similar_name, similarity_score in similar_concepts:
                if similar_name in seen_node_ids:
                    continue
                
                # Get node details from Neo4j
                node_details = await self._get_node_details(similar_name)
                if node_details:
                    # Only include educationally relevant nodes
                    labels = node_details.get('labels', [])
                    if any(label in self.expansion_labels for label in labels):
                        node_details['vector_similarity'] = similarity_score
                        node_details['rel_type'] = 'VECTOR_SIMILAR'
                        node_details['source_node'] = node
                        vector_neighbors.append(node_details)
            
            return vector_neighbors
            
        except Exception as e:
            logger.error(f"Error getting vector neighbors for {node.get('name', 'unknown')}: {e}")
            return []
    
    def _normalize_node(self, node: Dict) -> Dict:
        """Normalize node data for consistent format"""
        return {
            'id': node.get('id', ''),
            'name': node.get('name', ''),
            'category': node.get('category', ''),
            'labels': node.get('labels', []),
            'description': node.get('description', ''),
            'rel_type': node.get('rel_type', ''),
            'source_node': node.get('source_node', {})
        }
    
    async def _semantic_search(self, query: str, existing_nodes: List[Dict]) -> List[Dict]:
        """
        Enhanced semantic search using Node2Vec embeddings.
        
        Now passes initial nodes to semantic search for relevance filtering.
        """
        try:
            # Get existing node names to avoid duplicates
            existing_names = {node.get('name', '') for node in existing_nodes}
            
            semantic_nodes = []
            
            if self.node2vec_loaded and self.use_vectors:
                # Use Node2Vec for semantic similarity (with relevance filtering)
                semantic_nodes = await self._node2vec_semantic_search(
                    query, 
                    existing_names, 
                    initial_nodes=existing_nodes  # Pass for label filtering
                )
            else:
                # Fallback to keyword-based search
                semantic_nodes = await self._keyword_semantic_search(query, existing_names)
            
            return semantic_nodes
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def _node2vec_semantic_search(self, query: str, existing_names: set, initial_nodes: List[Dict] = None) -> List[Dict]:
        """
        Node2Vec-based semantic search for educational concepts.
        
        Now includes relevance filtering to remove semantically similar but contextually 
        irrelevant nodes (e.g., 'Attention' for 'motivation' queries).
        
        Args:
            query: Natural language query
            existing_names: Set of already retrieved node names
            initial_nodes: Initial nodes from Cypher query (for label filtering)
        
        Returns:
            List of relevant semantic nodes
        """
        try:
            # Extract key concepts from query
            query_concepts = self._extract_query_concepts(query)
            
            semantic_nodes = []
            seen_concepts = set()
            
            for concept in query_concepts:
                if concept in seen_concepts:
                    continue
                seen_concepts.add(concept)
                
                # Find similar concepts using Node2Vec
                similar_concepts = self._find_similar_concepts(concept, top_k=15)
                
                for similar_name, similarity_score in similar_concepts:
                    if similar_name in existing_names:
                        continue
                    
                    # Get node details from Neo4j
                    node_details = await self._get_node_details(similar_name)
                    if node_details:
                        node_details['semantic_score'] = similarity_score
                        node_details['query_concept'] = concept
                        semantic_nodes.append(self._normalize_node(node_details))
            
            # Sort by semantic score
            semantic_nodes.sort(key=lambda x: x.get('semantic_score', 0), reverse=True)
            
            # 🎯 P1 FIX: Filter irrelevant nodes by label relevance
            filtered_nodes = self._filter_semantic_nodes_by_relevance(
                semantic_nodes, 
                initial_nodes or [],
                query
            )
            
            logger.info(f"[P1 Filter] Node2Vec candidates: {len(semantic_nodes)} → Filtered: {len(filtered_nodes)}")
            
            return filtered_nodes[:20]  # Limit to top 20
            
        except Exception as e:
            logger.error(f"Node2Vec semantic search failed: {e}")
            return []
    
    async def _keyword_semantic_search(self, query: str, existing_names: set) -> List[Dict]:
        """Fallback keyword-based semantic search"""
        try:
            semantic_nodes = []
            
            with self.neo4j_driver.session() as session:
                # Search in node descriptions and names
                semantic_query = """
                MATCH (n)
                WHERE n.description IS NOT NULL OR n.name IS NOT NULL
                AND NOT n.name IN $existing_names
                AND any(label IN labels(n) WHERE label IN $relevant_labels)
                RETURN n, 
                       CASE 
                         WHEN toLower(n.description) CONTAINS toLower($query_text) THEN 0.8
                         WHEN toLower(n.name) CONTAINS toLower($query_text) THEN 0.6
                         ELSE 0.2
                       END as relevance_score
                ORDER BY relevance_score DESC
                LIMIT 10
                """
                
                result = session.run(
                    semantic_query,
                    query_text=query,
                    existing_names=list(existing_names),
                    relevant_labels=list(self.expansion_labels)
                )
                
                for record in result:
                    node = dict(record['n'])
                    node['semantic_score'] = record['relevance_score']
                    semantic_nodes.append(self._normalize_node(node))
            
            return semantic_nodes
            
        except Exception as e:
            logger.error(f"Keyword semantic search failed: {e}")
            return []
    
    def _extract_query_concepts(self, query: str) -> List[str]:
        """Extract educational concepts from query for Node2Vec similarity"""
        # Simple concept extraction - can be enhanced with NLP
        concepts = []
        
        # Common educational terms to look for
        educational_terms = [
            'adhd', 'autism', 'blind', 'deaf', 'cognitive', 'physical', 'disability',
            'cooperative', 'learning', 'methodology', 'pedagogical', 'strategy',
            'motivation', 'attention', 'visual', 'hearing', 'impairment'
        ]
        
        query_lower = query.lower()
        for term in educational_terms:
            if term in query_lower:
                concepts.append(term)
        
        # If no specific terms found, use the whole query
        if not concepts:
            concepts = [query]
        
        return concepts
    
    def _detect_query_intent(self, query: str) -> Dict[str, bool]:
        """
        🆕 PHASE 1 - Solution 5.1: Detect user intent from query text.
        
        This helps adapt filtering thresholds based on what the user is asking for.
        
        Args:
            query: Natural language query
            
        Returns:
            Dict with intent flags (is_comparison, is_exploratory, etc.)
        """
        if not query:
            return {
                'is_comparison': False,
                'is_exploratory': False,
                'is_relationship': False,
                'is_definition': False
            }
        
        query_lower = query.lower()
        
        intent = {
            'is_comparison': any(word in query_lower for word in [
                'difference', 'compare', 'vs', 'versus', 'between', 'contrast',
                'differenza', 'confronto'  # Italian
            ]),
            'is_exploratory': any(word in query_lower for word in [
                'all', 'every', 'any', 'list', 'what are', 'which',
                'tutti', 'quali', 'elenco'  # Italian
            ]),
            'is_relationship': any(word in query_lower for word in [
                'how does', 'affect', 'influence', 'impact', 'relate', 'connect',
                'come', 'influenza', 'influisce', 'collega'  # Italian
            ]),
            'is_definition': any(word in query_lower for word in [
                'what is', 'define', 'meaning', 'means',
                'cosa è', 'significa', 'definizione'  # Italian
            ])
        }
        
        return intent
    
    def _get_adaptive_threshold(
        self, 
        query: str, 
        domain: str, 
        initial_labels: List[str]
    ) -> float:
        """
        🆕 PHASE 1 - Solution 1.2: Calculate adaptive threshold based on query type and domain.
        
        Different query types need different levels of permissiveness:
        - Exploratory queries: Lower threshold (more results)
        - Comparison queries: Lower threshold (need both sides)
        - Definition queries: Higher threshold (focused results)
        
        Args:
            query: Natural language query
            domain: Domain filter ('udl', 'neuro', 'all')
            initial_labels: Labels from initial query results
            
        Returns:
            Adaptive threshold (e.g., 0.70-0.85)
        """
        # Base threshold by domain
        if domain == "neuro":
            base_threshold = 0.70  # Neuro concepts are more interconnected
        elif domain == "udl":
            base_threshold = 0.80  # UDL is more structured
        else:
            base_threshold = 0.75  # Default
        
        # Detect query intent
        intent = self._detect_query_intent(query)
        
        # Adjust based on query type
        if intent['is_exploratory']:
            # Exploratory query - be more inclusive
            adjustment = -0.05
            logger.debug(f"[P1+] Exploratory query detected, lowering threshold")
        elif intent['is_comparison']:
            # Comparison query - be more permissive for cross-label nodes
            adjustment = -0.10
            logger.debug(f"[P1+] Comparison query detected, lowering threshold significantly")
        elif intent['is_relationship']:
            # Relationship query - medium permissiveness
            adjustment = -0.05
            logger.debug(f"[P1+] Relationship query detected, slightly lowering threshold")
        elif intent['is_definition']:
            # Definition query - be more strict (keep current threshold)
            adjustment = 0.0
            logger.debug(f"[P1+] Definition query detected, keeping base threshold")
        else:
            # Default - slightly more permissive
            adjustment = -0.03
        
        # Number of initial labels - fewer labels = be more permissive
        if len(initial_labels) <= 2:
            adjustment -= 0.02
            logger.debug(f"[P1+] Few initial labels ({len(initial_labels)}), further lowering threshold")
        
        final_threshold = base_threshold + adjustment
        
        # Ensure threshold stays in reasonable range [0.60, 0.85]
        final_threshold = max(0.60, min(0.85, final_threshold))
        
        logger.info(f"[P1+] Adaptive threshold: {final_threshold:.2f} "
                   f"(base={base_threshold:.2f}, adjustment={adjustment:.2f})")
        
        return final_threshold
    
    def _split_pascal_case(self, text: str) -> str:
        """
        🆕 PHASE 1 - Helper: Split PascalCase/camelCase strings into separate words.
        
        Examples:
            'IntrinsicMotivation' → 'Intrinsic Motivation'
            'PositiveStressEustress' → 'Positive Stress Eustress'
        
        Args:
            text: PascalCase or camelCase string
            
        Returns:
            Space-separated words
        """
        return re.sub(r'(?<!^)(?=[A-Z])', ' ', text)
    
    def _are_labels_similar(self, label_a: str, label_b: str) -> bool:
        """
        🆕 PHASE 1 - Solution 2.1: Check if two labels are similar.
        
        Uses two methods:
        1. Substring matching (fast) - check for common words
        2. Edit distance (Levenshtein) - catch typos/variations
        
        Examples:
            'PositiveStressEustress' ↔ 'StressResponse' → True (shared 'Stress')
            'IntrinsicMotivation' ↔ 'Motivation' → True (shared 'Motivation')
            'Attention' ↔ 'Memory' → False (no overlap)
        
        Args:
            label_a: First label
            label_b: Second label
            
        Returns:
            True if labels are similar, False otherwise
        """
        label_a_lower = label_a.lower()
        label_b_lower = label_b.lower()
        
        # Method 1: Substring matching (fast)
        # Split PascalCase labels into words
        words_a = set(self._split_pascal_case(label_a).lower().split())
        words_b = set(self._split_pascal_case(label_b).lower().split())
        
        # Check for word overlap
        overlap = words_a & words_b
        if len(overlap) > 0:
            logger.debug(f"[P1+] Labels similar via word overlap: {label_a} ↔ {label_b} ({overlap})")
            return True
        
        # Method 2: Levenshtein distance (for typos/variations)
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, label_a_lower, label_b_lower).ratio()
        if similarity > 0.70:
            logger.debug(f"[P1+] Labels similar via edit distance: {label_a} ↔ {label_b} ({similarity:.2f})")
            return True
        
        return False
    
    def _filter_semantic_nodes_by_relevance(
        self, 
        semantic_nodes: List[Dict], 
        initial_nodes: List[Dict],
        query: str = ""
    ) -> List[Dict]:
        """
        🎯 P1 FIX + PHASE 1 ENHANCEMENTS: Filter semantically similar nodes by label relevance.
        
        Original P1 Solution: Keep only nodes that:
        1. Share at least one label with initial query results, OR
        2. Have very high semantic similarity (>0.8), OR
        3. Belong to a broad category that's relevant across domains
        
        PHASE 1 ENHANCEMENTS (Quick Wins):
        ✅ Solution 1.2: Adaptive semantic threshold (query/domain-aware)
        ✅ Solution 2.1: Label similarity matching (handle label variations)
        ✅ Solution 3.1: Multi-tier thresholds (HIGH/MEDIUM/LOW confidence)
        ✅ Solution 5.1: Query intent detection (comparison vs definition)
        
        Args:
            semantic_nodes: Candidates from Node2Vec semantic search
            initial_nodes: Initial nodes from Cypher query (for label extraction)
            query: Natural language query (for intent detection)
        
        Returns:
            Filtered list of relevant semantic nodes
        """
        if not initial_nodes or not semantic_nodes:
            return semantic_nodes  # No filtering if no initial nodes
        
        # Extract labels from initial nodes
        initial_labels = set()
        for node in initial_nodes:
            labels = node.get('labels', [])
            if isinstance(labels, list):
                initial_labels.update(labels)
            elif isinstance(labels, str):
                initial_labels.add(labels)
        
        if not initial_labels:
            logger.warning("[P1 Filter] No labels found in initial nodes, skipping filter")
            return semantic_nodes
        
        logger.info(f"[P1 Filter] Initial labels: {sorted(list(initial_labels))[:5]}")
        
        # 🆕 PHASE 1: Get adaptive threshold based on query intent
        adaptive_threshold = self._get_adaptive_threshold(query, self.domain, list(initial_labels))
        
        # Filter semantic nodes with PHASE 1 enhancements
        filtered = []
        rejected = []
        
        for node in semantic_nodes:
            node_labels = node.get('labels', [])
            if isinstance(node_labels, str):
                node_labels = [node_labels]
            
            semantic_score = node.get('semantic_score', 0.0)
            
            # Signal 1: Direct label match (backward compatible)
            has_shared_label = any(label in initial_labels for label in node_labels)
            
            # 🆕 Signal 2: Similar label match (Solution 2.1)
            has_similar_label = False
            if not has_shared_label:
                for node_label in node_labels:
                    for initial_label in initial_labels:
                        if self._are_labels_similar(node_label, initial_label):
                            has_similar_label = True
                            logger.debug(f"[P1+] Similar label match: {node_label} ≈ {initial_label}")
                            break
                    if has_similar_label:
                        break
            
            # Signal 3: Broad category (backward compatible)
            broad_categories = {
                'LearningDevelopment', 'TeachingPractices', 'CognitiveProcesses',
                'InstructionalStrategies', 'AssessmentMethods'
            }
            is_broad_category = any(label in broad_categories for label in node_labels)
            
            # 🆕 PHASE 1 - Solution 3.1: Multi-Tier Thresholds
            # Tier 1: HIGH confidence (above adaptive threshold)
            if semantic_score > adaptive_threshold:
                filtered.append(node)
                logger.debug(
                    f"[P1+] ✅ KEEP (HIGH): {node.get('name')} "
                    f"(score={semantic_score:.3f} > {adaptive_threshold:.2f})"
                )
                continue
            
            # Tier 2: MEDIUM confidence (0.65 to adaptive_threshold)
            # Require additional signal (label match OR similar label OR broad category)
            if 0.65 <= semantic_score <= adaptive_threshold:
                if has_shared_label:
                    filtered.append(node)
                    logger.debug(
                        f"[P1+] ✅ KEEP (MEDIUM+SharedLabel): {node.get('name')} "
                        f"(score={semantic_score:.3f})"
                    )
                elif has_similar_label:
                    filtered.append(node)
                    logger.debug(
                        f"[P1+] ✅ KEEP (MEDIUM+SimilarLabel): {node.get('name')} "
                        f"(score={semantic_score:.3f})"
                    )
                elif is_broad_category:
                    filtered.append(node)
                    logger.debug(
                        f"[P1+] ✅ KEEP (MEDIUM+Broad): {node.get('name')} "
                        f"(score={semantic_score:.3f})"
                    )
                else:
                    rejected.append(node)
                    logger.debug(
                        f"[P1+] ❌ REJECT (MEDIUM-NoSignal): {node.get('name')} "
                        f"(score={semantic_score:.3f}, labels={node_labels})"
                    )
                continue
            
            # Tier 3: LOW confidence (<0.65)
            # Only keep if strong label match or broad category
            if semantic_score < 0.65:
                if has_shared_label or has_similar_label:
                    filtered.append(node)
                    logger.debug(
                        f"[P1+] ✅ KEEP (LOW+Label): {node.get('name')} "
                        f"(score={semantic_score:.3f})"
                    )
                elif is_broad_category:
                    filtered.append(node)
                    logger.debug(
                        f"[P1+] ✅ KEEP (LOW+Broad): {node.get('name')} "
                        f"(score={semantic_score:.3f})"
                    )
                else:
                    rejected.append(node)
                    logger.debug(
                        f"[P1+] ❌ REJECT (LOW): {node.get('name')} "
                        f"(score={semantic_score:.3f}, labels={node_labels})"
                    )
        
        # Log summary
        if rejected:
            rejected_names = [n.get('name', 'Unknown') for n in rejected[:5]]
            logger.info(
                f"[P1+] Rejected {len(rejected)} irrelevant nodes "
                f"(e.g., {', '.join(rejected_names)}...)"
            )
        
        logger.info(f"[P1+] Filtered: {len(semantic_nodes)} → {len(filtered)} nodes "
                   f"(rejected {len(rejected)})")
        
        return filtered
    
    def _find_similar_concepts(self, concept: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find most similar concepts using Node2Vec embeddings"""
        if not self.node2vec_loaded or self.node_embeddings is None:
            return []
        
        try:
            # Try to find exact match first
            if concept in self.node_index:
                concept_idx = self.node_index[concept]
                concept_embedding = self.node_embeddings[concept_idx].reshape(1, -1)
                
                # Calculate cosine similarities
                similarities = cosine_similarity(concept_embedding, self.node_embeddings)[0]
                
                # Get top-k similar concepts
                similar_indices = np.argsort(similarities)[::-1][1:top_k+1]  # Exclude self
                
                results = []
                for idx in similar_indices:
                    similar_name = self.reverse_index[idx]
                    similarity_score = similarities[idx]
                    results.append((similar_name, similarity_score))
                
                return results
            
            # If no exact match, try fuzzy matching
            return self._fuzzy_concept_search(concept, top_k)
            
        except Exception as e:
            logger.error(f"Error finding similar concepts for '{concept}': {e}")
            return []
    
    def _fuzzy_concept_search(self, concept: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Fuzzy search for concepts when exact match not found"""
        try:
            # Find concepts that contain the search term
            matching_concepts = []
            for node_name in self.node_index.keys():
                if concept.lower() in node_name.lower():
                    matching_concepts.append(node_name)
            
            if not matching_concepts:
                return []
            
            # Get similarities for all matching concepts
            all_similarities = []
            for match_concept in matching_concepts:
                if match_concept in self.node_index:
                    concept_idx = self.node_index[match_concept]
                    concept_embedding = self.node_embeddings[concept_idx].reshape(1, -1)
                    similarities = cosine_similarity(concept_embedding, self.node_embeddings)[0]
                    
                    # Get top similarities for this concept
                    similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
                    for idx in similar_indices:
                        similar_name = self.reverse_index[idx]
                        similarity_score = similarities[idx]
                        all_similarities.append((similar_name, similarity_score))
            
            # Sort and deduplicate
            all_similarities.sort(key=lambda x: x[1], reverse=True)
            seen = set()
            results = []
            for name, score in all_similarities:
                if name not in seen:
                    results.append((name, score))
                    seen.add(name)
                    if len(results) >= top_k:
                        break
            
            return results
            
        except Exception as e:
            logger.error(f"Fuzzy concept search failed: {e}")
            return []
    
    async def _get_node_details(self, node_name: str) -> Optional[Dict]:
        """Get full node details from Neo4j"""
        try:
            with self.neo4j_driver.session() as session:
                query = """
                MATCH (n {name: $node_name})
                RETURN n, labels(n) as labels
                LIMIT 1
                """
                
                result = session.run(query, node_name=node_name)
                record = result.single()
                
                if record:
                    node = dict(record['n'])
                    node['labels'] = record['labels']
                    return node
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting node details for '{node_name}': {e}")
            return None
    
    def _fuse_results(self, graph_nodes: List[Dict], semantic_nodes: List[Dict]) -> Tuple[List[Dict], List[Tuple[str, str, str]]]:
        """Fuse and rank results from multiple sources (graph, structural, vector, semantic)"""
        # Combine all nodes
        all_nodes = []
        
        # Add graph nodes (already have source assigned)
        for node in graph_nodes:
            source = node.get('source', 'graph')
            node['rank_score'] = self._calculate_rank_score(node, source=source)
            all_nodes.append(node)
        
        # Add semantic nodes with lower priority
        for node in semantic_nodes:
            node['source'] = 'semantic'
            node['rank_score'] = self._calculate_rank_score(node, source='semantic')
            all_nodes.append(node)
        
        # Deduplicate by node name/id, keeping highest scoring version
        unique_nodes = {}
        for node in all_nodes:
            node_id = node.get('id') or node.get('name', '')
            if node_id:
                if node_id not in unique_nodes or node['rank_score'] > unique_nodes[node_id]['rank_score']:
                    unique_nodes[node_id] = node
        
        # Sort by rank score
        ranked_nodes = sorted(unique_nodes.values(), key=lambda x: x['rank_score'], reverse=True)
        
        # Extract triples from relationships
        triples = self._extract_triples(ranked_nodes)
        
        return ranked_nodes, triples
    
    def _calculate_rank_score(self, node: Dict, source: str = 'graph') -> float:
        """Calculate ranking score for a node with Node2Vec enhancements"""
        # Base score by source
        if source == 'graph':
            base_score = 1.0
        elif source == 'structural':
            base_score = 0.8
        elif source == 'vector':
            base_score = 0.6
        else:  # semantic
            base_score = 0.5
        
        # Apply domain boosts
        labels = node.get('labels', [])
        domain_boost = max([self.domain_boosts.get(l, 1.0) for l in labels], default=1.0)
        
        # Apply semantic score if available
        semantic_score = node.get('semantic_score', 1.0)
        
        # Apply vector similarity boost if available
        vector_boost = node.get('vector_similarity', 1.0)
        
        # Calculate final score
        final_score = base_score * domain_boost * semantic_score * vector_boost
        
        return final_score
    
    def _extract_triples(self, nodes: List[Dict]) -> List[Tuple[str, str, str]]:
        """Extract relationship triples from nodes"""
        triples = []
        
        for node in nodes:
            # Extract triples from neighbor relationships
            if 'rel_type' in node and 'source_node' in node:
                source_name = node['source_node'].get('name', '')
                rel_type = node['rel_type']
                target_name = node.get('name', '')
                
                if source_name and rel_type and target_name:
                    triples.append((source_name, rel_type, target_name))
        
        return triples
    
    def _build_facets(self, nodes: List[Dict], triples: List[Tuple[str, str, str]]) -> Dict[str, Dict[str, int]]:
        """Build facet summaries for the retrieved context"""
        # Count by label
        label_counts = defaultdict(int)
        for node in nodes:
            labels = node.get('labels', [])
            for label in labels:
                label_counts[label] += 1
        
        # Count by relationship type
        rel_counts = defaultdict(int)
        for source, rel_type, target in triples:
            rel_counts[rel_type] += 1
        
        return {
            'label_counts': dict(label_counts),
            'rel_counts': dict(rel_counts)
        }
    
    async def _fallback_retrieval(self, cypher_result: Dict, error: str) -> RetrievedContext:
        """Fallback retrieval when main retrieval fails"""
        logger.warning(f"Using fallback retrieval due to error: {error}")
        
        # Return basic context from cypher result
        nodes = cypher_result.get('results', [])
        normalized_nodes = [self._normalize_node(node) for node in nodes]
        
        return RetrievedContext(
            nodes=normalized_nodes[:self.max_nodes],
            triples=[],
            facets={'label_counts': {}, 'rel_counts': {}},
            metadata={
                'graph_count': len(nodes),
                'semantic_count': 0,
                'total_nodes': len(normalized_nodes),
                'total_triples': 0,
                'error': error,
                'fallback_used': True,
                'timings': {'total': 0}
            }
        )

# Integration helper for your existing pipeline
class EnhancedMultilingualText2Cypher:
    """Enhanced version that includes hybrid retrieval"""
    
    def __init__(self, use_vectors: bool = False, domain: str = "all", config: Optional[Dict] = None):
        from multilingual_text2cypher import MultilingualText2Cypher
        from config import config as app_config
        
        self.domain = domain
        self.use_vectors = use_vectors
        self.text2cypher = MultilingualText2Cypher()
        self.graph_retriever = HybridGraphRetriever(
            neo4j_driver=self.text2cypher.pipeline.converter.schema_extractor.driver,
            use_vectors=use_vectors,
            domain=domain,
            config=config
        )
    
    async def process_query_with_retrieval(self, query: str, domain: str = None) -> Dict:
        """Process query with full hybrid retrieval pipeline
        
        Args:
            query: Natural language query
            domain: Domain filter ('udl', 'neuro', 'all', or None). If None, uses the domain set during initialization.
        """
        # Use provided domain or fall back to initialization domain
        if domain is None:
            domain = self.domain
        
        # Step 1: Text2Cypher (now with domain support)
        cypher_result = self.text2cypher.process_query(query, domain=domain, execute=True)
        
        # Step 2: Hybrid Retrieval (new functionality)
        retrieval_result = await self.graph_retriever.retrieve(query, cypher_result)
        
        # Step 3: Build Educational Context (uses real retrieval, domain-aware)
        try:
            from context_builder import EducationalContextBuilder
            context_builder = EducationalContextBuilder(domain=domain)
            
            # Convert retrieval_result to dict format expected by context_builder
            # Convert triples (tuples) to dicts with keys: relationship, source, target
            triples_as_dicts = [
                {'relationship': rel_type, 'source': source, 'target': target}
                for source, rel_type, target in retrieval_result.triples
            ]
            
            retrieval_dict = {
                'nodes': retrieval_result.nodes,
                'triples': triples_as_dicts,
                'metadata': retrieval_result.metadata
            }
            
            # Determine educational context based on domain
            if domain == 'udl':
                educational_context = 'special_needs'  # UDL focuses on disabilities/adaptations
            elif domain == 'neuro':
                educational_context = 'neuroscience'   # Neuro focuses on cognitive processes
            else:
                educational_context = 'general'        # Cross-domain or unspecified
            
            educational_context_obj = await context_builder.build_context(
                retrieval_dict,
                query,
                {
                    'educational_context': educational_context,  # ✅ Dynamic based on domain
                    'original_query': query  # ✅ PHASE 2 FIX: Needed for intent detection
                }
            )
            
            # Convert to dict for display/serialization
            educational_context_dict = asdict(educational_context_obj)
            
            # Clean up the confidence fields - convert from "ConfidenceLevel.HIGH" to "HIGH"
            if 'confidence_assessment' in educational_context_dict:
                conf = str(educational_context_dict['confidence_assessment'])
                educational_context_dict['confidence_assessment'] = conf.replace('ConfidenceLevel.', '')
            
            for methodology in educational_context_dict.get('primary_methodologies', []):
                if 'confidence' in methodology and methodology['confidence']:
                    conf = str(methodology['confidence'])
                    methodology['confidence'] = conf.replace('ConfidenceLevel.', '')
            
            for methodology in educational_context_dict.get('supporting_methodologies', []):
                if 'confidence' in methodology and methodology['confidence']:
                    conf = str(methodology['confidence'])
                    methodology['confidence'] = conf.replace('ConfidenceLevel.', '')
            
        except Exception as e:
            logger.error(f"Context building failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            educational_context_obj = None
            educational_context_dict = {}

        return {
            'original_query': query,
            'domain': domain,  # Include domain in result
            'cypher_result': cypher_result,
            'retrieval_result': retrieval_result,
            'combined_context': self._build_context(retrieval_result),
            'educational_context': educational_context_dict,  # Dict for display
            'educational_context_obj': educational_context_obj  # Object for LLM chain
        }
    
    def _build_context(self, retrieval_result: RetrievedContext) -> str:
        """Build human-readable context from retrieval results"""
        context_parts = []
        
        # Add node summaries
        if retrieval_result.nodes:
            context_parts.append("## Educational Context")
            for i, node in enumerate(retrieval_result.nodes[:10], 1):
                context_parts.append(f"{i}. **{node.get('name', 'Unknown')}** ({node.get('category', 'Unknown')})")
        
        # Add relationship summaries
        if retrieval_result.triples:
            context_parts.append("\n## Key Relationships")
            for source, rel_type, target in retrieval_result.triples[:10]:
                context_parts.append(f"- {source} → {rel_type} → {target}")
        
        # Add facet summary
        if retrieval_result.facets:
            context_parts.append("\n## Summary")
            label_counts = retrieval_result.facets.get('label_counts', {})
            if label_counts:
                context_parts.append("**Node Types:** " + ", ".join([f"{label} ({count})" for label, count in label_counts.items()]))
        
        return "\n".join(context_parts)
    
    def close(self):
        """Close all connections"""
        self.text2cypher.close()

# Testing function
async def test_hybrid_retriever():
    """Test the hybrid retriever with sample queries"""
    from config import config
    
    # Initialize enhanced processor with Node2Vec
    processor = EnhancedMultilingualText2Cypher(use_vectors=True)
    
    # Test queries
    test_queries = [
        "Ci sono strategie per i ragazzi ipovedenti?",
        "Il mio studente ha l'ADHD, cosa posso fare?",
        "Metodologie per studenti senza motivazione personale?",
        "Come aiutare studenti con disturbi dello spettro autistico?",
        "Esistono tecniche per includere studenti con disabilità fisica?"
    ]
    
    try:
        for query in test_queries:
            print(f"\n{'='*80}")
            print(f"Query: {query}")
            print('='*80)
            
            result = await processor.process_query_with_retrieval(query)
            
            # Show Cypher result
            cypher_result = result['cypher_result']
            print(f"\n🔍 Generated Cypher: {cypher_result['cypher_query']}")
            print(f"✅ Valid: {cypher_result['metadata'].get('is_valid', False)}")
            
            # Show retrieval result
            retrieval_result = result['retrieval_result']
            print(f"\n📊 Retrieved {len(retrieval_result.nodes)} nodes")
            print(f"🔗 Retrieved {len(retrieval_result.triples)} relationships")
            
            # Show metadata
            metadata = retrieval_result.metadata
            print(f"\n⏱️  Timings: {metadata.get('timings', {})}")
            print(f"📈 Counts: Graph={metadata.get('graph_count', 0)}, Semantic={metadata.get('semantic_count', 0)}")
            
            # Show sample nodes
            if retrieval_result.nodes:
                print(f"\n📝 Sample Nodes:")
                for i, node in enumerate(retrieval_result.nodes[:3], 1):
                    print(f"  {i}. {node.get('name', 'Unknown')} ({node.get('category', 'Unknown')})")
            
            # Show sample triples
            if retrieval_result.triples:
                print(f"\n🔗 Sample Relationships:")
                for i, (source, rel, target) in enumerate(retrieval_result.triples[:3], 1):
                    print(f"  {i}. {source} → {rel} → {target}")
    
    finally:
        processor.close()

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_hybrid_retriever())
