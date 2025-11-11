#!/usr/bin/env python3
"""
Node2Vec Training Script for Educational Knowledge Graph.
Trains embeddings to capture educational concept relationships and semantic similarities
"""

import os
import pickle
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from neo4j import GraphDatabase
import networkx as nx
from node2vec import Node2Vec
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class EducationalNode2VecTrainer:
    """Train Node2Vec embeddings for educational concepts with domain awareness"""
    
    def __init__(self, neo4j_driver, domain: str = "all", config: Optional[Dict] = None):
        """Initialize Node2Vec trainer with domain-specific configuration
        
        Args:
            neo4j_driver: Neo4j driver instance
            domain: Domain to train on ('udl', 'neuro', 'all')
            config: Optional configuration overrides
        """
        self.driver = neo4j_driver
        self.domain = domain
        self.config = config or {}
        
        logger.info(f"Initializing Node2Vec trainer for domain: {domain}")
        
        # Node2Vec hyperparameters optimized for educational graphs
        self.node2vec_config = {
            'dimensions': self.config.get('dimensions', 128),  # Embedding dimension
            'walk_length': self.config.get('walk_length', 30),  # Length of random walks
            'num_walks': self.config.get('num_walks', 200),    # Number of walks per node
            'p': self.config.get('p', 1.0),                    # Return parameter (BFS-like)
            'q': self.config.get('q', 0.5),                    # In-out parameter (DFS-like)
            'workers': self.config.get('workers', 4),          # Parallel workers
            'window': self.config.get('window', 10),           # Skip-gram window
            'min_count': self.config.get('min_count', 1),      # Minimum word count
            'batch_words': self.config.get('batch_words', 10000) # Batch size
        }
        
        # Educational domain weights (higher = more important for walks)
        # UDL domain weights
        udl_weights = {
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
        
        # Neuro domain weights (synced with graph_retriever.py domain_boosts)
        # Based on neuro_audit_report.json: 478 nodes, 195 labels, 111 relationship types
        neuro_weights = {
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
        
        # Select domain weights based on training domain
        if self.domain == "neuro":
            self.domain_weights = neuro_weights
        elif self.domain == "udl":
            self.domain_weights = udl_weights
        else:  # "all" - combine both
            self.domain_weights = {**udl_weights, **neuro_weights}
        
        self.model = None
        self.node_embeddings = None
        self.node_index = None
        self.reverse_index = None
    
    def extract_graph_data(self) -> Tuple[nx.Graph, Dict[str, str]]:
        """Extract graph data from Neo4j and build NetworkX graph with domain filtering"""
        logger.info(f"Extracting graph data from Neo4j (domain: {self.domain})...")
        
        G = nx.Graph()
        node_labels = {}  # node_id -> label mapping
        
        with self.driver.session() as session:
            # Get nodes with optional domain filtering
            if self.domain and self.domain != "all":
                nodes_query = """
                MATCH (n {domain: $domain})
                RETURN id(n) as node_id, n.name as name, labels(n) as labels, 
                       n.category as category, n.description as description
                """
                result = session.run(nodes_query, domain=self.domain)
            else:
                nodes_query = """
                MATCH (n)
                RETURN id(n) as node_id, n.name as name, labels(n) as labels, 
                       n.category as category, n.description as description
                """
                result = session.run(nodes_query)
            for record in result:
                node_id = record['node_id']
                name = record['name']
                labels = record['labels']
                category = record['category']
                description = record['description']
                
                # Add node to graph
                G.add_node(node_id, 
                          name=name,
                          labels=labels,
                          category=category or '',
                          description=description or '')
                
                # Store label mapping
                node_labels[str(node_id)] = name
                
                # Add domain weight as node attribute
                if labels:
                    main_label = labels[0]
                    weight = self.domain_weights.get(main_label, 1.0)
                    G.nodes[node_id]['domain_weight'] = weight
            
            # Get relationships with optional domain filtering
            if self.domain and self.domain != "all":
                rels_query = """
                MATCH (a {domain: $domain})-[r]->(b {domain: $domain})
                RETURN id(a) as source_id, id(b) as target_id, type(r) as rel_type
                """
                result = session.run(rels_query, domain=self.domain)
            else:
                rels_query = """
                MATCH (a)-[r]->(b)
                RETURN id(a) as source_id, id(b) as target_id, type(r) as rel_type
                """
                result = session.run(rels_query)
            for record in result:
                source_id = record['source_id']
                target_id = record['target_id']
                rel_type = record['rel_type']
                
                # Add edge with relationship type
                G.add_edge(source_id, target_id, rel_type=rel_type)
        
        logger.info(f"Extracted graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G, node_labels
    
    def train_node2vec(self, G: nx.Graph) -> Node2Vec:
        """Train Node2Vec model on the educational graph"""
        logger.info("Training Node2Vec model...")
        
        # Initialize Node2Vec with educational-specific parameters
        node2vec = Node2Vec(
            G,
            dimensions=self.node2vec_config['dimensions'],
            walk_length=self.node2vec_config['walk_length'],
            num_walks=self.node2vec_config['num_walks'],
            p=self.node2vec_config['p'],
            q=self.node2vec_config['q'],
            workers=self.node2vec_config['workers'],
            weight_key='domain_weight'  # Use educational domain weights
        )
        
        # Train the model
        model = node2vec.fit(
            window=self.node2vec_config['window'],
            min_count=self.node2vec_config['min_count'],
            batch_words=self.node2vec_config['batch_words']
        )
        
        logger.info("Node2Vec training completed!")
        return model
    
    def build_embeddings_index(self, model: Node2Vec, node_labels: Dict[str, str]) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
        """Build embeddings index for fast similarity search"""
        logger.info("Building embeddings index...")
        
        # Get all node embeddings
        node_ids = list(node_labels.keys())
        embeddings = []
        node_index = {}  # node_name -> index
        reverse_index = {}  # index -> node_name
        
        for i, node_id in enumerate(node_ids):
            if node_id in model.wv:
                embedding = model.wv[node_id]
                embeddings.append(embedding)
                node_name = node_labels[node_id]
                node_index[node_name] = i
                reverse_index[i] = node_name
        
        embeddings_array = np.array(embeddings)
        
        logger.info(f"Built embeddings index: {len(embeddings_array)} nodes")
        return embeddings_array, node_index, reverse_index
    
    def find_similar_concepts(self, concept_name: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find most similar concepts using Node2Vec embeddings"""
        if not self.model or self.node_embeddings is None:
            logger.error("Model not trained yet!")
            return []
        
        # Find concept in index
        if concept_name not in self.node_index:
            logger.warning(f"Concept '{concept_name}' not found in embeddings")
            return []
        
        concept_idx = self.node_index[concept_name]
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
    
    def save_model(self, model_path: str = None):
        """Save trained model and embeddings with domain-specific path"""
        if model_path is None:
            model_path = f"models/{self.domain}_node2vec"
        
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else "models", exist_ok=True)
        
        # Save Node2Vec model
        model_file = f"{model_path}_model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save embeddings and indices
        embeddings_file = f"{model_path}_embeddings.npz"
        np.savez(embeddings_file,
                embeddings=self.node_embeddings,
                node_index=self.node_index,
                reverse_index=self.reverse_index)
        
        # Save configuration
        config_file = f"{model_path}_config.json"
        with open(config_file, 'w') as f:
            json.dump({
                'domain': self.domain,
                'node2vec_config': self.node2vec_config,
                'domain_weights': self.domain_weights,
                'training_date': datetime.now().isoformat(),
                'num_nodes': len(self.node_embeddings),
                'embedding_dim': self.node_embeddings.shape[1]
            }, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str = None):
        """Load pre-trained model and embeddings from domain-specific path"""
        if model_path is None:
            model_path = f"models/{self.domain}_node2vec"
        
        try:
            # Load Node2Vec model
            model_file = f"{model_path}_model.pkl"
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load embeddings and indices
            embeddings_file = f"{model_path}_embeddings.npz"
            data = np.load(embeddings_file)
            self.node_embeddings = data['embeddings']
            self.node_index = data['node_index'].item()
            self.reverse_index = data['reverse_index'].item()
            
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def train_and_save(self, model_path: str = None):
        """Complete training pipeline with domain-specific model path"""
        logger.info(f"Starting Node2Vec training pipeline for domain: {self.domain}...")
        
        if model_path is None:
            model_path = f"models/{self.domain}_node2vec"
        
        # Step 1: Extract graph data
        G, node_labels = self.extract_graph_data()
        
        # Step 2: Train Node2Vec
        self.model = self.train_node2vec(G)
        
        # Step 3: Build embeddings index
        self.node_embeddings, self.node_index, self.reverse_index = self.build_embeddings_index(self.model, node_labels)
        
        # Step 4: Save everything
        self.save_model(model_path)
        
        logger.info("Node2Vec training pipeline completed!")
        return self.model

def get_test_concepts(domain: str) -> List[str]:
    """Get domain-specific test concepts for validation
    
    Note: Use actual node NAMES from Neo4j, not label names!
    Node2Vec indexes by the 'name' property, not the label.
    """
    if domain == "neuro":
        return [
            "Intrinsic Motivation",
            "Growth Mindset",
            "Working Memory",
            "Attention",
            "Metacognition",
            "Optimal Arousal",  # PositiveStressEustress node
            "Executive Functions",
            "Emotional Regulation",
            "Critical Thinking",
            "Creativity"
        ]
    elif domain == "udl":
        return [
            "Adhd",
            "Autism spectrum disorder",
            "Cooperative Learning",
            "Blind",
            "Deaf",
            "Cognitive disability [mild, moderate, severe]",
            "Physical disability",
            "Flipped Classroom"
        ]
    else:  # "all"
        return [
            "Intrinsic Motivation",
            "Adhd",
            "Cooperative Learning",
            "Growth Mindset",
            "Autism spectrum disorder",
            "Working Memory",
            "Blind",
            "Metacognition"
        ]

def test_node2vec_similarities(trainer: EducationalNode2VecTrainer, test_concepts: List[str] = None):
    """Test Node2Vec with domain-specific concept similarities"""
    if test_concepts is None:
        test_concepts = get_test_concepts(trainer.domain)
    
    print(f"\n🔍 Node2Vec Similarity Test Results (Domain: {trainer.domain})")
    print("=" * 60)
    
    for concept in test_concepts:
        print(f"\n📚 Concept: {concept}")
        print("-" * 40)
        
        similar = trainer.find_similar_concepts(concept, top_k=5)
        if similar:
            for i, (similar_name, score) in enumerate(similar, 1):
                print(f"  {i}. {similar_name} (similarity: {score:.3f})")
        else:
            print(f"  ⚠️ Concept '{concept}' not found in embeddings")

def main(domain: str = "all"):
    """Main training function with domain selection
    
    Args:
        domain: Domain to train ('udl', 'neuro', 'all')
    """
    from config import config
    from neo4j import GraphDatabase
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info(f"=" * 60)
    logger.info(f"Node2Vec Training - Domain: {domain.upper()}")
    logger.info(f"=" * 60)
    
    # Create Neo4j driver
    neo4j_driver = GraphDatabase.driver(
        config.neo4j.uri,
        auth=(config.neo4j.user, config.neo4j.password)
    )
    
    # Initialize trainer with domain
    trainer = EducationalNode2VecTrainer(
        neo4j_driver=neo4j_driver,
        domain=domain,
        config={
            'dimensions': 128,
            'walk_length': 30,
            'num_walks': 200,
            'p': 1.0,
            'q': 0.5,
            'workers': 4
        }
    )
    
    try:
        # Train and save model (auto uses domain-specific path)
        model = trainer.train_and_save()
        
        # Test with domain-specific concepts
        test_concepts = get_test_concepts(domain)
        test_node2vec_similarities(trainer, test_concepts)
        
        logger.info(f"\n✅ Training completed successfully!")
        logger.info(f"📁 Model saved to: models/{domain}_node2vec")
        logger.info(f"🎯 Ready to use in graph retriever with use_vectors=True")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        trainer.driver.close()

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        domain = sys.argv[1].lower()
        if domain not in ['udl', 'neuro', 'all']:
            print(f"❌ Invalid domain: {domain}")
            print("Usage: python train_node2vec.py [udl|neuro|all]")
            print("  udl   - Train on UDL data only")
            print("  neuro - Train on Neuro data only")
            print("  all   - Train on all domains (default)")
            sys.exit(1)
    else:
        domain = "all"
        print("ℹ️  No domain specified, using 'all' (train on all domains)")
        print("Usage: python train_node2vec.py [udl|neuro|all]\n")
    
    main(domain)
