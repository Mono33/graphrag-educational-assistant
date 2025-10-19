#!/usr/bin/env python3
"""
Node2Vec Training Script for Educational Knowledge Graph
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
    """Train Node2Vec embeddings for educational concepts"""
    
    def __init__(self, neo4j_driver, config: Optional[Dict] = None):
        self.driver = neo4j_driver
        self.config = config or {}
        
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
        self.domain_weights = {
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
        
        self.model = None
        self.node_embeddings = None
        self.node_index = None
        self.reverse_index = None
    
    def extract_graph_data(self) -> Tuple[nx.Graph, Dict[str, str]]:
        """Extract graph data from Neo4j and build NetworkX graph"""
        logger.info("Extracting graph data from Neo4j...")
        
        G = nx.Graph()
        node_labels = {}  # node_id -> label mapping
        
        with self.driver.session() as session:
            # Get all nodes with their properties
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
            
            # Get all relationships
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
    
    def save_model(self, model_path: str = "models/educational_node2vec"):
        """Save trained model and embeddings"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
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
                'node2vec_config': self.node2vec_config,
                'domain_weights': self.domain_weights,
                'training_date': datetime.now().isoformat(),
                'num_nodes': len(self.node_embeddings),
                'embedding_dim': self.node_embeddings.shape[1]
            }, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str = "models/educational_node2vec"):
        """Load pre-trained model and embeddings"""
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
    
    def train_and_save(self, model_path: str = "models/educational_node2vec"):
        """Complete training pipeline"""
        logger.info("Starting Node2Vec training pipeline...")
        
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

def test_node2vec_similarities(trainer: EducationalNode2VecTrainer):
    """Test Node2Vec with educational concept similarities"""
    test_concepts = [
        "Adhd",
        "Autism spectrum disorder", 
        "Cooperative Learning",
        "Blind",
        "Cognitive disability [mild, moderate, severe]",
        "Physical disability",
        "Deaf",
        "Visual impairment"
    ]
    
    print("\n🔍 Node2Vec Similarity Test Results:")
    print("=" * 60)
    
    for concept in test_concepts:
        print(f"\n📚 Concept: {concept}")
        print("-" * 40)
        
        similar = trainer.find_similar_concepts(concept, top_k=5)
        for i, (similar_name, score) in enumerate(similar, 1):
            print(f"  {i}. {similar_name} (similarity: {score:.3f})")

def main():
    """Main training function"""
    from config import config
    from neo4j import GraphDatabase
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create Neo4j driver
    neo4j_driver = GraphDatabase.driver(
        config.neo4j.uri,
        auth=(config.neo4j.user, config.neo4j.password)
    )
    
    # Initialize trainer
    trainer = EducationalNode2VecTrainer(
        neo4j_driver=neo4j_driver,
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
        # Train and save model
        model = trainer.train_and_save("models/educational_node2vec")
        
        # Test similarities
        test_node2vec_similarities(trainer)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        trainer.driver.close()

if __name__ == "__main__":
    main()
