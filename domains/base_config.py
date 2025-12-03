#!/usr/bin/env python3
"""
Base Domain Configuration
All domains must extend this abstract base class to ensure consistent interface
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional


class BaseDomainConfig(ABC):
    """
    Abstract base class for domain configurations.
    
    Each domain (UDL, Neuro, Math, etc.) must implement all methods below.
    This ensures consistency and makes it easy to add new domains.
    """
    
    def __init__(self):
        """Initialize domain configuration"""
        self.name = self._get_name()
        self.display_name = self._get_display_name()
        self.icon = self._get_icon()
    
    # ============================================================
    # CORE DOMAIN IDENTITY
    # ============================================================
    
    @abstractmethod
    def _get_name(self) -> str:
        """
        Return domain identifier (lowercase, no spaces)
        
        Returns:
            str: Domain identifier (e.g., 'udl', 'neuro', 'math')
        """
        pass
    
    @abstractmethod
    def _get_display_name(self) -> str:
        """
        Return human-readable domain name
        
        Returns:
            str: Display name (e.g., 'UDL (Universal Design for Learning)')
        """
        pass
    
    @abstractmethod
    def _get_icon(self) -> str:
        """
        Return emoji icon for this domain
        
        Returns:
            str: Emoji icon (e.g., '🎯', '🧠', '➗')
        """
        pass
    
    # ============================================================
    # NODE2VEC CONFIGURATION (train_node2vec.py)
    # ============================================================
    
    @abstractmethod
    def get_node2vec_weights(self) -> Dict[str, float]:
        """
        Return Node2Vec label weights for this domain.
        Higher weights = more important for random walks.
        
        Returns:
            dict: Label -> weight mapping (e.g., {'Attention': 2.2, 'WorkingMemory': 1.7})
        """
        pass
    
    # ============================================================
    # GRAPH RETRIEVAL CONFIGURATION (graph_retriever.py)
    # ============================================================
    
    @abstractmethod
    def get_retrieval_boosts(self) -> Dict[str, Dict[str, float]]:
        """
        Return retrieval boosts for labels and relationships.
        Boosts prioritize certain node types and relationships during retrieval.
        
        Returns:
            dict: {
                'label_boosts': {'StudentWithSpecialNeeds': 1.5, ...},
                'relationship_boosts': {'hasMethodology': 1.4, ...}
            }
        """
        pass
    
    @abstractmethod
    def get_similarity_threshold(self) -> float:
        """
        Return semantic similarity threshold for Node2Vec.
        Higher = stricter (fewer similar concepts), Lower = broader (more similar concepts)
        
        Returns:
            float: Threshold between 0.0 and 1.0 (e.g., 0.70 for Neuro, 0.80 for UDL)
        """
        pass
    
    # ============================================================
    # TEXT2CYPHER CONFIGURATION (text2cypher.py)
    # ============================================================
    
    @abstractmethod
    def get_few_shot_examples(self, domain: str = None) -> str:
        """
        Return few-shot examples for Cypher query generation.
        These examples teach the LLM how to generate Cypher for this domain.
        
        Args:
            domain: Optional domain name for placeholder replacement (e.g., 'neuro')
        
        Returns:
            str: Examples in "Question: ... Cypher: ..." format
            
        Example:
            '''
            Question: "Ci sono strategie per studenti ipovedenti?"
            Cypher: MATCH (s:StudentWithSpecialNeeds {name: 'Blind'})-[r:SUGGESTS]->(m) RETURN m
            
            Question: "What is metacognition?"
            Cypher: MATCH (m:Metacognition) RETURN m, labels(m) as node_labels LIMIT 10
            '''
        """
        pass
    
    @abstractmethod
    def get_cypher_patterns(self) -> str:
        """
        Return domain-specific Cypher query patterns for system prompt.
        These patterns guide the LLM on typical query structures for this domain.
        
        Returns:
            str: Multi-line string with query patterns
            
        Example:
            '''
            QUERY PATTERNS (UDL):
            - Student needs: MATCH (s:StudentWithSpecialNeeds)-[r:SUGGESTS]->(m)
            - Methodologies: MATCH (m:PedagogicalMethodology)-[r]->(resource)
            '''
        """
        pass
    
    @abstractmethod
    def repair_cypher_query(self, query: str) -> str:
        """
        Domain-specific Cypher query repair logic.
        Fixes common errors in generated Cypher (e.g., case mismatches, typos).
        
        Args:
            query: Raw Cypher query (may have errors)
            
        Returns:
            str: Repaired Cypher query
            
        Example:
            Input:  "MATCH (s:studentwithspecialneeds)"
            Output: "MATCH (s:StudentWithSpecialNeeds)"
        """
        pass
    
    # ============================================================
    # MULTILINGUAL CONFIGURATION (multilingual_text2cypher.py)
    # ============================================================
    
    @abstractmethod
    def get_italian_terms(self) -> Dict[str, str]:
        """
        Return Italian→English term mapping for this domain.
        Used to translate Italian queries to English for better Neo4j matching.
        
        Returns:
            dict: Italian term -> English term mapping
            
        Example:
            {
                'ipovedenti': 'Blind',
                'ADHD': 'Adhd',
                'memoria di lavoro': 'Working Memory'
            }
        """
        pass
    
    @abstractmethod
    def get_query_context(self) -> str:
        """
        Return domain context string for query enhancement.
        Used in prompts to provide context about what this domain covers.
        
        Returns:
            str: Domain context (e.g., 'education and Universal Design for Learning')
        """
        pass
    
    # ============================================================
    # LLM CHAIN CONFIGURATION (llm_chain.py)
    # ============================================================
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Return domain-specific system prompt for LLM response generation.
        Defines the LLM's role and expertise for this domain.
        
        Returns:
            str: System prompt in Italian
            
        Example:
            '''Sei un esperto di neuroscienze dell'apprendimento italiano,
            specializzato nell'applicazione pratica delle scoperte neuroscientifiche
            all'educazione.'''
        """
        pass
    
    # ============================================================
    # CONTEXT BUILDER CONFIGURATION (context_builder.py)
    # ============================================================
    
    @abstractmethod
    def get_methodology_categories(self) -> Dict:
        """
        Return methodology categories for this domain.
        Defines teaching methodologies, their characteristics, and applications.
        
        Returns:
            dict: Methodology name -> configuration dict
            
        Example:
            {
                'Cooperative Learning': {
                    'category': 'Collaborative Pedagogy',
                    'best_for': ['social_interaction', 'peer_learning'],
                    'implementation': 'Organize students in groups...',
                    'applications': ['Jigsaw method', ...],
                    'special_needs_adaptations': [...]
                }
            }
        """
        pass
    
    @abstractmethod
    def get_special_needs_mapping(self) -> Dict:
        """
        Return special needs mapping for this domain.
        Defines student needs and recommended methodologies.
        
        Returns:
            dict: Special need name -> configuration dict
            
        Example:
            {
                'Adhd': {
                    'primary_characteristics': ['attention_difficulties', ...],
                    'recommended_methodologies': ['Cooperative Learning', ...],
                    'support_needs': [...]
                }
            }
        """
        pass
    
    @abstractmethod
    def get_educational_context_type(self) -> str:
        """
        Return educational context type for this domain.
        Used by context builder to determine focus area.
        
        Returns:
            str: Context type ('special_needs', 'neuroscience', 'general', etc.)
        """
        pass
    
    # ============================================================
    # HELPER METHODS (optional overrides)
    # ============================================================
    
    def get_description(self) -> str:
        """
        Return domain description (optional, can be overridden)
        
        Returns:
            str: Domain description
        """
        return f"{self.display_name} domain configuration"
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate domain configuration (optional, can be overridden)
        
        Returns:
            tuple: (is_valid, list_of_errors)
        """
        errors = []
        
        # Check that name is lowercase and alphanumeric
        if not self.name.islower() or not self.name.isalnum():
            errors.append(f"Domain name '{self.name}' must be lowercase alphanumeric")
        
        # Check that Node2Vec weights are valid
        weights = self.get_node2vec_weights()
        if not weights:
            errors.append("Node2Vec weights cannot be empty")
        for label, weight in weights.items():
            if not isinstance(weight, (int, float)) or weight <= 0:
                errors.append(f"Invalid weight for label '{label}': {weight}")
        
        # Check similarity threshold
        threshold = self.get_similarity_threshold()
        if not (0.0 <= threshold <= 1.0):
            errors.append(f"Similarity threshold must be between 0.0 and 1.0, got {threshold}")
        
        # Check few-shot examples
        examples = self.get_few_shot_examples()
        if not examples:
            errors.append("Few-shot examples cannot be empty")
        
        return len(errors) == 0, errors
    
    def __repr__(self) -> str:
        """String representation"""
        return f"<{self.__class__.__name__}: {self.display_name} ({self.name})>"


