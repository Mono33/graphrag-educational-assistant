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
        Used by the legacy GraphRAG mode (llm_chain.py) as the standalone system prompt.
        Fetched from Langfuse at runtime (prompt name: ``{domain}.system_prompt``).
        """
        pass

    @abstractmethod
    def get_writer_prompt(self) -> str:
        """
        Return the domain expertise block for the agent-mode writer.
        Appended to the base writer prompt in domain_prompts.get_domain_extension().
        Fetched from Langfuse at runtime (prompt name: ``{domain}.writer_prompt``).

        Starts with the same text as get_system_prompt() but can diverge independently
        in Langfuse to tune agent-specific behaviour without affecting legacy mode.
        """
        pass

    def get_response_template(self) -> str:
        """
        Return domain-specific response formatting instructions.
        Defines HOW the output should be structured for this domain.
        
        This method provides the domain-specific part of the user message
        that tells the LLM how to format and structure its response.
        Each domain can define its own output structure (e.g., lesson schema,
        UDL framework, etc.) while sharing the same KG context format.
        
        Override this in domain subclasses. The default provides generic
        formatting instructions suitable for any educational domain.
        
        Returns:
            str: Response formatting instructions in Italian
        """
        return """ISTRUZIONI PER LA RISPOSTA:

1. **Inizia con un'introduzione empatica** che riconosca la domanda dell'insegnante
2. **Presenta le metodologie principali** (massimo 3) in modo chiaro e strutturato:
   - Nome della metodologia
   - Perché è efficace per questo contesto specifico
   - Come implementarla in classe (passi concreti)
   - Adattamenti per bisogni speciali (se applicabile)
3. **Fornisci esempi pratici** per ogni metodologia
4. **Includi le basi teoriche** spiegando da dove provengono queste raccomandazioni
5. **Suggerisci un ordine di implementazione** con priorità chiare
6. **Aggiungi note sulla fiducia**: se la confidenza è bassa, suggerisci di consultare specialisti
7. **Usa un tono professionale ma accessibile**, evita il gergo eccessivo
8. **Formatta con elenchi puntati e sezioni chiare** per facilitare la lettura

IMPORTANTE:
- Rispondi SEMPRE in italiano
- Sii concreto e pratico, non teorico
- Fornisci azioni immediate che l'insegnante può prendere
- Adatta il linguaggio al contesto scolastico italiano (primaria, secondaria, etc.)
- Se la confidenza è BASSA o VERY_LOW, enfatizza la necessità di supporto specialistico"""
    
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
    
    def get_label_category_map(self) -> Dict[str, str]:
        """
        Return mapping from Neo4j node labels to human-readable category names.
        Used by context_builder to classify nodes properly instead of
        labeling everything as 'Educational Methodology'.
        
        Override in domain subclasses with domain-specific mappings.
        
        Returns:
            dict: Neo4j label -> display category (e.g., {'Attention': 'Cognitive Process'})
        """
        return {}
    
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


