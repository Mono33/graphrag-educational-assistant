"""
GraphRAG Tool - Bridge to Existing GraphRAG Engine

This tool wraps the existing GraphRAG stack (text2cypher, graph_retriever,
context_builder) and exposes it as a callable tool for the agent pipeline.

IMPORTANT: This file IMPORTS the existing GraphRAG code but does NOT modify it.
The original files remain 100% unchanged.
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GraphRAGResult:
    """Result from a GraphRAG search"""
    nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    confidence: str
    query_translated: Optional[str] = None
    cypher_query: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class GraphRAGTool:
    """
    Wraps the existing GraphRAG stack as an agent tool.
    
    This is the ONLY connection between the agent layer and your existing code.
    Your existing code (graph_retriever.py, context_builder.py, etc.) remains
    completely unchanged.
    
    Usage:
        tool = GraphRAGTool(domain="neuro")
        result = await tool.search("Strategie per studenti con ADHD")
    """
    
    name = "graphrag_search"
    description = """
    Search the educational knowledge graph for teaching strategies, 
    methodologies, and neuroscience concepts. Use this tool when you need
    evidence-based educational information to include in lesson plans.
    
    Input: A natural language query in Italian or English
    Output: Retrieved nodes, relationships, and recommendations
    """
    
    def __init__(self, domain: str = "neuro", use_vectors: bool = True):
        """
        Initialize the GraphRAG tool.
        
        Args:
            domain: Knowledge domain ("neuro" or "udl")
            use_vectors: Whether to use Node2Vec semantic search
        """
        self.domain = domain
        self.use_vectors = use_vectors
        self._initialized = False
        
        # Lazy initialization to avoid import errors during module load
        self._text2cypher = None
        self._retriever = None
        self._context_builder = None
    
    def _ensure_initialized(self):
        """Lazy initialization of GraphRAG components"""
        if self._initialized:
            return
        
        try:
            # Import your EXISTING GraphRAG components (no modifications!)
            # Note: EnhancedMultilingualText2Cypher is in graph_retriever.py, not multilingual_text2cypher.py
            from graph_retriever import EnhancedMultilingualText2Cypher
            from context_builder import EducationalContextBuilder
            
            logger.info(f"[GraphRAGTool] Initializing for domain: {self.domain}")
            
            # Initialize components using YOUR EXISTING code (same as streamlit_app.py)
            # EnhancedMultilingualText2Cypher creates its own HybridGraphRetriever internally
            self._text2cypher = EnhancedMultilingualText2Cypher(
                domain=self.domain,
                use_vectors=self.use_vectors,
                config={'max_nodes': 15, 'max_edges': 30}
            )
            
            # Use the retriever from text2cypher (it has the Neo4j driver already)
            self._retriever = self._text2cypher.graph_retriever
            self._context_builder = EducationalContextBuilder(domain=self.domain)
            
            self._initialized = True
            logger.info(f"[GraphRAGTool] Initialization complete")
            
        except ImportError as e:
            logger.error(f"[GraphRAGTool] Failed to import GraphRAG components: {e}")
            raise RuntimeError(
                "GraphRAG components not found. Make sure you're running from "
                "the graphaixlearning directory."
            ) from e
    
    async def search(self, query: str) -> GraphRAGResult:
        """
        Execute a GraphRAG search using the existing pipeline.
        
        Args:
            query: Natural language query in Italian or English
            
        Returns:
            GraphRAGResult with nodes, relationships, and recommendations
        """
        self._ensure_initialized()
        
        logger.info(f"[GraphRAGTool] Searching: {query[:50]}...")
        
        try:
            # Use the combined method from EnhancedMultilingualText2Cypher
            # This handles: Text2Cypher -> Retrieval -> Context Building
            result_dict = await self._text2cypher.process_query_with_retrieval(query, domain=self.domain)
            
            # Extract data from the NESTED result structure
            # The retrieval_result is a RetrievedContext object, not a dict
            retrieval_result = result_dict.get('retrieval_result')
            educational_context = result_dict.get('educational_context', {})
            cypher_result = result_dict.get('cypher_result', {})
            
            # Extract nodes from retrieval_result (which is a RetrievedContext object)
            nodes = []
            if retrieval_result:
                if hasattr(retrieval_result, 'nodes'):
                    nodes = retrieval_result.nodes  # List of dicts
                elif isinstance(retrieval_result, dict):
                    nodes = retrieval_result.get('nodes', [])
            
            # Extract triples/relationships from retrieval_result
            triples = []
            if retrieval_result:
                if hasattr(retrieval_result, 'triples'):
                    triples = retrieval_result.triples  # List of tuples: (source, rel_type, target)
                elif isinstance(retrieval_result, dict):
                    triples = retrieval_result.get('triples', [])
            
            # Convert triples (tuples) to relationships format
            relationships = []
            for triple in triples:
                if isinstance(triple, dict):
                    relationships.append({
                        'source': triple.get('source', ''),
                        'type': triple.get('relationship', ''),
                        'target': triple.get('target', '')
                    })
                elif isinstance(triple, (list, tuple)) and len(triple) >= 3:
                    relationships.append({
                        'source': str(triple[0]),
                        'type': str(triple[1]),
                        'target': str(triple[2])
                    })
            
            # Extract recommendations from educational_context
            # They are in primary_methodologies and supporting_methodologies
            recommendations = []
            if educational_context:
                primary = educational_context.get('primary_methodologies', [])
                supporting = educational_context.get('supporting_methodologies', [])
                recommendations = primary + supporting
            
            # Get confidence from educational_context
            confidence = educational_context.get('confidence_assessment', 'MEDIUM')
            if not confidence:
                confidence = 'MEDIUM'
            
            # Get Cypher query
            cypher_query = ''
            if cypher_result and isinstance(cypher_result, dict):
                cypher_query = cypher_result.get('cypher', '')
            
            result = GraphRAGResult(
                nodes=nodes,
                relationships=relationships,
                recommendations=recommendations,
                confidence=str(confidence),
                query_translated=None,
                cypher_query=cypher_query,
                metadata={
                    'total_nodes': len(nodes),
                    'total_relationships': len(relationships),
                    'total_recommendations': len(recommendations),
                    'domain': self.domain
                }
            )
            
            logger.info(
                f"[GraphRAGTool] Found {len(nodes)} nodes, "
                f"{len(relationships)} relationships, "
                f"{len(recommendations)} recommendations"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"[GraphRAGTool] Search failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def search_sync(self, query: str) -> GraphRAGResult:
        """
        Synchronous version of search for non-async contexts.
        """
        import asyncio
        return asyncio.run(self.search(query))
    
    def get_tool_schema(self) -> Dict[str, Any]:
        """
        Returns OpenAI function calling schema for this tool.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Educational query in Italian or English"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
