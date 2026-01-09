"""
Retriever Agent

Executes GraphRAG searches based on the planner's retrieval plan.
Acts as the bridge between planning and content generation.
"""

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from agent.tools.graphrag_tool import GraphRAGTool, GraphRAGResult
from agent.agents.planner_agent import RetrievalPlan

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Combined results from multiple GraphRAG searches"""
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    search_results: List[GraphRAGResult] = field(default_factory=list)
    confidence: str = "MEDIUM"
    
    @property
    def total_nodes(self) -> int:
        return len(self.nodes)
    
    @property
    def total_relationships(self) -> int:
        return len(self.relationships)
    
    def to_context_string(self) -> str:
        """Format results as context string for the writer"""
        lines = []
        
        # Recommendations
        if self.recommendations:
            lines.append("## Recommended Methodologies")
            for i, rec in enumerate(self.recommendations[:10], 1):
                name = rec.get('name', 'Unknown')
                desc = rec.get('description', '')[:200]
                confidence = rec.get('confidence', 'MEDIUM')
                lines.append(f"{i}. **{name}** ({confidence})")
                if desc:
                    lines.append(f"   {desc}")
        
        # Key nodes
        if self.nodes:
            lines.append("\n## Retrieved Concepts")
            seen_names = set()
            for node in self.nodes[:20]:
                name = node.get('name', 'Unknown')
                if name not in seen_names:
                    seen_names.add(name)
                    labels = node.get('labels', [])
                    label_str = ', '.join(labels) if labels else ''
                    lines.append(f"- {name} ({label_str})")
        
        return '\n'.join(lines)


class RetrieverAgent:
    """
    Retriever Agent - Executes knowledge graph searches.
    
    Responsibilities:
    1. Take the retrieval plan from PlannerAgent
    2. Execute multiple GraphRAG searches
    3. Combine and deduplicate results
    4. Prepare context for WriterAgent
    """
    
    def __init__(self, domain: str = "neuro"):
        """
        Initialize the Retriever Agent.
        
        Args:
            domain: Knowledge domain ("neuro" or "udl")
        """
        self.domain = domain
        self._tool: Optional[GraphRAGTool] = None
    
    def _get_tool(self) -> GraphRAGTool:
        """Lazy initialization of GraphRAG tool"""
        if self._tool is None:
            self._tool = GraphRAGTool(domain=self.domain)
        return self._tool
    
    async def retrieve(self, plan: RetrievalPlan) -> RetrievalResult:
        """
        Execute GraphRAG searches based on the retrieval plan.
        
        Args:
            plan: RetrievalPlan from PlannerAgent
            
        Returns:
            RetrievalResult with combined search results
        """
        logger.info(
            f"[RetrieverAgent] Executing {len(plan.search_queries)} searches..."
        )
        
        tool = self._get_tool()
        result = RetrievalResult()
        
        # Execute each search query
        for query in plan.search_queries:
            try:
                search_result = await tool.search(query)
                result.search_results.append(search_result)
                
                # Combine nodes (deduplicate by name)
                existing_names = {n.get('name') for n in result.nodes}
                for node in search_result.nodes:
                    if node.get('name') not in existing_names:
                        result.nodes.append(node)
                        existing_names.add(node.get('name'))
                
                # Combine relationships
                result.relationships.extend(search_result.relationships)
                
                # Combine recommendations
                existing_recs = {r.get('name') for r in result.recommendations}
                for rec in search_result.recommendations:
                    if rec.get('name') not in existing_recs:
                        result.recommendations.append(rec)
                        existing_recs.add(rec.get('name'))
                
                logger.info(
                    f"[RetrieverAgent] Query '{query[:30]}...' returned "
                    f"{len(search_result.nodes)} nodes"
                )
                
            except Exception as e:
                logger.error(f"[RetrieverAgent] Search failed for '{query}': {e}")
                continue
        
        # Determine overall confidence
        confidences = [r.confidence for r in result.search_results if r.confidence]
        if confidences:
            # Use highest confidence
            confidence_order = ['VERY_HIGH', 'HIGH', 'MEDIUM', 'LOW', 'VERY_LOW']
            result.confidence = min(confidences, key=lambda c: confidence_order.index(c) if c in confidence_order else 99)
        
        logger.info(
            f"[RetrieverAgent] Retrieved total: {result.total_nodes} nodes, "
            f"{result.total_relationships} relationships, "
            f"{len(result.recommendations)} recommendations"
        )
        
        return result
    
    async def retrieve_single(self, query: str) -> RetrievalResult:
        """
        Execute a single GraphRAG search.
        
        Args:
            query: Search query
            
        Returns:
            RetrievalResult
        """
        plan = RetrievalPlan(
            lesson_type="search",
            key_concepts=[],
            search_queries=[query]
        )
        return await self.retrieve(plan)
    
    def retrieve_sync(self, plan: RetrievalPlan) -> RetrievalResult:
        """Synchronous version of retrieve()"""
        import asyncio
        return asyncio.run(self.retrieve(plan))

