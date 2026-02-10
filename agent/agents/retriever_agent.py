"""
Retriever Agent

Executes GraphRAG searches based on the planner's retrieval plan.
Acts as the bridge between planning and content generation.

Enhanced with Media Lookup (Phase 1) for multimodal content support.
Media lookup is optional and fails gracefully for backward compatibility.
"""

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from agent.tools.graphrag_tool import GraphRAGTool, GraphRAGResult
from agent.agents.planner_agent import RetrievalPlan

# Optional media lookup - fails gracefully if not available
try:
    from agent.media import MediaLookup
    MEDIA_LOOKUP_AVAILABLE = True
except ImportError:
    MEDIA_LOOKUP_AVAILABLE = False
    MediaLookup = None

# Optional external APIs for hybrid retrieval (Phase A)
try:
    from agent.media.external_apis import ExternalMediaAPI
    EXTERNAL_APIS_AVAILABLE = True
except ImportError:
    EXTERNAL_APIS_AVAILABLE = False
    ExternalMediaAPI = None

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Combined results from multiple GraphRAG searches"""
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    search_results: List[GraphRAGResult] = field(default_factory=list)
    confidence: str = "MEDIUM"
    # NEW: Curated media from sidecar JSON (Phase 1 - optional, backward compatible)
    curated_media: Dict[str, Any] = field(default_factory=dict)
    # NEW Phase A: External resources for out-of-scope queries
    external_resources: Dict[str, Any] = field(default_factory=dict)
    # NEW Phase A: Scope status passed through from planner
    scope_status: str = "in_scope"
    
    @property
    def total_nodes(self) -> int:
        return len(self.nodes)
    
    @property
    def total_relationships(self) -> int:
        return len(self.relationships)
    
    @property
    def has_media(self) -> bool:
        """Check if curated media is available"""
        return bool(self.curated_media)
    
    @property
    def has_external_resources(self) -> bool:
        """Check if external resources were fetched (for out-of-scope queries)"""
        return bool(self.external_resources)
    
    @property
    def is_hybrid(self) -> bool:
        """Check if this is a hybrid result (KG + external sources)"""
        return self.scope_status in ("partial_scope", "out_of_scope") and self.has_external_resources
    
    def to_context_string(self, include_media: bool = True) -> str:
        """
        Format results as context string for the writer.
        
        Args:
            include_media: Whether to include curated media context (default True)
            
        Returns:
            Formatted context string
        """
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
        
        # NEW: Curated media context (Phase 1)
        if include_media and self.curated_media:
            lines.append("\n## 📚 Available Educational Media")
            
            # Videos
            videos = self.curated_media.get('videos', [])
            if videos:
                lines.append("\n### 🎥 Video suggeriti:")
                for v in videos[:5]:
                    title = v.get('title', 'Video')
                    url = v.get('url') or v.get('suggested_url')
                    duration = v.get('duration_hint', '')
                    if url:
                        lines.append(f"- [{title}]({url}) {f'({duration})' if duration else ''}")
                    else:
                        search_q = v.get('search_query', title)
                        lines.append(f"- Cerca su YouTube: \"{search_q}\"")
            
            # Resources
            resources = self.curated_media.get('resources', [])
            if resources:
                lines.append("\n### 🔗 Risorse educative:")
                for r in resources[:5]:
                    title = r.get('title', 'Resource')
                    url = r.get('url') or r.get('suggested_url')
                    if url:
                        lines.append(f"- [{title}]({url})")
                    else:
                        lines.append(f"- {title}")
            
            # Citations
            citations = self.curated_media.get('citations', [])
            if citations:
                lines.append("\n### 📖 Riferimenti scientifici:")
                for c in citations[:3]:
                    authors = c.get('authors', [])
                    authors_str = ', '.join(authors[:2])
                    if len(authors) > 2:
                        authors_str += ' et al.'
                    year = c.get('year', '')
                    title = c.get('title', '')
                    journal = c.get('journal', '')
                    doi = c.get('doi')
                    
                    cite_line = f"- {authors_str}"
                    if year:
                        cite_line += f" ({year})"
                    cite_line += f". *{title}*"
                    if journal:
                        cite_line += f". {journal}"
                    if doi:
                        cite_line += f" DOI: {doi}"
                    lines.append(cite_line)
            
            # Open Textbooks (OER)
            textbooks = self.curated_media.get('open_textbooks', [])
            if textbooks:
                lines.append("\n### 📚 Libri di testo aperti (OER):")
                for t in textbooks[:3]:
                    title = t.get('title', 'Textbook')
                    source = t.get('source', 'Unknown')
                    chapter = t.get('chapter', '')
                    url = t.get('url')
                    license_type = t.get('license', 'CC BY 4.0')
                    
                    book_line = f"- **{title}**"
                    if chapter:
                        book_line += f" - {chapter}"
                    book_line += f" ({source}, {license_type})"
                    if url:
                        book_line += f" [{url}]"
                    lines.append(book_line)
        
        return '\n'.join(lines)


class RetrieverAgent:
    """
    Retriever Agent - Executes knowledge graph searches.
    
    Responsibilities:
    1. Take the retrieval plan from PlannerAgent
    2. Execute multiple GraphRAG searches
    3. Combine and deduplicate results
    4. Prepare context for WriterAgent
    5. (NEW Phase 1) Fetch curated media from sidecar JSON
    6. (NEW Phase A) Hybrid retrieval: External APIs for out-of-scope subjects
    """
    
    def __init__(self, domain: str = "neuro", enable_media_lookup: bool = True, enable_external_apis: bool = True):
        """
        Initialize the Retriever Agent.
        
        Args:
            domain: Knowledge domain ("neuro" or "udl")
            enable_media_lookup: Whether to enable curated media lookup (default True)
            enable_external_apis: Whether to enable external API calls for out-of-scope (default True)
        """
        self.domain = domain
        self.enable_media_lookup = enable_media_lookup
        self.enable_external_apis = enable_external_apis
        self._tool: Optional[GraphRAGTool] = None
        self._media_lookup: Optional[Any] = None  # Lazy loaded MediaLookup
        self._external_api: Optional[Any] = None  # Lazy loaded ExternalMediaAPI
    
    def _get_tool(self) -> GraphRAGTool:
        """Lazy initialization of GraphRAG tool"""
        if self._tool is None:
            self._tool = GraphRAGTool(domain=self.domain)
        return self._tool
    
    def _get_media_lookup(self) -> Optional[Any]:
        """
        Lazy initialization of MediaLookup (Phase 1).
        Returns None if media lookup is disabled or unavailable.
        """
        if not self.enable_media_lookup:
            return None
        
        if not MEDIA_LOOKUP_AVAILABLE:
            logger.debug("[RetrieverAgent] MediaLookup not available")
            return None
        
        if self._media_lookup is None:
            try:
                self._media_lookup = MediaLookup(domain=self.domain)
                if self._media_lookup.loaded:
                    logger.info(
                        f"[RetrieverAgent] MediaLookup initialized: "
                        f"{len(self._media_lookup.media_by_concept)} concepts"
                    )
                else:
                    logger.warning("[RetrieverAgent] MediaLookup loaded but no data")
                    self._media_lookup = None
            except Exception as e:
                logger.warning(f"[RetrieverAgent] MediaLookup init failed: {e}")
                self._media_lookup = None
        
        return self._media_lookup
    
    def _get_external_api(self) -> Optional[Any]:
        """
        Lazy initialization of ExternalMediaAPI (Phase A).
        Returns None if external APIs are disabled or unavailable.
        """
        if not self.enable_external_apis:
            return None
        
        if not EXTERNAL_APIS_AVAILABLE:
            logger.debug("[RetrieverAgent] ExternalMediaAPI not available")
            return None
        
        if self._external_api is None:
            try:
                self._external_api = ExternalMediaAPI()
                logger.info("[RetrieverAgent] ExternalMediaAPI initialized")
            except Exception as e:
                logger.warning(f"[RetrieverAgent] ExternalMediaAPI init failed: {e}")
                self._external_api = None
        
        return self._external_api
    
    async def retrieve(self, plan: RetrievalPlan) -> RetrievalResult:
        """
        Execute GraphRAG searches based on the retrieval plan.
        
        Args:
            plan: RetrievalPlan from PlannerAgent
            
        Returns:
            RetrievalResult with combined search results (and curated media if available)
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
        
        # ============================================
        # NEW Phase 1: Curated Media Lookup
        # ============================================
        # This is additive - if it fails, retrieval still succeeds
        result.curated_media = self._fetch_curated_media(result.nodes, plan.key_concepts)
        
        # ============================================
        # NEW Phase A: Hybrid Retrieval for Out-of-Scope
        # ============================================
        # If scope is partial or out-of-scope, fetch external resources
        result.scope_status = plan.scope_status
        
        if plan.needs_external_apis:
            external_resources = await self._fetch_external_resources(
                subject_concepts=plan.subject_concepts or [],
                pedagogy_concepts=plan.pedagogy_concepts or plan.key_concepts
            )
            result.external_resources = external_resources
            
            scope_emoji = {"partial_scope": "⚠️", "out_of_scope": "❌"}.get(plan.scope_status, "❓")
            logger.info(
                f"[RetrieverAgent] {scope_emoji} HYBRID retrieval: "
                f"External resources fetched for {plan.subject_concepts}"
            )
        
        # Log results
        media_str = ""
        if result.curated_media:
            media_count = (
                len(result.curated_media.get('videos', [])) +
                len(result.curated_media.get('resources', [])) +
                len(result.curated_media.get('citations', []))
            )
            media_str = f", {media_count} media items"
        
        external_str = ""
        if result.external_resources:
            external_count = sum(len(v) if isinstance(v, list) else 1 for v in result.external_resources.values() if v)
            external_str = f", {external_count} external resources"
        
        logger.info(
            f"[RetrieverAgent] Retrieved total: {result.total_nodes} nodes, "
            f"{result.total_relationships} relationships, "
            f"{len(result.recommendations)} recommendations{media_str}{external_str}"
        )
        
        return result
    
    def _fetch_curated_media(
        self, 
        nodes: List[Dict[str, Any]], 
        key_concepts: List[str]
    ) -> Dict[str, Any]:
        """
        Fetch curated media from sidecar JSON based on retrieved concepts.
        
        This is a graceful operation - failures don't affect main retrieval.
        
        Args:
            nodes: Retrieved nodes from GraphRAG
            key_concepts: Key concepts from planner
            
        Returns:
            Dict with videos, resources, citations (empty dict on failure)
        """
        media_lookup = self._get_media_lookup()
        if not media_lookup:
            return {}
        
        try:
            # Collect concept names to look up
            concept_names = set()
            
            # From retrieved nodes
            for node in nodes[:15]:  # Limit to top 15 nodes
                name = node.get('name')
                if name:
                    concept_names.add(name)
            
            # From planner's key concepts
            for concept in key_concepts:
                concept_names.add(concept)
            
            if not concept_names:
                return {}
            
            # Get combined media for all concepts
            combined_media = media_lookup.get_combined_media(list(concept_names))
            
            if not combined_media.has_content():
                logger.debug("[RetrieverAgent] No curated media found for concepts")
                return {}
            
            # Convert to dict for state serialization
            media_dict = {
                'videos': [
                    {
                        'title': v.title,
                        'platform': v.platform,
                        'url': v.url,
                        'search_query': v.search_query,
                        'duration_hint': v.duration_hint
                    }
                    for v in combined_media.videos[:5]
                ],
                'images': [
                    {
                        'description': i.description,
                        'search_query': i.search_query,
                        'type': i.type
                    }
                    for i in combined_media.images[:3]
                ],
                'resources': [
                    {
                        'title': r.title,
                        'type': r.type,
                        'url': r.url,
                        'suggested_url': r.suggested_url
                    }
                    for r in combined_media.resources[:5]
                ],
                'citations': [
                    {
                        'title': c.title,
                        'authors': c.authors,
                        'year': c.year,
                        'journal': c.journal,
                        'doi': c.doi
                    }
                    for c in combined_media.citations[:3]
                ],
                'open_textbooks': [
                    {
                        'title': t.title,
                        'source': t.source,
                        'chapter': t.chapter,
                        'url': t.url,
                        'license': t.license,
                        'relevance': t.relevance
                    }
                    for t in combined_media.open_textbooks[:3]
                ]
            }
            
            logger.info(
                f"[RetrieverAgent] Found curated media: "
                f"{len(media_dict['videos'])} videos, "
                f"{len(media_dict['resources'])} resources, "
                f"{len(media_dict['citations'])} citations, "
                f"{len(media_dict['open_textbooks'])} textbooks"
            )
            
            return media_dict
            
        except Exception as e:
            logger.warning(f"[RetrieverAgent] Media lookup failed (non-critical): {e}")
            return {}
    
    async def _fetch_external_resources(
        self,
        subject_concepts: List[str],
        pedagogy_concepts: List[str]
    ) -> Dict[str, Any]:
        """
        Fetch external resources for out-of-scope subject content.
        
        This is Phase A hybrid retrieval - uses Wikipedia, Semantic Scholar
        to get subject content when the topic is outside the KG domain.
        
        Args:
            subject_concepts: Subject-specific concepts (e.g., "heliocentrism")
            pedagogy_concepts: Teaching strategies from KG (for reference)
            
        Returns:
            Dict with wikipedia, papers, oer_textbooks (empty dict on failure)
        """
        external_api = self._get_external_api()
        if not external_api:
            logger.warning("[RetrieverAgent] External API unavailable for hybrid retrieval")
            return {}
        
        resources = {
            'wikipedia': [],
            'papers': [],
            'oer_textbooks': [],
            'source_attribution': 'external'  # Mark as external source
        }
        
        try:
            # Fetch Wikipedia summaries for subject concepts
            for concept in subject_concepts[:3]:  # Limit to 3 concepts
                try:
                    wiki = await external_api.get_wikipedia_summary(concept, language="it")
                    if wiki:
                        resources['wikipedia'].append({
                            'title': wiki.title,
                            'summary': wiki.summary[:500],  # Truncate for context
                            'url': wiki.url,
                            'thumbnail_url': wiki.thumbnail_url,
                            'concept': concept
                        })
                        logger.info(f"[RetrieverAgent] Wikipedia found: {wiki.title}")
                except Exception as e:
                    logger.debug(f"[RetrieverAgent] Wikipedia failed for '{concept}': {e}")
            
            # Fetch academic papers from Semantic Scholar
            for concept in subject_concepts[:2]:  # Limit to 2 concepts
                try:
                    papers = await external_api.search_semantic_scholar(
                        query=f"{concept} education teaching",
                        max_results=3,
                        open_access_only=True
                    )
                    for paper in papers:
                        resources['papers'].append({
                            'title': paper.title,
                            'authors': paper.authors[:3],  # First 3 authors
                            'year': paper.year,
                            'abstract': (paper.abstract or '')[:300],
                            'url': paper.url,
                            'citation_count': paper.citation_count,
                            'concept': concept
                        })
                    if papers:
                        logger.info(f"[RetrieverAgent] Semantic Scholar found: {len(papers)} papers for '{concept}'")
                except Exception as e:
                    logger.debug(f"[RetrieverAgent] Semantic Scholar failed for '{concept}': {e}")
            
            # =========================================================
            # NEW: Fetch OER Textbooks (Domain Expert Trusted Sources)
            # =========================================================
            # These are from DOAB, Open Textbook Library, BC Campus
            # Approved by domain experts as copyright-safe sources
            for concept in subject_concepts[:2]:  # Limit to 2 concepts
                try:
                    textbooks = await external_api.search_oer_textbooks(
                        query=f"{concept} education",
                        max_results=3
                    )
                    for textbook in textbooks:
                        resources['oer_textbooks'].append({
                            'title': textbook.title,
                            'source': textbook.source,
                            'url': textbook.url,
                            'authors': textbook.authors,
                            'subject': textbook.subject,
                            'description': textbook.description,
                            'license': textbook.license,
                            'relevance_note': textbook.relevance_note,
                            'concept': concept
                        })
                    if textbooks:
                        logger.info(f"[RetrieverAgent] OER Textbooks found: {len(textbooks)} for '{concept}'")
                except Exception as e:
                    logger.debug(f"[RetrieverAgent] OER search failed for '{concept}': {e}")
            
            logger.info(
                f"[RetrieverAgent] External resources: "
                f"{len(resources['wikipedia'])} Wikipedia, "
                f"{len(resources['papers'])} papers, "
                f"{len(resources['oer_textbooks'])} OER textbooks"
            )
            
            return resources
            
        except Exception as e:
            logger.warning(f"[RetrieverAgent] External resource fetch failed: {e}")
            return {}
    
    async def retrieve_single(self, query: str) -> RetrievalResult:
        """
        Execute a single GraphRAG search.
        
        Args:
            query: Search query
            
        Returns:
            RetrievalResult
        """
        plan = RetrievalPlan(
            query_intent="search",
            key_concepts=[],
            search_queries=[query]
        )
        return await self.retrieve(plan)
    
    def retrieve_sync(self, plan: RetrievalPlan) -> RetrievalResult:
        """Synchronous version of retrieve()"""
        import asyncio
        return asyncio.run(self.retrieve(plan))

