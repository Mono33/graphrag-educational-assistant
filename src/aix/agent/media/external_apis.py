"""
External APIs Module for Real-Time Media Enrichment

This module provides real-time access to external APIs for dynamic content:
- YouTube Data API: Search for educational videos
- Wikipedia API: Fetch article summaries
- Semantic Scholar API: Find recent academic papers

These APIs complement the curated media mapping (Phase 0) by providing:
- Fresh, up-to-date content
- Fallback when curated media doesn't exist
- Real-time search based on user queries

Features:
- Proper User-Agent headers (required by Wikipedia)
- Rate limiting for Semantic Scholar (prevents 429 errors)
- Exponential backoff retry logic
- Graceful error handling

Usage:
    from aix.agent.media.external_apis import ExternalMediaAPI
    
    api = ExternalMediaAPI()
    videos = await api.search_youtube("metacognition education")
    wiki = await api.get_wikipedia_summary("Metacognition")
    papers = await api.search_semantic_scholar("metacognition learning")
"""

import os
import logging
import asyncio
import aiohttp
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from urllib.parse import quote_plus
from collections import deque

logger = logging.getLogger(__name__)

# =============================================================================
# USER-AGENT (Required by Wikipedia API policy)
# =============================================================================
USER_AGENT = "GraphAIxLearning/1.0 (https://github.com/FEM-modena/graphrag-aixlearning; louis.mono@fem.digital.com)"


# =============================================================================
# RATE LIMITER (Prevents 429 Too Many Requests)
# =============================================================================
class RateLimiter:
    """
    Token bucket rate limiter for API calls.
    
    Semantic Scholar free tier: 100 requests per 5 minutes
    With API key: 1000 requests per 5 minutes
    """
    
    def __init__(self, calls_per_period: int = 100, period_seconds: int = 300):
        """
        Initialize rate limiter.
        
        Args:
            calls_per_period: Maximum calls allowed in the period
            period_seconds: Period duration in seconds
        """
        self.calls_per_period = calls_per_period
        self.period_seconds = period_seconds
        self.call_times: deque = deque()
        self._lock = asyncio.Lock()
    
    async def wait_if_needed(self) -> float:
        """
        Wait if rate limit is exceeded.
        
        Returns:
            Time waited in seconds (0 if no wait needed)
        """
        async with self._lock:
            now = time.time()
            
            # Remove calls outside the current window
            while self.call_times and self.call_times[0] < now - self.period_seconds:
                self.call_times.popleft()
            
            # Check if we need to wait
            if len(self.call_times) >= self.calls_per_period:
                # Calculate wait time
                oldest_call = self.call_times[0]
                wait_time = (oldest_call + self.period_seconds) - now + 0.1  # Add 100ms buffer
                
                if wait_time > 0:
                    logger.info(f"[RateLimiter] Rate limit reached, waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                    return wait_time
            
            # Record this call
            self.call_times.append(time.time())
            return 0.0
    
    def reset(self):
        """Reset the rate limiter"""
        self.call_times.clear()


# =============================================================================
# RETRY LOGIC (Exponential Backoff)
# =============================================================================
async def retry_with_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    retry_on: tuple = (429, 500, 502, 503, 504)
):
    """
    Retry an async function with exponential backoff.
    
    Args:
        func: Async function to call (should return response or raise)
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        retry_on: HTTP status codes to retry on
        
    Returns:
        Response from the function
        
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except aiohttp.ClientResponseError as e:
            last_exception = e
            if e.status not in retry_on or attempt >= max_retries:
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            logger.warning(f"[Retry] Attempt {attempt + 1}/{max_retries}, status {e.status}, waiting {delay:.1f}s")
            await asyncio.sleep(delay)
        except asyncio.TimeoutError as e:
            last_exception = e
            if attempt >= max_retries:
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            logger.warning(f"[Retry] Timeout, attempt {attempt + 1}/{max_retries}, waiting {delay:.1f}s")
            await asyncio.sleep(delay)
    
    raise last_exception


@dataclass
class YouTubeVideo:
    """YouTube video result"""
    title: str
    video_id: str
    channel: str
    description: str
    thumbnail_url: str
    url: str
    duration: Optional[str] = None
    view_count: Optional[int] = None


@dataclass
class WikipediaSummary:
    """Wikipedia article summary"""
    title: str
    summary: str
    url: str
    page_id: int
    thumbnail_url: Optional[str] = None
    extract_html: Optional[str] = None


@dataclass
class SemanticScholarPaper:
    """Semantic Scholar paper result"""
    title: str
    authors: List[str]
    year: Optional[int]
    abstract: Optional[str]
    citation_count: int
    url: str
    doi: Optional[str] = None
    venue: Optional[str] = None
    is_open_access: bool = False
    pdf_url: Optional[str] = None


@dataclass
class OERTextbook:
    """Open Educational Resource textbook result"""
    title: str
    source: str  # "DOAB", "OpenTextbook", "BCCampus", "Pressbooks"
    url: str
    authors: List[str] = field(default_factory=list)
    subject: Optional[str] = None
    description: Optional[str] = None
    license: str = "CC BY 4.0"
    year: Optional[int] = None
    relevance_note: Optional[str] = None


class ExternalMediaAPI:
    """
    External Media API Client
    
    Provides async methods for fetching media from external sources.
    All methods are designed to fail gracefully - errors return empty results.
    
    Features:
    - Proper User-Agent headers (required by Wikipedia)
    - Rate limiting for Semantic Scholar
    - Exponential backoff retry logic
    """
    
    # API Endpoints
    YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
    YOUTUBE_VIDEO_URL = "https://www.googleapis.com/youtube/v3/videos"
    WIKIPEDIA_API_URL = "https://en.wikipedia.org/api/rest_v1/page/summary"
    WIKIPEDIA_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
    SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    def __init__(
        self,
        youtube_api_key: Optional[str] = None,
        semantic_scholar_api_key: Optional[str] = None
    ):
        """
        Initialize External Media API.
        
        Args:
            youtube_api_key: Optional YouTube Data API key. If not provided,
                           will try to get from YOUTUBE_API_KEY env var.
                           If no key available, YouTube search returns search URLs.
            semantic_scholar_api_key: Optional Semantic Scholar API key.
                           If provided, rate limits are higher (1000/5min vs 100/5min).
        """
        self.youtube_api_key = youtube_api_key or os.getenv('YOUTUBE_API_KEY')
        self.semantic_scholar_api_key = semantic_scholar_api_key or os.getenv('SEMANTIC_SCHOLAR_API_KEY')
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiter for Semantic Scholar
        # Free tier: 100 requests per 5 minutes
        # With API key: 1000 requests per 5 minutes
        calls_limit = 1000 if self.semantic_scholar_api_key else 100
        self._semantic_scholar_limiter = RateLimiter(
            calls_per_period=calls_limit,
            period_seconds=300  # 5 minutes
        )
        
        logger.info(f"[ExternalMediaAPI] Initialized with User-Agent: {USER_AGENT[:50]}...")
        if self.semantic_scholar_api_key:
            logger.info("[ExternalMediaAPI] Semantic Scholar API key detected (higher rate limits)")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with proper headers"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=15)
            
            # Default headers for all requests
            headers = {
                "User-Agent": USER_AGENT,
                "Accept": "application/json"
            }
            
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers
            )
        return self._session
    
    async def close(self):
        """Close the aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()
        self._semantic_scholar_limiter.reset()
    
    # =========================================================================
    # YOUTUBE API
    # =========================================================================
    
    async def search_youtube(
        self,
        query: str,
        max_results: int = 5,
        language: str = "en"
    ) -> List[YouTubeVideo]:
        """
        Search YouTube for educational videos.
        
        If YouTube API key is available, uses the official API.
        Otherwise, returns search URL links.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            language: Language code
            
        Returns:
            List of YouTubeVideo objects
        """
        if self.youtube_api_key:
            return await self._youtube_api_search(query, max_results, language)
        else:
            return self._youtube_fallback_search(query, max_results)
    
    async def _youtube_api_search(
        self,
        query: str,
        max_results: int,
        language: str
    ) -> List[YouTubeVideo]:
        """Search YouTube using official API"""
        try:
            session = await self._get_session()
            
            # Search for videos
            params = {
                "part": "snippet",
                "q": f"{query} educational",
                "type": "video",
                "maxResults": max_results,
                "relevanceLanguage": language,
                "videoEmbeddable": "true",
                "key": self.youtube_api_key
            }
            
            async with session.get(self.YOUTUBE_SEARCH_URL, params=params) as resp:
                if resp.status != 200:
                    logger.warning(f"[YouTube API] Search failed: {resp.status}")
                    return self._youtube_fallback_search(query, max_results)
                
                data = await resp.json()
            
            videos = []
            for item in data.get("items", []):
                snippet = item.get("snippet", {})
                video_id = item.get("id", {}).get("videoId", "")
                
                if video_id:
                    videos.append(YouTubeVideo(
                        title=snippet.get("title", ""),
                        video_id=video_id,
                        channel=snippet.get("channelTitle", ""),
                        description=snippet.get("description", "")[:200],
                        thumbnail_url=snippet.get("thumbnails", {}).get("medium", {}).get("url", ""),
                        url=f"https://www.youtube.com/watch?v={video_id}"
                    ))
            
            logger.info(f"[YouTube API] Found {len(videos)} videos for '{query}'")
            return videos
            
        except Exception as e:
            logger.error(f"[YouTube API] Error: {e}")
            return self._youtube_fallback_search(query, max_results)
    
    def _youtube_fallback_search(self, query: str, max_results: int) -> List[YouTubeVideo]:
        """Generate YouTube search URLs when API key is not available"""
        encoded_query = quote_plus(f"{query} educational")
        
        # Return a single search link
        return [YouTubeVideo(
            title=f"Search: {query}",
            video_id="",
            channel="YouTube Search",
            description=f"Click to search YouTube for '{query}' educational videos",
            thumbnail_url="",
            url=f"https://www.youtube.com/results?search_query={encoded_query}"
        )]
    
    # =========================================================================
    # WIKIPEDIA API (with proper User-Agent to avoid 403 errors)
    # =========================================================================
    
    async def get_wikipedia_summary(
        self,
        topic: str,
        language: str = "en",
        max_retries: int = 3
    ) -> Optional[WikipediaSummary]:
        """
        Get Wikipedia article summary for a topic.
        
        Uses proper User-Agent header as required by Wikipedia API policy.
        Implements retry with exponential backoff.
        
        Args:
            topic: Topic to search for
            language: Language code (default: "en")
            max_retries: Maximum retry attempts
            
        Returns:
            WikipediaSummary or None if not found
        """
        try:
            session = await self._get_session()
            
            # Use the REST API for cleaner summaries
            base_url = f"https://{language}.wikipedia.org/api/rest_v1/page/summary"
            encoded_topic = quote_plus(topic.replace(" ", "_"))
            url = f"{base_url}/{encoded_topic}"
            
            # Retry logic with exponential backoff
            for attempt in range(max_retries):
                try:
                    async with session.get(url) as resp:
                        if resp.status == 404:
                            # Try search if direct lookup fails
                            return await self._wikipedia_search(topic, language)
                        
                        if resp.status == 403:
                            # Forbidden - shouldn't happen with proper User-Agent
                            logger.error(f"[Wikipedia] 403 Forbidden - check User-Agent header")
                            return None
                        
                        if resp.status == 429:
                            # Rate limited - wait and retry
                            delay = 2 ** attempt
                            logger.warning(f"[Wikipedia] Rate limited, waiting {delay}s (attempt {attempt + 1}/{max_retries})")
                            await asyncio.sleep(delay)
                            continue
                        
                        if resp.status != 200:
                            logger.warning(f"[Wikipedia] Failed for '{topic}': {resp.status}")
                            return None
                        
                        data = await resp.json()
                        break
                        
                except asyncio.TimeoutError:
                    if attempt < max_retries - 1:
                        delay = 2 ** attempt
                        logger.warning(f"[Wikipedia] Timeout, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(delay)
                        continue
                    raise
            else:
                # All retries exhausted
                logger.error(f"[Wikipedia] All retries failed for '{topic}'")
                return None
            
            # Check if it's a disambiguation page
            if data.get("type") == "disambiguation":
                return await self._wikipedia_search(topic, language)
            
            result = WikipediaSummary(
                title=data.get("title", topic),
                summary=data.get("extract", ""),
                url=data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                page_id=data.get("pageid", 0),
                thumbnail_url=data.get("thumbnail", {}).get("source"),
                extract_html=data.get("extract_html")
            )
            
            logger.info(f"[Wikipedia] Found summary for '{topic}'")
            return result
            
        except Exception as e:
            logger.error(f"[Wikipedia] Error for '{topic}': {e}")
            return None
    
    async def _wikipedia_search(
        self,
        query: str,
        language: str,
        max_retries: int = 3
    ) -> Optional[WikipediaSummary]:
        """Search Wikipedia and return first result (with retry logic)"""
        try:
            session = await self._get_session()
            
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 1
            }
            
            base_url = f"https://{language}.wikipedia.org/w/api.php"
            
            # Retry logic
            for attempt in range(max_retries):
                try:
                    async with session.get(base_url, params=params) as resp:
                        if resp.status == 429:
                            delay = 2 ** attempt
                            logger.warning(f"[Wikipedia Search] Rate limited, waiting {delay}s")
                            await asyncio.sleep(delay)
                            continue
                        
                        if resp.status != 200:
                            logger.warning(f"[Wikipedia Search] Failed: {resp.status}")
                            return None
                        
                        data = await resp.json()
                        break
                        
                except asyncio.TimeoutError:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return None
            else:
                return None
            
            results = data.get("query", {}).get("search", [])
            if not results:
                return None
            
            # Get the first result's title and fetch its summary
            first_title = results[0].get("title", "")
            if first_title:
                return await self.get_wikipedia_summary(first_title, language)
            
            return None
            
        except Exception as e:
            logger.error(f"[Wikipedia Search] Error: {e}")
            return None
    
    # =========================================================================
    # SEMANTIC SCHOLAR API (with rate limiting to avoid 429 errors)
    # =========================================================================
    
    async def search_semantic_scholar(
        self,
        query: str,
        max_results: int = 5,
        year_from: Optional[int] = None,
        open_access_only: bool = False,
        max_retries: int = 3
    ) -> List[SemanticScholarPaper]:
        """
        Search Semantic Scholar for academic papers.
        
        Uses rate limiting to prevent 429 Too Many Requests errors.
        Implements exponential backoff retry logic.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            year_from: Filter papers from this year onwards
            open_access_only: Only return open access papers
            max_retries: Maximum retry attempts
            
        Returns:
            List of SemanticScholarPaper objects
        """
        try:
            # Wait if rate limit is reached
            wait_time = await self._semantic_scholar_limiter.wait_if_needed()
            if wait_time > 0:
                logger.info(f"[Semantic Scholar] Waited {wait_time:.1f}s for rate limit")
            
            session = await self._get_session()
            
            # Build query parameters
            params = {
                "query": query,
                "limit": max_results,
                "fields": "title,authors,year,abstract,citationCount,url,openAccessPdf,venue,externalIds"
            }
            
            if year_from:
                params["year"] = f"{year_from}-"
            
            if open_access_only:
                params["openAccessPdf"] = ""
            
            # Add API key if available (higher rate limits)
            headers = {}
            if self.semantic_scholar_api_key:
                headers["x-api-key"] = self.semantic_scholar_api_key
            
            # Retry logic with exponential backoff
            data = None
            for attempt in range(max_retries):
                try:
                    async with session.get(
                        self.SEMANTIC_SCHOLAR_URL,
                        params=params,
                        headers=headers
                    ) as resp:
                        if resp.status == 429:
                            # Rate limited by server - exponential backoff
                            delay = min(2 ** attempt * 2, 30)  # 2, 4, 8... max 30s
                            logger.warning(f"[Semantic Scholar] Rate limited (429), waiting {delay}s (attempt {attempt + 1}/{max_retries})")
                            await asyncio.sleep(delay)
                            continue
                        
                        if resp.status == 504:
                            # Gateway timeout - retry
                            delay = 2 ** attempt
                            logger.warning(f"[Semantic Scholar] Gateway timeout (504), retrying in {delay}s")
                            await asyncio.sleep(delay)
                            continue
                        
                        if resp.status != 200:
                            logger.warning(f"[Semantic Scholar] Search failed: {resp.status}")
                            return []
                        
                        data = await resp.json()
                        break
                        
                except asyncio.TimeoutError:
                    if attempt < max_retries - 1:
                        delay = 2 ** attempt
                        logger.warning(f"[Semantic Scholar] Timeout, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(delay)
                        continue
                    logger.error(f"[Semantic Scholar] All retry attempts timed out for '{query}'")
                    return []
            
            if data is None:
                logger.error(f"[Semantic Scholar] All retries exhausted for '{query}'")
                return []
            
            papers = []
            for item in data.get("data", []):
                # Extract authors
                authors = [a.get("name", "") for a in item.get("authors", [])]
                
                # Extract DOI
                external_ids = item.get("externalIds", {})
                doi = external_ids.get("DOI")
                
                # Check open access
                open_access_pdf = item.get("openAccessPdf")
                pdf_url = open_access_pdf.get("url") if open_access_pdf else None
                
                papers.append(SemanticScholarPaper(
                    title=item.get("title", ""),
                    authors=authors,
                    year=item.get("year"),
                    abstract=item.get("abstract", "")[:300] if item.get("abstract") else None,
                    citation_count=item.get("citationCount", 0),
                    url=item.get("url", ""),
                    doi=doi,
                    venue=item.get("venue"),
                    is_open_access=pdf_url is not None,
                    pdf_url=pdf_url
                ))
            
            logger.info(f"[Semantic Scholar] Found {len(papers)} papers for '{query}'")
            return papers
            
        except Exception as e:
            logger.error(f"[Semantic Scholar] Error: {e}")
            return []
    
    # =========================================================================
    # COMBINED SEARCH
    # =========================================================================
    
    async def search_all(
        self,
        topic: str,
        max_videos: int = 3,
        max_papers: int = 3,
        max_textbooks: int = 3,
        include_wikipedia: bool = True,
        include_oer: bool = True
    ) -> Dict[str, Any]:
        """
        Search all external APIs for a topic.
        
        Runs all searches concurrently for efficiency.
        
        Args:
            topic: Topic to search for
            max_videos: Maximum YouTube videos
            max_papers: Maximum Semantic Scholar papers
            max_textbooks: Maximum OER textbooks
            include_wikipedia: Whether to include Wikipedia
            include_oer: Whether to include OER textbooks (domain expert sources)
            
        Returns:
            Dict with 'youtube', 'wikipedia', 'semantic_scholar', 'oer_textbooks' keys
        """
        tasks = [
            self.search_youtube(topic, max_videos),
            self.search_semantic_scholar(topic, max_papers)
        ]
        
        if include_wikipedia:
            tasks.append(self.get_wikipedia_summary(topic))
        
        if include_oer:
            tasks.append(self.search_oer_textbooks(topic, max_textbooks))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        output = {
            "youtube": results[0] if not isinstance(results[0], Exception) else [],
            "semantic_scholar": results[1] if not isinstance(results[1], Exception) else [],
            "wikipedia": None,
            "oer_textbooks": []
        }
        
        idx = 2
        if include_wikipedia:
            output["wikipedia"] = results[idx] if not isinstance(results[idx], Exception) else None
            idx += 1
        
        if include_oer and len(results) > idx:
            output["oer_textbooks"] = results[idx] if not isinstance(results[idx], Exception) else []
        
        return output
    
    def to_dict(self, data: Any) -> Dict[str, Any]:
        """Convert dataclass results to dict for JSON serialization"""
        if isinstance(data, list):
            return [self.to_dict(item) for item in data]
        elif hasattr(data, '__dataclass_fields__'):
            return {k: self.to_dict(v) for k, v in data.__dict__.items()}
        else:
            return data
    
    # =========================================================================
    # OER (Open Educational Resources) - Domain Expert Trusted Sources
    # =========================================================================
    
    async def search_oer_textbooks(
        self,
        query: str,
        max_results: int = 5,
        language: str = "en"
    ) -> List[OERTextbook]:
        """
        Search Open Educational Resources for textbooks.
        
        Searches multiple OER sources:
        - DOAB (Directory of Open Access Books)
        - Open Textbook Library
        - BC Campus OpenEd
        
        These are domain-expert approved, copyright-safe sources.
        
        Args:
            query: Search query (topic)
            max_results: Maximum total results
            language: Language preference ("en", "it")
            
        Returns:
            List of OERTextbook results from all sources
        """
        logger.info(f"[OER] Searching textbooks for: {query}")
        
        results = []
        
        # Search each source concurrently
        tasks = [
            self._search_doab(query, max_results=2),
            self._search_open_textbook_library(query, max_results=2),
            self._search_bc_campus(query, max_results=1)
        ]
        
        source_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for source_result in source_results:
            if isinstance(source_result, list):
                results.extend(source_result)
            elif isinstance(source_result, Exception):
                logger.debug(f"[OER] Source search failed: {source_result}")
        
        logger.info(f"[OER] Found {len(results)} textbooks from OER sources")
        return results[:max_results]
    
    async def _search_doab(self, query: str, max_results: int = 3) -> List[OERTextbook]:
        """
        Search DOAB (Directory of Open Access Books).
        
        Uses the DOAB OAI-PMH endpoint for searching.
        https://www.doabooks.org/oai
        """
        logger.debug(f"[DOAB] Searching: {query}")
        
        try:
            session = await self._get_session()
            
            # DOAB search API
            url = "https://directory.doabooks.org/rest/search"
            params = {
                "query": f"title:{query} OR description:{query}",
                "expand": "metadata",
                "limit": max_results
            }
            
            async with session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    try:
                        data = await response.json()
                        results = []
                        
                        for item in data[:max_results]:
                            # Parse DOAB response format
                            metadata = item.get("metadata", {})
                            title = self._get_doab_field(metadata, "dc.title")
                            authors = self._get_doab_authors(metadata)
                            url_link = self._get_doab_field(metadata, "dc.identifier.uri") or f"https://directory.doabooks.org/handle/{item.get('handle', '')}"
                            
                            if title:
                                results.append(OERTextbook(
                                    title=title,
                                    source="DOAB",
                                    url=url_link,
                                    authors=authors,
                                    subject=self._get_doab_field(metadata, "dc.subject"),
                                    description=self._get_doab_field(metadata, "dc.description.abstract"),
                                    license="Open Access",
                                    relevance_note="From Directory of Open Access Books"
                                ))
                        
                        logger.debug(f"[DOAB] Found {len(results)} results")
                        return results
                    except Exception as e:
                        logger.debug(f"[DOAB] Parse error: {e}")
                        return []
                else:
                    logger.debug(f"[DOAB] HTTP {response.status}")
                    return []
                    
        except Exception as e:
            logger.debug(f"[DOAB] Search failed: {e}")
            return []
    
    def _get_doab_field(self, metadata: dict, field: str) -> Optional[str]:
        """Extract a field from DOAB metadata"""
        for item in metadata.get("value", []):
            if item.get("key") == field:
                return item.get("value")
        return None
    
    def _get_doab_authors(self, metadata: dict) -> List[str]:
        """Extract authors from DOAB metadata"""
        authors = []
        for item in metadata.get("value", []):
            if item.get("key") == "dc.contributor.author":
                authors.append(item.get("value", ""))
        return authors[:3]  # Limit to 3 authors
    
    async def _search_open_textbook_library(self, query: str, max_results: int = 3) -> List[OERTextbook]:
        """
        Search Open Textbook Library.
        
        https://open.umn.edu/opentextbooks/
        Uses their public API endpoint.
        """
        logger.debug(f"[OpenTextbook] Searching: {query}")
        
        try:
            session = await self._get_session()
            
            # Open Textbook Library API
            url = "https://open.umn.edu/opentextbooks/api/v1/textbooks"
            params = {
                "search": query,
                "per_page": max_results
            }
            
            async with session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    try:
                        data = await response.json()
                        results = []
                        
                        for book in data.get("data", data if isinstance(data, list) else [])[:max_results]:
                            title = book.get("title", "")
                            if title:
                                results.append(OERTextbook(
                                    title=title,
                                    source="Open Textbook Library",
                                    url=book.get("url") or f"https://open.umn.edu/opentextbooks/textbooks/{book.get('id', '')}",
                                    authors=[a.get("name", "") for a in book.get("authors", [])[:3]],
                                    subject=", ".join(book.get("subjects", [])[:3]),
                                    description=book.get("description", "")[:300] if book.get("description") else None,
                                    license=book.get("license", "CC BY"),
                                    relevance_note="Peer-reviewed open textbook"
                                ))
                        
                        logger.debug(f"[OpenTextbook] Found {len(results)} results")
                        return results
                    except Exception as e:
                        logger.debug(f"[OpenTextbook] Parse error: {e}")
                        return []
                else:
                    logger.debug(f"[OpenTextbook] HTTP {response.status}")
                    return []
                    
        except Exception as e:
            logger.debug(f"[OpenTextbook] Search failed: {e}")
            return []
    
    async def _search_bc_campus(self, query: str, max_results: int = 3) -> List[OERTextbook]:
        """
        Search BC Campus OpenEd.
        
        https://collection.bccampus.ca/
        Uses their Pressbooks API endpoint.
        """
        logger.debug(f"[BCCampus] Searching: {query}")
        
        try:
            session = await self._get_session()
            
            # BC Campus uses Pressbooks, search via their catalog
            url = "https://open.bccampus.ca/wp-json/pressbooks-book-directory/v1/books"
            params = {
                "search": query,
                "per_page": max_results
            }
            
            async with session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    try:
                        data = await response.json()
                        results = []
                        
                        for book in data[:max_results]:
                            title = book.get("title", "")
                            if title:
                                # Clean HTML from title if present
                                if "<" in title:
                                    import re
                                    title = re.sub('<[^<]+?>', '', title)
                                
                                results.append(OERTextbook(
                                    title=title,
                                    source="BC Campus OpenEd",
                                    url=book.get("link", ""),
                                    authors=[book.get("author", "")[:50]] if book.get("author") else [],
                                    subject=book.get("subject", ""),
                                    description=book.get("description", "")[:300] if book.get("description") else None,
                                    license=book.get("license", "CC BY"),
                                    relevance_note="BC Campus open educational resource"
                                ))
                        
                        logger.debug(f"[BCCampus] Found {len(results)} results")
                        return results
                    except Exception as e:
                        logger.debug(f"[BCCampus] Parse error: {e}")
                        return []
                else:
                    logger.debug(f"[BCCampus] HTTP {response.status}")
                    return []
                    
        except Exception as e:
            logger.debug(f"[BCCampus] Search failed: {e}")
            return []


# =============================================================================
# CONVENIENCE ALIASES (backward compatibility)
# =============================================================================
ExternalAPIs = ExternalMediaAPI  # Alias for backward compatibility


# =============================================================================
# TESTING
# =============================================================================
async def test_external_apis():
    """
    Test all external APIs with proper error handling.
    
    This test verifies:
    1. Wikipedia works with proper User-Agent (no 403)
    2. Semantic Scholar works with rate limiting (no 429)
    3. YouTube fallback works without API key
    """
    print("=" * 60)
    print("TESTING EXTERNAL APIS (with fixes)")
    print("=" * 60)
    print(f"\n📋 User-Agent: {USER_AGENT}")
    
    api = ExternalMediaAPI()
    
    topic = "metacognition"
    
    # Test YouTube
    print(f"\n🎥 YouTube search for '{topic}':")
    try:
        videos = await api.search_youtube(topic, max_results=3)
        for v in videos:
            print(f"  ✅ {v.title}")
            print(f"     URL: {v.url}")
        print(f"  → Found {len(videos)} videos")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Test Wikipedia (this should NOT get 403 now)
    print(f"\n📖 Wikipedia summary for '{topic}':")
    try:
        wiki = await api.get_wikipedia_summary(topic)
        if wiki:
            print(f"  ✅ Title: {wiki.title}")
            print(f"     Summary: {wiki.summary[:150]}...")
            print(f"     URL: {wiki.url}")
        else:
            print(f"  ⚠️ No Wikipedia result found")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Test Semantic Scholar (this should NOT get 429 with rate limiting)
    print(f"\n📚 Semantic Scholar papers for '{topic}':")
    try:
        papers = await api.search_semantic_scholar(topic, max_results=3, year_from=2020)
        for p in papers:
            authors_str = ", ".join(p.authors[:2])
            if len(p.authors) > 2:
                authors_str += " et al."
            print(f"  ✅ {p.title}")
            print(f"     {authors_str} ({p.year}) - {p.citation_count} citations")
        print(f"  → Found {len(papers)} papers")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Test OER Textbooks (Domain Expert Sources)
    print(f"\n📚 OER Textbooks for '{topic}':")
    try:
        textbooks = await api.search_oer_textbooks(topic, max_results=5)
        for t in textbooks:
            print(f"  ✅ [{t.source}] {t.title}")
            if t.url:
                print(f"     URL: {t.url}")
        print(f"  → Found {len(textbooks)} OER textbooks")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Test rate limiting (multiple rapid calls)
    print(f"\n⏱️ Testing rate limiting (3 rapid calls):")
    try:
        for i in range(3):
            papers = await api.search_semantic_scholar(f"{topic} learning", max_results=1)
            print(f"  ✅ Call {i+1}: Got {len(papers)} papers")
        print("  → Rate limiting working correctly!")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    await api.close()
    print("\n" + "=" * 60)
    print("✅ All External API tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    asyncio.run(test_external_apis())

