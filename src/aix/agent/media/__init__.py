"""
Media Module - Multimodal content enhancement for Agentic GraphRAG

This module provides:
1. MediaLookup - Sidecar JSON lookup for curated media content
2. ResourceLookup - Expert-vetted educational resources (NEW)
3. ExternalMediaAPI - Real-time API access (YouTube, Wikipedia, Semantic Scholar)
4. ImageGenerator - AI-powered educational diagram generation (DALL-E 3)
5. MermaidGenerator - FREE diagram generation with accurate text (Mermaid.js)
6. DiagramFactory - Unified interface for multiple diagram generators
7. Media mapping utilities for concept-to-media associations

The media mapping is stored separately from the Neo4j Knowledge Graph,
allowing domain experts to review and improve media recommendations
without affecting the core knowledge base.

Phase 0: Curated media from JSON (pre-generated, expert-reviewable)
Phase 0b: Expert-vetted resources from JSON (copyright-safe, reviewed)
Phase 4: External APIs for real-time content (dynamic, fresh)
Phase 5: AI Image Generation (DALL-E 3, Mermaid.js, Canva for educational diagrams)
"""

from .canva_generator import CanvaGenerator, CanvaResult
from .diagram_factory import (  # Alias
    DiagramFactory,
    DiagramGeneratorFactory,
    DiagramResult,
    GeneratorType,
)
from .image_generator import DiagramType, GeneratedImage, ImageGenerator
from .live_media import LiveMediaService, fetch_live_subject_resources, to_panel_media
from .media_cache import MediaCache, make_cache_key
from .media_config import MediaConfig
from .media_lookup import MediaContent, MediaLookup
from .mermaid_generator import MermaidDiagramType, MermaidGenerator, MermaidResult
from .resource_lookup import (
    AudienceLevel,
    ExpertResource,
    ResourceCollection,
    ResourceLookup,
    ResourceType,
)

__all__ = [
    # Media Lookup (Phase 0)
    "MediaLookup",
    "MediaContent",
    # Dynamic media layer scaffolding (Dynamic_Media_Retrieval_Plan.md — Phase 0)
    "MediaConfig",
    "MediaCache",
    "make_cache_key",
    # Live media layer (Dynamic_Media_Retrieval_Plan.md — Phase 1)
    "LiveMediaService",
    "fetch_live_subject_resources",
    "to_panel_media",
    # Resource Lookup (Phase 0b - Expert-Vetted)
    "ResourceLookup",
    "ResourceCollection",
    "ExpertResource",
    "ResourceType",
    "AudienceLevel",
]

_EXTERNAL_API_EXPORTS = frozenset(
    {
        "ExternalMediaAPI",
        "ExternalAPIs",
        "YouTubeVideo",
        "WikipediaSummary",
        "SemanticScholarPaper",
        "RateLimiter",
        "USER_AGENT",
    }
)


def __getattr__(name: str):
    """Lazy-load external_apis so `python -m aix.agent.media.external_apis` runs without RuntimeWarning."""
    if name in _EXTERNAL_API_EXPORTS:
        from . import external_apis as _ext

        return getattr(_ext, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Re-append external API names to __all__ (kept in one place for readers)
__all__ = __all__ + [
    # External APIs (Phase 4)
    "ExternalMediaAPI",
    "ExternalAPIs",  # Alias
    "YouTubeVideo",
    "WikipediaSummary",
    "SemanticScholarPaper",
    "RateLimiter",
    "USER_AGENT",
    # DALL-E Generator (Phase 5)
    "ImageGenerator",
    "GeneratedImage",
    "DiagramType",
    # Mermaid Generator (Phase 5 - FREE)
    "MermaidGenerator",
    "MermaidResult",
    "MermaidDiagramType",
    # Canva Generator (Phase 5 - Coming Soon)
    "CanvaGenerator",
    "CanvaResult",
    # Diagram Factory (Unified Interface)
    "DiagramFactory",
    "DiagramGeneratorFactory",  # Alias
    "DiagramResult",
    "GeneratorType",
]
