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

from .media_lookup import MediaLookup, MediaContent
from .resource_lookup import (
    ResourceLookup, 
    ResourceCollection, 
    ExpertResource,
    ResourceType,
    AudienceLevel
)
from .external_apis import (
    ExternalMediaAPI, 
    ExternalAPIs,  # Alias for backward compatibility
    YouTubeVideo, 
    WikipediaSummary, 
    SemanticScholarPaper,
    RateLimiter,
    USER_AGENT
)
from .image_generator import ImageGenerator, GeneratedImage, DiagramType
from .mermaid_generator import MermaidGenerator, MermaidResult, MermaidDiagramType
from .canva_generator import CanvaGenerator, CanvaResult
from .diagram_factory import (
    DiagramFactory, 
    DiagramResult, 
    GeneratorType,
    DiagramGeneratorFactory  # Alias
)

__all__ = [
    # Media Lookup (Phase 0)
    'MediaLookup', 
    'MediaContent',
    
    # Resource Lookup (Phase 0b - Expert-Vetted)
    'ResourceLookup',
    'ResourceCollection',
    'ExpertResource',
    'ResourceType',
    'AudienceLevel',
    
    # External APIs (Phase 4)
    'ExternalMediaAPI',
    'ExternalAPIs',  # Alias
    'YouTubeVideo',
    'WikipediaSummary', 
    'SemanticScholarPaper',
    'RateLimiter',
    'USER_AGENT',
    
    # DALL-E Generator (Phase 5)
    'ImageGenerator',
    'GeneratedImage',
    'DiagramType',
    
    # Mermaid Generator (Phase 5 - FREE)
    'MermaidGenerator',
    'MermaidResult',
    'MermaidDiagramType',
    
    # Canva Generator (Phase 5 - Coming Soon)
    'CanvaGenerator',
    'CanvaResult',
    
    # Diagram Factory (Unified Interface)
    'DiagramFactory',
    'DiagramGeneratorFactory',  # Alias
    'DiagramResult',
    'GeneratorType',
]


