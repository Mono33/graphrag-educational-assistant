"""
Media Lookup Module

Provides a sidecar lookup mechanism for media content associated with
Knowledge Graph concepts. The media data is stored in a separate JSON file,
allowing easy updates and domain expert review without modifying the core KG.

Architecture:
    kg_neuro_neo4j.json (Core KG) ←→ kg_neuro_media_mapping.json (Media Sidecar)

The separation allows:
1. Domain experts to review/improve media recommendations
2. Independent versioning of media and knowledge
3. No re-ingestion required for media updates
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class VideoResource:
    """Video resource recommendation"""

    title: str
    platform: str  # youtube, vimeo, etc.
    search_query: str
    url: Optional[str] = None
    embed_url: Optional[str] = None  # verified embed URL (new pool format)
    rights_status: Optional[str] = None  # youtube_embed | youtube_cc (new pool format)
    duration_hint: Optional[str] = None
    language: str = "en"
    educational_level: str = "general"
    # Engagement fields populated by 04_enrich_pool.py
    duration_seconds: Optional[int] = None
    quality_score: Optional[float] = None
    trusted_channel: bool = False


@dataclass
class ImageResource:
    """Image/diagram resource recommendation"""

    description: str
    search_query: str
    type: str = "diagram"  # diagram, infographic, illustration
    url: Optional[str] = None
    source: Optional[str] = None


@dataclass
class ExternalResource:
    """External resource link (Wikipedia, educational sites)"""

    title: str
    type: str  # wikipedia, educational, academic
    url: Optional[str] = None
    suggested_url: Optional[str] = None
    language: str = "en"


@dataclass
class Citation:
    """Academic citation/reference"""

    title: str
    authors: list[str]
    year: Optional[int] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    abstract_snippet: Optional[str] = None


@dataclass
class OpenTextbook:
    """Open Educational Resource (OER) textbook reference"""

    title: str
    source: str  # OpenStax, DOAB, Pressbooks, OpenTextbookLibrary, BCCampus
    chapter: Optional[str] = None
    url: Optional[str] = None
    license: str = "CC BY 4.0"
    relevance: Optional[str] = None


@dataclass
class MediaContent:
    """Complete media content for a concept"""

    concept_name: str
    concept_id: str
    category: str
    videos: list[VideoResource] = field(default_factory=list)
    images: list[ImageResource] = field(default_factory=list)
    resources: list[ExternalResource] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    open_textbooks: list[OpenTextbook] = field(default_factory=list)

    def has_content(self) -> bool:
        """Check if any media content exists"""
        return bool(
            self.videos or self.images or self.resources or self.citations or self.open_textbooks
        )

    def to_context_string(self, include_citations: bool = True) -> str:
        """Format media content for Writer Agent context"""
        lines = []

        if self.videos:
            lines.append(f"\n### 🎥 Video per '{self.concept_name}':")
            for v in self.videos[:3]:
                display_url = v.url or v.embed_url
                if display_url:
                    lines.append(f"- [{v.title}]({display_url}) ({v.duration_hint or 'video'})")
                else:
                    lines.append(f'- Cerca: "{v.search_query}" su {v.platform}')

        if self.images:
            lines.append(f"\n### 🖼️ Diagrammi/Immagini per '{self.concept_name}':")
            for img in self.images[:2]:
                lines.append(f"- {img.description}")
                if img.url:
                    lines.append(f"  URL: {img.url}")

        if self.resources:
            lines.append(f"\n### 📚 Risorse per '{self.concept_name}':")
            for r in self.resources[:3]:
                if r.url or r.suggested_url:
                    url = r.url or r.suggested_url
                    lines.append(f"- [{r.title}]({url})")
                else:
                    lines.append(f"- {r.title} ({r.type})")

        if include_citations and self.citations:
            lines.append(f"\n### 📖 Riferimenti scientifici per '{self.concept_name}':")
            for c in self.citations[:2]:
                authors_str = ", ".join(c.authors[:3])
                if len(c.authors) > 3:
                    authors_str += " et al."
                year_str = f" ({c.year})" if c.year else ""
                lines.append(f"- {authors_str}{year_str}. *{c.title}*")
                if c.journal:
                    lines.append(f"  {c.journal}")
                if c.doi:
                    doi_link = f"https://doi.org/{c.doi}"
                    lines.append(f"  [DOI: {c.doi}]({doi_link})")

        if self.open_textbooks:
            lines.append(f"\n### 📚 Libri di Testo Aperti (OER) per '{self.concept_name}':")
            for t in self.open_textbooks[:2]:
                if t.url:
                    lines.append(f"- [{t.title}]({t.url})")
                else:
                    lines.append(f"- {t.title}")
                if t.chapter:
                    lines.append(f"  Capitolo: {t.chapter}")
                lines.append(f"  Fonte: {t.source} | Licenza: {t.license}")

        return "\n".join(lines)


class MediaLookup:
    """
    Media Lookup Service - Sidecar JSON lookup for concept media.

    Loads media mapping at startup and provides fast lookup by concept name.
    The mapping is stored separately from the Neo4j KG for easy updates.

    Usage:
        lookup = MediaLookup(domain="neuro")
        media = lookup.find_media(["Working Memory", "Metacognition"])
        context = media.to_context_string()
    """

    def __init__(self, domain: str = "neuro", mapping_path: Optional[str] = None):
        """
        Initialize Media Lookup.

        Args:
            domain: Knowledge domain ("neuro" or "udl")
            mapping_path: Optional custom path to media mapping JSON
        """
        self.domain = domain
        self.media_by_concept: dict[str, dict] = {}
        self.media_by_id: dict[str, dict] = {}
        self.loaded = False
        self._pool_loaded = False  # True when loaded from the new verified pool format

        # Default path based on domain.
        # Media mapping JSONs live under <repo_root>/data/media/
        # (separated from data/kg/{domain}/ which holds the KG dump only).
        if mapping_path is None:
            # Walk up from src/aix/agent/media/ to the repo root (5 levels)
            repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
            pool_path = repo_root / "data" / "media" / f"kg_{domain}_media_pool.json"
            legacy_path = repo_root / "data" / "media" / f"kg_{domain}_media_mapping.json"
            # Prefer the verified pool if it exists
            mapping_path = pool_path if pool_path.exists() else legacy_path
        else:
            mapping_path = Path(mapping_path)

        self.mapping_path = mapping_path
        self._load_mapping()

    def _load_mapping(self) -> None:
        """Load media mapping from JSON file (verified pool or legacy mapping)."""
        if not self.mapping_path.exists():
            logger.warning(
                f"[MediaLookup] Media mapping not found at {self.mapping_path}. "
                "Run scripts/media_pool/01_run_pool_agent.py to generate the verified pool."
            )
            return

        try:
            with open(self.mapping_path, encoding="utf-8") as f:
                data = json.load(f)

            # Detect format: verified pool has top-level "entries" dict keyed by concept name
            if "entries" in data and isinstance(data["entries"], dict):
                self._load_pool_format(data)
            else:
                self._load_legacy_format(data)

        except Exception as e:
            logger.error(f"[MediaLookup] Failed to load media mapping: {e}")

    def _load_pool_format(self, data: dict) -> None:
        """Load the new verified pool format (kg_{domain}_media_pool.json)."""
        entries = data.get("entries", {})
        for concept_name, entry in entries.items():
            # Only include entries that have at least one verified item
            if not (entry.get("videos") or entry.get("citations") or entry.get("wikipedia")):
                continue
            # Store under normalised name; also keep original for exact-match
            key = concept_name.lower().strip()
            # Attach the concept name so _parse_media_content can read it
            entry_with_name = {"name": concept_name, "id": key, "category": "", **entry}
            self.media_by_concept[key] = entry_with_name

        self._pool_loaded = True
        self.loaded = True
        logger.info(
            f"[MediaLookup] Loaded verified pool ({data.get('generated_by', '?')}): "
            f"{len(self.media_by_concept)} concepts"
        )

    def _load_legacy_format(self, data: dict) -> None:
        """Load the legacy media mapping format (kg_{domain}_media_mapping.json)."""
        concepts = data.get("concepts", [])
        for concept in concepts:
            name = concept.get("name", "").lower().strip()
            concept_id = concept.get("id", "")
            if name:
                self.media_by_concept[name] = concept
            if concept_id:
                self.media_by_id[concept_id] = concept

        self.loaded = True
        logger.info(
            f"[MediaLookup] Loaded legacy media mapping: {len(self.media_by_concept)} concepts"
        )

    def find_media_for_concept(self, concept_name: str) -> Optional[MediaContent]:
        """
        Find media content for a single concept.

        Args:
            concept_name: Name of the concept

        Returns:
            MediaContent if found, None otherwise
        """
        if not self.loaded:
            return None

        key = concept_name.lower().strip()
        data = self.media_by_concept.get(key)

        if not data:
            # Try partial matching
            for stored_name, stored_data in self.media_by_concept.items():
                if key in stored_name or stored_name in key:
                    data = stored_data
                    break

        if not data:
            return None

        return self._parse_media_content(data)

    def find_media(self, concept_names: list[str]) -> dict[str, MediaContent]:
        """
        Find media for multiple concepts.

        Args:
            concept_names: List of concept names to look up

        Returns:
            Dictionary mapping concept names to MediaContent
        """
        result = {}

        for name in concept_names:
            media = self.find_media_for_concept(name)
            if media and media.has_content():
                result[name] = media

        return result

    def get_combined_media(self, concept_names: list[str]) -> MediaContent:
        """
        Get combined media for multiple concepts.

        Args:
            concept_names: List of concept names

        Returns:
            Combined MediaContent with all media from all concepts
        """
        combined = MediaContent(concept_name="Combined", concept_id="combined", category="Multiple")

        for name in concept_names:
            media = self.find_media_for_concept(name)
            if media:
                # Sort by quality_score descending (trusted + high-view first); pool JSON
                # is pre-sorted by 04_enrich_pool.py but in-memory sort is free and safe.
                sorted_videos = sorted(
                    media.videos,
                    key=lambda v: (v.trusted_channel, v.quality_score or 0),
                    reverse=True,
                )
                combined.videos.extend(sorted_videos[:2])
                combined.images.extend(media.images[:2])
                combined.resources.extend(media.resources[:2])
                combined.citations.extend(media.citations[:2])
                combined.open_textbooks.extend(media.open_textbooks[:2])

        # Deduplicate by title/description
        seen_videos = set()
        combined.videos = [
            v
            for v in combined.videos
            if v.title not in seen_videos and not seen_videos.add(v.title)
        ]

        seen_images = set()
        combined.images = [
            i
            for i in combined.images
            if i.description not in seen_images and not seen_images.add(i.description)
        ]

        seen_textbooks = set()
        combined.open_textbooks = [
            t
            for t in combined.open_textbooks
            if t.title not in seen_textbooks and not seen_textbooks.add(t.title)
        ]

        return combined

    def _parse_media_content(self, data: dict) -> MediaContent:
        """Parse raw JSON data into MediaContent dataclass (handles both pool and legacy formats)."""
        videos = []
        for v in data.get("videos", []):
            # New pool format has video_id + embed_url; skip entries without a verified date
            if self._pool_loaded and not v.get("verified_date"):
                continue
            videos.append(
                VideoResource(
                    title=v.get("title", ""),
                    platform=v.get("platform", "youtube"),
                    search_query=v.get("search_query", v.get("title", "")),
                    url=v.get("url"),
                    embed_url=v.get("embed_url"),
                    rights_status=v.get("rights_status"),
                    duration_hint=v.get("duration_hint"),
                    language=v.get("language", "en"),
                    educational_level=v.get("educational_level", "general"),
                    duration_seconds=v.get("duration_seconds"),
                    quality_score=v.get("quality_score"),
                    trusted_channel=v.get("trusted_channel", False),
                )
            )

        images = []
        for i in data.get("images", []):
            images.append(
                ImageResource(
                    description=i.get("description", ""),
                    search_query=i.get("search_query", ""),
                    type=i.get("type", "diagram"),
                    url=i.get("url"),
                    source=i.get("source"),
                )
            )

        resources = []
        for r in data.get("resources", []):
            resources.append(
                ExternalResource(
                    title=r.get("title", ""),
                    type=r.get("type", "educational"),
                    url=r.get("url"),
                    suggested_url=r.get("suggested_url"),
                    language=r.get("language", "en"),
                )
            )

        citations = []
        for c in data.get("citations", []):
            if self._pool_loaded and not c.get("verified_date"):
                continue
            citations.append(
                Citation(
                    title=c.get("title", ""),
                    authors=c.get("authors", []),
                    year=c.get("year"),
                    journal=c.get("journal"),
                    doi=c.get("doi"),
                    abstract_snippet=c.get("abstract_snippet"),
                )
            )

        open_textbooks = []
        for t in data.get("open_textbooks", []):
            open_textbooks.append(
                OpenTextbook(
                    title=t.get("title", ""),
                    source=t.get("source", "Unknown"),
                    chapter=t.get("chapter"),
                    url=t.get("url"),
                    license=t.get("license", "CC BY 4.0"),
                    relevance=t.get("relevance"),
                )
            )

        return MediaContent(
            concept_name=data.get("name", ""),
            concept_id=data.get("id", ""),
            category=data.get("category", ""),
            videos=videos,
            images=images,
            resources=resources,
            citations=citations,
            open_textbooks=open_textbooks,
        )

    def get_all_concepts(self) -> list[str]:
        """Get list of all concepts with media mappings"""
        return list(self.media_by_concept.keys())

    def get_stats(self) -> dict[str, int]:
        """Get statistics about the media mapping."""
        if not self.loaded:
            return {"loaded": False, "concepts": 0}

        total_videos = sum(len(c.get("videos", [])) for c in self.media_by_concept.values())
        total_citations = sum(len(c.get("citations", [])) for c in self.media_by_concept.values())

        stats: dict[str, Any] = {
            "loaded": True,
            "pool_format": self._pool_loaded,
            "concepts": len(self.media_by_concept),
            "total_videos": total_videos,
            "total_citations": total_citations,
        }

        if self._pool_loaded:
            total_wiki = sum(1 for c in self.media_by_concept.values() if c.get("wikipedia"))
            stats["total_wikipedia"] = total_wiki
        else:
            stats["total_images"] = sum(
                len(c.get("images", [])) for c in self.media_by_concept.values()
            )
            stats["total_resources"] = sum(
                len(c.get("resources", [])) for c in self.media_by_concept.values()
            )
            stats["total_open_textbooks"] = sum(
                len(c.get("open_textbooks", [])) for c in self.media_by_concept.values()
            )

        return stats
