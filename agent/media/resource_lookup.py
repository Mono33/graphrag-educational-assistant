"""
Resource Lookup Module

Provides lookup for expert-vetted educational resources. These resources are
curated by domain experts and are all copyright-safe (Open Access, Creative Commons,
or Educational Commons).

Architecture:
    kg_neuro_resources.json (Expert-Vetted Resources)
    
This module allows:
1. Topic-based resource discovery
2. Audience-appropriate filtering (K-12, university, teacher training)
3. Resource type filtering (textbooks, simulations, videos, etc.)
4. Integration with lesson generation for citations
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of educational resources"""
    TEXTBOOK = "textbook"
    WEBSITE = "website"
    INTERACTIVE_SIMULATION = "interactive_simulation"
    DATASET = "dataset"
    COURSE = "course"
    VIDEO_CHANNEL = "video_channel"
    ACADEMIC_PAPER = "academic_paper"


class AudienceLevel(Enum):
    """Target audience levels"""
    K12 = "K-12"
    UNIVERSITY = "university"
    TEACHER_TRAINING = "teacher_training"
    GENERAL_PUBLIC = "general_public"
    RESEARCH = "research"


@dataclass
class ExpertResource:
    """Expert-vetted educational resource"""
    id: str
    title: str
    url: str
    type: str
    license: str
    license_details: str
    copyright_safe: bool
    topics: List[str] = field(default_factory=list)
    kg_concepts: List[str] = field(default_factory=list)
    audience: List[str] = field(default_factory=list)
    educational_level: List[str] = field(default_factory=list)
    language: str = "en"
    description: str = ""
    source_org: str = ""
    recommended_for: List[str] = field(default_factory=list)
    usage_note: Optional[str] = None
    
    def matches_topic(self, topic: str) -> bool:
        """Check if resource matches a topic (case-insensitive)"""
        topic_lower = topic.lower().replace(" ", "_")
        return any(
            topic_lower in t.lower() or t.lower() in topic_lower 
            for t in self.topics + self.kg_concepts
        )
    
    def matches_audience(self, audience: str) -> bool:
        """Check if resource is appropriate for audience"""
        return audience.lower() in [a.lower() for a in self.audience + self.educational_level]
    
    def to_citation_string(self, style: str = "simple") -> str:
        """Format resource as a citation"""
        if style == "markdown":
            return f"[{self.title}]({self.url}) - {self.source_org} ({self.license})"
        elif style == "full":
            return f"{self.title}. {self.source_org}. {self.url} [License: {self.license}]"
        else:  # simple
            return f"{self.title} ({self.source_org})"


@dataclass
class ResourceCollection:
    """Collection of resources for a query"""
    topic: str
    resources: List[ExpertResource] = field(default_factory=list)
    
    def has_resources(self) -> bool:
        """Check if any resources exist"""
        return len(self.resources) > 0
    
    def filter_by_type(self, resource_type: str) -> List[ExpertResource]:
        """Filter resources by type"""
        return [r for r in self.resources if r.type == resource_type]
    
    def filter_by_audience(self, audience: str) -> List[ExpertResource]:
        """Filter resources by audience"""
        return [r for r in self.resources if r.matches_audience(audience)]
    
    def get_textbooks(self) -> List[ExpertResource]:
        """Get textbook resources"""
        return self.filter_by_type("textbook")
    
    def get_simulations(self) -> List[ExpertResource]:
        """Get interactive simulation resources"""
        return self.filter_by_type("interactive_simulation")
    
    def get_videos(self) -> List[ExpertResource]:
        """Get video resources"""
        return self.filter_by_type("video_channel")
    
    def get_courses(self) -> List[ExpertResource]:
        """Get course resources"""
        return self.filter_by_type("course")
    
    def to_context_string(self, max_resources: int = 5) -> str:
        """Format resources for Writer Agent context"""
        if not self.resources:
            return ""
        
        lines = [f"\n### 📚 Risorse Consigliate dagli Esperti per '{self.topic}':"]
        
        for resource in self.resources[:max_resources]:
            emoji = self._get_type_emoji(resource.type)
            lines.append(f"\n{emoji} **{resource.title}**")
            lines.append(f"   - URL: {resource.url}")
            lines.append(f"   - Fonte: {resource.source_org}")
            lines.append(f"   - Licenza: {resource.license} ✅")
            if resource.description:
                lines.append(f"   - {resource.description[:150]}...")
        
        return "\n".join(lines)
    
    def _get_type_emoji(self, resource_type: str) -> str:
        """Get emoji for resource type"""
        emoji_map = {
            "textbook": "📖",
            "website": "🌐",
            "interactive_simulation": "🎮",
            "dataset": "📊",
            "course": "🎓",
            "video_channel": "🎥",
            "academic_paper": "📄"
        }
        return emoji_map.get(resource_type, "📚")


class ResourceLookup:
    """
    Resource Lookup Service - Expert-vetted educational resources.
    
    Loads curated resources from JSON and provides filtered lookup by:
    - Topic/concept
    - Audience level (K-12, university, teacher training)
    - Resource type (textbook, simulation, video, etc.)
    
    Usage:
        lookup = ResourceLookup(domain="neuro")
        resources = lookup.find_resources_for_topic("metacognition")
        context = resources.to_context_string()
    """
    
    def __init__(self, domain: str = "neuro", resources_path: Optional[str] = None):
        """
        Initialize Resource Lookup.
        
        Args:
            domain: Knowledge domain ("neuro" or "udl")
            resources_path: Optional custom path to resources JSON
        """
        self.domain = domain
        self.resources: List[ExpertResource] = []
        self.resources_by_id: Dict[str, ExpertResource] = {}
        self.topic_mappings: Dict[str, List[str]] = {}
        self.by_type: Dict[str, List[str]] = {}
        self.by_audience: Dict[str, List[str]] = {}
        self.loaded = False
        self.metadata: Dict[str, Any] = {}
        
        # Default path based on domain
        if resources_path is None:
            base_path = Path(__file__).parent.parent.parent
            resources_path = base_path / f"kg_{domain}_resources.json"
        else:
            resources_path = Path(resources_path)
        
        self.resources_path = resources_path
        self._load_resources()
    
    def _load_resources(self) -> None:
        """Load resources from JSON file"""
        if not self.resources_path.exists():
            logger.warning(
                f"[ResourceLookup] Resources file not found at {self.resources_path}. "
                "Expert resources will not be available."
            )
            return
        
        try:
            with open(self.resources_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.metadata = data.get('metadata', {})
            self.topic_mappings = data.get('topic_mappings', {})
            self.by_type = data.get('by_type', {})
            self.by_audience = data.get('by_audience', {})
            
            resources_data = data.get('resources', [])
            
            for res_data in resources_data:
                resource = self._parse_resource(res_data)
                self.resources.append(resource)
                self.resources_by_id[resource.id] = resource
            
            self.loaded = True
            logger.info(
                f"[ResourceLookup] Loaded {len(self.resources)} expert resources "
                f"for domain '{self.domain}'"
            )
            
        except Exception as e:
            logger.error(f"[ResourceLookup] Failed to load resources: {e}")
    
    def _parse_resource(self, data: Dict) -> ExpertResource:
        """Parse JSON data into ExpertResource dataclass"""
        return ExpertResource(
            id=data.get('id', ''),
            title=data.get('title', ''),
            url=data.get('url', ''),
            type=data.get('type', 'website'),
            license=data.get('license', 'unknown'),
            license_details=data.get('license_details', ''),
            copyright_safe=data.get('copyright_safe', False),
            topics=data.get('topics', []),
            kg_concepts=data.get('kg_concepts', []),
            audience=data.get('audience', []),
            educational_level=data.get('educational_level', []),
            language=data.get('language', 'en'),
            description=data.get('description', ''),
            source_org=data.get('source_org', ''),
            recommended_for=data.get('recommended_for', []),
            usage_note=data.get('usage_note')
        )
    
    def find_resources_for_topic(
        self, 
        topic: str, 
        max_results: int = 5
    ) -> ResourceCollection:
        """
        Find resources for a topic/concept.
        
        Args:
            topic: Topic or concept name
            max_results: Maximum number of resources to return
            
        Returns:
            ResourceCollection with matching resources
        """
        if not self.loaded:
            return ResourceCollection(topic=topic)
        
        matching_resources = []
        topic_key = topic.lower().replace(" ", "_")
        
        # First, check topic mappings
        if topic_key in self.topic_mappings:
            resource_ids = self.topic_mappings[topic_key]
            for res_id in resource_ids:
                if res_id in self.resources_by_id:
                    matching_resources.append(self.resources_by_id[res_id])
        
        # Then, search by topic/concept match
        for resource in self.resources:
            if resource not in matching_resources and resource.matches_topic(topic):
                matching_resources.append(resource)
        
        # Limit results
        matching_resources = matching_resources[:max_results]
        
        return ResourceCollection(topic=topic, resources=matching_resources)
    
    def find_resources_for_concepts(
        self, 
        concepts: List[str], 
        max_per_concept: int = 2
    ) -> ResourceCollection:
        """
        Find resources for multiple concepts (combined).
        
        Args:
            concepts: List of concept names
            max_per_concept: Maximum resources per concept
            
        Returns:
            Combined ResourceCollection
        """
        all_resources = []
        seen_ids = set()
        
        for concept in concepts:
            collection = self.find_resources_for_topic(concept, max_results=max_per_concept)
            for resource in collection.resources:
                if resource.id not in seen_ids:
                    all_resources.append(resource)
                    seen_ids.add(resource.id)
        
        return ResourceCollection(
            topic=", ".join(concepts[:3]) + ("..." if len(concepts) > 3 else ""),
            resources=all_resources
        )
    
    def find_by_type(self, resource_type: str, max_results: int = 10) -> List[ExpertResource]:
        """
        Find resources by type.
        
        Args:
            resource_type: Type of resource (textbook, simulation, etc.)
            max_results: Maximum number of results
            
        Returns:
            List of matching resources
        """
        if not self.loaded:
            return []
        
        resource_ids = self.by_type.get(resource_type, [])
        return [
            self.resources_by_id[rid] 
            for rid in resource_ids[:max_results] 
            if rid in self.resources_by_id
        ]
    
    def find_by_audience(self, audience: str, max_results: int = 10) -> List[ExpertResource]:
        """
        Find resources by target audience.
        
        Args:
            audience: Target audience (K-12, university, teacher_training, etc.)
            max_results: Maximum number of results
            
        Returns:
            List of matching resources
        """
        if not self.loaded:
            return []
        
        resource_ids = self.by_audience.get(audience, [])
        return [
            self.resources_by_id[rid] 
            for rid in resource_ids[:max_results] 
            if rid in self.resources_by_id
        ]
    
    def get_recommended_for(
        self, 
        purpose: str, 
        max_results: int = 5
    ) -> List[ExpertResource]:
        """
        Find resources recommended for a specific purpose.
        
        Args:
            purpose: Purpose like "lesson_creation", "interactive_learning", etc.
            max_results: Maximum number of results
            
        Returns:
            List of matching resources
        """
        if not self.loaded:
            return []
        
        matching = [
            r for r in self.resources 
            if purpose.lower() in [p.lower() for p in r.recommended_for]
        ]
        
        return matching[:max_results]
    
    def get_all_resources(self) -> List[ExpertResource]:
        """Get all loaded resources"""
        return self.resources.copy()
    
    def get_resource_by_id(self, resource_id: str) -> Optional[ExpertResource]:
        """Get a specific resource by ID"""
        return self.resources_by_id.get(resource_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded resources"""
        if not self.loaded:
            return {"loaded": False, "total_resources": 0}
        
        type_counts = {}
        for resource in self.resources:
            type_counts[resource.type] = type_counts.get(resource.type, 0) + 1
        
        license_counts = {}
        for resource in self.resources:
            license_counts[resource.license] = license_counts.get(resource.license, 0) + 1
        
        return {
            "loaded": True,
            "domain": self.domain,
            "total_resources": len(self.resources),
            "by_type": type_counts,
            "by_license": license_counts,
            "topics_mapped": len(self.topic_mappings),
            "metadata": self.metadata
        }
    
    def to_streamlit_display(
        self, 
        resources: List[ExpertResource]
    ) -> List[Dict[str, Any]]:
        """
        Format resources for Streamlit display.
        
        Args:
            resources: List of resources to format
            
        Returns:
            List of dictionaries ready for Streamlit rendering
        """
        display_items = []
        
        for r in resources:
            emoji = ResourceCollection(topic="")._get_type_emoji(r.type)
            display_items.append({
                "emoji": emoji,
                "title": r.title,
                "url": r.url,
                "type": r.type,
                "source": r.source_org,
                "license": r.license,
                "description": r.description,
                "safe": r.copyright_safe
            })
        
        return display_items
