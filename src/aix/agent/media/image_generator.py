"""
AI Image Generation Module for Agentic GraphRAG

This module provides AI-powered educational diagram generation using DALL-E 3.
Images are generated only after content is approved by the Critic Agent (cost savings).

Features:
- Educational diagram generation based on concept descriptions
- Support for different diagram types (flowchart, infographic, concept map)
- Cost tracking and rate limiting
- Caching to avoid regenerating same images

Usage:
    from aix.agent.media.image_generator import ImageGenerator
    
    generator = ImageGenerator()
    result = await generator.generate_educational_diagram(
        concept="Metacognition",
        description="The process of thinking about thinking",
        diagram_type="concept_map"
    )
    print(result.url)
"""

import os
import logging
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class DiagramType(Enum):
    """Types of educational diagrams"""
    CONCEPT_MAP = "concept_map"
    FLOWCHART = "flowchart"
    INFOGRAPHIC = "infographic"
    PROCESS_DIAGRAM = "process_diagram"
    COMPARISON_CHART = "comparison_chart"
    HIERARCHY = "hierarchy"


@dataclass
class GeneratedImage:
    """Result from image generation"""
    url: str
    prompt: str
    concept: str
    diagram_type: str
    generated_at: str
    model: str = "dall-e-3"
    size: str = "1024x1024"
    quality: str = "standard"
    revised_prompt: Optional[str] = None
    cached: bool = False
    cost_estimate: float = 0.04  # $0.04 per standard DALL-E 3 image
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "prompt": self.prompt,
            "concept": self.concept,
            "diagram_type": self.diagram_type,
            "generated_at": self.generated_at,
            "model": self.model,
            "size": self.size,
            "quality": self.quality,
            "revised_prompt": self.revised_prompt,
            "cached": self.cached,
            "cost_estimate": self.cost_estimate
        }


# Prompt templates for different diagram types
DIAGRAM_PROMPTS = {
    DiagramType.CONCEPT_MAP: """
Create an educational concept map diagram for teachers about "{concept}".

The diagram should:
- Show "{concept}" as the central node
- Include 4-6 related concepts as connected nodes
- Use clear connecting lines with relationship labels
- Use a clean, professional color scheme (blues, greens)
- Include a simple legend
- Be suitable for teacher training materials
- Text should be clearly readable

Description: {description}

Style: Clean, minimalist educational diagram. White background. Professional font.
""",
    
    DiagramType.FLOWCHART: """
Create an educational flowchart diagram about "{concept}" for teachers.

The diagram should:
- Show the process or steps related to {concept}
- Use standard flowchart shapes (rectangles for steps, diamonds for decisions)
- Include clear arrows showing flow direction
- Use a professional color scheme
- Be easy to understand at a glance
- Be suitable for classroom presentation

Description: {description}

Style: Clean flowchart with clear shapes and arrows. White background.
""",
    
    DiagramType.INFOGRAPHIC: """
Create an educational infographic about "{concept}" for teacher training.

The infographic should:
- Present key information about {concept} visually
- Include icons and simple illustrations
- Use a clear visual hierarchy
- Include 3-5 key facts or statistics (you can use placeholder numbers)
- Use an appealing but professional color scheme
- Be suitable for educational materials

Description: {description}

Style: Modern educational infographic. Clean design. Easy to read text.
""",
    
    DiagramType.PROCESS_DIAGRAM: """
Create an educational process diagram showing how "{concept}" works.

The diagram should:
- Show the stages or phases of {concept}
- Use numbered steps or a clear sequence
- Include brief labels for each stage
- Use icons or simple illustrations
- Be suitable for explaining to teachers
- Professional educational style

Description: {description}

Style: Sequential process diagram. Clean and professional.
""",
    
    DiagramType.COMPARISON_CHART: """
Create an educational comparison chart about "{concept}" for teachers.

The chart should:
- Compare 2-3 related aspects of {concept}
- Use a clear table or side-by-side format
- Include visual icons or symbols
- Highlight key differences and similarities
- Be easy to understand quickly
- Professional educational design

Description: {description}

Style: Clean comparison chart. Professional colors. Clear layout.
""",
    
    DiagramType.HIERARCHY: """
Create an educational hierarchy diagram about "{concept}" for teacher training.

The diagram should:
- Show hierarchical relationships (parent-child concepts)
- Use a tree structure or pyramid layout
- Include clear labels at each level
- Use size or color to show importance
- Be suitable for educational materials
- Professional and clean design

Description: {description}

Style: Hierarchical diagram. Clean lines and shapes. Professional.
"""
}


class ImageGenerator:
    """
    AI Image Generator using DALL-E 3
    
    Generates educational diagrams on-demand. Designed to be cost-efficient:
    - Only generates after content is approved
    - Caches generated images to avoid duplicates
    - Tracks costs per session
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        enable_caching: bool = True
    ):
        """
        Initialize Image Generator.
        
        Args:
            cache_dir: Directory to cache image metadata
            enable_caching: Whether to use caching (default: True)
        """
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.enable_caching = enable_caching
        self.session_cost = 0.0
        self.images_generated = 0
        
        # Setup cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(__file__).parent.parent.parent / "generated_images"
        
        if self.enable_caching:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_file = self.cache_dir / "image_cache.json"
            self._load_cache()
        else:
            self.cache: Dict[str, Dict] = {}
    
    def _load_cache(self) -> None:
        """Load image cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                logger.info(f"[ImageGenerator] Loaded {len(self.cache)} cached images")
            except Exception as e:
                logger.warning(f"[ImageGenerator] Failed to load cache: {e}")
                self.cache = {}
        else:
            self.cache = {}
    
    def _save_cache(self) -> None:
        """Save image cache to disk"""
        if self.enable_caching:
            try:
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.cache, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"[ImageGenerator] Failed to save cache: {e}")
    
    def _get_cache_key(self, concept: str, description: str, diagram_type: str) -> str:
        """Generate a cache key for the image request"""
        content = f"{concept}:{description}:{diagram_type}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[GeneratedImage]:
        """Check if image exists in cache"""
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            return GeneratedImage(
                url=cached_data['url'],
                prompt=cached_data['prompt'],
                concept=cached_data['concept'],
                diagram_type=cached_data['diagram_type'],
                generated_at=cached_data['generated_at'],
                revised_prompt=cached_data.get('revised_prompt'),
                cached=True,
                cost_estimate=0.0  # No cost for cached images
            )
        return None
    
    async def generate_educational_diagram(
        self,
        concept: str,
        description: str,
        diagram_type: DiagramType = DiagramType.CONCEPT_MAP,
        size: str = "1024x1024",
        quality: str = "standard"
    ) -> Optional[GeneratedImage]:
        """
        Generate an educational diagram using DALL-E 3.
        
        Args:
            concept: The concept to visualize
            description: Description of the concept
            diagram_type: Type of diagram to generate
            size: Image size (1024x1024, 1024x1792, 1792x1024)
            quality: Image quality (standard, hd)
            
        Returns:
            GeneratedImage or None on failure
        """
        if not self.api_key:
            logger.error("[ImageGenerator] OPENAI_API_KEY not set")
            return None
        
        # Check cache first
        cache_key = self._get_cache_key(concept, description, diagram_type.value)
        cached = self._check_cache(cache_key)
        if cached:
            logger.info(f"[ImageGenerator] Using cached image for '{concept}'")
            return cached
        
        # Build prompt
        prompt_template = DIAGRAM_PROMPTS.get(diagram_type, DIAGRAM_PROMPTS[DiagramType.CONCEPT_MAP])
        prompt = prompt_template.format(concept=concept, description=description)
        
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.api_key)
            
            logger.info(f"[ImageGenerator] Generating {diagram_type.value} for '{concept}'...")
            
            response = await client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=size,
                quality=quality,
                n=1
            )
            
            image_url = response.data[0].url
            revised_prompt = response.data[0].revised_prompt
            
            # Calculate cost
            cost = 0.04 if quality == "standard" else 0.08  # HD costs more
            self.session_cost += cost
            self.images_generated += 1
            
            result = GeneratedImage(
                url=image_url,
                prompt=prompt,
                concept=concept,
                diagram_type=diagram_type.value,
                generated_at=datetime.now().isoformat(),
                model="dall-e-3",
                size=size,
                quality=quality,
                revised_prompt=revised_prompt,
                cached=False,
                cost_estimate=cost
            )
            
            # Cache the result
            self.cache[cache_key] = result.to_dict()
            self._save_cache()
            
            logger.info(
                f"[ImageGenerator] Generated image for '{concept}' "
                f"(session total: ${self.session_cost:.2f})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"[ImageGenerator] Failed to generate image: {e}")
            return None
    
    async def generate_for_concepts(
        self,
        concepts: List[Dict[str, Any]],
        diagram_type: DiagramType = DiagramType.CONCEPT_MAP,
        max_images: int = 3
    ) -> List[GeneratedImage]:
        """
        Generate diagrams for multiple concepts.
        
        Args:
            concepts: List of concept dicts with 'name' and 'description'
            diagram_type: Type of diagram to generate
            max_images: Maximum images to generate (cost control)
            
        Returns:
            List of GeneratedImage objects
        """
        results = []
        
        for i, concept in enumerate(concepts[:max_images]):
            name = concept.get('name', '')
            description = concept.get('description', '')
            
            if not name:
                continue
            
            result = await self.generate_educational_diagram(
                concept=name,
                description=description,
                diagram_type=diagram_type
            )
            
            if result:
                results.append(result)
            
            # Rate limiting - wait between generations
            if i < len(concepts) - 1:
                await asyncio.sleep(1.0)
        
        return results
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics for current session"""
        return {
            "images_generated": self.images_generated,
            "session_cost": self.session_cost,
            "cached_images": len(self.cache),
            "cache_enabled": self.enable_caching
        }
    
    def estimate_cost(self, num_images: int, quality: str = "standard") -> float:
        """Estimate cost for generating images"""
        cost_per_image = 0.04 if quality == "standard" else 0.08
        return num_images * cost_per_image


# Convenience function for testing
async def test_image_generator():
    """Test the image generator"""
    print("=" * 60)
    print("TESTING IMAGE GENERATOR (DALL-E 3)")
    print("=" * 60)
    
    generator = ImageGenerator()
    
    # Test with a simple concept
    result = await generator.generate_educational_diagram(
        concept="Working Memory",
        description="The cognitive system responsible for temporarily holding and manipulating information",
        diagram_type=DiagramType.CONCEPT_MAP
    )
    
    if result:
        print(f"\n✅ Image generated successfully!")
        print(f"   Concept: {result.concept}")
        print(f"   Type: {result.diagram_type}")
        print(f"   URL: {result.url[:80]}...")
        print(f"   Cost: ${result.cost_estimate:.2f}")
        print(f"   Cached: {result.cached}")
    else:
        print("\n❌ Image generation failed")
    
    stats = generator.get_session_stats()
    print(f"\n📊 Session Stats:")
    print(f"   Images generated: {stats['images_generated']}")
    print(f"   Session cost: ${stats['session_cost']:.2f}")


if __name__ == "__main__":
    asyncio.run(test_image_generator())


