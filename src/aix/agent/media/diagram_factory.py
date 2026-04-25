"""
Diagram Factory - Unified interface for multiple diagram generators

This factory provides a single interface to generate educational diagrams
using different generators:
- Mermaid.js (FREE, accurate text, SVG output)
- DALL-E 3 (Paid, visually rich, raster output)
- Canva (Paid, professional templates - coming soon)

Usage:
    from aix.agent.media.diagram_factory import DiagramFactory, GeneratorType
    
    factory = DiagramFactory()
    
    # Generate with Mermaid (FREE)
    result = await factory.generate(
        generator_type=GeneratorType.MERMAID,
        concept="metacognition",
        diagram_type="mindmap"
    )
    
    # Generate with DALL-E ($0.04)
    result = await factory.generate(
        generator_type=GeneratorType.DALLE,
        concept="metacognition",
        diagram_type="concept_map"
    )
"""

import logging
from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from .mermaid_generator import MermaidGenerator, MermaidResult
from .image_generator import ImageGenerator, DiagramType as DalleDiagramType
from .canva_generator import CanvaGenerator, CanvaResult

logger = logging.getLogger(__name__)


class GeneratorType(Enum):
    """Available diagram generators"""
    MERMAID = "mermaid"
    DALLE = "dalle"
    CANVA = "canva"


@dataclass
class DiagramResult:
    """Unified result from any diagram generator"""
    success: bool
    generator_type: str
    concept: str
    diagram_type: str
    
    # Image output (URL or base64)
    image_url: Optional[str] = None
    svg_url: Optional[str] = None
    png_url: Optional[str] = None
    
    # For Mermaid: raw code (copyable/editable)
    mermaid_code: Optional[str] = None
    
    # Cost tracking
    cost: float = 0.0
    
    # Error handling
    error_message: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "generator_type": self.generator_type,
            "concept": self.concept,
            "diagram_type": self.diagram_type,
            "image_url": self.image_url,
            "svg_url": self.svg_url,
            "png_url": self.png_url,
            "mermaid_code": self.mermaid_code,
            "cost": self.cost,
            "error_message": self.error_message,
            "metadata": self.metadata
        }


# Mapping of diagram types between generators
DIAGRAM_TYPE_MAPPING = {
    # User-friendly name -> (mermaid_type, dalle_type)
    "Mappa Concettuale": ("mindmap", "concept_map"),
    "Concept Map": ("mindmap", "concept_map"),
    "Diagramma di Flusso": ("flowchart", "flowchart"),
    "Flowchart": ("flowchart", "flowchart"),
    "Gerarchia": ("hierarchy", "hierarchy"),
    "Hierarchy": ("hierarchy", "hierarchy"),
    "Sequenza": ("sequence", "process"),
    "Sequence": ("sequence", "process"),
    "Timeline": ("timeline", "infographic"),
    "Confronto": ("comparison", "comparison"),
    "Comparison": ("comparison", "comparison"),
    "Processo": ("process", "process"),
    "Process": ("process", "process"),
    "Infografica": ("mindmap", "infographic"),
    "Infographic": ("mindmap", "infographic"),
}


class DiagramFactory:
    """
    Factory for creating educational diagrams.
    
    Provides a unified interface to multiple diagram generators,
    allowing easy switching between free (Mermaid) and paid (DALL-E) options.
    """
    
    def __init__(self):
        """Initialize all generators"""
        self._mermaid: Optional[MermaidGenerator] = None
        self._dalle: Optional[ImageGenerator] = None
        self._canva: Optional[CanvaGenerator] = None
        
        # Track total costs across all generators
        self._total_cost = 0.0
        self._generation_count = {
            "mermaid": 0,
            "dalle": 0,
            "canva": 0
        }
        
        logger.info("[DiagramFactory] Initialized with multi-generator support")
    
    @property
    def mermaid(self) -> MermaidGenerator:
        """Lazy-load Mermaid generator"""
        if self._mermaid is None:
            self._mermaid = MermaidGenerator()
        return self._mermaid
    
    @property
    def dalle(self) -> ImageGenerator:
        """Lazy-load DALL-E generator"""
        if self._dalle is None:
            self._dalle = ImageGenerator()
        return self._dalle
    
    @property
    def canva(self) -> CanvaGenerator:
        """Lazy-load Canva generator"""
        if self._canva is None:
            self._canva = CanvaGenerator()
        return self._canva
    
    async def close(self):
        """Close all generators"""
        if self._mermaid:
            await self._mermaid.close()
        if self._dalle:
            await self._dalle.close()
        if self._canva:
            await self._canva.close()
    
    def _get_diagram_types(self, diagram_type: str) -> tuple:
        """Get generator-specific diagram types from user-friendly name"""
        mapping = DIAGRAM_TYPE_MAPPING.get(diagram_type)
        if mapping:
            return mapping
        
        # Default fallback
        return (diagram_type.lower(), diagram_type.lower())
    
    async def generate(
        self,
        generator_type: GeneratorType,
        concept: str,
        diagram_type: str,
        related_concepts: Optional[List[str]] = None,
        description: Optional[str] = None
    ) -> DiagramResult:
        """
        Generate a diagram using the specified generator.
        
        Args:
            generator_type: Which generator to use (MERMAID, DALLE, CANVA)
            concept: Main concept for the diagram
            diagram_type: Type of diagram (will be mapped to generator-specific type)
            related_concepts: Optional list of related concepts
            description: Optional description for context
            
        Returns:
            DiagramResult with image URLs and metadata
        """
        mermaid_type, dalle_type = self._get_diagram_types(diagram_type)
        
        try:
            if generator_type == GeneratorType.MERMAID:
                return await self._generate_mermaid(
                    concept, mermaid_type, related_concepts
                )
            elif generator_type == GeneratorType.DALLE:
                return await self._generate_dalle(
                    concept, dalle_type, description
                )
            elif generator_type == GeneratorType.CANVA:
                return await self._generate_canva(
                    concept, diagram_type, related_concepts
                )
            else:
                raise ValueError(f"Unknown generator type: {generator_type}")
                
        except Exception as e:
            logger.error(f"[DiagramFactory] Error with {generator_type.value}: {e}")
            return DiagramResult(
                success=False,
                generator_type=generator_type.value,
                concept=concept,
                diagram_type=diagram_type,
                error_message=str(e)
            )
    
    async def _generate_mermaid(
        self,
        concept: str,
        diagram_type: str,
        related_concepts: Optional[List[str]]
    ) -> DiagramResult:
        """Generate diagram using Mermaid.js"""
        result = await self.mermaid.generate(
            concept=concept,
            diagram_type=diagram_type,
            related_concepts=related_concepts
        )
        
        self._generation_count["mermaid"] += 1
        
        return DiagramResult(
            success=result.success,
            generator_type="mermaid",
            concept=concept,
            diagram_type=diagram_type,
            svg_url=result.svg_url,
            png_url=result.png_url,
            mermaid_code=result.mermaid_code,
            cost=0.0,
            error_message=result.error_message,
            metadata={"generator": "mermaid.ink"}
        )
    
    async def _generate_dalle(
        self,
        concept: str,
        diagram_type: str,
        description: Optional[str]
    ) -> DiagramResult:
        """Generate diagram using DALL-E 3"""
        # Map string type to enum
        dalle_type = DalleDiagramType.CONCEPT_MAP  # Default
        for dt in DalleDiagramType:
            if dt.value == diagram_type:
                dalle_type = dt
                break
        
        generated = await self.dalle.generate_educational_diagram(
            concept=concept,
            description=description or "",
            diagram_type=dalle_type,
        )

        cost = 0.04  # DALL-E 3 cost per image
        self._total_cost += cost
        self._generation_count["dalle"] += 1

        if generated:
            return DiagramResult(
                success=True,
                generator_type="dalle",
                concept=concept,
                diagram_type=diagram_type,
                image_url=generated.url,
                cost=cost,
                metadata={"generator": "DALL-E 3", "model": "dall-e-3"}
            )
        else:
            return DiagramResult(
                success=False,
                generator_type="dalle",
                concept=concept,
                diagram_type=diagram_type,
                cost=cost,
                error_message="DALL-E generation failed"
            )
    
    async def _generate_canva(
        self,
        concept: str,
        diagram_type: str,
        related_concepts: Optional[List[str]]
    ) -> DiagramResult:
        """Generate diagram using Canva API"""
        result = await self.canva.generate(
            concept=concept,
            diagram_type=diagram_type,
            related_concepts=related_concepts
        )
        
        self._generation_count["canva"] += 1
        
        return DiagramResult(
            success=result.success,
            generator_type="canva",
            concept=concept,
            diagram_type=diagram_type,
            image_url=result.image_url,
            cost=result.cost,
            error_message=result.error_message,
            metadata={"generator": "Canva Connect API"}
        )
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get factory statistics"""
        return {
            "total_cost": self._total_cost,
            "generation_count": self._generation_count,
            "generators_available": ["mermaid", "dalle", "canva"]
        }
    
    @staticmethod
    def get_generator_info() -> List[Dict[str, Any]]:
        """Get information about available generators"""
        return [
            {
                "id": "mermaid",
                "name": "Mermaid.js",
                "description": "Diagrammi con testo preciso, formato SVG",
                "cost": "Gratuito",
                "cost_per_image": 0.0,
                "features": ["Testo accurato", "SVG scalabile", "Codice modificabile"],
                "best_for": ["Mappe concettuali", "Diagrammi di flusso", "Gerarchie"],
                "available": True
            },
            {
                "id": "dalle",
                "name": "DALL-E 3",
                "description": "Immagini AI visivamente ricche",
                "cost": "$0.04/immagine",
                "cost_per_image": 0.04,
                "features": ["Visivamente attraente", "Stile artistico", "Colori vivaci"],
                "best_for": ["Poster decorativi", "Infografiche", "Ispirazione visiva"],
                "available": True
            },
            {
                "id": "canva",
                "name": "Canva",
                "description": "Template professionali personalizzabili",
                "cost": "Abbonamento richiesto",
                "cost_per_image": None,
                "features": ["Template professionali", "Brand consistency", "Export multipli"],
                "best_for": ["Materiali didattici formali", "Presentazioni"],
                "available": False  # Coming soon
            }
        ]


# Convenience alias
DiagramGeneratorFactory = DiagramFactory


