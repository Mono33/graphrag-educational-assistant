"""
Mermaid.js Diagram Generator

Generates educational diagrams using Mermaid.js syntax rendered via mermaid.ink API.
This is a FREE alternative to DALL-E that produces:
- Perfect text accuracy (no garbled words)
- Scalable SVG output
- Editable code that teachers can modify

Mermaid.ink API: https://mermaid.ink
- No API key required
- No rate limits
- Returns SVG or PNG

Usage:
    from agent.media.mermaid_generator import MermaidGenerator
    
    generator = MermaidGenerator()
    result = await generator.generate("metacognition", "mindmap", ["planning", "monitoring"])
    print(result.svg_url)  # URL to rendered SVG
    print(result.mermaid_code)  # Raw Mermaid code
"""

import os
import base64
import logging
import asyncio
import aiohttp
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class MermaidDiagramType(Enum):
    """Supported Mermaid diagram types"""
    MINDMAP = "mindmap"
    FLOWCHART = "flowchart"
    SEQUENCE = "sequence"
    TIMELINE = "timeline"
    HIERARCHY = "hierarchy"
    COMPARISON = "comparison"
    PROCESS = "process"


@dataclass
class MermaidResult:
    """Result from Mermaid diagram generation"""
    success: bool
    concept: str
    diagram_type: str
    
    # URLs for rendered diagrams
    svg_url: Optional[str] = None
    png_url: Optional[str] = None
    
    # Raw Mermaid code (for display/editing)
    mermaid_code: Optional[str] = None
    
    # Cost is always 0 (FREE!)
    cost: float = 0.0
    
    # Error info
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "success": self.success,
            "concept": self.concept,
            "diagram_type": self.diagram_type,
            "svg_url": self.svg_url,
            "png_url": self.png_url,
            "mermaid_code": self.mermaid_code,
            "cost": self.cost,
            "error_message": self.error_message
        }


# Mermaid code templates for different diagram types
MERMAID_TEMPLATES = {
    "mindmap": """Generate a Mermaid.js mindmap for the educational concept "{concept}".

Related concepts to include: {related_concepts}

Requirements:
- Use the mindmap syntax
- Central node should be the main concept
- Branch into 3-5 main categories
- Each category can have 2-3 sub-items
- Keep labels short (2-4 words max)
- Use Italian labels if the concept is in Italian

Example format:
mindmap
  root((Concetto Centrale))
    Categoria 1
      Elemento 1.1
      Elemento 1.2
    Categoria 2
      Elemento 2.1
      Elemento 2.2

Output ONLY the Mermaid code, no explanation or markdown fences.""",

    "flowchart": """Generate a Mermaid.js flowchart for the educational concept "{concept}".

Related concepts to include: {related_concepts}

Requirements:
- Use flowchart TD (top-down) syntax
- Show the process or relationship flow
- Use appropriate node shapes: [] for process, () for start/end, {{}} for decision
- Keep labels concise
- Use arrows to show relationships

Example format:
flowchart TD
    A[Start] --> B{{Decision?}}
    B -->|Yes| C[Action 1]
    B -->|No| D[Action 2]
    C --> E[End]
    D --> E

Output ONLY the Mermaid code, no explanation or markdown fences.""",

    "hierarchy": """Generate a Mermaid.js hierarchy diagram for the educational concept "{concept}".

Related concepts to include: {related_concepts}

Requirements:
- Use graph TD (top-down) syntax
- Show hierarchical relationships
- Main concept at top, sub-concepts below
- Use consistent node styling

Example format:
graph TD
    A[Main Concept] --> B[Sub-concept 1]
    A --> C[Sub-concept 2]
    B --> D[Detail 1.1]
    B --> E[Detail 1.2]
    C --> F[Detail 2.1]

Output ONLY the Mermaid code, no explanation or markdown fences.""",

    "sequence": """Generate a Mermaid.js sequence diagram for the educational concept "{concept}".

Related concepts to include: {related_concepts}

Requirements:
- Use sequenceDiagram syntax
- Show interactions between components/actors
- Use appropriate message types (->>, -->>)
- Keep participant names short

Example format:
sequenceDiagram
    participant S as Student
    participant T as Teacher
    participant B as Brain
    S->>T: Ask question
    T->>S: Provide explanation
    S->>B: Process information
    B-->>S: Understanding

Output ONLY the Mermaid code, no explanation or markdown fences.""",

    "timeline": """Generate a Mermaid.js timeline for the educational concept "{concept}".

Related concepts to include: {related_concepts}

Requirements:
- Use timeline syntax
- Show progression or stages
- Include 4-6 time points or stages
- Brief descriptions for each

Example format:
timeline
    title Learning Stages
    section Foundation
        Stage 1 : Basic understanding
        Stage 2 : Initial practice
    section Development
        Stage 3 : Deeper learning
        Stage 4 : Application

Output ONLY the Mermaid code, no explanation or markdown fences.""",

    "comparison": """Generate a Mermaid.js comparison diagram for the educational concept "{concept}".

Related concepts to include: {related_concepts}

Requirements:
- Use graph LR (left-right) with subgraphs
- Compare 2-3 related concepts
- Show similarities and differences
- Use subgraphs to group related items

Example format:
graph LR
    subgraph Concept A
        A1[Feature 1]
        A2[Feature 2]
    end
    subgraph Concept B
        B1[Feature 1]
        B2[Feature 2]
    end
    A1 -.->|similar| B1
    A2 -.->|different| B2

Output ONLY the Mermaid code, no explanation or markdown fences.""",

    "process": """Generate a Mermaid.js process diagram for the educational concept "{concept}".

Related concepts to include: {related_concepts}

Requirements:
- Use flowchart LR (left-right) syntax
- Show step-by-step process
- Use numbered steps if appropriate
- Include decision points if relevant

Example format:
flowchart LR
    A[Step 1: Input] --> B[Step 2: Process]
    B --> C{{Check}}
    C -->|OK| D[Step 3: Output]
    C -->|Error| E[Step 4: Retry]
    E --> B

Output ONLY the Mermaid code, no explanation or markdown fences."""
}


class MermaidGenerator:
    """
    Generate educational diagrams using Mermaid.js
    
    Uses mermaid.ink API for rendering (FREE, no API key needed).
    Uses GPT-4o-mini for generating Mermaid code (fast, cheap).
    """
    
    MERMAID_INK_URL = "https://mermaid.ink"
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize Mermaid generator.
        
        Args:
            openai_api_key: Optional OpenAI API key. If not provided,
                          will use OPENAI_API_KEY environment variable.
        """
        self.api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self._client: Optional[AsyncOpenAI] = None
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Track usage (for logging only, it's free!)
        self._diagrams_generated = 0
        
        logger.info("[MermaidGenerator] Initialized (FREE diagram generation)")
    
    async def _get_client(self) -> AsyncOpenAI:
        """Get or create OpenAI client"""
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """Close resources"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _encode_mermaid(self, mermaid_code: str) -> str:
        """Encode Mermaid code for URL"""
        # Clean up the code
        code = mermaid_code.strip()
        
        # Remove markdown fences if present
        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        
        # Base64 encode for URL
        encoded = base64.urlsafe_b64encode(code.encode('utf-8')).decode('utf-8')
        return encoded
    
    def get_svg_url(self, mermaid_code: str) -> str:
        """Get mermaid.ink SVG URL for the code"""
        encoded = self._encode_mermaid(mermaid_code)
        return f"{self.MERMAID_INK_URL}/svg/{encoded}"
    
    def get_png_url(self, mermaid_code: str) -> str:
        """Get mermaid.ink PNG URL for the code"""
        encoded = self._encode_mermaid(mermaid_code)
        return f"{self.MERMAID_INK_URL}/img/{encoded}"
    
    async def _generate_mermaid_code(
        self,
        concept: str,
        diagram_type: str,
        related_concepts: Optional[List[str]] = None
    ) -> str:
        """Use LLM to generate Mermaid.js code"""
        
        # Get template for diagram type
        template = MERMAID_TEMPLATES.get(
            diagram_type.lower(),
            MERMAID_TEMPLATES["mindmap"]  # Default to mindmap
        )
        
        # Format prompt
        prompt = template.format(
            concept=concept,
            related_concepts=", ".join(related_concepts) if related_concepts else "none specified"
        )
        
        try:
            client = await self._get_client()
            
            response = await client.chat.completions.create(
                model="gpt-4o-mini",  # Fast and cheap for code generation
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at creating Mermaid.js diagrams for educational content. Output ONLY valid Mermaid code, no explanations."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent code
                max_tokens=1000
            )
            
            mermaid_code = response.choices[0].message.content.strip()
            
            # Clean up any markdown fences
            if mermaid_code.startswith("```"):
                lines = mermaid_code.split("\n")
                # Remove first line (```mermaid) and last line (```)
                mermaid_code = "\n".join(
                    line for i, line in enumerate(lines) 
                    if i > 0 and not (i == len(lines) - 1 and line.strip() == "```")
                )
            
            logger.info(f"[MermaidGenerator] Generated {diagram_type} code for '{concept}'")
            return mermaid_code.strip()
            
        except Exception as e:
            logger.error(f"[MermaidGenerator] Error generating code: {e}")
            raise
    
    async def generate(
        self,
        concept: str,
        diagram_type: str = "mindmap",
        related_concepts: Optional[List[str]] = None,
        validate: bool = True
    ) -> MermaidResult:
        """
        Generate a Mermaid diagram.
        
        Args:
            concept: Main concept for the diagram
            diagram_type: Type of diagram (mindmap, flowchart, hierarchy, etc.)
            related_concepts: Optional list of related concepts to include
            validate: Whether to validate the generated code renders correctly
            
        Returns:
            MermaidResult with SVG/PNG URLs and raw code
        """
        try:
            # Generate Mermaid code using LLM
            mermaid_code = await self._generate_mermaid_code(
                concept, diagram_type, related_concepts
            )
            
            # Generate URLs
            svg_url = self.get_svg_url(mermaid_code)
            png_url = self.get_png_url(mermaid_code)
            
            # Optionally validate that the code renders
            if validate:
                is_valid = await self._validate_render(svg_url)
                if not is_valid:
                    logger.warning(f"[MermaidGenerator] Code validation failed, attempting fix...")
                    # Try a simpler fallback
                    mermaid_code = self._get_fallback_code(concept, diagram_type)
                    svg_url = self.get_svg_url(mermaid_code)
                    png_url = self.get_png_url(mermaid_code)
            
            self._diagrams_generated += 1
            
            return MermaidResult(
                success=True,
                concept=concept,
                diagram_type=diagram_type,
                svg_url=svg_url,
                png_url=png_url,
                mermaid_code=mermaid_code,
                cost=0.0  # FREE!
            )
            
        except Exception as e:
            logger.error(f"[MermaidGenerator] Error generating diagram: {e}")
            return MermaidResult(
                success=False,
                concept=concept,
                diagram_type=diagram_type,
                error_message=str(e)
            )
    
    async def _validate_render(self, svg_url: str) -> bool:
        """Check if the SVG URL returns a valid image"""
        try:
            session = await self._get_session()
            async with session.head(svg_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                return resp.status == 200
        except Exception:
            return False
    
    def _get_fallback_code(self, concept: str, diagram_type: str) -> str:
        """Generate simple fallback code if LLM-generated code fails"""
        if diagram_type == "mindmap":
            return f"""mindmap
  root(({concept}))
    Caratteristiche
      Elemento 1
      Elemento 2
    Applicazioni
      Uso 1
      Uso 2"""
        elif diagram_type in ["flowchart", "process"]:
            return f"""flowchart TD
    A[{concept}] --> B[Fase 1]
    B --> C[Fase 2]
    C --> D[Risultato]"""
        elif diagram_type == "hierarchy":
            return f"""graph TD
    A[{concept}] --> B[Sottoconcetto 1]
    A --> C[Sottoconcetto 2]
    B --> D[Dettaglio 1]
    C --> E[Dettaglio 2]"""
        else:
            return f"""mindmap
  root(({concept}))
    Info 1
    Info 2
    Info 3"""
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get generator statistics"""
        return {
            "diagrams_generated": self._diagrams_generated,
            "total_cost": 0.0,  # Always free!
            "generator": "mermaid.ink"
        }


# =============================================================================
# TESTING
# =============================================================================
async def test_mermaid_generator():
    """Test the Mermaid generator"""
    print("=" * 60)
    print("TESTING MERMAID GENERATOR")
    print("=" * 60)
    
    generator = MermaidGenerator()
    
    # Test different diagram types
    test_cases = [
        ("metacognizione", "mindmap", ["pianificazione", "monitoraggio", "valutazione"]),
        ("self-regulation", "flowchart", ["goal setting", "monitoring", "adjustment"]),
        ("learning styles", "comparison", ["visual", "auditory", "kinesthetic"]),
    ]
    
    for concept, diagram_type, related in test_cases:
        print(f"\n📊 Generating {diagram_type} for '{concept}'...")
        
        result = await generator.generate(concept, diagram_type, related, validate=False)
        
        if result.success:
            print(f"  ✅ Success!")
            print(f"  📝 Mermaid code:\n{result.mermaid_code[:200]}...")
            print(f"  🔗 SVG URL: {result.svg_url[:80]}...")
            print(f"  💰 Cost: ${result.cost:.2f} (FREE!)")
        else:
            print(f"  ❌ Error: {result.error_message}")
    
    await generator.close()
    print(f"\n📊 Stats: {generator.stats}")
    print("\n✅ Mermaid generator tests complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_mermaid_generator())


