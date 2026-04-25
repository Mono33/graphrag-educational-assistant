"""
Media Mapping Generator for Agentic GraphRAG

This script generates a sidecar media mapping JSON file from the existing
Knowledge Graph. It uses GPT-4o to act as a neuroscience/cognitive psychology
expert to recommend:

1. Educational videos (YouTube search queries)
2. Diagrams/images (descriptions for generation or search)
3. External resources (Wikipedia, educational sites)
4. Academic citations (seminal papers with DOIs)

The output JSON can be reviewed and improved by domain experts before
being used in the multimodal pipeline.

Usage:
    python generate_media_mapping.py --domain neuro --batch-size 10

Output:
    kg_{domain}_media_mapping.json
"""

import os
import sys
import json
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from openai import AsyncOpenAI
from dotenv import load_dotenv

# Respect LLM_MODEL env var (set in .env); fall back to gpt-4o via OpenRouter
_DEFAULT_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o")

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# NEUROSCIENCE EXPERT PROMPT
# ============================================================================

MEDIA_GENERATOR_SYSTEM_PROMPT = """You are an expert educational content curator specializing in **cognitive neuroscience, educational psychology, and evidence-based learning**.

Your task is to generate high-quality educational media recommendations for neuroscience/cognitive concepts used in teacher training. You must act as a domain expert who understands:

- Cognitive processes (attention, memory, executive functions)
- Learning theories (cognitive load, metacognition, self-regulation)
- Motivation and emotion in learning
- Neuroscience of education (neuroplasticity, brain development)
- Evidence-based teaching strategies

For each concept, generate:

1. **Videos**: Suggest 2-3 specific educational YouTube videos or search queries
   - Prefer established educational channels (CrashCourse, TED-Ed, Khan Academy, Sprouts)
   - Include estimated duration
   - Focus on teacher-appropriate content

2. **Images/Diagrams**: Describe 1-2 ideal educational diagrams
   - Be specific about what the diagram should show
   - Include search queries for finding similar diagrams

3. **External Resources**: Suggest 2-3 quality educational resources
   - Wikipedia articles (use actual URLs when confident)
   - Educational websites (Simply Psychology, Verywell Mind, etc.)
   - Teacher-focused resources

4. **Academic Citations**: Suggest 2-3 seminal papers
   - Include real authors, years, and journals when you know them
   - Include DOI if known
   - Focus on foundational, highly-cited papers

5. **Open Textbooks (OER)**: Suggest 1-2 relevant chapters from open access textbooks
   - Use these COPYRIGHT-SAFE sources (CC BY or open access):
     * OpenStax Psychology (https://openstax.org/subjects/social-sciences)
     * DOAB - Directory of Open Access Books (https://www.doabooks.org)
     * Pressbooks Psychology (https://pressbooks.directory/?subj=Psychology)
     * Open Textbook Library (https://open.umn.edu/opentextbooks/subjects/psychology)
     * BC Campus OpenEd (https://collection.bccampus.ca)
   - Suggest specific chapters or sections relevant to the concept
   - Include the license type (CC BY, CC BY-SA, etc.)

Be SPECIFIC and EDUCATIONAL. These recommendations will help teachers create better lessons.

Output JSON format:
{
  "videos": [
    {
      "title": "Video title",
      "platform": "youtube",
      "search_query": "exact search query",
      "url": "https://youtube.com/..." or null,
      "duration_hint": "5:32" or "~10 min",
      "language": "en",
      "educational_level": "teacher_training"
    }
  ],
  "images": [
    {
      "description": "Detailed description of ideal diagram",
      "search_query": "search query for similar images",
      "type": "diagram|infographic|illustration"
    }
  ],
  "resources": [
    {
      "title": "Resource title",
      "type": "wikipedia|educational|academic",
      "suggested_url": "https://...",
      "language": "en"
    }
  ],
  "citations": [
    {
      "title": "Paper title",
      "authors": ["First Author", "Second Author"],
      "year": 2010,
      "journal": "Journal Name",
      "doi": "10.xxxx/xxxxx" or null,
      "abstract_snippet": "Brief relevance note"
    }
  ],
  "open_textbooks": [
    {
      "title": "Textbook title",
      "source": "OpenStax|DOAB|Pressbooks|OpenTextbookLibrary|BCCampus",
      "chapter": "Chapter X: Title or Section name",
      "url": "https://..." or null,
      "license": "CC BY 4.0|CC BY-SA|Open Access",
      "relevance": "Brief note on why this chapter is relevant"
    }
  ]
}
"""


async def generate_media_for_concept(
    client: AsyncOpenAI,
    concept: Dict[str, Any],
    model: str = _DEFAULT_MODEL
) -> Optional[Dict[str, Any]]:
    """
    Generate media recommendations for a single concept using GPT-4o.
    
    Args:
        client: OpenAI async client
        concept: Concept data from KG
        model: OpenAI model to use
        
    Returns:
        Media recommendations dict or None on failure
    """
    name = concept.get('name', 'Unknown')
    category = concept.get('category', 'Unknown')
    description = concept.get('description', 'No description')
    label = concept.get('label', '')
    
    user_prompt = f"""Generate educational media recommendations for this neuroscience/cognitive concept:

**Concept Name:** {name}
**Category:** {category}
**Label/Domain:** {label}
**Description:** {description}

Generate specific, high-quality educational media recommendations that would help teachers understand and teach this concept effectively.

Remember to:
- Suggest real, findable videos from educational channels (CrashCourse, TED-Ed, Khan Academy)
- Describe diagrams that would clarify the concept
- Include Wikipedia and educational resource links
- Cite real, seminal academic papers when you know them
- **IMPORTANT**: Include relevant chapters from Open Textbooks (OER):
  * OpenStax Psychology 2e (https://openstax.org/details/books/psychology-2e)
  * Introduction to Psychology by University of Minnesota (Open Textbook Library)
  * Any relevant DOAB or Pressbooks psychology textbooks
  * These are COPYRIGHT-SAFE resources with CC licenses

Output ONLY valid JSON.
"""

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": MEDIA_GENERATOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=1500
        )
        
        content = response.choices[0].message.content
        media_data = json.loads(content)
        
        return {
            "id": concept.get('id', ''),
            "name": name,
            "category": category,
            "label": label,
            "videos": media_data.get('videos', []),
            "images": media_data.get('images', []),
            "resources": media_data.get('resources', []),
            "citations": media_data.get('citations', []),
            "open_textbooks": media_data.get('open_textbooks', [])
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for '{name}': {e}")
        return None
    except Exception as e:
        logger.error(f"Error generating media for '{name}': {e}")
        return None


def extract_unique_concepts(kg_path: str) -> List[Dict[str, Any]]:
    """
    Extract unique concepts from Knowledge Graph JSON.
    
    Args:
        kg_path: Path to kg_neuro_neo4j.json
        
    Returns:
        List of unique concepts with their properties
    """
    with open(kg_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    nodes = data.get('nodes', [])
    
    # Deduplicate by name and extract relevant properties
    seen_names = set()
    concepts = []
    
    for node in nodes:
        props = node.get('properties', {})
        name = props.get('name', '')
        
        # Skip if already seen or empty
        if not name or name.lower() in seen_names:
            continue
        
        seen_names.add(name.lower())
        
        concepts.append({
            'id': props.get('id', ''),
            'name': name,
            'category': props.get('category', ''),
            'description': props.get('description', ''),
            'label': node.get('label', ''),
            'domain': props.get('domain', 'neuro')
        })
    
    logger.info(f"Extracted {len(concepts)} unique concepts from KG")
    return concepts


async def process_concepts_batch(
    client: AsyncOpenAI,
    concepts: List[Dict[str, Any]],
    batch_size: int = 5
) -> List[Dict[str, Any]]:
    """
    Process concepts in batches with rate limiting.
    
    Args:
        client: OpenAI async client
        concepts: List of concepts to process
        batch_size: Number of concurrent requests
        
    Returns:
        List of media mappings
    """
    results = []
    total = len(concepts)
    
    for i in range(0, total, batch_size):
        batch = concepts[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} concepts)...")
        
        # Process batch concurrently
        tasks = [generate_media_for_concept(client, c) for c in batch]
        batch_results = await asyncio.gather(*tasks)
        
        # Filter successful results
        for result in batch_results:
            if result:
                results.append(result)
        
        logger.info(f"Batch {batch_num} complete. Total processed: {len(results)}/{total}")
        
        # Rate limiting - pause between batches
        if i + batch_size < total:
            await asyncio.sleep(1.0)
    
    return results


def prioritize_concepts(concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prioritize concepts for processing - core educational concepts first.
    
    Args:
        concepts: List of all concepts
        
    Returns:
        Prioritized list with important concepts first
    """
    # High-priority categories (core neuroscience/education concepts)
    priority_categories = [
        'attention types', 'memory systems', 'executive functions',
        'cognitive processes', 'learning processes', 'motivational types',
        'metacognitive processes', 'belief systems', 'emotions',
        'neuroscience foundations', 'cognitive load'
    ]
    
    # High-priority labels
    priority_labels = [
        'Attention', 'Memory', 'ExecutiveFunctions', 'Metacognition',
        'Motivation', 'Emotions', 'CognitiveLoad', 'Neuroplasticity',
        'Creativity', 'LearningStrategies'
    ]
    
    def get_priority(concept: Dict) -> int:
        category = concept.get('category', '').lower()
        label = concept.get('label', '')
        
        # Check category priority
        for i, cat in enumerate(priority_categories):
            if cat in category:
                return i
        
        # Check label priority
        for i, lab in enumerate(priority_labels):
            if lab.lower() == label.lower():
                return len(priority_categories) + i
        
        return 999
    
    return sorted(concepts, key=get_priority)


async def main():
    """Main entry point for media mapping generation"""
    parser = argparse.ArgumentParser(
        description='Generate media mapping for Knowledge Graph concepts'
    )
    parser.add_argument(
        '--domain', type=str, default='neuro',
        help='Knowledge domain (neuro or udl)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=5,
        help='Batch size for concurrent API calls'
    )
    parser.add_argument(
        '--limit', type=int, default=None,
        help='Limit number of concepts to process (for testing)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output file path (default: kg_{domain}_media_mapping.json)'
    )
    parser.add_argument(
        '--model', type=str, default=_DEFAULT_MODEL,
        help='Model to use (via OpenRouter if OPENROUTER_API_KEY is set)'
    )
    
    args = parser.parse_args()
    
    # Paths
    base_path = Path(__file__).parent
    kg_path = base_path / f"kg_{args.domain}_neo4j.json"
    output_path = args.output or base_path / f"kg_{args.domain}_media_mapping.json"
    
    if not kg_path.exists():
        logger.error(f"Knowledge Graph not found: {kg_path}")
        sys.exit(1)
    
    # Check API key — prefer OpenRouter, fall back to direct OpenAI
    api_key = os.getenv('OPENROUTER_API_KEY') or os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENROUTER_API_KEY (or OPENAI_API_KEY) not set in environment")
        sys.exit(1)
    use_openrouter = bool(os.getenv('OPENROUTER_API_KEY'))
    
    logger.info(f"=" * 60)
    logger.info(f"Media Mapping Generator for Agentic GraphRAG")
    logger.info(f"=" * 60)
    logger.info(f"Domain: {args.domain}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"KG source: {kg_path}")
    logger.info(f"Output: {output_path}")
    
    # Extract and prioritize concepts
    concepts = extract_unique_concepts(str(kg_path))
    concepts = prioritize_concepts(concepts)
    
    if args.limit:
        concepts = concepts[:args.limit]
        logger.info(f"Limited to {args.limit} concepts")
    
    # Initialize client — point to OpenRouter when OPENROUTER_API_KEY is set
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1" if use_openrouter else None,
    )
    
    # Process concepts
    logger.info(f"\nProcessing {len(concepts)} concepts...")
    start_time = datetime.now()
    
    media_mappings = await process_concepts_batch(
        client, concepts, args.batch_size
    )
    
    elapsed = datetime.now() - start_time
    
    # Build output structure
    output_data = {
        "metadata": {
            "domain": args.domain,
            "generated_at": datetime.now().isoformat(),
            "model": args.model,
            "total_concepts": len(media_mappings),
            "generation_time_seconds": elapsed.total_seconds(),
            "version": "1.0.0",
            "description": "Media mapping for Knowledge Graph concepts. "
                          "Can be reviewed and improved by domain experts."
        },
        "concepts": media_mappings
    }
    
    # Calculate statistics
    total_videos = sum(len(c.get('videos', [])) for c in media_mappings)
    total_images = sum(len(c.get('images', [])) for c in media_mappings)
    total_resources = sum(len(c.get('resources', [])) for c in media_mappings)
    total_citations = sum(len(c.get('citations', [])) for c in media_mappings)
    total_textbooks = sum(len(c.get('open_textbooks', [])) for c in media_mappings)
    
    output_data["metadata"]["statistics"] = {
        "total_videos": total_videos,
        "total_images": total_images,
        "total_resources": total_resources,
        "total_citations": total_citations,
        "total_open_textbooks": total_textbooks
    }
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n" + "=" * 60)
    logger.info(f"GENERATION COMPLETE")
    logger.info(f"=" * 60)
    logger.info(f"Concepts processed: {len(media_mappings)}")
    logger.info(f"Total videos: {total_videos}")
    logger.info(f"Total images: {total_images}")
    logger.info(f"Total resources: {total_resources}")
    logger.info(f"Total citations: {total_citations}")
    logger.info(f"Total open textbooks: {total_textbooks}")
    logger.info(f"Time elapsed: {elapsed}")
    logger.info(f"Output saved to: {output_path}")
    logger.info(f"\n📚 OER Sources included:")
    logger.info(f"   - OpenStax Psychology")
    logger.info(f"   - DOAB (Directory of Open Access Books)")
    logger.info(f"   - Pressbooks Psychology")
    logger.info(f"   - Open Textbook Library")
    logger.info(f"   - BC Campus OpenEd")
    logger.info(f"\nThe mapping can now be reviewed by domain experts!")


if __name__ == "__main__":
    asyncio.run(main())


