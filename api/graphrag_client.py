"""
GraphRAG Client - Helper module for DEV team integration

This module provides a simple interface to get GraphRAG context.
DEV team can either:
1. Use this client directly (if API is deployed)
2. Copy this file to their project

Usage:
    from graphrag_client import get_graphrag_context, format_for_prompt
    
    # Get context
    context = get_graphrag_context("Quali strategie per studenti con ADHD?", domain="neuro")
    
    # Use in prompt
    prompt_section = format_for_prompt(context)
"""

import requests
from typing import Optional, Dict, Any


# Configure this based on deployment
GRAPHRAG_API_URL = "http://localhost:8000/api/v1"


def get_graphrag_context(
    query: str,
    domain: str = "neuro",
    language: str = "it",
    include_raw_nodes: bool = False,
    max_methodologies: int = 5,
    api_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get educational context from GraphRAG API
    
    Args:
        query: The educational query (Italian or English)
        domain: Knowledge domain ("neuro", "udl", or "all")
        language: Query language ("it" or "en")
        include_raw_nodes: Whether to include raw node data
        max_methodologies: Maximum methodologies to return
        api_url: Override default API URL
    
    Returns:
        Dict with context data ready for prompt injection
        
    Example:
        context = get_graphrag_context(
            "Come gestire studenti con ADHD?",
            domain="neuro"
        )
        
        # Access structured data
        methodologies = context["context"]["primary_methodologies"]
        evidence = context["context"]["evidence_summary"]
        
        # Or use pre-formatted prompt section
        prompt_text = context["formatted_prompt_section"]
    """
    url = api_url or GRAPHRAG_API_URL
    
    try:
        response = requests.post(
            f"{url}/context",
            json={
                "query": query,
                "domain": domain,
                "language": language,
                "include_raw_nodes": include_raw_nodes,
                "max_methodologies": max_methodologies
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": str(e),
            "context": {
                "educational_context_type": "error",
                "student_profile": "",
                "primary_methodologies": [],
                "supporting_methodologies": [],
                "evidence_summary": "Unable to retrieve context from knowledge graph",
                "implementation_priority": [],
                "confidence_level": "very_low",
                "fallback_strategies": ["Please try again later"]
            },
            "formatted_prompt_section": "⚠️ Knowledge graph context unavailable"
        }


def format_for_prompt(context: Dict[str, Any], language: str = "it") -> str:
    """
    Format GraphRAG context as a prompt section
    
    If the API response includes formatted_prompt_section, returns that.
    Otherwise, builds a basic formatted section from the context data.
    
    Args:
        context: Response from get_graphrag_context()
        language: Output language ("it" or "en")
    
    Returns:
        Formatted string ready to inject into prompt
    """
    # Use pre-formatted section if available
    if context.get("formatted_prompt_section"):
        return context["formatted_prompt_section"]
    
    # Otherwise build from context data
    ctx = context.get("context", {})
    
    if language == "it":
        lines = [
            "## CONTESTO DAL KNOWLEDGE GRAPH",
            "",
            f"**Contesto:** {ctx.get('educational_context_type', 'N/A')}",
            ""
        ]
        
        methodologies = ctx.get("primary_methodologies", [])
        if methodologies:
            lines.append("**Metodologie Raccomandate:**")
            for i, m in enumerate(methodologies, 1):
                lines.append(f"{i}. {m.get('name', 'N/A')} ({m.get('category', '')})")
            lines.append("")
        
        if ctx.get("evidence_summary"):
            lines.append(f"**Evidenza:** {ctx['evidence_summary']}")
            lines.append("")
        
        lines.append(f"**Confidenza:** {ctx.get('confidence_level', 'N/A').upper()}")
        
        return "\n".join(lines)
    
    else:  # English
        lines = [
            "## KNOWLEDGE GRAPH CONTEXT",
            "",
            f"**Context:** {ctx.get('educational_context_type', 'N/A')}",
            ""
        ]
        
        methodologies = ctx.get("primary_methodologies", [])
        if methodologies:
            lines.append("**Recommended Methodologies:**")
            for i, m in enumerate(methodologies, 1):
                lines.append(f"{i}. {m.get('name', 'N/A')} ({m.get('category', '')})")
            lines.append("")
        
        if ctx.get("evidence_summary"):
            lines.append(f"**Evidence:** {ctx['evidence_summary']}")
        
        return "\n".join(lines)


# ============================================================================
# EXAMPLE INTEGRATION FOR DEV TEAM
# ============================================================================

def example_integration():
    """
    Example showing how DEV team can integrate this into their prompt_assemblers.py
    
    They would add something like:
    
    ```python
    # In prompt_assemblers.py
    from graphrag_client import get_graphrag_context, format_for_prompt
    
    def create_graphrag_prompt(user_query: str) -> str:
        '''Get knowledge graph context for a query'''
        context = get_graphrag_context(user_query, domain="neuro")
        
        if context.get("success", False):
            return format_for_prompt(context)
        else:
            return ""  # Graceful fallback
    
    # Then in their main assembler:
    def get_assistant_prompt_assembler(tool, lesson_plan, user_prompt, ...):
        prompt = ""
        
        # ... existing code ...
        
        # Add GraphRAG context
        graphrag_section = create_graphrag_prompt(user_prompt)
        if graphrag_section:
            prompt += "\\n\\n" + graphrag_section
        
        return prompt
    ```
    """
    print("=" * 60)
    print("GraphRAG Client - Integration Example")
    print("=" * 60)
    
    # Example query
    query = "Quali strategie per studenti con ADHD?"
    print(f"\n📝 Query: {query}")
    
    # Get context
    print("\n🔄 Fetching context from GraphRAG API...")
    context = get_graphrag_context(query, domain="neuro")
    
    if context.get("success"):
        print("✅ Context retrieved successfully!")
        print(f"\n📊 Metrics:")
        print(f"   - Nodes: {context.get('metrics', {}).get('total_nodes', 0)}")
        print(f"   - Relationships: {context.get('metrics', {}).get('total_relationships', 0)}")
        print(f"   - Processing time: {context.get('metrics', {}).get('processing_time_ms', 0)}ms")
        
        print("\n" + "=" * 60)
        print("📄 FORMATTED PROMPT SECTION:")
        print("=" * 60)
        print(format_for_prompt(context))
    else:
        print(f"❌ Error: {context.get('error', 'Unknown error')}")


if __name__ == "__main__":
    example_integration()

