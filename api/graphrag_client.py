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
from typing import Optional, Dict, Any, List


# Configure this based on deployment
GRAPHRAG_API_URL = "http://localhost:8000/api/v1"


def get_graphrag_context(
    query: str,
    domain: str = "neuro",
    language: str = "it",
    include_raw_nodes: bool = False,
    max_methodologies: int = 10,
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
    # Use pre-formatted section if available (now domain-aware)
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


def get_domain_prompt_context(context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Get domain-specific prompt context from API response (Option B).
    
    This provides the full domain expertise for production integration:
    - system_prompt: Rich domain role + principles + meta-rules
    - response_template: Domain-specific output structure (e.g., I Do/We Do/You Do)
    - kg_context_formatted: KG data formatted for the domain
    
    Args:
        context: Response from get_graphrag_context()
    
    Returns:
        Dict with domain_prompt_context fields, or None if not available
    
    Example:
        response = get_graphrag_context("Come gestire ADHD?", domain="neuro")
        domain_ctx = get_domain_prompt_context(response)
        
        if domain_ctx:
            # Use the rich system prompt
            system_prompt = domain_ctx["system_prompt"]
            
            # Get the lesson schema template
            response_template = domain_ctx["response_template"]
            
            # Get KG data formatted for this domain
            kg_block = domain_ctx["kg_context_formatted"]
    """
    return context.get("domain_prompt_context")


# ============================================================================
# EXAMPLE INTEGRATION FOR DEV TEAM
# ============================================================================

def example_integration():
    """
    Example showing how DEV team can integrate this into their prompt_assemblers.py
    
    Three integration options:
    
    Option 1 - Simple (one block):
    ```python
    from graphrag_client import get_graphrag_context, format_for_prompt
    
    def create_graphrag_prompt(user_query: str, domain: str = "neuro") -> str:
        context = get_graphrag_context(user_query, domain=domain)
        if context.get("success", False):
            return format_for_prompt(context)
        return ""
    ```
    
    Option 2 - Granular (inject at specific points):
    ```python
    from graphrag_client import get_graphrag_context
    
    def assemble_prompt(user_query: str, base_prompt: str) -> str:
        response = get_graphrag_context(user_query, domain="neuro")
        if response["success"]:
            ctx = response["context"]
            # Build text from raw methodologies
            methods = "\\n".join(
                f"- {m['name']}: {m['implementation_guidance']}"
                for m in ctx["primary_methodologies"]
            )
            # Inject at [usa le info dal GraphRag] points
            prompt = base_prompt.replace(
                "[usa le info dal GraphRag]",
                f"\\n{methods}"
            )
            return prompt
        return base_prompt
    ```
    
    Option 3 - Full alignment (use domain_prompt_context):
    ```python
    from graphrag_client import get_graphrag_context, get_domain_prompt_context
    
    def assemble_prompt_full(user_query: str, base_prompt: str) -> str:
        response = get_graphrag_context(user_query, domain="neuro")
        domain_ctx = get_domain_prompt_context(response)
        
        if domain_ctx:
            # Rich system prompt with RUOLO, TAG-CLOUD, PRINCIPI
            system_prompt = domain_ctx["system_prompt"]
            
            # Lesson schema (I Do / We Do / You Do)
            response_template = domain_ctx["response_template"]
            
            # KG data already formatted for the domain
            kg_block = domain_ctx["kg_context_formatted"]
            
            return f"{base_prompt}\\n\\n{kg_block}"
        return base_prompt
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
        print("📄 FORMATTED PROMPT SECTION (domain-aware):")
        print("=" * 60)
        print(format_for_prompt(context))
        
        # Show domain prompt context (Option B)
        domain_ctx = get_domain_prompt_context(context)
        if domain_ctx:
            print("\n" + "=" * 60)
            print("🏗️ DOMAIN PROMPT CONTEXT (Option B):")
            print("=" * 60)
            print(f"   Domain: {domain_ctx.get('domain')}")
            print(f"   Display Name: {domain_ctx.get('domain_display_name')}")
            print(f"   System Prompt: {len(domain_ctx.get('system_prompt', ''))} chars")
            print(f"   Response Template: {len(domain_ctx.get('response_template', ''))} chars")
            print(f"   KG Context: {len(domain_ctx.get('kg_context_formatted', ''))} chars")
    else:
        print(f"❌ Error: {context.get('error', 'Unknown error')}")


if __name__ == "__main__":
    example_integration()

