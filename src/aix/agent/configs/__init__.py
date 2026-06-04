"""
Agent Configurations Module

Domain-specific extensions for the Agentic GraphRAG system.
These extensions are ONLY used in Agent Mode (not GraphRAG Mode).
"""

from .domain_prompts import (
    DOMAIN_EXTENSIONS,
    NEURO_CRITIC_EXTENSION,
    NEURO_WRITER_EXTENSION,
    get_domain_extension,
)

__all__ = [
    "get_domain_extension",
    "NEURO_WRITER_EXTENSION",
    "NEURO_CRITIC_EXTENSION",
    "DOMAIN_EXTENSIONS",
]
