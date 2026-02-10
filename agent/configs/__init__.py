"""
Agent Configurations Module

Domain-specific extensions for the Agentic GraphRAG system.
These extensions are ONLY used in Agent Mode (not GraphRAG Mode).
"""

from .domain_prompts import (
    get_domain_extension,
    NEURO_WRITER_EXTENSION,
    NEURO_CRITIC_EXTENSION,
    DOMAIN_EXTENSIONS
)

__all__ = [
    'get_domain_extension',
    'NEURO_WRITER_EXTENSION', 
    'NEURO_CRITIC_EXTENSION',
    'DOMAIN_EXTENSIONS'
]


