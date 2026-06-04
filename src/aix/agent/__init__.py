"""
Agent Layer for Agentic GraphRAG

This module provides a multi-agent pipeline for educational content generation.
It uses the existing GraphRAG engine as a tool while adding planning, writing,
and quality review capabilities.

Architecture:
    Teacher Query → Planner → Retriever → Writer → Critic → Lesson Plan

Usage:
    from aix.agent import AgentOrchestrator

    orchestrator = AgentOrchestrator(domain="neuro")
    result = await orchestrator.create_lesson_plan(
        "Crea una lezione sulla motivazione per studenti con ADHD"
    )

The existing GraphRAG engine (graph_retriever.py, context_builder.py, etc.)
remains completely unchanged. This agent layer imports and uses it as a tool.
"""

from aix.agent.orchestrator import AgentOrchestrator

__all__ = ["AgentOrchestrator"]
__version__ = "0.1.0"
