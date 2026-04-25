"""graphaixlearning — top-level `aix` package.

Phase 3C consolidates the codebase under a single importable namespace:

    aix.core         — shared config + utilities
    aix.retrieval    — GraphRAG retrieval layer (graph_retriever, context_builder,
                       text2cypher, multilingual_text2cypher, query_metrics)
    aix.generation   — LLM response generation (llm_chain)
    aix.agent        — Agentic GraphRAG (multi-agent pipeline)
    aix.api          — FastAPI service
    aix.domains      — UDL + Neuro domain configurations

After `pip install -e .[dev]`, the package is importable from any working
directory:

    >>> from aix.core.config import config
    >>> from aix.agent import AgentOrchestrator
"""

__version__ = "0.2.0"
