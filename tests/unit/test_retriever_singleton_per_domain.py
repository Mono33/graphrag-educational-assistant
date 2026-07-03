"""#38 — ``get_retriever`` per-domain cache regression coverage.

Before #38, ``get_retriever`` held a *single* module-level slot that was
swapped whenever the requested domain changed. Once #31 allows several
pipelines in flight at once, two interleaved runs in different domains
(``neuro`` + ``udl``) would clobber each other's retriever mid-run → a
wrong-domain retrieval correctness bug.

These are pure unit tests: ``RetrieverAgent.__init__`` is cheap (no Neo4j /
no LLM at construction — the heavy ``GraphRAGTool`` is lazy-built), so they
run in milliseconds with no external dependencies. They lock in the
per-domain cache: distinct instances per domain, stable across calls, and
never clobbered by interleaving.
"""

from __future__ import annotations

import pytest

from aix.agent.agents.retriever_agent import RetrieverAgent
from aix.agent.graph import nodes

pytestmark = pytest.mark.unit


def test_get_retriever_caches_per_domain_and_is_stable():
    nodes._retrievers.clear()

    neuro = nodes.get_retriever("neuro")
    udl = nodes.get_retriever("udl")

    assert isinstance(neuro, RetrieverAgent) and neuro.domain == "neuro"
    assert isinstance(udl, RetrieverAgent) and udl.domain == "udl"
    assert neuro is not udl  # distinct instances — not one shared swap-slot

    # Same domain → same cached instance (no costly re-instantiation).
    assert nodes.get_retriever("neuro") is neuro
    assert nodes.get_retriever("udl") is udl


def test_interleaving_domains_does_not_clobber():
    """The core #38 regression: a ``udl`` call between two ``neuro`` calls must
    not swap out the ``neuro`` retriever (which the old single-slot did)."""
    nodes._retrievers.clear()

    neuro_first = nodes.get_retriever("neuro")
    _udl = nodes.get_retriever("udl")  # under the old code this clobbered neuro
    neuro_again = nodes.get_retriever("neuro")

    assert neuro_again is neuro_first
    assert neuro_again.domain == "neuro"
