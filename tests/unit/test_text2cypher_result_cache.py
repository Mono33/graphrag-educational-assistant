"""The graph is asked a repeated question once: what the answer depends on is in the key.

The lesson asks short keywords that come back turn after turn. A repeat must skip
the translation, the Cypher generation and the Neo4j query, but never serve an
error, and never survive a change of domain, of model or of graph version.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aix.agent.media.media_cache import MediaCache
from aix.core import result_cache
from aix.retrieval import graph_retriever
from aix.retrieval.graph_retriever import EnhancedMultilingualText2Cypher

pytestmark = pytest.mark.unit


class _Memory:
    backend_name = "memory"

    def __init__(self):
        self.data = {}
        self.ttls = []

    async def get(self, key):
        return self.data.get(key)

    async def set(self, key, value, ttl):
        self.data[key] = value
        self.ttls.append(ttl)


class _Stop(Exception):
    """Raised by the fake retriever: the test is about what happens before it."""


def _answer(**over):
    base = {
        "results": [{"name": "ADHD"}],
        "metadata": {"is_valid": True},
        "execution_error": None,
        "enhanced_query": None,
    }
    base.update(over)
    return base


def _processor(answers):
    """The wrapper with a fake text2cypher that returns ``answers`` in order."""
    calls = []

    def process_query(query, domain=None, execute=True):
        calls.append((query, domain))
        return answers[min(len(calls), len(answers)) - 1]

    async def retrieve(*args, **kwargs):
        raise _Stop

    wrapper = object.__new__(EnhancedMultilingualText2Cypher)
    wrapper.domain = "udl"
    wrapper.text2cypher = SimpleNamespace(process_query=process_query)
    wrapper.graph_retriever = SimpleNamespace(retrieve=retrieve)
    return wrapper, calls


async def _ask(wrapper, query, domain=None):
    with pytest.raises(_Stop):
        await wrapper.process_query_with_retrieval(query, domain=domain)


@pytest.fixture
def memory(monkeypatch):
    monkeypatch.setenv("AIX_RESULT_CACHE_ENABLED", "true")
    monkeypatch.delenv("AIX_KG_VERSION", raising=False)
    monkeypatch.setenv("TEXT2CYPHER_MODEL", "google/gemini-3.1-flash-lite")
    cache = MediaCache(_Memory(), "test", 100)
    monkeypatch.setattr(result_cache, "_cache_for", lambda namespace: cache)
    return cache


async def test_a_repeated_question_asks_the_graph_once(memory):
    wrapper, calls = _processor([_answer()])

    await _ask(wrapper, "ADHD")
    await _ask(wrapper, "  adhd ")

    assert len(calls) == 1


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.setenv("AIX_KG_VERSION", "2026-10-07"),
        lambda m: m.setenv("TEXT2CYPHER_MODEL", "google/gemini-3.5-flash-lite"),
    ],
    ids=["a new graph version", "another model"],
)
async def test_a_change_of_graph_or_model_is_not_answered_from_the_old(memory, monkeypatch, change):
    wrapper, calls = _processor([_answer()])

    await _ask(wrapper, "ADHD")
    change(monkeypatch)
    await _ask(wrapper, "ADHD")

    assert len(calls) == 2


async def test_another_domain_is_another_question(memory):
    wrapper, calls = _processor([_answer()])

    await _ask(wrapper, "memoria", domain="udl")
    await _ask(wrapper, "memoria", domain="neuro")

    assert len(calls) == 2


@pytest.mark.parametrize(
    "bad",
    [
        _answer(metadata={"is_valid": False}),
        _answer(metadata={"is_valid": True, "error": "boom"}),
        _answer(execution_error="syntax error"),
    ],
    ids=["invalid query", "error in metadata", "execution error"],
)
async def test_a_failure_is_asked_again(memory, bad):
    wrapper, calls = _processor([bad])

    await _ask(wrapper, "ADHD")
    await _ask(wrapper, "ADHD")

    assert len(calls) == 2


async def test_an_empty_answer_is_kept_for_less_time_than_a_full_one(memory):
    wrapper, _ = _processor([_answer(results=[])])
    await _ask(wrapper, "ADHD")
    wrapper, _ = _processor([_answer()])
    await _ask(wrapper, "dislessia")

    empty, full = memory._backend.ttls
    assert empty < full


def test_the_helpers_agree_on_what_is_worth_keeping():
    assert graph_retriever._worth_keeping(_answer())
    assert not graph_retriever._worth_keeping({"metadata": {}, "results": []})
