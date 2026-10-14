"""The same pair is not paid twice, and the judge's instructions are read from the provider's cache.

What the verdicts are is pinned in ``test_citation_support.py``; here only what
the two caches add. The prompt text is the one the golden cases measured, so the
split in blocks must give it back unchanged.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aix.agent.media.media_cache import MediaCache
from aix.agent.validation import citation_support
from aix.agent.validation.citation_support import drop_unsupported_citations
from aix.core import result_cache

pytestmark = pytest.mark.unit

YALTA = "[[material:a#page=1]]"
ITEMS = [{"citation": YALTA, "text": "A Yalta l'Europa fu divisa in sfere d'influenza."}]
DRAFT = f"La conferenza di Yalta divise l'Europa {YALTA}."
SONNET = "anthropic/claude-sonnet-5"


def _answer(text: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _client(create) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


class _Memory:
    backend_name = "memory"

    def __init__(self):
        self.data = {}

    async def get(self, key):
        return self.data.get(key)

    async def set(self, key, value, ttl):
        self.data[key] = value


@pytest.fixture
def remembered(monkeypatch):
    monkeypatch.setenv("AIX_RESULT_CACHE_ENABLED", "true")
    cache = MediaCache(_Memory(), "test", 100)
    monkeypatch.setattr(result_cache, "_cache_for", lambda namespace: cache)
    return cache


@pytest.fixture
def marked(monkeypatch):
    monkeypatch.setenv("AIX_PROMPT_CACHE_ENABLED", "true")
    monkeypatch.setenv("AIX_CITATION_CHECK_MODEL", SONNET)


async def test_the_same_sentence_and_passage_are_judged_once(remembered):
    create = AsyncMock(side_effect=[_answer("unsupported")])
    client = _client(create)

    first = await drop_unsupported_citations(DRAFT, ITEMS, client=client)
    second = await drop_unsupported_citations(DRAFT, ITEMS, client=client)

    assert YALTA not in first and YALTA not in second
    assert create.await_count == 1


async def test_a_failed_call_is_not_remembered_as_supported(remembered):
    broken = _client(AsyncMock(side_effect=RuntimeError("network")))
    kept = await drop_unsupported_citations(DRAFT, ITEMS, client=broken)
    assert YALTA in kept, "a failed check leaves the lesson as it was"

    out = await drop_unsupported_citations(
        DRAFT, ITEMS, client=_client(AsyncMock(side_effect=[_answer("unsupported")]))
    )

    assert YALTA not in out, "the hiccup of the first call must not outlive it"


async def test_from_three_citations_the_first_goes_alone_to_write_the_cache(marked):
    log = []

    async def create(**kwargs):
        log.append("start")
        await asyncio.sleep(0)
        log.append("end")
        return _answer("supported")

    pairs = [{"claim": f"c{i}", "passage": f"p{i}"} for i in range(4)]

    assert await citation_support.judge(pairs, _client(create)) == ["supported"] * 4
    assert log[:2] == ["start", "end"], "the first call finishes before any other starts"
    assert log.count("start") == 4


async def test_the_instructions_are_one_marked_block_and_the_pair_is_not(marked):
    seen = {}

    async def create(**kwargs):
        seen.update(kwargs)
        return _answer("supported")

    pairs = [{"claim": f"c{i}", "passage": f"p{i}"} for i in range(3)]
    await citation_support.judge(pairs, _client(create))

    head, tail = seen["messages"][0]["content"]
    assert head["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in tail
    assert head["text"] + tail["text"] == citation_support._PROMPT.format(claim="c2", passage="p2")


async def test_the_judge_asks_for_one_provider_first_and_keeps_its_reasoning(marked, monkeypatch):
    monkeypatch.delenv("AIX_CITATION_CHECK_PROVIDER", raising=False)
    seen = {}

    async def create(**kwargs):
        seen.update(kwargs)
        return _answer("supported")

    await citation_support.judge([{"claim": "c", "passage": "p"}], _client(create))

    assert seen["extra_body"]["provider"] == {
        "order": ["google-vertex/global"],
        "allow_fallbacks": True,
    }
    assert seen["extra_body"]["reasoning"] == {"effort": "low"}


@pytest.mark.parametrize("value", ["", "none"])
async def test_the_provider_can_be_left_to_openrouter(marked, monkeypatch, value):
    monkeypatch.setenv("AIX_CITATION_CHECK_PROVIDER", value)
    seen = {}

    async def create(**kwargs):
        seen.update(kwargs)
        return _answer("supported")

    await citation_support.judge([{"claim": "c", "passage": "p"}], _client(create))

    assert "provider" not in seen.get("extra_body", {})


async def test_fewer_than_three_citations_are_sent_as_plain_text(marked):
    seen = {}

    async def create(**kwargs):
        seen.update(kwargs)
        return _answer("supported")

    await citation_support.judge([{"claim": "c", "passage": "p"}], _client(create))

    assert isinstance(seen["messages"][0]["content"], str)


async def test_a_model_that_does_not_read_the_mark_is_sent_plain_text(monkeypatch):
    monkeypatch.setenv("AIX_PROMPT_CACHE_ENABLED", "true")
    monkeypatch.setenv("AIX_CITATION_CHECK_MODEL", "openai/gpt-4o")
    seen = {}

    async def create(**kwargs):
        seen.update(kwargs)
        return _answer("supported")

    pairs = [{"claim": f"c{i}", "passage": f"p{i}"} for i in range(4)]
    await citation_support.judge(pairs, _client(create))

    assert isinstance(seen["messages"][0]["content"], str)


async def test_the_same_sentence_and_passage_twice_is_asked_once():
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return _answer("unsupported")

    pairs = [
        {"claim": "Yalta divise l'Europa", "passage": "p"},
        {"claim": "altra frase", "passage": "p"},
        {"claim": "Yalta divise l'Europa", "passage": "p"},
    ]

    verdicts = await citation_support.judge(pairs, _client(create))

    assert len(calls) == 2, "the repeated pair is not paid twice"
    assert verdicts == ["unsupported", "unsupported", "unsupported"], (
        "every occurrence gets its answer"
    )


async def test_a_repeated_unsupported_citation_is_removed_everywhere():
    draft = f"La conferenza di Yalta divise l'Europa {YALTA} {YALTA}."
    create = AsyncMock(side_effect=[_answer("unsupported")])

    out = await drop_unsupported_citations(draft, ITEMS, client=_client(create))

    assert YALTA not in out
    assert create.await_count == 1
