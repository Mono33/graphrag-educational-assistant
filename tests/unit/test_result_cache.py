"""A deterministic answer is paid once: the key says what it depends on, and a failure is not an answer."""

from __future__ import annotations

import pytest

from aix.agent.media.media_cache import MediaCache
from aix.core import result_cache

pytestmark = pytest.mark.unit


class MemoryBackend:
    """What diskcache or Redis would be, in a dict."""

    backend_name = "memory"

    def __init__(self) -> None:
        self.data: dict = {}
        self.ttls: dict = {}

    async def get(self, key):
        return self.data.get(key)

    async def set(self, key, value, ttl):
        self.data[key] = value
        self.ttls[key] = ttl

    async def close(self):
        return None


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setenv("AIX_RESULT_CACHE_ENABLED", "true")
    return MediaCache(MemoryBackend(), "test", 100)


def _counter():
    calls = []

    async def compute():
        calls.append(1)
        return {"n": len(calls)}

    return calls, compute


async def test_the_second_ask_is_answered_without_computing(store):
    calls, compute = _counter()

    first = await result_cache.remember("t", ("adhd", "udl"), compute, cache=store)
    second = await result_cache.remember("t", ("adhd", "udl"), compute, cache=store)

    assert first == second == {"n": 1}
    assert len(calls) == 1


@pytest.mark.parametrize("other", [("adhd", "neuro"), ("autismo", "udl"), ("adhd", "udl", "v2")])
async def test_anything_the_answer_depends_on_is_part_of_the_key(store, other):
    calls, compute = _counter()

    await result_cache.remember("t", ("adhd", "udl"), compute, cache=store)
    await result_cache.remember("t", other, compute, cache=store)

    assert len(calls) == 2


async def test_what_keep_refuses_is_not_stored(store):
    calls, compute = _counter()

    await result_cache.remember("t", ("k",), compute, keep=lambda value: False, cache=store)
    await result_cache.remember("t", ("k",), compute, keep=lambda value: False, cache=store)

    assert len(calls) == 2


async def test_a_value_that_is_not_json_is_returned_but_not_stored(store):
    async def compute():
        return {"node": object()}

    value = await result_cache.remember("t", ("k",), compute, cache=store)

    assert "node" in value
    assert store._backend.data == {}


async def test_an_empty_answer_can_live_shorter(store):
    async def compute():
        return {"results": []}

    await result_cache.remember(
        "t", ("k",), compute, ttl=lambda v: 60 if not v["results"] else 600, cache=store
    )

    assert list(store._backend.ttls.values()) == [60]


async def test_a_cache_that_breaks_is_a_miss_not_an_error(store, monkeypatch):
    async def boom(key):
        raise RuntimeError("redis is down")

    monkeypatch.setattr(store._backend, "get", boom)
    calls, compute = _counter()

    assert await result_cache.remember("t", ("k",), compute, cache=store) == {"n": 1}


async def test_off_computes_every_time(store, monkeypatch):
    monkeypatch.setenv("AIX_RESULT_CACHE_ENABLED", "false")
    calls, compute = _counter()

    await result_cache.remember("t", ("k",), compute, cache=store)
    await result_cache.remember("t", ("k",), compute, cache=store)

    assert len(calls) == 2


def test_case_and_spacing_do_not_change_a_question():
    assert result_cache.normalize("  ADHD \n a scuola ") == result_cache.normalize("adhd a scuola")
