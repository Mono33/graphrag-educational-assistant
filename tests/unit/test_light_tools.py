"""A light turn looks the documents up only when the model asks for them."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from aix.webui.agent import light_tools

pytestmark = pytest.mark.unit


def _chunk(content=None, tool=None):
    calls = None
    if tool:
        index, call_id, name, arguments = tool
        calls = [
            SimpleNamespace(
                index=index, id=call_id, function=SimpleNamespace(name=name, arguments=arguments)
            )
        ]
    delta = SimpleNamespace(content=content, tool_calls=calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class _Stream:
    def __init__(self, chunks):
        self._chunks = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None


class _Client:
    """Answers each call with the next scripted stream, and records what it was sent."""

    def __init__(self, *rounds):
        self.rounds = list(rounds)
        self.sent = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.sent.append(kwargs)
        return _Stream(self.rounds.pop(0))


class _Box(light_tools.Toolbox):
    def __init__(self):
        super().__init__(lesson=SimpleNamespace(domain="udl"))

    async def _documents(self, query):
        self.items.append({"citation": "[[material:a#page=1]]", "text": "passaggio"})
        return f"PASSAGGI per «{query}»"


async def _collect(client, box):
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    return [d async for d in light_tools.stream_with_tools(client, messages, {"model": "m"}, box)]


async def test_an_answer_that_needs_nothing_is_one_call_and_streams_as_is():
    client = _Client([_chunk("La prima "), _chunk("attività serve a...")])

    text = await _collect(client, _Box())

    assert "".join(text) == "La prima attività serve a..."
    assert len(client.sent) == 1
    assert client.sent[0]["tools"] == light_tools.SCHEMAS


async def test_a_document_question_looks_them_up_then_answers():
    args = json.dumps({"richiesta": "che cosa dice il documento sulla Stasi"})
    client = _Client(
        [_chunk(tool=(0, "call_1", "cerca_nei_", "")), _chunk(tool=(0, None, "documenti", args))],
        [_chunk("Il documento dice... [[material:a#page=1]]")],
    )
    box = _Box()

    text = await _collect(client, box)

    assert "".join(text).startswith("Il documento dice")
    assert box.calls == ["cerca_nei_documenti"]
    assert box.items, "the passages found go to the citation check"
    second = client.sent[1]["messages"]
    assert second[-2]["tool_calls"][0]["function"]["name"] == "cerca_nei_documenti"
    assert second[-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "PASSAGGI per «che cosa dice il documento sulla Stasi»",
    }


async def test_the_last_round_offers_no_tools_so_an_answer_always_comes():
    looking = [_chunk(tool=(0, "c", "cerca_nei_documenti", json.dumps({"richiesta": "x"})))]
    client = _Client(looking, looking, [_chunk("Risposta.")])

    text = await _collect(client, _Box())

    assert "".join(text) == "Risposta."
    assert len(client.sent) == light_tools.MAX_ROUNDS + 1
    assert "tools" not in client.sent[-1]


async def test_a_broken_look_up_is_told_to_the_model_not_raised():
    class Broken(_Box):
        async def _documents(self, query):
            raise RuntimeError("database down")

    result = await Broken().run("cerca_nei_documenti", json.dumps({"richiesta": "x"}))

    assert "non è riuscita" in result
