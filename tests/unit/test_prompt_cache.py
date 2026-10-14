"""The prompt is the same text; only the stable head is marked for the provider's cache."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from aix.core import prompt_cache

pytestmark = pytest.mark.unit

SONNET = "anthropic/claude-sonnet-5"
GEMINI = "google/gemini-3.1-flash-lite"


@pytest.fixture(autouse=True)
def cache_on(monkeypatch):
    # The suite runs with the flag off (tests/conftest.py); these tests are about it on.
    monkeypatch.setenv("AIX_PROMPT_CACHE_ENABLED", "true")


def _plain():
    return [
        {"role": "system", "content": "regole stabili"},
        {"role": "user", "content": "PREFISSO stabile. domanda del turno"},
    ]


def test_the_system_message_becomes_one_marked_block():
    out = prompt_cache.cacheable(_plain(), model=SONNET)

    assert out[0]["content"] == [
        {"type": "text", "text": "regole stabili", "cache_control": {"type": "ephemeral"}}
    ]
    assert out[1] == _plain()[1]


def test_the_input_is_never_changed_in_place():
    messages = _plain()
    prompt_cache.cacheable(messages, model=SONNET, stable_user_prefix="PREFISSO stabile.")

    assert messages == _plain()


def test_the_user_message_is_split_after_the_stable_prefix():
    out = prompt_cache.cacheable(_plain(), model=SONNET, stable_user_prefix="PREFISSO stabile.")

    head, tail = out[1]["content"]
    assert head["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in tail
    assert head["text"] + tail["text"] == _plain()[1]["content"]


def test_a_prefix_the_message_does_not_start_with_is_ignored():
    out = prompt_cache.cacheable(_plain(), model=SONNET, stable_user_prefix="altro")

    assert out[1] == _plain()[1]


def test_the_one_hour_option_is_carried_on_the_mark():
    out = prompt_cache.cacheable(_plain(), model=SONNET, ttl="1h")

    assert out[0]["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


@pytest.mark.parametrize("model", ["openai/gpt-4o", "deepseek/deepseek-chat", "", None])
def test_a_model_that_does_not_read_the_mark_gets_plain_strings(model):
    assert prompt_cache.cacheable(_plain(), model=model) == _plain()


def test_off_means_plain_strings(monkeypatch):
    monkeypatch.setenv("AIX_PROMPT_CACHE_ENABLED", "false")

    assert prompt_cache.cacheable(_plain(), model=SONNET) == _plain()


def test_a_message_that_is_already_blocks_is_left_alone():
    blocks = [{"role": "system", "content": [{"type": "text", "text": "x"}]}]

    assert prompt_cache.cacheable(blocks, model=SONNET) == blocks


def test_langchain_system_message_is_marked_and_the_human_one_is_not():
    out = prompt_cache.cacheable_lc(
        [SystemMessage(content="schema"), HumanMessage(content="Question: ADHD")],
        model=GEMINI,
    )

    assert out[0].content == [
        {"type": "text", "text": "schema", "cache_control": {"type": "ephemeral"}}
    ]
    assert out[1].content == "Question: ADHD"


def test_langchain_messages_stay_plain_for_other_providers():
    messages = [SystemMessage(content="schema"), HumanMessage(content="q")]

    assert prompt_cache.cacheable_lc(messages, model="openai/gpt-4o") is messages
