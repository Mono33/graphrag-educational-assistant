"""Claude 5 is told how much to think; nothing else about the call changes, and no other model is touched.

On 2026-10-07 ``AIX_THINKING_EFFORT=low`` turned out to reach only the Claude 4
family: Sonnet 5 thought as much as it liked, a fifth of the bill went on
reasoning tokens and lessons were cut off by ``max_tokens``. Pinned here: the
effort reaches Claude 5.x, temperature and the JSON mode stay as asked, the
Claude 4 branch is unchanged, and a Gemini call built from the same helper never
inherits the Writer's setting.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aix.core import config as config_module
from aix.core.config import config, reasoning_extra_body

pytestmark = pytest.mark.unit

SONNET_5 = "anthropic/claude-sonnet-5"
SONNET_55 = "anthropic/claude-sonnet-5.5"
SONNET_4 = "anthropic/claude-sonnet-4-6"
GEMINI = "google/gemini-3.1-flash-lite"


@pytest.fixture(autouse=True)
def effort_low(monkeypatch):
    monkeypatch.delenv("AIX_THINKING_EFFORT", raising=False)


@pytest.mark.parametrize("model", [SONNET_5, SONNET_55, "anthropic/claude-opus-5"])
def test_claude_5_gets_the_effort_and_keeps_temperature_and_json(model):
    kwargs = config.openai.build_completion_kwargs(
        temperature=0.3, max_tokens=4000, json_mode=True, model_override=model
    )

    assert kwargs["extra_body"] == {"reasoning": {"effort": "low"}}
    assert kwargs["temperature"] == 0.3
    assert kwargs["max_tokens"] == 4000, "no thinking buffer: low effort does not need one"
    assert kwargs["response_format"] == {"type": "json_object"}, "the planner keeps its JSON mode"


def test_the_effort_applies_even_when_the_thinking_is_not_returned():
    kwargs = config.openai.build_completion_kwargs(
        temperature=0.4, max_tokens=3000, include_reasoning=False, model_override=SONNET_5
    )

    assert kwargs["extra_body"] == {"reasoning": {"effort": "low"}}


@pytest.mark.parametrize("effort", ["medium", "high"])
def test_the_effort_is_the_one_configured(monkeypatch, effort):
    monkeypatch.setenv("AIX_THINKING_EFFORT", effort)

    assert reasoning_extra_body(SONNET_55) == {"reasoning": {"effort": effort}}


def test_default_sends_nothing_which_is_the_old_behavior(monkeypatch):
    monkeypatch.setenv("AIX_THINKING_EFFORT", "default")

    kwargs = config.openai.build_completion_kwargs(model_override=SONNET_5)

    assert "extra_body" not in kwargs


def test_claude_4_keeps_its_own_branch():
    kwargs = config.openai.build_completion_kwargs(
        temperature=0.3, max_tokens=4000, json_mode=True, model_override=SONNET_4
    )

    assert kwargs["temperature"] == 1
    assert kwargs["max_tokens"] == 6000
    assert "response_format" not in kwargs
    assert kwargs["extra_body"] == {"reasoning": {"effort": "low"}}


def test_claude_4_with_default_falls_back_to_low_not_to_an_invalid_effort(monkeypatch):
    monkeypatch.setenv("AIX_THINKING_EFFORT", "default")

    kwargs = config.openai.build_completion_kwargs(model_override=SONNET_4)

    assert kwargs["extra_body"] == {"reasoning": {"effort": "low"}}


@pytest.mark.parametrize("model", [GEMINI, "openai/gpt-4o", "anthropic/claude-haiku-4.5"])
def test_other_models_get_no_reasoning_setting(model):
    assert reasoning_extra_body(model) is None
    assert "extra_body" not in config.openai.build_completion_kwargs(model_override=model)


def test_the_translation_is_built_for_its_own_model(monkeypatch):
    """It used to be built for the Writer's model and then sent to Gemini."""
    from aix.retrieval import multilingual_text2cypher as m

    monkeypatch.setattr(config.openai, "model", SONNET_5)
    monkeypatch.setattr(config.openai, "api_key", "test")
    monkeypatch.setattr(config.text2cypher, "model", GEMINI)
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="working memory"))]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(config.openai, "get_client", lambda: client)

    translator = object.__new__(m.MultilingualText2Cypher)
    translator._translate_with_openai("memoria di lavoro", "neuro")

    assert sent["model"] == GEMINI
    assert "extra_body" not in sent, "Gemini must not receive Claude's reasoning setting"
    assert sent["temperature"] == 0


async def test_the_citation_judge_passes_the_effort(monkeypatch):
    from aix.agent.validation import citation_support

    monkeypatch.setenv("AIX_CITATION_CHECK_MODEL", SONNET_55)
    sent = {}

    async def create(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="supported"))]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    await citation_support.judge([{"claim": "c", "passage": "p"}], client)

    assert sent["extra_body"] == {"reasoning": {"effort": "low"}}
    assert sent["temperature"] == 0


def test_the_pattern_reads_versions_not_substrings():
    assert config_module.is_adaptive_claude("anthropic/claude-sonnet-5.5")
    assert config_module.is_adaptive_claude("anthropic/claude-opus-5")
    assert not config_module.is_adaptive_claude("anthropic/claude-sonnet-4-6")
    assert not config_module.is_adaptive_claude("anthropic/claude-haiku-4.5")
    assert not config_module.is_adaptive_claude(None)


@pytest.mark.parametrize("model", ["google/gemini-3.5-flash", "google/gemini-3.5-pro"])
def test_gemini_flash_and_pro_get_the_effort(model):
    """They think by default and cannot be switched off; low trims it (-18% on a real plan)."""
    assert reasoning_extra_body(model) == {"reasoning": {"effort": "low"}}


@pytest.mark.parametrize("model", ["google/gemini-3.5-flash-lite", "google/gemini-3.1-flash-lite"])
def test_gemini_lite_is_left_alone(model):
    """The Critic and the translation run on these: they do not think, nothing is sent."""
    assert reasoning_extra_body(model) is None


@pytest.mark.parametrize(
    "model",
    ["mistralai/mistral-large-4-0", "mistralai/mistral-large-5", "deepseek/deepseek-v4.1-flash"],
)
def test_mistral_large_4_and_deepseek_v4_are_told_not_to_reason(model):
    """By default it spent its whole budget reasoning and answered nothing (2026-10-13)."""
    assert reasoning_extra_body(model) == {"reasoning": {"enabled": False}}
    kwargs = config.openai.build_completion_kwargs(model_override=model)
    assert kwargs["extra_body"] == {"reasoning": {"enabled": False}}


def test_older_mistral_is_left_alone():
    assert reasoning_extra_body("mistralai/mistral-large-2512") is None


def test_qwen_3_gets_the_effort():
    assert reasoning_extra_body("qwen/qwen3.8-27b") == {"reasoning": {"effort": "low"}}
