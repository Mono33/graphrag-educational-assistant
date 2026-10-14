"""Langfuse tracing: off by default, per teacher and lesson when on, never fatal."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from aix.core import tracing

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    tracing.get_langfuse.cache_clear()
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    yield
    getattr(tracing.get_langfuse, "cache_clear", lambda: None)()  # may still be patched


def test_on_by_default_when_langfuse_is_configured(monkeypatch):
    monkeypatch.delenv("AIX_TRACING_ENABLED", raising=False)

    assert tracing.tracing_enabled()


def test_without_langfuse_keys_it_stays_off(monkeypatch):
    monkeypatch.delenv("AIX_TRACING_ENABLED", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)

    assert not tracing.tracing_enabled()


def test_off_when_switched_off(monkeypatch):
    monkeypatch.setenv("AIX_TRACING_ENABLED", "false")
    import openai

    assert not tracing.tracing_enabled()
    assert tracing.openai_classes() == (openai.OpenAI, openai.AsyncOpenAI)
    assert tracing.graph_callbacks() == []
    run = tracing.start_run("lesson.turn", user_id="u")
    assert run.trace_id is None
    run.end(output="x")  # a no-op, never an error


def test_without_keys_it_stays_off(monkeypatch):
    monkeypatch.setenv("AIX_TRACING_ENABLED", "true")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY")

    assert not tracing.tracing_enabled()


def test_on_the_model_clients_are_langfuse_s_traced_drop_ins(monkeypatch):
    monkeypatch.setenv("AIX_TRACING_ENABLED", "true")
    import langfuse.openai

    assert tracing.openai_classes() == (langfuse.openai.OpenAI, langfuse.openai.AsyncOpenAI)


def test_the_turn_is_traced_under_the_teacher_and_the_lesson(monkeypatch):
    monkeypatch.setenv("AIX_TRACING_ENABLED", "true")
    seen: dict = {}

    class _Span:
        trace_id = "trace-1"

        def update(self, **kwargs):
            seen["update"] = kwargs

    class _Client:
        @contextmanager
        def start_as_current_observation(self, **kwargs):
            seen["observation"] = kwargs
            yield _Span()
            seen["closed"] = True

    @contextmanager
    def _propagate(**kwargs):
        seen["attributes"] = kwargs
        yield

    monkeypatch.setattr(tracing, "get_langfuse", lambda: _Client())
    monkeypatch.setattr("langfuse.propagate_attributes", _propagate)

    run = tracing.start_run(
        "lesson.turn",
        user_id="anna-uuid",
        session_id="lesson-uuid",
        tags=("webui", "neuro"),
        metadata={"turn": 2, "intent": None},
        input="crea una lezione",
    )
    assert run.trace_id == "trace-1"
    run.end(output="# Lezione", metadata={"approved": True})

    assert seen["attributes"]["user_id"] == "anna-uuid"
    assert seen["attributes"]["session_id"] == "lesson-uuid"
    assert seen["attributes"]["tags"] == ["webui", "neuro"]
    assert seen["attributes"]["metadata"] == {"turn": "2"}, "values are strings, None dropped"
    assert seen["observation"]["input"] == "crea una lezione"
    assert seen["update"]["output"] == "# Lezione"
    assert seen["closed"] is True


def test_a_broken_langfuse_never_breaks_a_run(monkeypatch):
    monkeypatch.setenv("AIX_TRACING_ENABLED", "true")

    def _boom():
        raise RuntimeError("langfuse down")

    monkeypatch.setattr(tracing, "get_langfuse", _boom)

    run = tracing.start_run("lesson.turn", user_id="u")
    run.end(output="x")
    assert tracing.graph_callbacks() == []


def test_content_can_be_kept_out_of_the_traces(monkeypatch):
    monkeypatch.setenv("AIX_TRACING_CONTENT", "none")

    assert not tracing._content_enabled()
    assert tracing._drop_content(data="il testo della docente") != "il testo della docente"
    assert tracing._drop_content(data=None) is None


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("rehearsal", "rehearsal"),
        ("Production", "production"),
        ("pilot FEM", "pilot-fem"),
        ("langfuse", "env-langfuse"),
        (None, "development"),
    ],
)
def test_the_environment_name_is_one_langfuse_accepts(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("ENVIRONMENT", raising=False)
    else:
        monkeypatch.setenv("ENVIRONMENT", raw)

    assert tracing._environment() == expected
