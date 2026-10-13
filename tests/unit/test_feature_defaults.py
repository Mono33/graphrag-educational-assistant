"""The defaults are the pilot's (decided by Angelo, 2026-10-05).

Until then every new surface was off unless an ``.env`` switched it on, so a
deploy with a forgotten line silently served the previous product. Now a
feature that is ready is on, and ``false`` in the ``.env`` is how it is turned
off. Every place a flag is read must agree: some are read in two modules.
"""

from __future__ import annotations

import pytest

from aix.agent.agents import planner_agent
from aix.agent.graph import nodes
from aix.agent.media.media_config import MediaConfig
from aix.agent.validation import validator
from aix.core import prompt_cache, result_cache, tracing
from aix.webui import features
from aix.webui.materials import embedding, storage
from aix.webui.materials.parsing import vision_ocr

pytestmark = pytest.mark.unit

ON_BY_DEFAULT = {
    "AIX_UPLOAD_PARSER_ENABLED": [features.upload_parser_enabled],
    "AIX_MATERIALS_ENABLED": [storage.materials_enabled],
    "AIX_PRIVATE_RAG_ENABLED": [embedding.private_rag_enabled, nodes._private_rag_enabled],
    "AIX_CONDITIONAL_RETRIEVAL_ENABLED": [
        features.conditional_retrieval_enabled,
        nodes._conditional_retrieval_enabled,
    ],
    "AIX_LIGHT_TURNS_ENABLED": [features.light_turns_enabled],
    "AIX_MEDIA_LIVE_ENABLED": [lambda: MediaConfig.from_env().live_enabled],
    "AIX_MEDIA_RERANK_ENABLED": [lambda: MediaConfig.from_env().rerank_enabled],
    "AIX_PROMPT_CACHE_ENABLED": [prompt_cache.enabled],
    "AIX_RESULT_CACHE_ENABLED": [result_cache.enabled],
    "AIX_PLANNER_ESCALATE": [planner_agent.escalate_enabled],
}


@pytest.mark.parametrize("flag", sorted(ON_BY_DEFAULT))
def test_on_when_the_env_says_nothing_and_off_when_it_says_false(flag, monkeypatch):
    monkeypatch.delenv(flag, raising=False)
    assert all(read() for read in ON_BY_DEFAULT[flag]), f"{flag} unset should be on"

    monkeypatch.setenv(flag, "false")
    assert not any(read() for read in ON_BY_DEFAULT[flag]), f"{flag}=false should be off"


def test_the_vision_model_reads_the_scans_by_default(monkeypatch):
    monkeypatch.delenv("AIX_OCR_BACKEND", raising=False)
    assert vision_ocr.backend() == "vision"


def test_tracing_is_on_by_default_but_needs_its_keys(monkeypatch):
    monkeypatch.delenv("AIX_TRACING_ENABLED", raising=False)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    assert tracing.tracing_enabled()

    monkeypatch.delenv("LANGFUSE_SECRET_KEY")
    assert not tracing.tracing_enabled()


@pytest.mark.parametrize(
    "flag",
    [
        "AIX_VALIDATOR_ENABLED",
        "AIX_CORRECTIVE_RAG_ENABLED",
        "AIX_LIGHT_TOOLS_ENABLED",
        "AIX_WRITER_GRAPH_RELATIONS",
    ],
)
def test_what_the_pilot_keeps_off_stays_off(flag, monkeypatch):
    from aix.agent.agents import writer_agent
    from aix.webui.agent import light_tools

    monkeypatch.delenv(flag, raising=False)
    readers = {
        "AIX_VALIDATOR_ENABLED": validator.validator_enabled,
        "AIX_CORRECTIVE_RAG_ENABLED": nodes._corrective_rag_enabled,
        # Until measured on real lessons (2026-10-13).
        "AIX_LIGHT_TOOLS_ENABLED": light_tools.enabled,
        "AIX_WRITER_GRAPH_RELATIONS": writer_agent.graph_relations_enabled,
    }
    assert not readers[flag]()
