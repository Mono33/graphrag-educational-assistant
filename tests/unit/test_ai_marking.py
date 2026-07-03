"""EU AI Act Article 50 marking helpers (#21) — unit coverage.

Pure-function tests for ``aix.core.ai_marking``: the machine-readable comment
is well-formed, prepending is idempotent (so re-runs / re-processing never
double-stamp), stripping is the inverse for plain-text export, and blank input
is left untouched.
"""

from __future__ import annotations

import pytest

from aix.core import ai_marking as m

pytestmark = pytest.mark.unit


def test_build_comment_contains_system_and_trace_id():
    comment = m.build_marking_comment("abc-123")
    assert comment.startswith("<!-- ai-generated: true")
    assert f"system: {m.AI_SYSTEM_NAME}" in comment
    assert "trace_id: abc-123" in comment
    assert comment.endswith("-->")


def test_build_comment_defaults_trace_id_when_missing():
    assert "trace_id: n/a" in m.build_marking_comment(None)


def test_ensure_marking_prepends_once_and_preserves_body():
    body = "# Lezione\n\nContenuto."
    marked = m.ensure_marking(body, trace_id="t1")

    assert m.is_marked(marked)
    assert marked.endswith(body)  # original content preserved verbatim
    assert "trace_id: t1" in marked


def test_ensure_marking_is_idempotent():
    body = "# Lezione\n\nContenuto."
    once = m.ensure_marking(body, trace_id="t1")
    twice = m.ensure_marking(once, trace_id="t1")

    assert once == twice  # second call is a no-op
    assert twice.count("<!-- ai-generated:") == 1


def test_ensure_marking_leaves_blank_untouched():
    assert m.ensure_marking("", trace_id="t1") == ""
    assert m.ensure_marking("   \n ", trace_id="t1") == "   \n "


def test_strip_marking_round_trips():
    body = "# Lezione\n\nContenuto."
    marked = m.ensure_marking(body, trace_id="t1")
    assert m.strip_marking(marked) == body


def test_strip_marking_noop_without_comment():
    body = "# Lezione senza marcatura"
    assert m.strip_marking(body) == body


def test_header_constants():
    assert m.AI_GENERATED_HEADER == "X-AI-Generated"
    assert m.AI_GENERATED_HEADER_VALUE == "true"
