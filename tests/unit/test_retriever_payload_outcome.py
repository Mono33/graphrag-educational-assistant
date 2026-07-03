"""
CORE 2 #9.UX-2 + #9.UX-3 — retriever payload outcome regression coverage
=========================================================================

Locks in the four mutually exclusive retrieval-outcome tokens computed
by ``aix.webui.agent.service._build_retriever_payload`` (via
``_compute_retrieval_outcome``), plus the ``retrieval_attempts_max``
field that drives the ``Tentativi: N/M`` badge in the chat card.

The four outcomes drive the chat card's color, headline, and copy:

    | Outcome              | Color  | Trigger                                       |
    |----------------------|--------|-----------------------------------------------|
    | "success"            | green  | grade=relevant (or grading didn't run)        |
    | "adapted_with_hybrid"| blue   | grade=ambiguous|irrelevant + hybrid kicked in |
    | "limited_kg_only"    | amber  | grade=ambiguous|irrelevant + no hybrid        |
    | "grader_error"       | red    | grader LLM threw, reason has sentinel prefix  |

Plus a few edge-case tests:
  * Tentativi badge data shape (attempts vs attempts_max).
  * Backward-compat: when the corrective-RAG flag is OFF (grade is None),
    outcome defaults to "success" but the template will skip the row
    entirely thanks to its outer ``{% if p.retrieval_grade %}`` guard.
  * ``_grader_will_retry`` semantics — paired with #9.UX-2's gate.

These are pure unit tests — no LLM, no DB, no network. They run in
milliseconds and are stable in CI.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from aix.webui.agent.service import (
    _build_retriever_payload,
    _classify_coverage_tier,
    _compute_retrieval_outcome,
    _grader_will_retry,
    _resolve_domain_labels,
    _resolve_max_attempts,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(**overrides: Any) -> dict[str, Any]:
    """Build a minimal post-retrieve state dict and apply overrides.

    All fields are optional — :class:`AgentState` is a ``TypedDict`` with
    ``total=False`` so the loose dict shape is the runtime contract.
    """
    base: dict[str, Any] = {
        "retrieved_nodes": [{"title": "Scaffolding"}, {"title": "Working Memory"}],
        "retrieved_relationships": [{"src": "a", "dst": "b"}],
        "recommendations": [{"strategy": "chunking"}],
        "curated_media": {},
        "external_resources": None,
        "retrieval_confidence": "medium",
        "retrieval_grade": None,
        "retrieval_grade_reason": None,
        "retrieval_attempts": None,
        "retrieval_rewritten_query": None,
        "retrieval_warning": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# The four canonical outcomes — one test per outcome
# ---------------------------------------------------------------------------

def test_outcome_success_when_grade_relevant() -> None:
    """grade=relevant → outcome="success" (green ✅).

    This is the happy path for in-scope queries (e.g. ADHD, UDL,
    scaffolding). Behaviour unchanged from pre-#9.UX-3 — same green
    rendering, just expressed via the outcome token now.
    """
    state = _state(
        retrieval_grade="relevant",
        retrieval_grade_reason=(
            "The retrieved concepts directly cover the core pedagogical and "
            "neurological dimensions of the query: ADHD characteristics, "
            "inclusive teaching strategies, and self-regulation approaches."
        ),
        retrieval_attempts=1,
    )

    payload = _build_retriever_payload(state)

    assert payload["retrieval_outcome"] == "success"
    assert payload["retrieval_grade"] == "relevant"
    assert payload["retrieval_grade_label"] == "Rilevante"
    assert payload["retrieval_grade_emoji"] == "✅"
    assert payload["retrieval_attempts"] == 1
    # The badge is driven by retrieval_attempts_max — must always be set.
    assert isinstance(payload["retrieval_attempts_max"], int)
    assert payload["retrieval_attempts_max"] >= 1


def test_outcome_adapted_with_hybrid_when_irrelevant_with_external_resources() -> None:
    """grade=irrelevant + external_resources populated → outcome="adapted_with_hybrid".

    This is the "fotosintesi clorofiliana" scenario: the KG is
    out-of-scope for the disciplinary content, but the hybrid
    retrieval path landed Wikipedia / OER / S2 papers that filled the
    gap. The card should show BLUE ℹ️ "Adattamento riuscito" — NOT
    red — because no error occurred, the system gracefully adapted.
    """
    state = _state(
        retrieval_grade="irrelevant",
        retrieval_grade_reason=(
            "The retrieved concepts focus on accessibility features and "
            "ADHD-related challenges rather than the photosynthesis "
            "scientific content requested."
        ),
        retrieval_attempts=2,
        retrieval_warning=True,
        retrieval_rewritten_query=(
            "fotosintesi clorofiliana UDL scaffolding DSA ADHD scuola "
            "secondaria strategie inclusive"
        ),
        # Hybrid path landed Wikipedia + papers + OER:
        external_resources={
            "wikipedia": [{"title": "Fotosintesi"}, {"title": "Clorofilla"}],
            "semantic_scholar": [{"title": "Photosynthesis pedagogy paper"}],
            "oer": [{"title": "OpenStax Biology - Photosynthesis"}],
        },
        curated_media={
            "videos": [{"title": "Crash course photosynthesis"}],
            "citations": [{"title": "Paper 1"}, {"title": "Paper 2"}],
            "resources": [{"title": "OpenStax chapter"}],
            "open_textbooks": [],
        },
    )

    payload = _build_retriever_payload(state)

    assert payload["retrieval_outcome"] == "adapted_with_hybrid"
    assert payload["retrieval_grade"] == "irrelevant"
    assert payload["retrieval_attempts"] == 2
    # The rewrite must surface so the bonus rewrite-UX line renders.
    assert payload["retrieval_rewritten_query"]
    # Hybrid tally is what the blue copy cites in its bullet list.
    assert payload["media_counts"]["articles"] == 2
    assert payload["media_counts"]["oer"] == 1


def test_outcome_limited_kg_only_when_irrelevant_without_external_resources() -> None:
    """grade=irrelevant + no external resources → outcome="limited_kg_only".

    Edge case: the KG was sparse for the query AND the hybrid
    retrieval path didn't land anything (network failure, query
    rewrite still off-target, etc.). The card should show AMBER ⚠️
    "Copertura limitata" with an explicit "verify manually" prompt.
    Distinct from grader_error — the run completed cleanly, the
    outcome is just genuinely poor.
    """
    state = _state(
        retrieval_grade="irrelevant",
        retrieval_grade_reason=(
            "Neither the KG nodes nor the recommendations cover the "
            "subject content of the query."
        ),
        retrieval_attempts=2,
        retrieval_warning=True,
        # No external resources, no hybrid media.
        external_resources=None,
        curated_media={},
    )

    payload = _build_retriever_payload(state)

    assert payload["retrieval_outcome"] == "limited_kg_only"
    assert payload["retrieval_grade"] == "irrelevant"
    assert payload["retrieval_attempts"] == 2
    assert payload["media_counts"]["articles"] == 0
    assert payload["media_counts"]["oer"] == 0


def test_outcome_grader_error_when_reason_has_sentinel_prefix() -> None:
    """Grader exception → outcome="grader_error" (red ❌).

    When the grader LLM throws (timeout, parse failure, etc.),
    ``grade_retrieval_node`` falls back to ``grade=relevant`` so the
    pipeline never blocks — but stamps the reason with the sentinel
    prefix ``"Grader exception:"``. The outcome detector must see
    through that fail-open green light and surface a legitimate red
    error card so the teacher knows the lesson was generated WITHOUT
    grading. This is the only outcome that should ever show RED.
    """
    state = _state(
        retrieval_grade="relevant",  # fail-open default
        retrieval_grade_reason="Grader exception: TimeoutError",
        retrieval_attempts=1,
    )

    payload = _build_retriever_payload(state)

    assert payload["retrieval_outcome"] == "grader_error"
    # Note: grade itself is still "relevant" (fail-open) — only the
    # outcome token reflects the error. This keeps the writer's
    # downstream behaviour unchanged (no caveat injected).
    assert payload["retrieval_grade"] == "relevant"
    assert payload["retrieval_grade_reason"] == "Grader exception: TimeoutError"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_outcome_success_when_grade_is_none() -> None:
    """Corrective-RAG flag OFF → grade is None → outcome="success".

    The template's outer ``{% if p.retrieval_grade %}`` guard skips
    the entire grading block, so the rendered card is bit-for-bit
    identical to pre-#9. The outcome token is irrelevant in that
    case but we still return "success" for consistency / forward-
    compat (a future caller that ignores the outer guard would see
    a sane default).
    """
    state = _state(retrieval_grade=None, retrieval_grade_reason=None)

    payload = _build_retriever_payload(state)

    assert payload["retrieval_outcome"] == "success"
    assert payload["retrieval_grade"] is None


def test_outcome_adapted_with_hybrid_when_irrelevant_via_curated_media_only() -> None:
    """external_resources empty but curated_media has external buckets → blue.

    Belt-and-braces: the hybrid retrieval path *might* land its
    output in ``curated_media`` (citations + resources + open_textbooks)
    instead of the ``external_resources`` field, depending on the
    retriever code path. We accept either signal as evidence that
    the gap was filled.
    """
    state = _state(
        retrieval_grade="ambiguous",
        retrieval_attempts=2,
        external_resources=None,  # not set
        curated_media={
            "citations": [{"title": "Paper A"}, {"title": "Paper B"}],
            "resources": [{"title": "Khan Academy module"}],
        },
    )

    payload = _build_retriever_payload(state)

    assert payload["retrieval_outcome"] == "adapted_with_hybrid"
    assert payload["media_counts"]["articles"] == 2
    assert payload["media_counts"]["oer"] == 1


def test_compute_retrieval_outcome_pure_function() -> None:
    """``_compute_retrieval_outcome`` is the pure decision function —
    test it in isolation from the payload-shaping wrapper."""
    # success — relevant
    assert _compute_retrieval_outcome(
        {"retrieval_grade": "relevant", "retrieval_grade_reason": "ok"},
        {"articles": 0, "oer": 0, "videos": 0},
    ) == "success"

    # success — flag off
    assert _compute_retrieval_outcome(
        {"retrieval_grade": None, "retrieval_grade_reason": None},
        {"articles": 0, "oer": 0, "videos": 0},
    ) == "success"

    # adapted_with_hybrid — irrelevant + external dict
    assert _compute_retrieval_outcome(
        {
            "retrieval_grade": "irrelevant",
            "retrieval_grade_reason": "out of scope",
            "external_resources": {"wikipedia": [{"x": 1}]},
        },
        {"articles": 0, "oer": 0, "videos": 0},
    ) == "adapted_with_hybrid"

    # limited_kg_only — irrelevant + nothing else
    assert _compute_retrieval_outcome(
        {
            "retrieval_grade": "irrelevant",
            "retrieval_grade_reason": "out of scope",
            "external_resources": None,
        },
        {"articles": 0, "oer": 0, "videos": 0},
    ) == "limited_kg_only"

    # grader_error — sentinel reason wins over grade=relevant
    assert _compute_retrieval_outcome(
        {
            "retrieval_grade": "relevant",
            "retrieval_grade_reason": "Grader exception: ConnectionError",
        },
        {"articles": 0, "oer": 0, "videos": 0},
    ) == "grader_error"


# ---------------------------------------------------------------------------
# Tentativi: N/M badge data + retry semantics (#9.UX-2)
# ---------------------------------------------------------------------------

def test_retrieval_attempts_max_resolves_from_env() -> None:
    """``retrieval_attempts_max`` must reflect the configured env var so
    the chat card shows the actual budget (e.g. ``2/2`` not ``2/?``)."""
    state = _state(retrieval_grade="relevant", retrieval_attempts=1)

    with patch.dict("os.environ", {"AIX_CORRECTIVE_RAG_MAX_ATTEMPTS": "3"}):
        payload = _build_retriever_payload(state)
        assert payload["retrieval_attempts_max"] == 3
        assert _resolve_max_attempts() == 3

    with patch.dict("os.environ", {"AIX_CORRECTIVE_RAG_MAX_ATTEMPTS": "1"}):
        assert _resolve_max_attempts() == 1


def test_retrieval_attempts_max_clamps_invalid_env() -> None:
    """Garbage in env must clamp to the safe default of 2."""
    with patch.dict("os.environ", {"AIX_CORRECTIVE_RAG_MAX_ATTEMPTS": "not-a-number"}):
        assert _resolve_max_attempts() == 2

    # Above the clamp ceiling (4) collapses to 4.
    with patch.dict("os.environ", {"AIX_CORRECTIVE_RAG_MAX_ATTEMPTS": "99"}):
        assert _resolve_max_attempts() == 4

    # Below 1 collapses to 1.
    with patch.dict("os.environ", {"AIX_CORRECTIVE_RAG_MAX_ATTEMPTS": "0"}):
        assert _resolve_max_attempts() == 1


def test_grader_will_retry_mirrors_router_logic() -> None:
    """The SSE-layer ``_grader_will_retry`` must return the same answer
    as the graph's ``should_retry_retrieval`` router so the #9.UX-2
    gate matches the actual graph traversal exactly. This is the
    invariant that prevents the duplicate-card regression."""
    # relevant → never retry, regardless of attempts.
    assert _grader_will_retry({"retrieval_grade": "relevant", "retrieval_attempts": 1}) is False
    assert _grader_will_retry({"retrieval_grade": "relevant", "retrieval_attempts": 2}) is False

    with patch.dict("os.environ", {"AIX_CORRECTIVE_RAG_MAX_ATTEMPTS": "2"}):
        # irrelevant + attempts < max → retry.
        assert _grader_will_retry(
            {"retrieval_grade": "irrelevant", "retrieval_attempts": 1}
        ) is True
        # irrelevant + attempts == max → do NOT retry (loop terminates).
        assert _grader_will_retry(
            {"retrieval_grade": "irrelevant", "retrieval_attempts": 2}
        ) is False
        # ambiguous behaves like irrelevant.
        assert _grader_will_retry(
            {"retrieval_grade": "ambiguous", "retrieval_attempts": 1}
        ) is True


# ---------------------------------------------------------------------------
# Backward-compat
# ---------------------------------------------------------------------------

def test_payload_shape_unchanged_when_grading_off() -> None:
    """When corrective-RAG is OFF (grade=None), every grading-related
    payload key is still present (with None / falsy values) so the
    template's defensive ``payload.get(...)`` calls keep working.
    Locks the contract that pre-#9 callers still see the same shape."""
    state = _state()  # all None for grading

    payload = _build_retriever_payload(state)

    # Pre-#9 keys still there.
    assert "nodes_count" in payload
    assert "media_counts" in payload
    assert "top_concepts" in payload
    # Grading keys present but None / False.
    assert payload["retrieval_grade"] is None
    assert payload["retrieval_grade_label"] is None
    assert payload["retrieval_grade_emoji"] is None
    assert payload["retrieval_grade_reason"] is None
    assert payload["retrieval_warning"] is False
    # Outcome defaults to "success" but the template's outer guard
    # ({% if p.retrieval_grade %}) skips the row entirely.
    assert payload["retrieval_outcome"] == "success"
    # attempts_max is always set so the badge logic doesn't crash on
    # legacy state diffs missing the field.
    assert isinstance(payload["retrieval_attempts_max"], int)


# ---------------------------------------------------------------------------
# CORE 2 #9.UX-5 — domain-aware coverage banner (Corrective-RAG-OFF only)
# ---------------------------------------------------------------------------

def test_coverage_tier_classifier_boundaries() -> None:
    """Pure tier classifier locks the three coverage-band boundaries:
    0 nodes → out_of_scope, 1..4 → limited, ≥5 → healthy.

    These cutoffs drive the CR-OFF banner color (slate-blue / amber /
    sage) and the teacher copy ("La lezione si baserà su…" /
    "Copertura parziale…" / "Ricerca completata…"). Changing them
    breaks the explainability contract, so they're locked here."""
    # Zero KG nodes → blue "out of scope" banner.
    assert _classify_coverage_tier(0) == "out_of_scope"

    # 1..4 KG nodes → amber "limited coverage" banner. Lower bound and
    # upper bound both exercised (the default threshold is 5).
    assert _classify_coverage_tier(1) == "limited"
    assert _classify_coverage_tier(4) == "limited"

    # ≥5 KG nodes → sage "healthy" banner. Threshold itself + above.
    assert _classify_coverage_tier(5) == "healthy"
    assert _classify_coverage_tier(11) == "healthy"  # the ADHD smoke
    assert _classify_coverage_tier(64) == "healthy"  # the user's mockup example


def test_coverage_tier_classifier_threshold_configurable_from_env() -> None:
    """Threshold can be tuned via ``AIX_COVERAGE_HEALTHY_THRESHOLD`` for
    ops experiments. Clamped to 1..50 so a typo (e.g. ``"999"``) can't
    silently turn every lesson into ``limited`` and a negative value
    can't push ``out_of_scope`` into the positive band."""
    # Raise the threshold to 10: 5-9 nodes are now "limited", not "healthy".
    with patch.dict("os.environ", {"AIX_COVERAGE_HEALTHY_THRESHOLD": "10"}):
        assert _classify_coverage_tier(5) == "limited"
        assert _classify_coverage_tier(9) == "limited"
        assert _classify_coverage_tier(10) == "healthy"

    # Garbage clamps to the default (5).
    with patch.dict("os.environ", {"AIX_COVERAGE_HEALTHY_THRESHOLD": "abc"}):
        assert _classify_coverage_tier(5) == "healthy"
        assert _classify_coverage_tier(4) == "limited"

    # Out-of-range clamps to the valid window.
    with patch.dict("os.environ", {"AIX_COVERAGE_HEALTHY_THRESHOLD": "999"}):
        # 50 is the upper clamp ceiling.
        assert _classify_coverage_tier(49) == "limited"
        assert _classify_coverage_tier(50) == "healthy"


def test_resolve_domain_labels_known_and_unknown() -> None:
    """Domain label dictionary is the single source of truth for the
    teacher-facing domain names. Locks both the known domains' short
    + long forms and the graceful fallback for unknown domains."""
    # UDL — short form is the bare acronym; long form adds the
    # explanatory parenthetical used in the Tier 0 "out of scope" copy.
    udl = _resolve_domain_labels("udl")
    assert udl["short"] == "UDL"
    assert udl["long"]  == "UDL (pedagogia inclusiva)"

    # Case-insensitive lookup — state["domain"] may arrive as "UDL".
    assert _resolve_domain_labels("UDL") == udl

    # Neuro — short and long are identical (user decision, see
    # ClickUp #9.UX-5: "Neuro is fine").
    neuro = _resolve_domain_labels("neuro")
    assert neuro["short"] == "Neuro"
    assert neuro["long"]  == "Neuro"

    # Unknown / None / empty — fall back gracefully to either the raw
    # value (so a future "stem" domain renders "stem" until labelled)
    # or a generic phrase. Never raises KeyError.
    unknown = _resolve_domain_labels("stem")
    assert unknown["short"] == "stem"
    assert unknown["long"]  == "stem"

    none_labels = _resolve_domain_labels(None)
    assert none_labels["short"] == "il dominio attivo"
    assert none_labels["long"]  == "il dominio attivo"


def test_retriever_payload_carries_domain_and_coverage_tier() -> None:
    """End-to-end shape: the public payload contract surfaces the four
    #9.UX-5 fields (``domain``, ``domain_label_short``,
    ``domain_label_long``, ``coverage_tier``) plus ``media_total``, so
    the template can render the CR-OFF banner + domain-aware footer
    without any further lookups.

    Three flavours exercised, one per coverage tier, on two distinct
    domains so the domain-label plumbing is covered too:
        1. udl + 11 nodes  → healthy + "UDL"
        2. udl + 3 nodes   → limited + "UDL"
        3. neuro + 0 nodes → out_of_scope + "Neuro"
    """
    # 1. Healthy on UDL (the ADHD smoke shape).
    healthy = _build_retriever_payload(_state(
        domain="udl",
        retrieved_nodes=[{"title": f"Node {i}"} for i in range(11)],
        recommendations=[{"strategy": f"S{i}"} for i in range(88)],
        curated_media={
            "videos":    [{"title": "v1"}, {"title": "v2"}],
            "citations": [{"title": "c1"}, {"title": "c2"}, {"title": "c3"}],
            "resources": [{"title": "r1"}],
        },
    ))
    assert healthy["domain"] == "udl"
    assert healthy["domain_label_short"] == "UDL"
    assert healthy["domain_label_long"]  == "UDL (pedagogia inclusiva)"
    assert healthy["coverage_tier"] == "healthy"
    assert healthy["nodes_count"] == 11
    assert healthy["recommendations_count"] == 88
    # media_total is the sum of videos + articles + oer (2 + 3 + 1 = 6).
    assert healthy["media_total"] == 6

    # 2. Limited on UDL — partial coverage.
    limited = _build_retriever_payload(_state(
        domain="udl",
        retrieved_nodes=[{"title": "Working Memory"}, {"title": "Scaffolding"}, {"title": "Chunking"}],
        recommendations=[{"strategy": "s1"}, {"strategy": "s2"}],
        curated_media={},
    ))
    assert limited["domain"] == "udl"
    assert limited["domain_label_short"] == "UDL"
    assert limited["coverage_tier"] == "limited"
    assert limited["nodes_count"] == 3
    assert limited["media_total"] == 0

    # 3. Out of scope on neuro — KG returned nothing.
    oos = _build_retriever_payload(_state(
        domain="neuro",
        retrieved_nodes=[],
        recommendations=[],
        curated_media={},
    ))
    assert oos["domain"] == "neuro"
    assert oos["domain_label_short"] == "Neuro"
    assert oos["domain_label_long"]  == "Neuro"
    assert oos["coverage_tier"] == "out_of_scope"
    assert oos["nodes_count"] == 0
    assert oos["media_total"] == 0
    # Sanity: even when grade is None (CR OFF), the existing outcome
    # token still returns "success" — proves the new banner block
    # doesn't conflict with the CR-ON outcome callout's token.
    assert oos["retrieval_outcome"] == "success"
    assert oos["retrieval_grade"] is None
