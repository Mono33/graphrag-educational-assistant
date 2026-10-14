"""The retriever card's payload: its shape, the coverage tier and the domain labels.

The graded outcomes of corrective RAG (``AIX_CORRECTIVE_RAG_ENABLED``, kept off
by the pilot) are no longer covered here (2026-10-13): with grading off the
payload must keep every key, which is what is pinned. Pure unit tests -- no
LLM, no DB, no network.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from aix.webui.agent.service import (
    _build_retriever_payload,
    _classify_coverage_tier,
    _resolve_domain_labels,
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
    assert udl["long"] == "UDL (pedagogia inclusiva)"

    # Case-insensitive lookup — state["domain"] may arrive as "UDL".
    assert _resolve_domain_labels("UDL") == udl

    # Neuro — short and long are identical (user decision, see
    # ClickUp #9.UX-5: "Neuro is fine").
    neuro = _resolve_domain_labels("neuro")
    assert neuro["short"] == "Neuro"
    assert neuro["long"] == "Neuro"

    # Unknown / None / empty — fall back gracefully to either the raw
    # value (so a future "stem" domain renders "stem" until labelled)
    # or a generic phrase. Never raises KeyError.
    unknown = _resolve_domain_labels("stem")
    assert unknown["short"] == "stem"
    assert unknown["long"] == "stem"

    none_labels = _resolve_domain_labels(None)
    assert none_labels["short"] == "il dominio attivo"
    assert none_labels["long"] == "il dominio attivo"


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
    healthy = _build_retriever_payload(
        _state(
            domain="udl",
            retrieved_nodes=[{"title": f"Node {i}"} for i in range(11)],
            recommendations=[{"strategy": f"S{i}"} for i in range(88)],
            curated_media={
                "videos": [{"title": "v1"}, {"title": "v2"}],
                "citations": [{"title": "c1"}, {"title": "c2"}, {"title": "c3"}],
                "resources": [{"title": "r1"}],
            },
        )
    )
    assert healthy["domain"] == "udl"
    assert healthy["domain_label_short"] == "UDL"
    assert healthy["domain_label_long"] == "UDL (pedagogia inclusiva)"
    assert healthy["coverage_tier"] == "healthy"
    assert healthy["nodes_count"] == 11
    assert healthy["recommendations_count"] == 88
    # media_total is the sum of videos + articles + oer (2 + 3 + 1 = 6).
    assert healthy["media_total"] == 6

    # 2. Limited on UDL — partial coverage.
    limited = _build_retriever_payload(
        _state(
            domain="udl",
            retrieved_nodes=[
                {"title": "Working Memory"},
                {"title": "Scaffolding"},
                {"title": "Chunking"},
            ],
            recommendations=[{"strategy": "s1"}, {"strategy": "s2"}],
            curated_media={},
        )
    )
    assert limited["domain"] == "udl"
    assert limited["domain_label_short"] == "UDL"
    assert limited["coverage_tier"] == "limited"
    assert limited["nodes_count"] == 3
    assert limited["media_total"] == 0

    # 3. Out of scope on neuro — KG returned nothing.
    oos = _build_retriever_payload(
        _state(
            domain="neuro",
            retrieved_nodes=[],
            recommendations=[],
            curated_media={},
        )
    )
    assert oos["domain"] == "neuro"
    assert oos["domain_label_short"] == "Neuro"
    assert oos["domain_label_long"] == "Neuro"
    assert oos["coverage_tier"] == "out_of_scope"
    assert oos["nodes_count"] == 0
    assert oos["media_total"] == 0
    # Sanity: even when grade is None (CR OFF), the existing outcome
    # token still returns "success" — proves the new banner block
    # doesn't conflict with the CR-ON outcome callout's token.
    assert oos["retrieval_outcome"] == "success"
    assert oos["retrieval_grade"] is None


# ---------------------------------------------------------------------------
# «Fuori ambito» and «the database is down» are not the same sentence
# ---------------------------------------------------------------------------


def test_an_unreachable_graph_is_not_reported_as_an_out_of_scope_question() -> None:
    """Zero concepts has two causes and the teacher is owed the right one.

    Both produce ``coverage_tier == "out_of_scope"``, whose copy tells the
    teacher their question lies outside the Knowledge Graph. When the database
    simply did not answer that is false, and it sends them rewriting a question
    that was never the problem (2026-09-21).
    """
    outage = _build_retriever_payload(_state(retrieved_nodes=[], kg_unreachable=True))
    genuinely_out_of_scope = _build_retriever_payload(
        _state(retrieved_nodes=[], kg_unreachable=False)
    )

    assert outage["kg_unreachable"] is True
    assert genuinely_out_of_scope["kg_unreachable"] is False
    # The tier is unchanged in both: the banner branches on the new flag, so
    # nothing that reads ``coverage_tier`` has to learn a fourth value.
    assert outage["coverage_tier"] == genuinely_out_of_scope["coverage_tier"] == "out_of_scope"


def test_a_search_that_found_something_never_claims_an_outage() -> None:
    assert _build_retriever_payload(_state())["kg_unreachable"] is False
