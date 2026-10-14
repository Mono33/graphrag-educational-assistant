"""The chat has to show a reused turn as a reused turn.

T6 lets a follow-up skip the Knowledge Graph. The graph did that correctly on
the first live run and the chat still showed "Retriever Agent · ricerca" with a
spinner, forever: the SSE loop drops any node it does not recognise, ``reuse``
was not in ``PHASE_LABELS``, so no event was ever emitted and the pending
placeholder was never replaced. The teacher was left watching a search that had
already been decided against.

Two halves are pinned here — that the node is recognised at all, and that the
card it produces describes what happened rather than reusing the copy of a
search ("graphrag.search", "Ricerca completata", "10-30 secondi").
"""

from __future__ import annotations

from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

from aix.webui.agent.service import PHASE_LABELS, _build_retriever_payload

pytestmark = pytest.mark.unit

TEMPLATES = Path(__file__).resolve().parents[2] / "src" / "aix" / "webui" / "templates"


def _state(**over):
    state = {
        "retrieved_nodes": [{"name": "Energie rinnovabili"}, {"name": "Scaffolding"}],
        "retrieved_relationships": [{"type": "RELATES_TO"}],
        "recommendations": [{"name": "Chunking"}],
        "domain": "neuro",
        "retrieval_confidence": "high",
    }
    state.update(over)
    return state


def _render(payload):
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)))
    return env.get_template("partials/chat_retriever_card.html").render(payload=payload)


# ---------------------------------------------------------------------------
# The node has to reach the SSE loop at all
# ---------------------------------------------------------------------------


def test_reuse_is_a_recognised_phase():
    """``if node_name not in PHASE_LABELS: continue`` is what stranded the card."""
    assert "reuse" in PHASE_LABELS


# ---------------------------------------------------------------------------
# The payload
# ---------------------------------------------------------------------------


def test_payload_marks_a_reused_turn():
    payload = _build_retriever_payload(
        _state(retrieval_reused=True, plan={"retrieval_reason": "solo una traduzione"})
    )

    assert payload["reused"] is True
    assert payload["reuse_reason"] == "solo una traduzione"
    assert payload["nodes_count"] == 2, "the counts are the corpus in hand, not zero"


def test_a_searched_turn_is_not_marked():
    payload = _build_retriever_payload(_state())

    assert payload["reused"] is False
    assert payload["reuse_reason"] == ""


def test_a_missing_reason_does_not_break_the_card():
    payload = _build_retriever_payload(_state(retrieval_reused=True, plan={}))

    assert payload["reused"] is True
    assert payload["reuse_reason"] == ""
    assert "Nessuna nuova ricerca" in _render(payload)


# ---------------------------------------------------------------------------
# The card
# ---------------------------------------------------------------------------


def test_reused_card_does_not_claim_a_search_ran():
    html = _render(
        _build_retriever_payload(
            _state(retrieval_reused=True, plan={"retrieval_reason": "solo una traduzione"})
        )
    )

    assert "graphrag.search" not in html, "no tool was called this turn"
    assert "Ricerca completata" not in html
    assert "Contesto riutilizzato" in html
    assert "Nessuna nuova ricerca per questa richiesta" in html
    assert "solo una traduzione" in html


def test_reused_card_still_shows_the_evidence_behind_the_lesson():
    """Reuse is not "no sources" — it is the same sources as the turn before."""
    html = _render(_build_retriever_payload(_state(retrieval_reused=True)))

    assert "Energie rinnovabili" in html
    assert "concetti gi" in html, "the role line says where the concepts come from"


def test_searched_card_is_unchanged():
    html = _render(_build_retriever_payload(_state()))

    assert "graphrag.search" in html
    assert "Ricerca nel Knowledge Graph" in html
    assert "Contesto riutilizzato" not in html
    assert "Nessuna nuova ricerca" not in html
