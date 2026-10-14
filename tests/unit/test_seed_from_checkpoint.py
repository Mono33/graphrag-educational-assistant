"""What a turn takes from the previous one's checkpoint (2026-10-14).

A failed turn left its error in the checkpoint; the next turn, seeded from it,
was declared failed after writing and approving its lesson.
"""

from __future__ import annotations

import pytest

from aix.webui.agent.service import _fail_if_the_turn_produced_nothing, _seed_from_checkpoint

pytestmark = pytest.mark.unit


def test_the_error_of_a_failed_turn_does_not_fail_the_next():
    final_state = {"error": None, "query": "nuova richiesta"}

    _seed_from_checkpoint(final_state, {"error": "'wikipedia_content'"})

    assert final_state["error"] is None
    _fail_if_the_turn_produced_nothing(final_state)  # does not raise


def test_the_evidence_a_follow_up_reuses_is_carried_over():
    final_state = {}

    _seed_from_checkpoint(
        final_state,
        {
            "retrieved_nodes": [{"name": "Muro di Berlino"}],
            "external_resources": {"wikipedia": [{"title": "Guerra fredda"}]},
            "final_lesson_plan": "# La lezione di prima",
        },
    )

    assert final_state["retrieved_nodes"] == [{"name": "Muro di Berlino"}]
    assert final_state["external_resources"]["wikipedia"][0]["title"] == "Guerra fredda"
    assert "final_lesson_plan" not in final_state, "the previous lesson is not this turn's"
