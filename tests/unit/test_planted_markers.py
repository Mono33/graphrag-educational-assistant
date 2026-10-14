"""A citation marker written inside a document cannot become a citation.

A marker is plain text and so is a teacher's document. A file carrying
``[[material:<someone else's id>#page=1]]`` in its body hands the Writer a
marker nobody retrieved. The WebUI renderer drops it, because it knows which
documents the lesson is bound to; the public API renders without that list, so
it became "(materiale allegato, p. 1)" -- a citation to nothing -- and the
support check, which only judges markers it knows, let it through (2026-09-25).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from aix.agent.validation.citation_support import drop_unknown_markers

pytestmark = pytest.mark.unit

MINE = "[[material:11111111-1111-1111-1111-111111111111#page=2&block=a1]]"
PLANTED = "[[material:99999999-9999-9999-9999-999999999999#page=1]]"
ITEMS = [{"citation": MINE, "text": "La geografia studia i rapporti fra società e pianeta."}]


def test_a_marker_the_writer_was_given_stays():
    draft = f"La geografia studia le società {MINE}."

    assert drop_unknown_markers(draft, ITEMS) == draft


def test_a_planted_marker_goes_and_the_sentence_stays():
    draft = f"La geografia studia le società {MINE}. Un'altra frase {PLANTED}."

    out = drop_unknown_markers(draft, ITEMS)

    assert PLANTED not in out
    assert MINE in out
    assert out.endswith("Un'altra frase."), "no space left before the full stop"


def test_without_retrieved_material_every_marker_is_unknown():
    """No private search this turn: nothing may be cited as private material."""
    assert drop_unknown_markers(f"Frase {PLANTED}.", []) == "Frase."
    assert drop_unknown_markers(f"Frase {PLANTED}.", None) == "Frase."


def test_the_same_document_with_another_page_is_still_unknown():
    """The Writer must copy the marker it was handed, not compose a neighbour."""
    neighbour = MINE.replace("page=2", "page=9")

    assert neighbour not in drop_unknown_markers(f"Frase {neighbour}.", ITEMS)


async def test_write_node_removes_a_planted_marker(monkeypatch):
    from aix.agent.graph import nodes as graph_nodes

    monkeypatch.setenv("AIX_CITATION_CHECK_ENABLED", "false")
    writer = AsyncMock(write=AsyncMock(return_value=f"# Lezione\n\nUna frase {PLANTED}.\n"))
    state = {
        "teacher_query": "crea una lezione",
        "plan": {"query_intent": "lesson_creation"},
        "query_intent": "lesson_creation",
        "language": "it",
        "material_context_items": ITEMS,
    }

    with patch.object(graph_nodes, "get_writer", return_value=writer):
        updates = await graph_nodes.write_node(state)

    assert PLANTED not in updates["lesson_plan_draft"]
    assert "Una frase." in updates["lesson_plan_draft"]
