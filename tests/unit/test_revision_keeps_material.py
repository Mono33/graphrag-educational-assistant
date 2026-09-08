"""A Critic-triggered revision must keep the teacher's uploaded material.

``WriterAgent.revise()` did not accept ``teacher_provided_context`` at all, and
``write_node`` did not pass it. So the material reached the first draft and then
vanished: the moment the Critic asked for a change, the revision was produced
from the draft and the critique alone.

Any constraint the upload carried stopped applying at that point — a syllabus,
a class limitation, a passage the lesson had to quote. From the teacher's side
it reads as the assistant forgetting their file, with nothing in the interface
to explain why.

The tests below pin the fix at all three levels: the agent, the node that calls
it, and a full run through the compiled graph.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from aix.agent.agents.writer_agent import WriterAgent
from aix.agent.graph import nodes as graph_nodes
from aix.agent.graph.lesson_planner_graph import build_lesson_planner_graph, reset_compiled_graph
from aix.agent.graph.state import create_initial_state

pytestmark = pytest.mark.unit

MATERIAL = "Programma di classe: si affronta solo la memoria di lavoro, non l'attenzione."


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("AIX_CORRECTIVE_RAG_ENABLED", raising=False)
    monkeypatch.delenv("AIX_CONDITIONAL_RETRIEVAL_ENABLED", raising=False)
    reset_compiled_graph()
    yield
    reset_compiled_graph()


def _completion(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")]
    )


async def _revise_capturing_prompt(**kwargs) -> str:
    """Run ``revise()`` against a mocked completion; return the user prompt sent."""
    captured: dict = {}

    async def _fake_completion(_client, *, messages, **_kw):
        captured["messages"] = messages
        return _completion("# Lezione rivista")

    writer = WriterAgent()
    with (
        patch.object(WriterAgent, "_get_client", return_value=object()),
        patch("aix.agent.agents.writer_agent.guarded_chat_completion", _fake_completion),
    ):
        await writer.revise(
            current_draft="# Bozza",
            critique="manca la valutazione",
            revision_instructions="aggiungi una sezione di valutazione",
            **kwargs,
        )

    return next(m["content"] for m in captured["messages"] if m["role"] == "user")


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------


async def test_revise_carries_the_material_into_the_prompt():
    prompt = await _revise_capturing_prompt(teacher_provided_context=MATERIAL)

    assert MATERIAL in prompt
    assert "Materiale caricato dall'insegnante" in prompt, (
        "reuses write()'s appendix, so both paths frame the material identically"
    )


async def test_revise_without_material_is_unchanged():
    """The no-upload path must stay byte-identical to the pre-fix behaviour."""
    prompt = await _revise_capturing_prompt()

    assert "Materiale caricato dall'insegnante" not in prompt
    assert "# Bozza" in prompt
    assert "manca la valutazione" in prompt


@pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
async def test_blank_material_adds_no_appendix(blank):
    prompt = await _revise_capturing_prompt(teacher_provided_context=blank)
    assert "Materiale caricato dall'insegnante" not in prompt


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------


async def test_write_node_passes_the_material_on_a_revision():
    writer = AsyncMock(revise=AsyncMock(return_value="# Rivista"))
    state = dict(create_initial_state(query="q", max_revisions=2))
    state.update(
        teacher_provided_context=MATERIAL,
        lesson_plan_draft="# Bozza",
        revision_count=1,  # write_node branches on this to choose revise()
        critique="manca la valutazione",
        revision_instructions="aggiungi la valutazione",
    )

    with patch.object(graph_nodes, "get_writer", return_value=writer):
        await graph_nodes.write_node(state)

    writer.revise.assert_awaited_once()
    assert writer.revise.await_args.kwargs["teacher_provided_context"] == MATERIAL


async def test_write_node_still_writes_from_scratch_on_the_first_pass():
    writer = AsyncMock(write=AsyncMock(return_value="# Bozza"))
    state = dict(create_initial_state(query="q"))
    state.update(teacher_provided_context=MATERIAL, retrieved_nodes=[], recommendations=[])

    with patch.object(graph_nodes, "get_writer", return_value=writer):
        await graph_nodes.write_node(state)

    writer.write.assert_awaited_once()
    writer.revise.assert_not_awaited()
    assert writer.write.await_args.kwargs["teacher_provided_context"] == MATERIAL


# ---------------------------------------------------------------------------
# The whole run
# ---------------------------------------------------------------------------


async def test_material_survives_a_full_run_with_one_revision():
    """The regression in situ: draft, Critic rejects, revision still has the file."""
    from tests.unit.agent_doubles import _critique, _plan, _retrieval_result

    planner = AsyncMock(plan=AsyncMock(return_value=_plan()))
    retriever = AsyncMock(retrieve=AsyncMock(return_value=_retrieval_result()))
    writer = AsyncMock(
        write=AsyncMock(return_value="# Bozza"),
        revise=AsyncMock(return_value="# Rivista"),
    )
    critic = AsyncMock(critique=AsyncMock(side_effect=[_critique(False), _critique(True)]))

    with (
        patch.object(graph_nodes, "get_planner", return_value=planner),
        patch.object(graph_nodes, "get_retriever", return_value=retriever),
        patch.object(graph_nodes, "get_writer", return_value=writer),
        patch.object(graph_nodes, "get_critic", return_value=critic),
    ):
        final = await build_lesson_planner_graph().ainvoke(
            create_initial_state(
                query="Crea una lezione",
                teacher_provided_context=MATERIAL,
                max_revisions=2,
            )
        )

    assert writer.write.await_args.kwargs["teacher_provided_context"] == MATERIAL
    assert writer.revise.await_args.kwargs["teacher_provided_context"] == MATERIAL, (
        "the revision used to run without the teacher's file"
    )
    assert final["final_lesson_plan"] == "# Rivista"
