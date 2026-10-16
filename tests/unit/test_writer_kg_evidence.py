"""The Writer reads what the graph says about this class, not the head of a list."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aix.agent.agents.writer_agent import WriterAgent, _format_relationships

pytestmark = pytest.mark.unit

NODES = [
    {"name": "Focus Tasks", "labels": ["InstructionalTechnique"]},
    {"name": "Long Frontal Lesson", "labels": ["EducationalApproach"]},
    {"name": "Difficulty sustaining focus", "labels": ["Adhd"]},
    {"name": "Group till  15", "labels": ["Context"]},
    {"name": "Group  till 30", "labels": ["Context"]},
]
RELS = [
    {"source": "Difficulty sustaining focus", "type": "SUGGESTS", "target": "Focus Tasks"},
    {
        "source": "Difficulty sustaining focus",
        "type": "NO_SUGGESTS",
        "target": "Long Frontal Lesson",
    },
    {"source": "Group  till 30", "type": "SUGGESTS", "target": "Focus Tasks"},
    {"source": "Group till  15", "type": "VECTOR_SIMILAR", "target": "Group  till 30"},
]
RECS = [
    {
        "name": "Focus Tasks",
        "implementation_guidance": "One short task, one goal",
        "confidence": "HIGH",
    },
    {
        "name": "Group till  15",
        "category": "Contesto Classe",
        "implementation_guidance": "Apply Group till  15 with appropriate adaptations",
    },
]
PROFILE = {"group": {"students_number": 22, "disabilities": ["ADHD"]}, "time_available_minutes": 60}


def _plan(intent: str = "lesson_creation"):
    from tests.unit.agent_doubles import _plan as base

    plan = base()
    plan.query_intent = intent
    return plan


async def _messages(profile=PROFILE, intent: str = "lesson_creation") -> list[dict]:
    result = SimpleNamespace(
        nodes=NODES, relationships=RELS, recommendations=RECS, confidence="HIGH"
    )
    seen = []

    async def _fake_stream(_self, _client, messages, _kwargs, _bus):
        seen.append(messages)
        return ("# Lezione", "stop")

    with (
        patch.object(WriterAgent, "_get_client", return_value=object()),
        patch.object(WriterAgent, "_stream_completion", _fake_stream),
    ):
        await WriterAgent().write(
            teacher_query="Crea una lezione",
            plan=_plan(intent),
            retrieval_result=result,
            educational_profile=profile,
            token_bus=asyncio.Queue(),
        )
    return seen[0]


def _text(message) -> str:
    content = message["content"]
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content)
    return content


async def test_the_needs_map_reaches_the_writer():
    user = _text((await _messages())[1])

    assert (
        "**ADHD · Difficulty sustaining focus** → il KG suggerisce: Focus Tasks; sconsiglia: Long Frontal Lesson"
        in user
    )
    assert "**Group till 30** (profilo della classe) → suggerisce: Focus Tasks" in user


async def test_recommendations_carry_their_guidance_and_no_class_profile():
    user = _text((await _messages())[1])

    assert "### Focus Tasks (Confidence: HIGH)\nOne short task, one goal" in user
    assert "### Group till  15" not in user
    assert "appropriate adaptations" not in user


async def test_class_profile_nodes_are_not_listed_as_concepts():
    user = _text((await _messages())[1])
    listed = user.split("### Altri nodi recuperati", 1)[1]

    assert "Group till  15" not in listed
    assert "**Focus Tasks** [InstructionalTechnique]\n  One short task, one goal" in listed


async def test_the_rules_for_the_evidence_are_in_the_system_prompt():
    system = _text((await _messages())[0])

    assert "## Using the Knowledge Graph evidence" in system


async def test_the_learning_environment_is_asked_for_when_the_class_needs_it():
    assert "## Learning environment" in _text((await _messages())[1])
    calm = {"group": {"students_number": 22, "disabilities": ["DLDS"]}}
    assert "## Learning environment" not in _text((await _messages(calm))[1])


async def test_not_for_a_definition():
    messages = await _messages(intent="definition")

    assert "## Learning environment" not in _text(messages[1])
    assert "## Using the Knowledge Graph evidence" not in _text(messages[0])


def test_vector_similarity_is_not_a_relationship_the_writer_reads():
    text = _format_relationships(RELS)

    assert "VECTOR_SIMILAR" not in text
    assert "Difficulty sustaining focus —NO_SUGGESTS→ Long Frontal Lesson" in text
