"""What the graph returned, ranked and shaped for the lesson (staging case, 2026-10-16)."""

from __future__ import annotations

import pytest

from aix.agent import kg_evidence
from aix.agent.kg_evidence import (
    digest,
    environment_needed,
    kind,
    rank_nodes,
    rank_recommendations,
    render_evidence,
)

pytestmark = pytest.mark.unit

#: The shape of the staging lesson's retrieval: the class-profile search ran
#: first, so its nodes headed the merged list.
NODES = [
    {"name": "Group till  15", "labels": ["Context"], "rank_score": 1.6},
    {"name": "Group till  20", "labels": ["Context"], "rank_score": 1.6},
    {"name": "Group  till 30", "labels": ["Context"], "rank_score": 1.6},
    {"name": "Split in groups", "labels": ["Context"], "rank_score": 1.6},
    {"name": "Motivated", "labels": ["Context"], "rank_score": 1.6},
    {
        "name": "Station Rotation",
        "labels": ["LearningMethodology"],
        "rank_score": 1.52,
        "description": "Offers movement, autonomy, and lower-stress interactions.",
    },
    {"name": "Cooperative Learning", "labels": ["LearningMethodology"], "rank_score": 1.52},
    {"name": "Difficulty sustaining focus", "labels": ["Adhd"], "rank_score": 1.6},
    {"name": "Limited academic vocabulary", "labels": ["ForeignStudents"], "rank_score": 1.44},
    {"name": "Challenges in joint attention", "labels": ["AutismSpectrum"], "rank_score": 1.6},
    {"name": "Focus Tasks", "labels": ["InstructionalTechnique"], "rank_score": 1.7},
    {"name": "Multisensory Activities", "labels": ["InstructionalStrategy"], "rank_score": 2},
    {"name": "Long Frontal Lesson", "labels": ["EducationalApproach"], "rank_score": 2},
    {"name": "Glossaries", "labels": ["UniversaryTool"], "rank_score": 1},
    {"name": "Peer-Mediated Learning", "labels": ["EducationalApproach"], "rank_score": 2},
]
RELS = [
    {"source": "Group  till 30", "type": "SUGGESTS", "target": "Cooperative Learning"},
    {"source": "Group  till 30", "type": "NO_SUGGESTS", "target": "Station Rotation"},
    {"source": "Group till  15", "type": "SUGGESTS", "target": "Station Rotation"},
    {"source": "Group till  15", "type": "VECTOR_SIMILAR", "target": "Group till  20"},
    {"source": "Split in groups", "type": "SUGGESTS", "target": "Cooperative Learning"},
    {"source": "Difficulty sustaining focus", "type": "SUGGESTS", "target": "Focus Tasks"},
    {
        "source": "Multisensory Activities",
        "type": "SUGGESTS",
        "target": "Difficulty sustaining focus",
    },
    {
        "source": "Difficulty sustaining focus",
        "type": "NO_SUGGESTS",
        "target": "Long Frontal Lesson",
    },
    {"source": "Limited academic vocabulary", "type": "SUGGESTS", "target": "Glossaries"},
    {
        "source": "Challenges in joint attention",
        "type": "SUGGESTS",
        "target": "Peer-Mediated Learning",
    },
]
PROFILE = {
    "group": {
        "students_number": 22,
        "disabilities": ["DLDS", "ADHD"],
        "class_features": ["DIVISA_IN_GRUPPI"],
    }
}


def test_kinds():
    assert kind({"labels": ["Context"]}) == "context"
    assert kind({"labels": ["Adhd"]}) == "need"
    assert kind({"labels": ["InstructionalTechnique"]}) == "strategy"
    assert kind({"labels": ["UniversaryTool"]}) == "tool"
    assert kind({"labels": ["Colour"]}) == "environment"
    assert kind({"labels": ["Checkpoint"]}) == "udl"
    assert kind({"labels": ["Neuroplasticity"]}) == "other"


def test_strategies_outrank_the_class_profile():
    ranked = [n["name"] for n in rank_nodes(NODES)]

    assert ranked[:3] == [
        "Multisensory Activities",
        "Long Frontal Lesson",
        "Peer-Mediated Learning",
    ]
    assert set(ranked[-5:]) == {n["name"] for n in NODES if n["labels"] == ["Context"]}
    assert ranked.index("Difficulty sustaining focus") < ranked.index("Glossaries")


def test_ranking_is_stable_and_keeps_every_node():
    assert len(rank_nodes(NODES)) == len(NODES)
    assert rank_nodes([]) == []
    assert rank_nodes(None) == []


def test_recommendations_that_are_class_profile_nodes_go_last():
    recs = [
        {"name": "Group till  15", "category": "Contesto Classe", "relevance_score": 0.5},
        {"name": "Cooperative Learning", "relevance_score": 0.9},
        {"name": "Chunking", "relevance_score": 1},
        {"name": "Motivated", "relevance_score": 0.95},
    ]
    names = [r["name"] for r in rank_recommendations(recs, NODES)]

    assert names == ["Chunking", "Cooperative Learning", "Motivated", "Group till  15"]


def test_the_needs_of_this_class_with_what_the_graph_suggests_and_advises_against():
    text = render_evidence(NODES, RELS, PROFILE)

    assert (
        "**ADHD · Difficulty sustaining focus** → il KG suggerisce: Focus Tasks, Multisensory Activities"
        in text
    )
    assert "sconsiglia: Long Frontal Lesson" in text
    assert (
        "**Alunni non italofoni · Limited academic vocabulary** → il KG suggerisce: Glossaries"
        in text
    )
    # The class has no pupil on the autism spectrum.
    assert "joint attention" not in text


def test_only_the_class_profile_nodes_of_this_class():
    text = render_evidence(NODES, RELS, PROFILE)

    # 22 pupils: the «till 30» band, and the feature the teacher ticked. The
    # graph's names carry double spaces («Group  till 30»); they are collapsed.
    assert (
        "**Group till 30** (profilo della classe) → suggerisce: Cooperative Learning; sconsiglia: Station Rotation"
        in text
    )
    assert "**Split in groups**" in text
    assert "Group till 15" not in text
    assert "Motivated" not in text
    assert "VECTOR_SIMILAR" not in text


def test_a_strategy_advised_against_for_this_class_is_not_offered_as_one():
    text = render_evidence(NODES, RELS, PROFILE)
    strategies = text.split("### Strategie e strumenti recuperati", 1)[1]

    assert "Long Frontal Lesson" not in strategies
    assert "Station Rotation" not in strategies
    assert "**Focus Tasks** (strategia)" in strategies
    assert "**Glossaries** (strumento)" in strategies


def test_a_node_without_description_takes_the_recommendation_guidance():
    recs = [
        {"name": "Focus Tasks", "implementation_guidance": "Short tasks with one clear goal"},
        {
            "name": "Multisensory Activities",
            "implementation_guidance": "Apply Multisensory Activities with appropriate adaptations",
        },
    ]
    text = render_evidence(NODES, RELS, PROFILE, recs)

    assert "**Focus Tasks** (strategia): Short tasks with one clear goal" in text
    assert "appropriate adaptations" not in text


def test_without_a_profile_every_need_is_shown_and_no_class_profile():
    text = render_evidence(NODES, RELS, None)

    assert "joint attention" in text
    assert "Profilo della classe" not in text


def test_environment_and_udl_sections_when_retrieved():
    nodes = NODES + [
        {
            "name": "Neutral tones",
            "labels": ["Colour"],
            "description": "Neutral colours reduce overstimulation, aiding focus.",
        },
        {
            "name": "address distractions",
            "labels": ["Checkpoint"],
            "guideline": "recruit interest",
            "description": "Minimize sensory, emotional, or cognitive distractions.",
        },
    ]
    text = render_evidence(nodes, RELS, PROFILE)

    assert (
        "### Ambiente di apprendimento (nodi del KG)\n- Colour · **Neutral tones**: Neutral colours"
        in text
    )
    assert "- Checkpoint **address distractions** (linea guida: recruit interest)" in text


def test_bounded():
    many = [
        {"name": f"Strategia {i}", "labels": ["InstructionalStrategy"], "description": "x" * 300}
        for i in range(200)
    ]
    text = render_evidence(many, [], None, max_chars=2000)

    assert len(text) <= 2000
    assert text.count("\n- ") <= kg_evidence._MAX_STRATEGIES


def test_nothing_retrieved_is_nothing():
    assert render_evidence([], [], PROFILE) == ""
    assert digest({"retrieved_nodes": []}) is None


def test_digest_counts_and_text():
    state = {
        "retrieved_nodes": NODES,
        "retrieved_relationships": RELS,
        "recommendations": [{"name": "Chunking"}],
        "educational_profile": PROFILE,
    }
    kept = digest(state)

    assert kept["nodes"] == len(NODES)
    assert kept["relationships"] == len(RELS)
    assert kept["recommendations"] == 1
    assert "Difficulty sustaining focus" in kept["text"]


@pytest.mark.parametrize(
    ("profile", "needed"),
    [
        ({"group": {"disabilities": ["ADHD"]}}, True),
        ({"group": {"disabilities": ["SA"]}}, True),
        ({"group": {"disabilities": ["DLDS"], "class_features": ["ELEMENTI_DI_DISTURBO"]}}, True),
        ({"group": {"disabilities": ["DLDS"]}}, False),
        ({}, False),
        (None, False),
    ],
)
def test_environment_needed(profile, needed):
    assert environment_needed(profile) is needed
