"""The neuroscience graph, read for what its nodes are (2026-10-16).

The neuro graph says what a node is with ``category``, not with labels, and
files several neuromyths under «Teaching Practices». The nodes below are copied
from the graph (names, categories, descriptions shortened).
"""

from __future__ import annotations

import pytest

from aix.agent import kg_evidence
from aix.agent.kg_evidence import advised_against, kind, rank_nodes, render_evidence

pytestmark = pytest.mark.unit


def _node(name, category, description="", labels=None):
    return {
        "name": name,
        "category": category,
        "description": description,
        "labels": labels or ["TeachingPractices"],
    }


MICRO_CHUNKS = _node("Micro-chunks", "Teaching Practices", "Short units of content.")
STRUCTURED_ROUTINES = _node("Structured Routines", "Support Systems", "A predictable sequence.")
BRAIN_GYM = _node(
    "Pseudoscientific exercises",
    "Teaching Practices",
    "A commercial program suggesting that specific body movements boost brain integration.",
    ["BrainGym"],
)
LEARNING_STYLES = _node(
    "Modality preference myth",
    "Teaching Practices",
    "The idea that teaching students in their preferred style (visual, auditory) improves learning.",
    ["LearningStyles"],
)
MOZART = _node("Music-IQ myth", "Teaching Practices", "The Mozart effect.", ["MozartEffect"])
SUGAR = _node(
    "Sugar Effect",
    "Neuromyths",
    "The sugar-hyperactivity link is unsupported.",
    ["SugarCausesHyperactivity"],
)
FRUSTRATION = _node(
    "Difficulty Managing Frustration or Stress", "ADHD", "", ["EmotionalRegulation"]
)
CREATIVITY = _node("High Creativity", "ADHD", "", ["Creativity"])
DYSLEXIA = _node("Phonological Awareness Deficits", "Dyslexia", "", ["LanguageProcessing"])
REDUCED_ANXIETY = _node("Reduced Anxiety", "Pedagogical Strategies", "", ["AffectiveProcesses"])
DISTRACTIONS = _node("External Distractions", "External Influences", "", ["BottomUpSalience"])
RETENTION = _node(
    "Retention",
    "Not evidence-based",
    "Learning by osmosis is the false belief…",
    ["MemoryFormation"],
)

NODES = [
    RETENTION,
    MOZART,
    REDUCED_ANXIETY,
    BRAIN_GYM,
    FRUSTRATION,
    MICRO_CHUNKS,
    LEARNING_STYLES,
    DISTRACTIONS,
    SUGAR,
    CREATIVITY,
    DYSLEXIA,
    STRUCTURED_ROUTINES,
]
EDGES = [
    {"relationship": "TRIGGERS", "source": FRUSTRATION["name"], "target": "Anxiety"},
    {"relationship": "TRIGGERS", "source": FRUSTRATION["name"], "target": "Mood Swings"},
    {
        "relationship": "FACILITATES",
        "source": CREATIVITY["name"],
        "target": "Innovative Problem-solving",
    },
]
ADHD_CLASS = {"group": {"disabilities": ["ADHD"], "students_number": 22}}


@pytest.mark.parametrize(
    ("node", "expected"),
    [
        (MICRO_CHUNKS, "strategy"),
        (STRUCTURED_ROUTINES, "strategy"),
        (BRAIN_GYM, "myth"),
        (LEARNING_STYLES, "myth"),
        (SUGAR, "myth"),
        (FRUSTRATION, "need"),
        (_node("Adhd", "Neurodevelopmental Processes"), "need"),
        (REDUCED_ANXIETY, "outcome"),
        (DISTRACTIONS, "environment"),
        (_node("Educational Environment", "Teaching Practices"), "environment"),
        (RETENTION, "noise"),
    ],
)
def test_a_neuro_node_is_what_its_category_says(node, expected):
    assert kind(node) == expected


def test_a_myth_filed_as_a_teaching_practice_never_heads_the_strategies():
    ranked = [n["name"] for n in rank_nodes(NODES)]

    assert ranked[:2] == ["Micro-chunks", "Structured Routines"]
    assert ranked.index("Pseudoscientific exercises") > ranked.index("Reduced Anxiety")


def test_myths_and_their_noise_are_never_offered_to_the_writer():
    names = advised_against(NODES, EDGES, ADHD_CLASS)

    assert {"Pseudoscientific exercises", "Modality preference myth", "Sugar Effect"} <= names
    assert "Retention" in names
    assert "Micro-chunks" not in names


def test_the_evidence_says_what_an_adhd_difficulty_brings_about():
    text = render_evidence(NODES, EDGES, ADHD_CLASS)

    assert (
        "**ADHD · Difficulty Managing Frustration or Stress** → nel KG: scatena Anxiety; "
        "scatena Mood Swings" in text
    )
    assert "favorisce Innovative Problem-solving" in text, "a strength is kept"
    assert "Phonological Awareness Deficits" not in text, "dyslexia is not this class's need"


def _myth_lines(nodes):
    text = render_evidence(nodes, EDGES, ADHD_CLASS)
    section = text.split("### Da evitare: neuromiti (nodi del KG)\n", 1)[1].split("\n\n", 1)[0]
    return section.splitlines()


def test_only_the_myths_a_lesson_for_this_class_could_slip_into_are_shown():
    many = NODES + [
        _node(f"Myth {i}", "Neuromyths", "Unrelated claim.", ["EducationalMyths"])
        for i in range(10)
    ]
    lines = _myth_lines(many)

    assert lines[0].startswith("- **Modality preference myth**"), "learning styles, in any class"
    assert {line.split("**")[1] for line in lines[1:]} == {
        "Sugar Effect",
        "Pseudoscientific exercises",
    }, "and, with ADHD, sugar and Brain Gym; not Mozart, not the unrelated ones"


def test_the_myths_shown_are_bounded():
    many = NODES + [
        _node(f"Attention myth {i}", "Neuromyths", "About attention.", ["EducationalMyths"])
        for i in range(10)
    ]

    assert len(_myth_lines(many)) == kg_evidence._MAX_MYTHS


def test_effects_are_reasons_not_strategies():
    text = render_evidence(NODES, EDGES, ADHD_CLASS)
    strategies = text.split("### Strategie e strumenti recuperati\n", 1)[1].split("\n\n", 1)[0]

    assert "Reduced Anxiety" not in strategies
    assert "### Effetti attesi secondo il KG (motivazioni, non attività)\n- Reduced Anxiety" in text
    assert "Pseudoscientific exercises" not in strategies


def test_the_room_from_the_neuro_graph_is_the_environment_section():
    text = render_evidence(NODES, EDGES, ADHD_CLASS)

    assert (
        "### Ambiente di apprendimento (nodi del KG)\n- Ambiente · **External Distractions**"
        in text
    )


def test_the_profile_codes_name_the_needs_of_both_graphs():
    assert kg_evidence.class_need_groups({"group": {"disabilities": ["ADHD", "CL", "DSA"]}}) == {
        "ADHD",
        "Competenze linguistiche",
        "Dislessia",
        "Discalculia",
    }


def test_a_class_profile_node_reaches_the_teacher_in_italian():
    """«Group till 30» reached the teacher verbatim in 3 of 8 evaluated lessons."""
    from aix.agent.kg_evidence import context_label

    assert context_label({"name": "Group till 30"}) == "Classe fino a 30 alunni"
    assert context_label({"name": "Split in groups"}) == "Classe divisa in gruppi"
    assert context_label({"name": "Something else"}) == "Something else"
