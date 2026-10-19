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

    # A strategy node first; one with no node after it, however relevant.
    assert names == ["Cooperative Learning", "Chunking", "Motivated", "Group till  15"]


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
        "**Classe fino a 30 alunni** (profilo della classe) → suggerisce: Cooperative Learning; sconsiglia: Station Rotation"
        in text
    )
    assert "**Classe divisa in gruppi**" in text
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


def test_advised_against_for_this_class():
    from aix.agent.kg_evidence import advised_against

    assert advised_against(NODES, RELS, PROFILE) == {"Long Frontal Lesson", "Station Rotation"}
    # Without a profile the class-size band is unknown; a need named in the
    # request still speaks, and with none named nothing is advised against.
    request = "Lezione per una classe con 2 bambini con ADHD"
    assert advised_against(NODES, RELS, None, request) == {"Long Frontal Lesson"}
    assert advised_against(NODES, RELS, None) == set()


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


def test_a_class_with_no_need_declared_or_named_gets_none():
    """Until 2026-10-19 every need retrieved was shown: the follow-up of a class
    with none declared listed ADHD, autism and non-native speakers as its own
    (``scripts/eval/quality_ab.py``, case 4)."""
    text = render_evidence(NODES, RELS, None)

    assert "Bisogni della classe" not in text
    assert "joint attention" not in text
    assert "Profilo della classe" not in text


def test_a_need_named_only_in_the_request_is_the_class_s():
    """The WebUI profile form is optional: teachers write the needs in the text."""
    text = render_evidence(
        NODES, RELS, None, request="Ci sono 2 bambini con adhd e 2 non italofoni"
    )

    assert "**ADHD ·" in text and "**Alunni non italofoni ·" in text
    assert "joint attention" not in text, "autism was not named"


@pytest.mark.parametrize(
    ("request_text", "expected"),
    [
        ("due alunni con ADHD", {"ADHD"}),
        ("alunni stranieri appena arrivati", {"Alunni non italofoni"}),
        ("madre lingua spagnola", {"Alunni non italofoni"}),
        ("madrelingua italiana", set()),
        ("un alunno con DSA", {"Dislessia", "Discalculia"}),
        ("una ragazza con sindrome di Asperger", {"Spettro autistico"}),
        ("uno studente ad alto potenziale", {"Plusdotazione"}),
        ("la rivoluzione francese per una terza", set()),
    ],
)
def test_needs_named_in_a_request(request_text, expected):
    from aix.agent.kg_evidence import class_need_groups

    assert class_need_groups(None, request_text) == expected


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


def test_the_graph_s_typos_never_reach_the_writer():
    """«Gammification» reached two of eight evaluated lessons (2026-10-16)."""
    nodes = [
        {"name": "Gammification", "labels": ["PedagogicalMethodology"], "description": "Poject"}
    ]

    text = render_evidence(nodes, [], None)

    assert "Gamification" in text and "Gammification" not in text
    assert "Project" in text and "Poject" not in text


def test_the_lesson_s_use_of_the_graph_is_computed_not_asked():
    """The follow-up's counts did not match what it listed (spec §3)."""
    from aix.agent.kg_evidence import render_usage, usage

    lesson = (
        "## Attività 1\nCompiti brevi (Focus Tasks) con un solo obiettivo.\n"
        "## Attività 2\nApprendimento cooperativo (Cooperative Learning) a gruppi di 4."
    )
    found = usage(NODES, RELS, PROFILE, lesson)

    assert set(found["used"]) >= {"Focus Tasks", "Cooperative Learning"}
    assert "Station Rotation" in found["advised_against_avoided"]
    assert found["advised_against_used"] == []
    assert any(item.startswith("Glossaries") for item in found["unused_for_needs"])
    text = render_usage(found)
    assert f"Usati nella lezione ({len(found['used'])})" in text
    assert "ATTENZIONE" not in text


def test_a_strategy_the_graph_advises_against_is_flagged_when_used():
    from aix.agent.kg_evidence import render_usage, usage

    found = usage(NODES, RELS, PROFILE, "Rotazione tra postazioni (Station Rotation).")

    assert found["advised_against_used"] == ["Station Rotation"]
    assert "ATTENZIONE, sconsigliati dal grafo ma presenti nella lezione: Station Rotation" in (
        render_usage(found)
    )


def test_a_station_rotation_under_an_italian_name_is_still_flagged():
    """«Le fasi a stazioni»: groups rotating over four cartellini-stazione (2026-10-19)."""
    from aix.agent.kg_evidence import usage

    lesson = "**Attività 1 – Le fasi a stazioni**\nOgni gruppo ruota su 4 cartellini-stazione."
    found = usage(NODES, RELS, PROFILE, lesson)

    assert found["advised_against_used"] == ["Station Rotation"]


def test_what_the_graph_advises_against_says_what_it_looks_like():
    text = render_evidence(NODES, RELS, PROFILE)

    assert "Station Rotation (gruppi che ruotano tra stazioni o postazioni)" in text


@pytest.mark.parametrize(
    ("request_text", "expected"),
    [
        ("un alunno con DSA (discalculia)", {"Discalculia"}),
        ("un alunno con DSA, dislessico", {"Dislessia"}),
        ("un alunno con DSA", {"Dislessia", "Discalculia"}),
    ],
)
def test_dsa_is_the_specific_disorder_the_request_names(request_text, expected):
    from aix.agent.kg_evidence import class_need_groups

    assert class_need_groups({"group": {"disabilities": ["DSA"]}}, request_text) == expected


@pytest.mark.parametrize(
    "lesson",
    [
        # Every group at its own table: the title says «a stazioni», nothing rotates.
        "**Attività 2 — Caccia alle fasi a stazioni**\nOgni gruppo riordina le carte al suo banco.",
        # Named only to say it was left out, in the sources.
        "## Attività\nLavoro a coppie.\n\n*Fonti:*\n- Station Rotation (non applicata, esclusa).",
    ],
)
def test_no_false_alarm_for_station_rotation(lesson):
    from aix.agent.kg_evidence import usage

    assert usage(NODES, RELS, PROFILE, lesson)["advised_against_used"] == []


@pytest.mark.parametrize(
    "lesson",
    [
        # «Stem» inside «system» of the AI-disclosure comment.
        "<!-- ai-generated: true, system: agentic-graphrag -->\n# Lezione\nLavoro a coppie.",
        # A video's title is not the method being applied.
        "# Lezione\n- Video: [What is Service Learning?](https://youtu.be/x) per la docente.",
    ],
)
def test_a_node_is_used_only_when_the_lesson_applies_it(lesson):
    """Second blind read: «Stem» and «Service Learning» declared used, used nowhere."""
    from aix.agent.kg_evidence import usage

    nodes = NODES + [
        {"name": "Stem", "labels": ["PedagogicalMethodology"]},
        {"name": "Service Learning", "labels": ["PedagogicalMethodology"]},
    ]
    used = usage(nodes, RELS, PROFILE, lesson)["used"]

    assert "Stem" not in used and "Service Learning" not in used


def test_the_rest_is_what_the_follow_up_does_not_list():
    """«Il resto: 55» of 56, with 13 items listed: the counts must add up."""
    from aix.agent.kg_evidence import usage

    found = usage(NODES, RELS, PROFILE, "## Attività\nCompiti brevi (Focus Tasks).")
    listed = {
        name.split(" (")[0]
        for key, names in found.items()
        if key != "needs" and isinstance(names, list)
        for name in names
    }

    assert found["other"] + len(listed) == len({n["name"] for n in NODES})


def test_the_follow_up_gets_where_each_strategy_is_and_its_unused_already_chosen():
    """53 returned, 2 used, 5 unused the follow-up chose itself, «il resto 51»;
    and Adaptive Strategies listed as unused while the lesson names it (2026-10-19)."""
    from aix.agent.kg_evidence import render_usage, usage

    lesson = (
        "# Lezione\n\n**Attività 1 – Compiti brevi (15 min)**\n- Compiti brevi (Focus Tasks).\n"
        "\n## Differenziazione\n- Gruppi (Cooperative Learning)."
    )
    found = usage(NODES, RELS, {}, lesson)
    text = render_usage(found)

    assert found["where"]["Focus Tasks"] == "Attività 1 – Compiti brevi"
    assert found["where"]["Cooperative Learning"] == "Differenziazione"
    assert found["unused_other"] and not set(found["unused_other"]) & set(found["used"])
    shown = len(found["unused_for_needs"]) + len(found["unused_other"])
    assert 0 < shown <= 5 and f"Non usati, tra i più pertinenti ({shown};" in text
    assert "Focus Tasks [in «Attività 1 – Compiti brevi»]" in text
    assert f"Elementi restituiti dal grafo: {found['total']}" in text


@pytest.mark.parametrize(
    "sources",
    [
        "*Fonti: Focus Tasks (Strategie); Cooperative Learning.*",
        "## Risorse di approfondimento per il docente\n- Focus Tasks; Cooperative Learning.",
        "**Fonti:**\n- Focus Tasks; Cooperative Learning.",
    ],
)
def test_what_the_sources_list_is_not_used_by_the_lesson(sources):
    """«*Fonti: Cooperative learning (…)*» on one line, or a section of resources,
    was read as the body: every node it listed counted as used (2026-10-19)."""
    from aix.agent.kg_evidence import lesson_body, usage

    lesson = f"# Lezione\n\n## Attività (20 minuti)\n- Fonti di energia: il sole.\n\n{sources}\n"

    assert usage(NODES, RELS, PROFILE, lesson)["used"] == []
    assert "Fonti di energia" in lesson_body(lesson)


def test_a_suggestion_the_lesson_labels_is_used_whatever_kind_of_node_it_is():
    """Adaptive Strategies and Awareness of Errors are not «strategy» nodes: the
    lesson labelled them, and the follow-up listed them as unused (2026-10-19)."""
    from aix.agent.kg_evidence import usage

    nodes = NODES + [{"name": "Adaptive Strategies", "labels": ["Outcome"], "rank_score": 1}]
    rels = RELS + [
        {
            "source": "Difficulty sustaining focus",
            "type": "SUGGESTS",
            "target": "Adaptive Strategies",
        }
    ]
    lesson = "## Differenziazione\n- La scheda divisa in passi (Adaptive Strategies)."

    found = usage(nodes, rels, PROFILE, lesson)

    assert "Adaptive Strategies" in found["used"]
    assert not any(u.startswith("Adaptive Strategies") for u in found["unused_for_needs"])
    assert found["where"]["Adaptive Strategies"] == "Differenziazione"
