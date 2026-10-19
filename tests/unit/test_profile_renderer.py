"""The educational profile as the Writer reads it (Q1 · 1.1).

The reference profile below is the classroom that made the problem visible:
students without devices, one LIM, no WiFi, fixed desks, twenty-five students,
and three learner needs whose acronyms the model used to misread. Each group of
tests pins one of the two failures the renderer exists to end.

**The Writer was never shown the classroom.** Only five fields reached the
prompt, so "no personal devices" was not ignored — it was absent.

**Learner needs arrived as raw codes.** ``DCGM`` came back as dyspraxia and
``DLDS`` as a language disorder, because the model expanded the acronyms from
memory. The Italian labels already existed; nothing on the agent path used them.
"""

from __future__ import annotations

import pytest

from aix.agent.prompts.profile_renderer import (
    CONSTRAINTS_HEADER,
    PROFILE_HEADER,
    render_classroom_constraints,
    render_profile_section,
)
from aix.api.schemas.educational_profile import EducationalProfile

pytestmark = pytest.mark.unit

#: The classroom in which four activities out of five could not be run.
REFERENCE = {
    "group": {
        "students_number": 25,
        "grade": "SECONDARIA_II_GRADO",
        "disabilities": ["ADHD", "DCGM", "DLDS"],
        "class_features": ["DIVISA_IN_GRUPPI", "ELEMENTI_DI_DISTURBO"],
        "student_attributes": ["TIMIDEZZA_O_CHIUSURA"],
    },
    "classroom": {
        "forniture_mobility": "NO",
        "has_lim": True,
        "has_wifi": False,
        "has_suite": False,
        "pc_station": False,
        "own_device": "NO",
    },
    "time_available_minutes": 120,
    "subject_area": "Scienze",
    "specific_topic": "Memoria di lavoro",
}


@pytest.fixture
def section() -> str:
    return render_profile_section(REFERENCE)


# ---------------------------------------------------------------------------
# The classroom reaches the model
# ---------------------------------------------------------------------------


def test_no_personal_devices_is_a_stated_constraint(section):
    assert "do NOT have personal devices" in section
    assert "No activity may require a student to use one" in section


def test_the_lim_is_named_as_the_shared_device(section):
    assert "interactive whiteboard (LIM) is available" in section


def test_no_wifi_forbids_activities_that_need_internet(section):
    assert "NO WiFi" in section


def test_fixed_furniture_forbids_rearranging_the_room(section):
    assert "CANNOT be moved" in section


def test_class_size_bounds_the_groups(section):
    """The jigsaw built for eight, in a class of twenty-five."""
    assert "Class size: 25 students" in section
    assert "cooperative groups must not exceed 5" in section


def test_the_grouping_comes_worked_out_and_is_applied_silently(section):
    """22 pupils on staging (2026-10-16): «4 gruppi: 3 da 5 e 1 da 7 non ammesso» in the lesson."""
    assert "A grouping that works: 5 groups of 5." in section
    assert "never write the arithmetic" in section


@pytest.mark.parametrize(
    ("students", "split"),
    [
        (22, "2 groups of 5 and 3 groups of 4"),
        (25, "5 groups of 5"),
        (23, "3 groups of 5 and 2 groups of 4"),
        (6, "2 groups of 3"),
        (11, "2 groups of 4 and 1 group of 3"),
        (5, ""),
        (1, ""),
    ],
)
def test_group_split(students, split):
    from aix.agent.prompts.profile_renderer import group_split

    assert group_split(students) == split


def test_constraints_are_imperative_not_key_value(section):
    assert CONSTRAINTS_HEADER in section
    assert "own_device" not in section
    assert "has_wifi" not in section


# ---------------------------------------------------------------------------
# No acronym reaches the model unexpanded
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", ["DCGM", "DLDS"])
def test_ambiguous_bes_codes_never_reach_the_model(section, code):
    """The two the model reliably misread."""
    assert code not in section


def test_learner_needs_carry_their_italian_labels(section):
    assert "Disabilità cognitiva media" in section
    assert "Difficoltà linguistiche (straniero/a)" in section
    assert "Disturbo dell'attenzione (ADHD)" in section


def test_grade_and_class_codes_are_resolved_too(section):
    assert "Secondaria di secondo grado" in section
    assert "SECONDARIA_II_GRADO" not in section
    assert "Divisa in gruppi" in section
    assert "Timidezza / chiusura relazionale" in section


def test_an_unknown_code_is_kept_rather_than_dropped():
    """Something the teacher declared must not vanish because a dictionary lags."""
    section = render_profile_section({"group": {"disabilities": ["NUOVO_CODICE"]}})
    assert "NUOVO_CODICE" in section


# ---------------------------------------------------------------------------
# What the system prompt depends on
# ---------------------------------------------------------------------------


def test_the_headings_the_system_prompt_anchors_to_are_kept(section):
    """``writer_prompt.py`` names both, as the anchor of the hard duration rule."""
    assert PROFILE_HEADER == "## Teacher's Educational Profile"
    assert PROFILE_HEADER in section
    assert "- Duration: 120 minutes" in section


def test_the_section_is_framed_by_newlines_for_the_template(section):
    assert section.startswith("\n")
    assert section.endswith("\n")


# ---------------------------------------------------------------------------
# Unknown is not no
# ---------------------------------------------------------------------------


def test_an_unanswered_field_produces_no_constraint():
    """A half-filled profile must not forbid half the classroom."""
    lines = render_classroom_constraints({"classroom": {"has_lim": None, "has_wifi": None}})

    assert not any("LIM" in line for line in lines)
    assert not any("WiFi" in line for line in lines)


def test_an_explicit_false_does():
    lines = render_classroom_constraints({"classroom": {"has_lim": False}})
    assert any("NO interactive whiteboard" in line for line in lines)


@pytest.mark.parametrize(
    "device, expected",
    [
        ("NO", "do NOT have personal devices"),
        ("BES", "Only students with special educational needs"),
        ("YES", "may use their own devices"),
    ],
)
def test_each_device_policy_says_something_different(device, expected):
    lines = render_classroom_constraints({"classroom": {"own_device": device}})
    assert any(expected in line for line in lines)


# ---------------------------------------------------------------------------
# Shapes it has to accept
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("empty", [None, {}, "not a profile"])
def test_no_profile_means_no_section(empty):
    assert render_profile_section(empty) == ""


def test_the_pydantic_model_renders_the_same_as_its_dict():
    model = EducationalProfile.model_validate(REFERENCE)
    assert render_profile_section(model) == render_profile_section(REFERENCE)


def test_legacy_top_level_keys_still_work():
    section = render_profile_section({"disabilities": ["DSA"], "lesson_duration": 45})

    assert "Disturbi specifici dell'apprendimento (DSA)" in section
    assert "- Duration: 45 minutes" in section


def test_keys_the_renderer_does_not_know_are_ignored():
    """Stored profiles carry ``__refinement__`` and similar; none of it is a fact."""
    section = render_profile_section({"subject_area": "Scienze", "__refinement__": "accorcia"})

    assert "accorcia" not in section
    assert "Subject: Scienze" in section


@pytest.mark.parametrize(
    ("duration", "parts"),
    [(60, (10, 40, 10)), (45, (5, 35, 5)), (90, (15, 60, 15)), (120, (20, 80, 20))],
)
def test_the_duration_comes_with_a_split_that_adds_up(duration, parts):
    """Every lesson regenerated on 2026-10-16 came out 5-7 minutes short of 60."""
    from aix.agent.prompts.profile_renderer import time_plan

    plan = time_plan(duration)
    intro, activities, closing = parts

    assert intro + activities + closing == duration
    assert (
        f"Introduzione {intro} minuti + Attività {activities} minuti + Valutazione {closing} minuti = {duration} minuti"
        in plan
    )


def test_a_short_or_missing_duration_has_no_split():
    from aix.agent.prompts.profile_renderer import time_plan

    assert time_plan(15) == "" and time_plan(None) == ""


def test_a_section_timed_with_a_prime_is_counted():
    """«## Introduzione (10′)»: the notation a Writer copied from the time plan."""
    from aix.agent.validation.rules import section_minutes

    lesson = "# L\n\n## Introduzione (10′)\n\n## Attività (40′)\n\n## Valutazione (10 minuti)\n"
    assert [m for _t, m in section_minutes(lesson)] == [10, 40, 10]


@pytest.mark.parametrize(
    ("profile", "levels"),
    [
        (
            {"pedagogical_intent": "intro", "group": {"grade": "PRIMARIA"}},
            ("ricordare", "comprendere", "applicare"),
        ),
        (
            {"pedagogical_intent": "deepening", "group": {"grade": "SECONDARIA_II_GRADO"}},
            ("applicare", "analizzare", "valutare"),
        ),
        # A deepening in primary school still stops at «applicare».
        ({"pedagogical_intent": "deepening", "group": {"grade": "PRIMARIA"}}, ("applicare",)),
        ({"group": {"grade": "SECONDARIA_I_GRADO"}}, ("comprendere", "applicare", "analizzare")),
        (
            {"pedagogical_intent": "misconception: l'acqua sparisce"},
            ("comprendere", "analizzare", "valutare"),
        ),
        ({}, ()),
    ],
)
def test_the_objectives_climb_the_bloom_levels_this_lesson_needs(profile, levels):
    from aix.agent.prompts.profile_renderer import bloom_levels

    assert bloom_levels(profile) == levels


def test_the_bloom_levels_reach_the_writer_and_are_not_a_fixed_list():
    section = render_profile_section(
        {
            "pedagogical_intent": "intro",
            "group": {"grade": "PRIMARIA"},
            "time_available_minutes": 60,
        }
    )
    deeper = render_profile_section(
        {"pedagogical_intent": "deepening", "group": {"grade": "SECONDARIA_II_GRADO"}}
    )

    assert "levels of Bloom's taxonomy, lowest first: ricordare, comprendere, applicare" in section
    assert "applicare, analizzare, valutare" in deeper
