"""References that cannot back a lesson leave before the Writer (2026-10-19).

Titles in the shape of those the evaluation's lessons cited (rodents, ChatGPT,
college students in a primary lesson).
"""

from __future__ import annotations

import pytest

from aix.agent.reference_filter import keep_references, off_topic

pytestmark = pytest.mark.unit

PRIMARY = {"group": {"grade": "PRIMARIA"}}
UNIVERSITY = {"group": {"grade": "UNIVERSITA"}}


@pytest.mark.parametrize(
    "title",
    [
        "Object recognition memory in rodents: a review",
        "Can ChatGPT support lesson planning?",
        "A Conceptual Framework for Assessing Motivation in College Students",
        "Universal Design for Learning in postsecondary STEM education",
    ],
)
def test_a_reference_about_something_else_is_dropped_for_a_school_class(title):
    assert off_topic({"title": title}, PRIMARY)


def test_a_school_reference_stays():
    assert off_topic({"title": "Scaffolding in Teacher-Student Interaction"}, PRIMARY) is None


def test_studies_of_university_students_stay_for_a_university_class():
    title = "A Conceptual Framework for Assessing Motivation in College Students"

    assert off_topic({"title": title}, UNIVERSITY) is None
    assert off_topic({"title": "Recognition memory in rats"}, UNIVERSITY) == "not about pupils"


def test_keep_references_filters_a_list():
    refs = [{"title": "Working memory in children"}, {"title": "Spatial learning in mice"}]

    assert keep_references(refs, PRIMARY) == [{"title": "Working memory in children"}]
