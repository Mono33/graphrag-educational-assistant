"""A class profile as the AIxLearning platform stores it today (2026-10-16).

The schema was a 1:1 port of the platform's models, which then moved on: the BES
codes are now eight (``SC`` and ``CL`` replaced the cognitive and language codes),
the classroom questions are YES / NO / UNKNOWN, a class can have 0 students
(nobody entered the size), and the classroom says whether the lesson is in
presence, remote or hybrid. Each of these used to be a 422 for the platform, or
worse, a constraint invented for the model. See
``docs/product/Analisi_Integrazione_AIxLearning.md`` §3 A–C.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from tests.unit.test_api_sessions import _as, api  # noqa: F401 — the fixture

from aix.agent.prompts.profile_renderer import render_classroom_constraints, render_profile_section
from aix.api.schemas.educational_profile import (
    DISABILITY_CHOICES,
    ClassroomEnvironment,
    EducationalGroup,
    EducationalProfile,
)

pytestmark = pytest.mark.unit

#: What the platform builds from a ``Party`` and a ``Classroom`` it has today.
PLATFORM_PROFILE = {
    "group": {
        "title": "2B",
        "students_number": 0,
        "grade": "SECONDARIA_I_GRADO",
        "disabilities": ["SC", "CL", "ADHD"],
        "class_features": ["COESA"],
    },
    "classroom": {
        "title": "Laboratorio",
        "forniture_mobility": "none",
        "has_lim": "YES",
        "has_wifi": "UNKNOWN",
        "has_suite": None,
        "pc_station": "NO",
        "own_device": "UNKNOWN",
        "environment_type": "HYBRID",
        "participation_type": "ASYNC",
        "prevalent_modality_type": "MOSTLY_LOCAL",
    },
}


def test_the_platform_s_profile_validates():
    profile = EducationalProfile.model_validate(PLATFORM_PROFILE)

    assert [d.value for d in profile.group.disabilities] == ["SC", "CL", "ADHD"]
    assert profile.group.students_number is None
    room = profile.classroom
    assert (room.has_lim, room.has_wifi, room.pc_station) == (True, None, False)
    assert (room.forniture_mobility, room.own_device) == (None, None)
    assert room.environment_type.value == "HYBRID"


def test_unknown_is_never_turned_into_a_prohibition():
    lines = render_classroom_constraints(EducationalProfile.model_validate(PLATFORM_PROFILE))
    text = "\n".join(lines)

    assert "interactive whiteboard (LIM) is available" in text
    assert "NO PC workstations" in text
    # Not said: no device rule, no furniture rule, no WiFi rule, no class size.
    assert "personal devices" not in text
    assert "furniture" not in text
    assert "WiFi" not in text
    assert "students:" not in text


def test_a_hybrid_asynchronous_class_is_told_to_the_model():
    text = "\n".join(render_classroom_constraints(PLATFORM_PROFILE))

    assert "HYBRID" in text and "Most students are in the room" in text
    assert "ASYNCHRONOUSLY" in text


def test_the_new_codes_reach_the_prompt_as_words():
    section = render_profile_section(EducationalProfile.model_validate(PLATFORM_PROFILE))

    assert "Elaborazione cognitiva" in section
    assert "Competenze linguistiche" in section
    assert "SC;" not in section and "; CL" not in section
    assert "Class size" not in section


def test_profiles_saved_with_the_old_codes_still_validate():
    group = EducationalGroup.model_validate({"disabilities": ["DCGM", "DLDS"]})

    assert [d.value for d in group.disabilities] == ["DCGM", "DLDS"]
    assert "DCGM" not in DISABILITY_CHOICES  # accepted, no longer offered


def test_a_blank_classroom_question_no_longer_defaults_to_no():
    """The WebUI goes through the same model: a classroom with only a LIM used to
    come back with ``own_device=NO`` and ``forniture_mobility=NO``."""
    room = ClassroomEnvironment(has_lim=True).model_dump(exclude_none=True)

    assert room == {"has_lim": True}
    assert EducationalGroup(grade="PRIMARIA").students_number is None


def test_a_value_the_platform_does_not_have_is_still_refused():
    with pytest.raises(ValidationError):
        EducationalGroup.model_validate({"disabilities": ["XYZ"]})
    with pytest.raises(ValidationError):
        ClassroomEnvironment.model_validate({"has_lim": "FORSE"})


def test_the_api_takes_the_platform_s_profile_and_keeps_only_what_was_said(api):  # noqa: F811
    response = api.post(
        "/api/v1/sessions",
        json={"domain": "udl", "educational_profile": PLATFORM_PROFILE},
        headers=_as(),
    )

    assert response.status_code == 201, response.text
    detail = api.get(f"/api/v1/sessions/{response.json()['session_id']}", headers=_as()).json()
    room = detail["educational_profile"]["classroom"]
    assert room["has_lim"] is True and room["pc_station"] is False
    assert "has_wifi" not in room and "own_device" not in room
    assert "students_number" not in detail["educational_profile"]["group"]


def test_the_catalog_offers_the_platform_s_codes(api):  # noqa: F811
    profile = api.get("/api/v1/catalog", headers=_as()).json()["profile"]

    assert list(profile["disabilities"]) == ["DSA", "ADHD", "DOP", "DF", "SC", "PD", "SA", "CL"]
    assert set(profile["disabilities_legacy"]) == {"DCGL", "DCGM", "DCGS", "DLDS"}
    assert set(profile["environment_type"]) == {"LOCAL", "REMOTE", "HYBRID"}
    assert profile["three_state"] == ["YES", "NO", "UNKNOWN"]
