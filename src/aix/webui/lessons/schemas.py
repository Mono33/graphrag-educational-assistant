"""
Form ↔ EducationalProfile coercion (CORE 2 #6.6 P1).

The HTML form posts a flat ``FormData`` object: nested keys are not natively
supported by ``application/x-www-form-urlencoded``. We use a flat naming
convention with subform prefixes:

    title                                 (lesson title — top-level on Lesson row)
    domain                                ("neuro" / "udl" / "all")
    group_title, group_students_number, group_grade,
    group_disabilities[]                  (multi-select, repeated key)
    group_class_features[]                (multi-select, repeated key)
    group_student_attributes[]            (multi-select, repeated key)
    classroom_title, classroom_forniture_mobility, classroom_own_device,
    classroom_has_lim, classroom_has_wifi, classroom_has_suite,
    classroom_pc_station                  (checkboxes — present="on", absent=False)
    time_available_minutes, subject_area, specific_topic

``form_to_profile_dict`` validates the input by constructing a real
``EducationalProfile`` (Pydantic enforces enum membership and the int range
on ``time_available_minutes``), then ``model_dump`` returns a JSON-serializable
dict ready for the ``Lesson.educational_profile_json`` column.

Validation errors are surfaced as Pydantic ``ValidationError``; the route
handler catches that and re-renders the form with friendly Italian messages.
"""

from __future__ import annotations

from typing import Any, Optional

from starlette.datastructures import FormData

from aix.api.schemas.educational_profile import (
    ClassroomEnvironment,
    EducationalGroup,
    EducationalProfile,
)


_TRUTHY_FORM = {"on", "true", "1", "yes"}


def _str_or_none(value: Optional[str]) -> Optional[str]:
    """Empty string → None so Pydantic treats the field as 'omitted'."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _checkbox(form: FormData, key: str) -> Optional[bool]:
    """
    HTML checkbox idiom: the field is only present when checked. To support a
    tri-state ("yes" / "no" / "unknown") on the classroom flags, we accept a
    paired hidden field ``<key>_present`` that the template emits to signal
    "the user saw this question". When ``<key>_present`` is missing entirely
    we return ``None`` (unset). When it's set, ``<key>`` controls the bool.
    """
    if f"{key}_present" not in form:
        return None
    raw = form.get(key)
    if raw is None:
        return False
    return str(raw).lower() in _TRUTHY_FORM


def _int_or_none(value: Optional[str]) -> Optional[int]:
    """Parse an integer string, returning None for empty/invalid input."""
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def form_to_profile_dict(form: FormData) -> dict[str, Any]:
    """
    Build a validated EducationalProfile dict from a posted HTML form.

    Returns the result of ``EducationalProfile.model_dump(exclude_none=True)``
    so we don't persist explicit ``null`` for fields the teacher left blank.

    Raises ``pydantic.ValidationError`` when any field violates its enum or
    range constraint.
    """
    # Drop fields the user left blank so Pydantic uses its own defaults
    # (or leaves the sub-model out entirely when nothing was provided).
    group_kwargs: dict[str, Any] = {
        "title": _str_or_none(form.get("group_title")),
        "students_number": _int_or_none(form.get("group_students_number")),
        "grade": _str_or_none(form.get("group_grade")),
        "disabilities": [v for v in form.getlist("group_disabilities") if v],
        "class_features": [v for v in form.getlist("group_class_features") if v],
        "student_attributes": [v for v in form.getlist("group_student_attributes") if v],
    }
    group_kwargs = {k: v for k, v in group_kwargs.items() if v not in (None, [])}

    classroom_kwargs: dict[str, Any] = {
        "title": _str_or_none(form.get("classroom_title")),
        "forniture_mobility": _str_or_none(form.get("classroom_forniture_mobility")),
        "has_lim": _checkbox(form, "classroom_has_lim"),
        "has_wifi": _checkbox(form, "classroom_has_wifi"),
        "has_suite": _checkbox(form, "classroom_has_suite"),
        "pc_station": _checkbox(form, "classroom_pc_station"),
        "own_device": _str_or_none(form.get("classroom_own_device")),
    }
    classroom_kwargs = {k: v for k, v in classroom_kwargs.items() if v is not None}

    profile_kwargs: dict[str, Any] = {
        "time_available_minutes": _int_or_none(form.get("time_available_minutes")),
        "subject_area": _str_or_none(form.get("subject_area")),
        "specific_topic": _str_or_none(form.get("specific_topic")),
    }
    profile_kwargs = {k: v for k, v in profile_kwargs.items() if v is not None}

    # Only attach the sub-models when at least one field was provided. An
    # empty group / classroom would otherwise persist as ``{}`` in JSON,
    # which is harmless but noisy.
    if group_kwargs:
        profile_kwargs["group"] = EducationalGroup(**group_kwargs)
    if classroom_kwargs:
        profile_kwargs["classroom"] = ClassroomEnvironment(**classroom_kwargs)

    profile = EducationalProfile(**profile_kwargs)
    return profile.model_dump(mode="json", exclude_none=True)
