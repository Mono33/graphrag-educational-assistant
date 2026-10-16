"""The teacher's educational profile, as the model actually needs to read it (Q1 · 1.1).

The Writer used to build this section itself, from five fields: topic, subject,
grade, learner needs and duration. Everything else the form collects — how many
students there are, whether they have their own devices, whether there is a LIM,
WiFi, PC stations, whether the desks can be moved, what the class is like —
reached the database, the sidebar and the teacher's eyes, and never the model.

So a lesson could ask for Kahoot on students' phones in a class whose profile says
they have none. The model was not ignoring the constraint; it had never been shown
it. The same omission put an eight-student jigsaw into a class of twenty-five.

The learner needs had a second problem. They were joined as raw codes —
``ADHD, DCGM, DLDS`` — and the model re-expanded the acronyms from memory: DCGM
became dyspraxia instead of a moderate cognitive disability, DLDS a language
disorder instead of a student learning Italian as a second language. The
Italian labels that fix this already existed, in ``educational_profile.py``;
they were only ever used by the templates.

This module renders both: every field the profile carries, with codes resolved
to the labels the teacher chose from, and the classroom facts restated as
**constraints** — imperative sentences rather than key/value pairs, because a
model weighs "Students do NOT have personal devices" very differently from
``own_device: NO``.

Two rules keep it honest:

* **Unknown is not no.** ``has_lim=None`` means the teacher did not say, and
  produces no line at all. Only an explicit ``False`` becomes "not available";
  otherwise a profile left half-filled would forbid half the classroom.
* **The headings the system prompt names stay put.** ``writer_prompt.py`` refers
  to "the Teacher's Educational Profile" and its **Duration** by name, as the
  anchor of the hard duration constraint. Renaming either would quietly detach
  the rule from the value it governs.
"""

from __future__ import annotations

from typing import Any, Optional

from aix.api.schemas.educational_profile import (
    CLASS_FEATURE_LABELS,
    DISABILITY_LABELS,
    GRADE_LABELS,
    STUDENT_ATTR_LABELS,
)

#: Above this, cooperative groups stop being cooperative: the strategy document's
#: rule, and the fix for the jigsaw that assumed a class of eight.
MAX_GROUP_SIZE = 5

#: The header ``writer_prompt.py`` anchors its duration rule to. Do not rename.
PROFILE_HEADER = "## Teacher's Educational Profile"
CONSTRAINTS_HEADER = "## Classroom Constraints (NON-NEGOTIABLE)"


def _as_dict(profile: Any) -> dict[str, Any]:
    """Accept the stored dict or the Pydantic model, and nothing else."""
    if profile is None:
        return {}
    if hasattr(profile, "model_dump"):
        return profile.model_dump(mode="json")
    return profile if isinstance(profile, dict) else {}


def _label(code: Any, labels: dict[str, str]) -> Optional[str]:
    """The teacher-facing label for a code, or the code itself if it has none.

    Falling back to the code rather than dropping it: a value the dictionaries do
    not know yet is still something the teacher declared, and losing it silently
    is worse than showing it unexpanded.
    """
    if code is None:
        return None
    raw = getattr(code, "value", code)
    text = str(raw).strip()
    if not text:
        return None
    return labels.get(text, text)


def _labels(codes: Any, labels: dict[str, str]) -> list[str]:
    if not codes:
        return []
    if not isinstance(codes, (list, tuple)):
        codes = [codes]
    return [label for code in codes if (label := _label(code, labels))]


def _code(value: Any) -> str:
    """An enum or its stored string, as the bare code (``""`` when unset)."""
    return str(getattr(value, "value", value) or "")


def render_classroom_constraints(profile: Any) -> list[str]:
    """The classroom facts as constraint sentences, one per line.

    Public on its own because the Planner needs them too: a constraint that only
    reaches the Writer lets the Planner retrieve evidence for activities the
    classroom cannot run, and the Writer then has nothing better to write from.
    """
    ep = _as_dict(profile)
    group = ep.get("group") or {}
    room = ep.get("classroom") or {}
    lines: list[str] = []

    device = str(getattr(room.get("own_device"), "value", room.get("own_device")) or "")
    if device == "NO":
        lines.append(
            "Students do NOT have personal devices (smartphones, tablets, laptops). "
            "No activity may require a student to use one."
        )
    elif device == "BES":
        lines.append(
            "Only students with special educational needs may use a personal device, "
            "as a compensatory tool. Activities for the whole class must not require one."
        )
    elif device == "YES":
        lines.append("Students may use their own devices during the lesson.")

    if room.get("has_lim") is True:
        lines.append("An interactive whiteboard (LIM) is available for the whole class.")
    elif room.get("has_lim") is False:
        lines.append("There is NO interactive whiteboard (LIM) and no shared projection.")

    if room.get("has_wifi") is False:
        lines.append("There is NO WiFi connection: no activity may depend on internet access.")
    elif room.get("has_wifi") is True:
        lines.append("A WiFi connection is available.")

    if room.get("pc_station") is False:
        lines.append("There are NO PC workstations for students.")
    elif room.get("pc_station") is True:
        lines.append("PC workstations are available for students.")

    if room.get("has_suite") is False:
        lines.append("No office suite (Google Workspace / Microsoft 365) is available.")

    # Where the lesson happens: a remote class cannot pass a sheet around, and a
    # hybrid one must not leave the students at home watching the others work.
    where = _code(room.get("environment_type"))
    participation = _code(room.get("participation_type"))
    if where == "REMOTE":
        lines.append(
            "The lesson is held REMOTELY: no activity may require physical presence, "
            "shared physical materials or moving around a room."
        )
    elif where == "HYBRID":
        prevalence = {
            "MOSTLY_LOCAL": " Most students are in the room.",
            "MOSTLY_REMOTE": " Most students are connected remotely.",
        }.get(_code(room.get("prevalent_modality_type")), "")
        lines.append(
            "The lesson is HYBRID: some students are in the room and some are connected "
            "remotely. Every activity must work for both, and groups must be able to "
            f"include remote students.{prevalence}"
        )
    if where in ("REMOTE", "HYBRID") and participation == "ASYNC":
        lines.append(
            "Remote students follow ASYNCHRONOUSLY (recorded content): no activity for them "
            "may depend on live interaction with the teacher or classmates."
        )

    mobility = str(
        getattr(room.get("forniture_mobility"), "value", room.get("forniture_mobility")) or ""
    )
    if mobility == "NO":
        lines.append(
            "The furniture CANNOT be moved: no activity may require rearranging the desks."
        )
    elif mobility == "PARTIALLY":
        lines.append(
            "The furniture can only be partially moved: prefer activities that work "
            "with small local changes, not a full rearrangement of the room."
        )

    students = group.get("students_number")
    if isinstance(students, int) and students > 0:
        lines.append(
            f"The class has {students} students: every grouping must account for all "
            f"{students}, and cooperative groups must not exceed {MAX_GROUP_SIZE} members."
        )

    return lines


def render_profile_section(profile: Any) -> str:
    """The profile section for the Writer's prompt, or ``""`` when there is none.

    The returned string starts and ends with a newline, matching what the
    template around ``{educational_profile_section}`` expects.
    """
    ep = _as_dict(profile)
    if not ep:
        return ""

    group = ep.get("group") or {}
    facts: list[str] = []

    if ep.get("specific_topic"):
        facts.append(f"- Topic: {ep['specific_topic']}")
    if ep.get("subject_area"):
        facts.append(f"- Subject: {ep['subject_area']}")
    if grade := _label(group.get("grade"), GRADE_LABELS):
        facts.append(f"- Grade level: {grade}")
    if isinstance(group.get("students_number"), int):
        facts.append(f"- Class size: {group['students_number']} students")

    # ``group.disabilities`` is canonical; the top-level key is a legacy payload.
    needs = _labels(group.get("disabilities") or ep.get("disabilities"), DISABILITY_LABELS)
    if needs:
        facts.append(f"- Learner needs (BES): {'; '.join(needs)}")

    if features := _labels(group.get("class_features"), CLASS_FEATURE_LABELS):
        facts.append(f"- Class characteristics: {'; '.join(features)}")
    if attributes := _labels(group.get("student_attributes"), STUDENT_ATTR_LABELS):
        facts.append(f"- Student population: {'; '.join(attributes)}")

    # ``time_available_minutes`` is canonical; ``lesson_duration`` is legacy.
    duration = ep.get("time_available_minutes") or ep.get("lesson_duration")
    if duration:
        facts.append(f"- Duration: {duration} minutes")

    constraints = render_classroom_constraints(ep)

    if not facts and not constraints:
        return ""

    blocks: list[str] = []
    if facts:
        blocks.append("\n".join([PROFILE_HEADER, *facts]))
    if constraints:
        blocks.append("\n".join([CONSTRAINTS_HEADER, *(f"- {line}" for line in constraints)]))

    return "\n" + "\n\n".join(blocks) + "\n"
