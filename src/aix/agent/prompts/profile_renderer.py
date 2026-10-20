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

    # Pupils this young do not handle what can hurt them: the water-cycle lesson of
    # 2026-10-19 handed hot water to every group of seven-year-olds.
    if _code(group.get("grade")) in ("INFANZIA", "PRIMARIA"):
        lines.append(
            "Pupils this young never handle hot water, flames, heat sources, sharp tools "
            "or chemicals: any demonstration with them is done by the teacher."
        )

    students = group.get("students_number")
    if isinstance(students, int) and students > 0:
        line = (
            f"The class has {students} students: every grouping must account for all "
            f"{students}, and cooperative groups must not exceed {MAX_GROUP_SIZE} members."
        )
        if split := group_split(students):
            line += (
                f" A grouping that works: {split}. State the grouping you use as a plain "
                "fact; never write the arithmetic, the options you discarded or this rule "
                "in the lesson."
            )
        # Second blind read (2026-10-19): five roles for groups of four, two roles
        # both keeping time, a lesson that never said how big its groups were.
        line += (
            " Every group activity says how many pupils per group; a group has no more "
            "roles than its smallest group has members, and no two roles do the same job."
        )
        lines.append(line)

    return lines


def group_split(students: int, max_size: int = MAX_GROUP_SIZE) -> str:
    """The most even split of the class into groups of at most ``max_size``.

    Handed to the model ready-made because the arithmetic leaked into the
    lesson. The staging lesson of 2026-10-16 (22 pupils) read «Gruppi di 4–5
    studenti (5 gruppi da 4 e 1 da 2, oppure 4 gruppi: 3 da 5 e 1 da 7 non
    ammesso; usare 5 gruppi da 4 e 1 da 2 oppure 2 gruppi da 5 e 3 da 4…)»: the
    rule, worked out in front of the teacher. ``""`` below two groups.
    """
    if students <= max_size:
        return ""
    count = -(-students // max_size)
    small, larger = divmod(students, count)
    parts = []
    if larger:
        parts.append(f"{larger} group{'s' if larger > 1 else ''} of {small + 1}")
    if count - larger:
        parts.append(f"{count - larger} group{'s' if count - larger > 1 else ''} of {small}")
    return " and ".join(parts)


def _five(minutes: float) -> int:
    return max(5, int(round(minutes / 5.0)) * 5)


def time_plan(duration: Any) -> str:
    """A split of the duration that adds up, for the Writer to fill in.

    «The total MUST equal the Duration exactly» was in the prompt, and every
    lesson regenerated on 2026-10-16 still came out at 53–55 minutes of 60: the
    model does not add. The group split came worked out for the same reason (see
    ``render_classroom_constraints``) and the arithmetic left the lesson; so does this.
    The proportions are a starting point the Writer may move, as long as the sum
    stays exact.
    """
    if not isinstance(duration, int) or duration < 20:
        return ""
    intro = _five(duration * 0.15)
    closing = _five(duration * 0.15)
    activities = duration - intro - closing
    return (
        f"Time plan: Introduzione {intro} minuti + Attività {activities} minuti + Valutazione {closing} minuti "
        f"= {duration} minuti. The minutes of the activities inside «Attività» add up to "
        f"{activities}, and every activity states its own minutes; move minutes between "
        f"sections if the lesson needs it, but the section timings must add up to "
        f"exactly {duration}."
    )


#: Bloom's levels in Italian, lowest first.
BLOOM_LEVELS = ("ricordare", "comprendere", "applicare", "analizzare", "valutare", "creare")

#: The levels a lesson's objectives should use, by pedagogical intent.
_BLOOM_BY_INTENT: dict[str, tuple[str, ...]] = {
    "intro": ("ricordare", "comprendere", "applicare"),
    "deepening": ("applicare", "analizzare", "valutare"),
    "misconception": ("comprendere", "analizzare", "valutare"),
    "assessment": ("ricordare", "comprendere", "applicare"),
    "practice": ("applicare", "analizzare"),
    "interdisciplinary": ("comprendere", "analizzare", "creare"),
}
_BLOOM_BY_GRADE: dict[str, tuple[str, ...]] = {
    "INFANZIA": ("ricordare", "comprendere"),
    "PRIMARIA": ("ricordare", "comprendere", "applicare"),
    "SECONDARIA_I_GRADO": ("comprendere", "applicare", "analizzare"),
    "SECONDARIA_II_GRADO": ("applicare", "analizzare", "valutare"),
    "UNIVERSITA": ("analizzare", "valutare", "creare"),
    "FORMAZIONE_SUL_LAVORO": ("applicare", "analizzare", "valutare"),
}
#: Young pupils: no higher than this, whatever the intent says.
_BLOOM_CEILING = {"INFANZIA": "comprendere", "PRIMARIA": "applicare"}


def bloom_levels(profile: Any) -> tuple[str, ...]:
    """The Bloom levels for this lesson's objectives: from the intent, else the grade.

    The blind read of 2026-10-16 preferred the lesson whose objectives climbed
    Bloom's levels (ricordare, comprendere, applicare, esprimere) -- but not as a
    formula repeated in every lesson: the levels follow what the teacher wants
    (an introduction remembers and understands, a deepening analyses and judges)
    and how old the pupils are. The Writer chooses the verbs.
    """
    ep = _as_dict(profile)
    group = ep.get("group") or {}
    grade = _code(group.get("grade"))
    intent = str(ep.get("pedagogical_intent") or "").split(":", 1)[0].strip()
    levels = _BLOOM_BY_INTENT.get(intent) or _BLOOM_BY_GRADE.get(grade) or ()
    ceiling = _BLOOM_CEILING.get(grade)
    if ceiling:
        top = BLOOM_LEVELS.index(ceiling)
        levels = tuple(level for level in levels if BLOOM_LEVELS.index(level) <= top)
        levels = levels or (ceiling,)
    return levels


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

    # The profile's topic and subject were set when the session began: a request
    # that names another wins. A conversation moved from the water cycle to
    # volcanoes, and a refinement of the lesson on volcanoes came back on the
    # water cycle (API replay, 2026-10-20).
    if ep.get("specific_topic"):
        facts.append(
            f"- Topic set at the start of the session: {ep['specific_topic']} "
            "(if the teacher's request or the lesson being refined is about another "
            "subject, that subject is the topic)"
        )
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
        if plan := time_plan(duration):
            facts.append(f"- {plan}")

    if levels := bloom_levels(ep):
        facts.append(
            "- Learning objectives: 2–4, observable, each opening with a verb of these "
            f"levels of Bloom's taxonomy, lowest first: {', '.join(levels)}. Choose the "
            "verbs for this lesson (ricordare → elencare, nominare; comprendere → spiegare "
            "con parole proprie, riordinare; applicare → usare in un caso nuovo…), and do "
            "not go beyond what the request asks."
        )

    constraints = render_classroom_constraints(ep)

    if not facts and not constraints:
        return ""

    blocks: list[str] = []
    if facts:
        blocks.append("\n".join([PROFILE_HEADER, *facts]))
    if constraints:
        blocks.append("\n".join([CONSTRAINTS_HEADER, *(f"- {line}" for line in constraints)]))

    return "\n" + "\n\n".join(blocks) + "\n"
