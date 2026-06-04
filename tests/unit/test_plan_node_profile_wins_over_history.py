"""
CORE 2 #12b.3 — plan_node duration-precedence regression coverage
==================================================================

Locks in the user-facing matrix that landed with #12b.3:

    | Profile | Current turn says | History says | Fixed behaviour       |
    |---------|-------------------|--------------|-----------------------|
    | 60 min  | (no duration)     | (no dur.)    | 60 min   (profile)    |
    | 60 min  | (no duration)     | "45 min"     | 60 min   (profile)    | ← the fix
    | 60 min  | "30 min"          | "45 min"     | 30 min   (current)    |
    | (none)  | (no duration)     | "45 min"     | 45 min   (planner)    |
    | (none)  | "30 min"          | (no dur.)    | 30 min   (planner)    |

Plus a few edge cases (Italian "minuti", hours phrasing, legacy callers
that don't populate ``raw_user_turn``).

These are pure unit tests — the Planner agent is fully mocked, so they
run in milliseconds and are stable in CI.
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import AsyncMock, patch

import pytest

from aix.agent.agents.planner_agent import RetrievalPlan
from aix.agent.graph import nodes as plan_nodes
from aix.agent.graph.state import create_initial_state

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_state(
    *,
    raw_user_turn: Optional[str],
    augmented_query: str,
    profile_minutes: Optional[int],
):
    """Create an AgentState shaped like a real follow-up run.

    ``augmented_query`` is what ``state["teacher_query"]`` ends up as after the
    service-layer's ``_augment_query_with_history`` glues prior turns onto the
    raw_user_turn — pass the raw turn alone for a first-turn-style state.
    """
    profile = None
    if profile_minutes is not None:
        profile = {
            "time_available_minutes": profile_minutes,
            "subject_area": "Scienza",
            "specific_topic": "ADHD",
            "group": {"disabilities": ["ADHD"]},
        }

    return create_initial_state(
        query=augmented_query,
        domain="udl",
        language="it",
        educational_profile=profile,
        raw_user_turn=raw_user_turn,
    )


def _make_plan(time_constraints: Optional[str]) -> RetrievalPlan:
    """Build a minimal RetrievalPlan with whatever ``time_constraints`` the
    Planner would have extracted from the augmented query."""
    return RetrievalPlan(
        query_intent="lesson_creation",
        key_concepts=["ADHD"],
        search_queries=["ADHD"],
        lesson_type="full_lesson",
        target_grade=None,
        special_needs=["ADHD"],
        time_constraints=time_constraints,
        intent_confidence="HIGH",
        reasoning="mock",
        scope_status="in_scope",
        scope_confidence=1.0,
        subject_concepts=["ADHD"],
        pedagogy_concepts=[],
        response_language="it",
        language_confidence="HIGH",
    )


async def _run_plan_node(state, planner_extracted: Optional[str]) -> dict:
    """Run plan_node with the Planner mocked to return a plan whose
    ``time_constraints`` is ``planner_extracted`` (simulating extraction
    from the augmented query)."""
    fake_planner = AsyncMock()
    fake_planner.plan = AsyncMock(return_value=_make_plan(planner_extracted))
    with patch.object(plan_nodes, "get_planner", return_value=fake_planner):
        return await plan_nodes.plan_node(state)


# ---------------------------------------------------------------------------
# Sniffer (regex helper) sanity checks — cheap, pure-function tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        # Italian phrasings
        ("ora rifalla in 30 minuti", True),
        ("una lezione di 45 min", True),
        ("dura 1 ora", True),
        ("una lezione da 90 minuti per favore", True),
        # English phrasings
        ("make it 30 minutes", True),
        ("90 min lesson on ADHD", True),
        ("a 2 hour workshop", True),
        # No duration mentions
        ("crea una lezione su disturbi da deficit di attenzione", False),
        ("ora adattala per ADHD", False),
        ("fammi una panoramica su UDL", False),
        # Stray digits NOT matching the pattern (must not false-positive)
        ("UDL 2.0 framework", False),
        ("WCAG 2.1 guidelines", False),
        ("studenti delle medie con BES", False),
    ],
)
def test_duration_sniffer(text: str, expected: bool):
    state = {"raw_user_turn": text, "teacher_query": text}
    assert plan_nodes._current_turn_mentions_duration(state) is expected


def test_duration_sniffer_falls_back_to_teacher_query_when_raw_missing():
    """Legacy callers that don't populate ``raw_user_turn`` must still get
    the right answer on first-turn-style states (where ``teacher_query`` IS
    the raw turn)."""
    state = {"teacher_query": "fai una lezione di 30 minuti"}
    assert plan_nodes._current_turn_mentions_duration(state) is True

    state2 = {"teacher_query": "fai una lezione su ADHD"}
    assert plan_nodes._current_turn_mentions_duration(state2) is False


# ---------------------------------------------------------------------------
# The 5-row matrix from the user's spec
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_row1_profile_set_no_durations_anywhere():
    """Row 1 — profile=60, current turn silent, history silent → 60."""
    state = _build_state(
        raw_user_turn="crea una lezione su ADHD",
        augmented_query="crea una lezione su ADHD",  # first-turn-shaped
        profile_minutes=60,
    )
    updates = await _run_plan_node(state, planner_extracted=None)

    assert updates["plan"]["time_constraints"] == "60 minutes"


@pytest.mark.asyncio
async def test_row2_profile_set_history_leaks_duration_THE_FIX():
    """Row 2 — THE BUG WE'RE FIXING. Profile=60, current turn silent,
    history says 45. Without #12b.3 the Planner extracts "45 minutes"
    from the augmented blob and the profile is silently ignored.
    With #12b.3 the profile (60) wins because the *current turn* has
    no duration."""
    state = _build_state(
        raw_user_turn="crea una lezione su disturbi da deficit di attenzione",
        augmented_query=(
            "## Conversazione precedente\n"
            "### Sintesi dei turni più vecchi\n"
            "Il docente ha richiesto una lezione di 45 minuti su ADHD.\n"
            "## Nuova richiesta del docente\n"
            "crea una lezione su disturbi da deficit di attenzione"
        ),
        profile_minutes=60,
    )
    # Simulate Planner extracting 45 from the augmented blob (the actual
    # observed bug behaviour from the 2026-05-10 12:19 run).
    updates = await _run_plan_node(state, planner_extracted="45 minutes")

    assert updates["plan"]["time_constraints"] == "60 minutes", (
        "Profile (60 min) must override the history-leaked '45 minutes' "
        "when the current turn has no duration mention."
    )


@pytest.mark.asyncio
async def test_row3_current_turn_explicit_duration_beats_profile_and_history():
    """Row 3 — profile=60, current turn says "30 minuti", history says "45 minuti"
    → 30 wins (the teacher just said it explicitly)."""
    state = _build_state(
        raw_user_turn="ora rifalla in 30 minuti",
        augmented_query=(
            "## Conversazione precedente\n"
            "### Turno 1 — Docente\nuna lezione di 45 minuti\n"
            "## Nuova richiesta del docente\n"
            "ora rifalla in 30 minuti"
        ),
        profile_minutes=60,
    )
    # Planner correctly extracts the explicit current-turn duration.
    updates = await _run_plan_node(state, planner_extracted="30 minutes")

    assert updates["plan"]["time_constraints"] == "30 minutes"


@pytest.mark.asyncio
async def test_row4_no_profile_history_only_planner_wins():
    """Row 4 — no profile set, current turn silent, history says "45 minuti"
    → 45 wins (no profile to defer to)."""
    state = _build_state(
        raw_user_turn="crea una lezione su ADHD",
        augmented_query=(
            "## Conversazione precedente\n"
            "Una lezione di 45 minuti era stata richiesta in passato.\n"
            "## Nuova richiesta del docente\ncrea una lezione su ADHD"
        ),
        profile_minutes=None,
    )
    updates = await _run_plan_node(state, planner_extracted="45 minutes")

    assert updates["plan"]["time_constraints"] == "45 minutes"


@pytest.mark.asyncio
async def test_row5_no_profile_current_turn_says_30_planner_wins():
    """Row 5 — no profile, current turn says "30 minuti" → 30."""
    state = _build_state(
        raw_user_turn="una lezione di 30 minuti su ADHD",
        augmented_query="una lezione di 30 minuti su ADHD",
        profile_minutes=None,
    )
    updates = await _run_plan_node(state, planner_extracted="30 minutes")

    assert updates["plan"]["time_constraints"] == "30 minutes"


# ---------------------------------------------------------------------------
# Backward-compat: callers that don't populate raw_user_turn
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_legacy_caller_no_raw_user_turn_first_turn_path():
    """Pre-#12b.3 callers that omit ``raw_user_turn`` must still get the
    right answer on first-turn requests (no history). ``teacher_query`` IS
    the raw turn in that case, so the sniffer falls back to it cleanly."""
    # Simulate a legacy caller: build state and then strip raw_user_turn.
    state = _build_state(
        raw_user_turn="crea una lezione su ADHD",
        augmented_query="crea una lezione su ADHD",
        profile_minutes=60,
    )
    state["raw_user_turn"] = None  # legacy caller didn't set it

    updates = await _run_plan_node(state, planner_extracted=None)
    assert updates["plan"]["time_constraints"] == "60 minutes"


@pytest.mark.asyncio
async def test_legacy_caller_first_turn_with_explicit_duration():
    """Same legacy caller, but the first-turn query mentions a duration.
    Sniffer falls back to ``teacher_query``, sees the duration, lets the
    Planner-extracted value win — same behaviour as #12b.1 first-turn path."""
    state = _build_state(
        raw_user_turn="lezione di 45 minuti su ADHD",
        augmented_query="lezione di 45 minuti su ADHD",
        profile_minutes=60,
    )
    state["raw_user_turn"] = None  # legacy caller

    updates = await _run_plan_node(state, planner_extracted="45 minutes")
    assert updates["plan"]["time_constraints"] == "45 minutes"
