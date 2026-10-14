"""A turn of a lesson's conversation: accepting it, and running it (2026-10-14).

Until now this lived inside two WebUI routes, ``POST /webui/lesson/{id}/run`` and
``GET /webui/lesson/{id}/stream``, mixed with the HTML they answer with. The API
for chatbots needs the same turn -- the same checks, the same light-or-full
decision, the same messages written, the same slot and claim -- and a second
copy would drift from the first on the next change. So the decisions live here,
and each route is an adapter: HTML cards for the WebUI, JSON for the API.

* :func:`accept_turn` -- everything the run POST decided: the request, the
  checks (an empty request, a run already going, documents still being
  prepared, the rate limit), the light-or-full choice, the intent, a refinement
  chip, the Knowledge Graph switch, the user's message, ``status=running``.
  A refusal comes back as :class:`TurnRefused`, with a stable ``code`` for a
  machine and the Italian sentence for a person.
* :func:`drive_turn` -- everything the stream GET did around the agent: the
  global slot, the claim on the lesson, the heartbeat, then the light answer or
  the whole chain. The services it calls already persist what they produce.

The intent card the WebUI shows before a first lesson stays in the WebUI route:
it is a question asked of a person through the page. The API takes the intent
in the profile instead (``ask_intent=False``).

``form`` is any mapping with the WebUI form's field names (``query``,
``full_run``, ``light_answer``, ``refinement_code``, ``refinement_detail``,
``graph_search_choice``, ``graph_search``, ``intent_confirmed``,
``pedagogical_intent_code``, ``pedagogical_intent_detail``): the route passes
the posted form, the API builds one from its JSON.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from aix.core.concurrency import (
    AtCapacity,
    RateLimited,
    acquire_run_slot,
    check_user_rate_limit,
    release_run_slot,
)
from aix.core.run_registry import get_run_registry, heartbeat_loop
from aix.webui.agent.light import LIGHT_KEY, run_light_turn, take_light_turn, wants_light_turn
from aix.webui.agent.service import StreamEvent, run_agent_stream
from aix.webui.lessons.models import Lesson, LessonMessage
from aix.webui.materials.readiness import still_preparing

logger = logging.getLogger(__name__)

#: The longest request a turn keeps, the textarea's ``maxlength``.
MAX_QUERY_CHARS = 2000

#: What the chat shows, and the Writer reads, for a refinement chip.
REFINEMENT_LABELS = {
    "simplify": "Semplifica la lezione",
    "deepen": "Approfondisci la lezione",
    "more_activities": "Aggiungi attività interattive",
    "adapt_class": "Adatta la lezione alla classe",
    "custom": "Rifinisci la lezione",
}

#: What the Writer is told for each chip.
REFINEMENT_PROMPTS = {
    "simplify": "Reduce length by ~30%, simplify vocabulary, keep only essential concepts",
    "deepen": "Add scientific depth, more precise terminology, nuanced examples",
    "more_activities": "Replace passive sections with at least 2 interactive activities or exercises",
    "adapt_class": "Re-calibrate to the class profile: BES accommodations, time constraint, available tech",
}


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnRefused:
    """A turn that does not start, and why: ``code`` for a machine, ``message`` for a person."""

    code: str
    message: str
    status_code: int
    retry_after: Optional[int] = None
    #: The documents still being prepared, for ``materials_preparing``.
    preparing: tuple[str, ...] = ()


@dataclass(frozen=True)
class TurnAccepted:
    """A turn written and waiting to be driven."""

    #: ``new`` (first turn), ``follow_up`` (the whole chain again) or ``light``.
    mode: str
    turn_index: int
    has_new_query: bool
    #: The WebUI asks for the intent before the first lesson (see the module docstring).
    ask_intent: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Small pieces the routes used to own
# ---------------------------------------------------------------------------


def refinement_label(code: str, detail: Optional[str] = None) -> str:
    """The teacher's turn for a chip: «Semplifica la lezione: più breve»."""
    label = REFINEMENT_LABELS.get(code, "Rifinisci la lezione")
    if code == "custom" and detail:
        return detail[:MAX_QUERY_CHARS]
    return f"{label}: {detail}"[:MAX_QUERY_CHARS] if detail else label


def remember_graph_search(lesson: Lesson, form: Any) -> None:
    """Keep the teacher's Knowledge Graph switch on the lesson, if the form had one.

    Only the follow-up input shows the switch and sends ``graph_search_choice``;
    an unticked checkbox sends nothing, which is why the marker is needed to
    tell "switched off" from "not asked". Kept in the profile and left there, so
    the switch renders as she last set it on the turns after.
    """
    if form.get("graph_search_choice") != "1":
        return
    profile = dict(lesson.educational_profile_json or {})
    profile["__graph_search__"] = form.get("graph_search") == "on"
    lesson.educational_profile_json = profile


def legacy_query_for_backfill(lesson: Lesson, new_query: Optional[str]) -> str:
    """
    Recover the previous turn's user query for the backfill path when
    ``lesson.teacher_query`` has already been overwritten by the new query.

    Pre-#10.3 lessons that DO have a ``lesson_plan_md`` but lost their
    ``teacher_query`` fall back to a placeholder, so the backfill row stays
    non-empty (NOT NULL constraint).
    """
    if new_query and (lesson.teacher_query or "").strip() == new_query.strip():
        return "[richiesta precedente non persistita]"
    return (lesson.teacher_query or "[richiesta precedente non persistita]").strip()


def _text(form: Mapping[str, Any], name: str) -> Optional[str]:
    raw = form.get(name)
    return raw.strip() if isinstance(raw, str) and raw.strip() else None


# ---------------------------------------------------------------------------
# Accepting a turn
# ---------------------------------------------------------------------------


async def accept_turn(
    session: AsyncSession,
    lesson: Lesson,
    form: Mapping[str, Any],
    *,
    rate_key: str,
    ask_intent: bool = True,
) -> Union[TurnAccepted, TurnRefused]:
    """Check, decide and write the turn; commit; or say why not.

    Multi-turn semantics (CORE 2 #10.3c): the mode follows ``lesson.status`` --
    draft is the first turn, complete or error a follow-up, running a refusal.
    Each turn writes a ``LessonMessage(role="user")``; the services write the
    assistant's on success.

    Pre-#10.3 lessons have no message rows but do have ``teacher_query`` and
    ``lesson_plan_md``: on their first follow-up, turn 1 is backfilled from
    those, once.
    """
    new_query = _text(form, "query")
    # Decided before anything is written: a light turn answers about the lesson
    # and leaves ``teacher_query`` -- the lesson's topic -- as it was.
    light = wants_light_turn(lesson, form, has_query=new_query is not None)

    if new_query is not None:
        # No minimum length: a reply as short as «sì» is a real answer in a
        # conversation (2026-10-14).
        new_query = new_query[:MAX_QUERY_CHARS]
        if not light:
            lesson.teacher_query = new_query

    if not (lesson.teacher_query or "").strip():
        return TurnRefused(
            "empty_query",
            "Scrivi prima la tua richiesta nel campo della chat, poi premi Invia.",
            422,
        )

    if lesson.status == "running":
        return TurnRefused(
            "already_running", "Una generazione è già in corso. Attendi il termine.", 409
        )

    # §6.1.4 / decision U1 — a lesson is generated from material that is ready.
    # Without this, a request sent seconds after the upload is written from the
    # pasted text while the teacher believes the document was read: exactly what
    # happened on 2026-09-16.
    preparing = await still_preparing(session, lesson_id=lesson.id, owner_id=lesson.owner_id)
    if preparing:
        return TurnRefused(
            "materials_preparing",
            f"Sto ancora preparando {', '.join(preparing)}. Parte da solo appena è pronto.",
            409,
            preparing=tuple(preparing),
        )

    # Per-user rate limit (#33): checked when a run starts, never on a stream
    # reconnect.
    try:
        check_user_rate_limit(rate_key)
    except RateLimited as exc:
        wait = int(exc.retry_after) + 1
        return TurnRefused(
            "rate_limited",
            f"Hai avviato troppe generazioni in poco tempo. Attendi {wait} secondi e riprova.",
            429,
            retry_after=wait,
        )

    is_follow_up = lesson.status in ("complete", "error")
    mode = "follow_up" if is_follow_up else "new"

    # The pedagogical intent, asked before the first lesson (WebUI only). A
    # teacher who already chose it in the profile has answered the question the
    # card asks: asking again cost a whole Planner call (measured: 78 s).
    intent_confirmed = form.get("intent_confirmed") == "1"
    already_chosen = str(
        (lesson.educational_profile_json or {}).get("pedagogical_intent") or ""
    ).strip()
    ask = ask_intent and mode == "new" and not intent_confirmed and not already_chosen
    if ask:
        profile = dict(lesson.educational_profile_json or {})
        profile["__ask_intent__"] = True
        lesson.educational_profile_json = profile
    elif intent_confirmed:
        intent_code = _text(form, "pedagogical_intent_code")
        if intent_code and intent_code != "skip":
            intent_detail = _text(form, "pedagogical_intent_detail")
            profile = dict(lesson.educational_profile_json or {})
            profile.pop("__ask_intent__", None)
            profile.pop("__planner_only__", None)
            profile["pedagogical_intent"] = (
                f"{intent_code}: {intent_detail}" if intent_detail else intent_code
            )
            lesson.educational_profile_json = profile

    # F3b: a refinement chip, kept transiently in the profile for the service.
    refinement_code = _text(form, "refinement_code")
    refinement_detail = _text(form, "refinement_detail")
    chip_label: Optional[str] = None
    if refinement_code:
        if refinement_code == "custom" and refinement_detail:
            instruction = refinement_detail
        else:
            base = REFINEMENT_PROMPTS.get(refinement_code, "")
            instruction = f"{base} ({refinement_detail})" if refinement_detail else base
        profile = dict(lesson.educational_profile_json or {})
        profile["__refinement__"] = instruction
        lesson.educational_profile_json = profile
        # A chip sends no text: the chat shows the chip, not a repeat of the
        # previous request. ``teacher_query`` stays the topic the planner needs.
        if new_query is None:
            chip_label = refinement_label(refinement_code, refinement_detail)

    # The Knowledge Graph switch on a follow-up, left on the lesson as last set
    # (2026-09-29). Then the light turn's own settings, or «Lezione completa».
    remember_graph_search(lesson, form)
    if light:
        profile = dict(lesson.educational_profile_json or {})
        profile[LIGHT_KEY] = {"answer_here": form.get("light_answer") == "1"}
        lesson.educational_profile_json = profile
    elif form.get("full_run") == "1":
        profile = dict(lesson.educational_profile_json or {})
        profile["__graph_search__"] = True
        lesson.educational_profile_json = profile

    if is_follow_up:
        existing = await session.scalar(
            select(func.count(LessonMessage.id)).where(LessonMessage.lesson_id == lesson.id)
        )
        if (
            (existing or 0) == 0
            and (lesson.teacher_query or "").strip()
            and (lesson.lesson_plan_md or "").strip()
        ):
            logger.info("[turns] backfilling turn 1 from legacy fields lesson_id=%s", lesson.id)
            backfill_query = (
                new_query is None
                and lesson.teacher_query
                or legacy_query_for_backfill(lesson, new_query)
            )
            session.add(
                LessonMessage(
                    lesson_id=lesson.id, role="user", content_md=backfill_query, turn_index=1
                )
            )
            session.add(
                LessonMessage(
                    lesson_id=lesson.id,
                    role="assistant",
                    content_md=lesson.lesson_plan_md,
                    turn_index=1,
                    agent_kind="writer",
                    meta_json={"approved": True, "backfilled": True},
                )
            )
            await session.flush()

    latest = (
        await session.scalar(
            select(func.max(LessonMessage.turn_index)).where(LessonMessage.lesson_id == lesson.id)
        )
        or 0
    )
    turn_index = latest + 1

    # Phase 2 of the intent card is not a new user turn: the message was written
    # in phase 1, and a duplicate would make the assistant's reply match the
    # wrong turn.
    if not intent_confirmed:
        session.add(
            LessonMessage(
                lesson_id=lesson.id,
                role="user",
                content_md=new_query or chip_label or lesson.teacher_query,
                turn_index=turn_index,
            )
        )

    # Running from now: the previous lesson stays in the history (#10.3); a
    # first turn clears what an aborted attempt may have left.
    lesson.status = "running"
    lesson.error_message = None
    if mode == "new":
        lesson.lesson_plan_md = None
    await session.commit()
    await session.refresh(lesson)

    logger.info(
        "▶️  Lesson run accepted: id=%s owner=%s mode=%s turn=%s has_new_query=%s uploads=%s",
        lesson.id,
        lesson.owner_id,
        "light" if light else mode,
        turn_index,
        bool(new_query),
        len(lesson.uploaded_files_json or []),
    )
    return TurnAccepted(
        mode="light" if light else mode,
        turn_index=turn_index,
        has_new_query=new_query is not None,
        ask_intent=ask,
    )


# ---------------------------------------------------------------------------
# Driving a turn
# ---------------------------------------------------------------------------

#: Said when another connection or worker is already running this lesson.
BUSY = TurnRefused(
    "already_running",
    "Una generazione è già in corso in un'altra finestra. Attendi il termine e ricarica la pagina.",
    409,
)

#: Said when every generation slot is taken past the queue window (#34).
AT_CAPACITY = TurnRefused(
    "at_capacity",
    "Il sistema è momentaneamente occupato: troppe generazioni in corso in questo momento. "
    "Attendi qualche istante e riprova.",
    503,
)


async def drive_turn(
    lesson: Lesson,
    session: AsyncSession,
    *,
    is_disconnected=None,
) -> AsyncIterator[Union[StreamEvent, TurnRefused]]:
    """Run an accepted turn: slot, claim, heartbeat, then the light answer or the whole chain.

    Yields the services' :class:`StreamEvent` objects as they come, or a single
    :class:`TurnRefused` (``already_running``, ``at_capacity``) when the turn
    cannot start. ``is_disconnected`` is an optional async callable: when it
    says the listener left, the loop stops reading -- the services still bring
    the lesson to a terminal state.
    """
    registry = get_run_registry()
    if await registry.is_active(lesson.id):
        yield BUSY
        return

    try:
        await acquire_run_slot(label=f"webui:{lesson.id}")
    except AtCapacity:
        # Shed gracefully (#34). The accepted turn flipped the row to "running";
        # revert it so a retry takes the right mode -- "draft" for a first
        # turn, "complete" for a follow-up -- instead of a stale spinner.
        lesson.status = "complete" if (lesson.lesson_plan_md or "").strip() else "draft"
        await session.commit()
        yield AT_CAPACITY
        return

    token = await registry.claim(lesson.id, owner_id=lesson.owner_id)
    if token is None:
        release_run_slot()
        yield BUSY
        return

    heartbeat = asyncio.create_task(heartbeat_loop(registry, lesson.id, token))
    light = take_light_turn(lesson)
    events = (
        run_light_turn(lesson, session, **light)
        if light is not None
        else run_agent_stream(lesson, session)
    )
    try:
        async for event in events:
            if is_disconnected is not None and await is_disconnected():
                logger.info(
                    "[turns] listener gone lesson_id=%s; the run goes on to its end", lesson.id
                )
                break
            yield event
    finally:
        heartbeat.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await heartbeat
        await registry.release(lesson.id, token)
        release_run_slot()
