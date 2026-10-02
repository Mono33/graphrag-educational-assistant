"""Light follow-up turns: one streamed answer instead of the whole chain.

After the first lesson every request used to run planner, Knowledge Graph,
Writer and Critic again: 50–70 s, of which the Writer rewriting the whole lesson
was 65–85 % (``scripts/eval/measure_followup_turns.py``), for questions such as
«perché hai scelto questa attività?» or «che cosa dice il capitolo 4?».

A light turn is one call to ``AIX_CHAT_MODEL`` (``LLM_MODEL`` when unset) that
reads the lesson, the conversation and the attached documents -- searched again
on every turn, so a document added a minute ago is there -- and streams an
answer into the chat. The lesson is not changed. When the request is a new
lesson the model says so with :data:`NEW_LESSON` and the teacher chooses: the
full chain, or an answer here.

Not a LangGraph node: nothing here needs the graph's state, and the turn must
not pay for compiling it.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import AsyncIterator
from typing import Any, Optional

from sqlalchemy import delete, func, select

from aix.agent.prompts.light_prompt import (
    ANSWER_HERE_RULE,
    ESCALATION_RULE,
    LIGHT_SYSTEM_PROMPT,
    LIGHT_USER_TEMPLATE,
    NEW_LESSON,
)
from aix.webui.agent.service import StreamEvent
from aix.webui.errors import report as _report_failure
from aix.webui.features import light_turns_enabled

logger = logging.getLogger(__name__)

#: The flag the route leaves on the lesson for the stream to read, once.
LIGHT_KEY = "__light__"

#: The lesson is the working document: whole, up to a size no lesson reaches.
_LESSON_CHARS = 60000
_TURNS_KEPT = 6
_USER_CHARS = 1500
_ANSWER_CHARS = 2500
_MAX_TOKENS = int(os.getenv("AIX_CHAT_MAX_TOKENS", "3000"))
_MARKER = re.compile(r"\[\[material:[^\]]*\]\]")


def chat_model() -> str:
    from aix.core.config import config as app_config

    return os.getenv("AIX_CHAT_MODEL") or app_config.openai.model


def wants_light_turn(lesson: Any, form: Any, *, has_query: bool) -> bool:
    """Whether this turn is answered by the light path.

    A follow-up on a finished lesson, with something typed, is light -- unless
    the teacher ticked «Lezione completa», picked an intent, or pressed a
    refinement chip (the chips still rewrite the whole lesson).
    """
    if not light_turns_enabled() or not has_query:
        return False
    if lesson.status != "complete" or not (lesson.lesson_plan_md or "").strip():
        return False
    return not any(form.get(name) for name in ("full_run", "intent_confirmed", "refinement_code"))


def take_light_turn(lesson: Any) -> Optional[dict[str, Any]]:
    """The light-turn flag, removed from the saved profile: it is for one turn."""
    profile = dict(lesson.educational_profile_json or {})
    if LIGHT_KEY not in profile:
        return None
    flag = profile.pop(LIGHT_KEY)
    lesson.educational_profile_json = profile
    return flag if isinstance(flag, dict) else {}


class EscalationGate:
    """Holds the first tokens back until they cannot be :data:`NEW_LESSON`.

    The marker arrives in pieces; streaming them would flash «<<<NUOVA» in the
    chat before the hand-off card replaces it.
    """

    def __init__(self) -> None:
        self._head = ""
        self.open = False
        self.escalated = False

    def feed(self, delta: str) -> str:
        if self.open:
            return delta
        self._head += delta
        probe = self._head.lstrip()
        if probe.startswith(NEW_LESSON):
            self.escalated = True
            return ""
        if NEW_LESSON.startswith(probe):
            return ""
        self.open = True
        head, self._head = self._head, ""
        return head

    def flush(self) -> str:
        if self.open or self.escalated:
            return ""
        head, self._head = self._head, ""
        return head


def _turn_line(msg: Any, lesson_md: str) -> Optional[str]:
    text = (msg.content_md or "").strip()
    if not text:
        return None
    if msg.role == "user":
        return f"**Docente:** {text[:_USER_CHARS]}"
    if msg.agent_kind == "tutor":
        clipped = text[:_ANSWER_CHARS] + ("…" if len(text) > _ANSWER_CHARS else "")
        return f"**Tu:** {clipped}"
    if text == lesson_md.strip():
        return "**Tu:** (la lezione: è il documento di lavoro qui sopra)"
    return "**Tu:** (una versione precedente della lezione)"


async def _history(session: Any, lesson: Any, before_turn: int) -> str:
    from aix.webui.lessons.models import LessonMessage

    rows = (
        await session.execute(
            select(LessonMessage)
            .where(
                LessonMessage.lesson_id == lesson.id,
                LessonMessage.turn_index < before_turn,
                LessonMessage.turn_index >= before_turn - _TURNS_KEPT,
            )
            .order_by(LessonMessage.turn_index, LessonMessage.created_at)
        )
    ).scalars()
    lines = [line for msg in rows if (line := _turn_line(msg, lesson.lesson_plan_md or ""))]
    return "\n## La conversazione finora\n\n" + "\n\n".join(lines) + "\n" if lines else ""


async def _documents(lesson: Any, instruction: str) -> tuple[str, list[dict[str, Any]]]:
    """The attached documents, searched for this request, as the Writer reads them."""
    from aix.webui.materials.embedding import private_rag_enabled

    if not private_rag_enabled():
        return "", []
    from aix.agent.agents.writer_agent import teacher_material_appendix
    from aix.webui.materials.retrieval import search_for_lesson

    search = search_for_lesson(lesson.id, lesson.owner_id)
    try:
        items = await search(instruction, request=instruction)
        maps = await search.maps()
    except Exception as exc:  # noqa: BLE001 — the answer still has the lesson
        logger.warning("[light] documents not searched (%s: %s)", type(exc).__name__, exc)
        return "", []
    if not items and not maps:
        return "", []
    return teacher_material_appendix(items, None, phase="light", material_maps=maps), items


def _messages(
    lesson: Any, instruction: str, history: str, documents: str, *, answer_here: bool
) -> list[dict[str, str]]:
    from aix.agent.prompts.profile_renderer import render_profile_section
    from aix.agent.prompts.today import today_note

    profile = {
        k: v for k, v in (lesson.educational_profile_json or {}).items() if not k.startswith("__")
    }
    system = LIGHT_SYSTEM_PROMPT.format(
        scope_rule=ANSWER_HERE_RULE if answer_here else ESCALATION_RULE
    )
    user = LIGHT_USER_TEMPLATE.format(
        today=today_note().strip(),
        profile=render_profile_section(profile),
        lesson=(lesson.lesson_plan_md or "")[:_LESSON_CHARS],
        documents=documents,
        history=history,
        instruction=instruction,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


async def _finish(answer: str, items: list[dict[str, Any]], lesson_md: str) -> str:
    """The same clean-up a Writer answer gets, minus the Critic."""
    from aix.agent.validation.answer_shape import tidy_answer
    from aix.agent.validation.citation_support import (
        drop_unknown_markers,
        drop_unsupported_citations,
    )

    # The lesson's own markers are known too: an answer may point at a page the
    # lesson already cites.
    known = items + [{"citation": m} for m in set(_MARKER.findall(lesson_md))]
    answer = drop_unknown_markers(tidy_answer(answer), known)
    return await drop_unsupported_citations(answer, items)


async def run_light_turn(
    lesson: Any, session: Any, *, answer_here: bool = False
) -> AsyncIterator[StreamEvent]:
    """Answer the latest request on ``lesson`` in one streamed call.

    The route has saved the request as the turn's user message and set the
    lesson to ``running``. On the way out the lesson is ``complete`` again with
    its plan untouched; the answer is a ``tutor`` message of the same turn.
    """
    from aix.agent.graph import write_stream
    from aix.core.concurrency import llm_slot
    from aix.core.config import config as app_config
    from aix.core.tracing import start_run
    from aix.webui.lessons.models import LessonMessage

    started = time.monotonic()
    lid = str(lesson.id)
    model = chat_model()
    turn = (
        await session.scalar(
            select(func.max(LessonMessage.turn_index)).where(
                LessonMessage.lesson_id == lesson.id, LessonMessage.role == "user"
            )
        )
        or 1
    )
    instruction = (
        await session.scalar(
            select(LessonMessage.content_md)
            .where(
                LessonMessage.lesson_id == lesson.id,
                LessonMessage.role == "user",
                LessonMessage.turn_index == turn,
            )
            .limit(1)
        )
        or ""
    )

    bus = write_stream.register(lid)
    yield StreamEvent(kind="light_pending", payload={"turn": turn})
    trace = start_run(
        "lesson.light",
        user_id=lesson.owner_id,
        session_id=lesson.id,
        tags=("webui", "light"),
        metadata={"turn": turn, "model": model},
        input=instruction,
    )
    output: Optional[str] = None
    error = ""
    try:
        documents, items = await _documents(lesson, instruction)
        messages = _messages(
            lesson,
            instruction,
            await _history(session, lesson, turn),
            documents,
            answer_here=answer_here,
        )
        kwargs = app_config.openai.build_completion_kwargs(
            temperature=0.4,
            max_tokens=_MAX_TOKENS,
            include_reasoning=False,
            model_override=model,
        )
        gate = EscalationGate()
        parts: list[str] = []
        first_token: Optional[float] = None
        client = app_config.openai.get_async_client()
        async with llm_slot():
            stream = await client.chat.completions.create(messages=messages, stream=True, **kwargs)
            async for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None
                delta = (choice.delta.content or "") if choice and choice.delta else ""
                if not delta:
                    continue
                shown = gate.feed(delta)
                if gate.escalated:
                    break
                if shown:
                    first_token = first_token or time.monotonic() - started
                    parts.append(shown)
                    bus.put_nowait(("content", shown))
            if tail := gate.flush():
                parts.append(tail)
                bus.put_nowait(("content", tail))

        if gate.escalated:
            # The request leaves this turn: the card offers to send it again,
            # to the full chain or back here. A user message with no answer
            # would sit in the history as a question nobody took.
            await session.execute(
                delete(LessonMessage).where(
                    LessonMessage.lesson_id == lesson.id,
                    LessonMessage.turn_index == turn,
                )
            )
            lesson.status = "complete"
            await session.commit()
            output = NEW_LESSON
            logger.info("[light] lesson_id=%s turn=%s handed to the full chain", lid, turn)
            yield StreamEvent(kind="light_escalate", payload={"query": instruction})
            return

        answer = (await _finish("".join(parts), items, lesson.lesson_plan_md or "")).strip()
        if not answer:
            raise RuntimeError("Il modello non ha prodotto una risposta.")
        meta = {
            "kind": "answer",
            "model": model,
            "seconds": round(time.monotonic() - started, 1),
            "first_token_seconds": round(first_token, 1) if first_token else None,
            "passages": len(items),
        }
        session.add(
            LessonMessage(
                lesson_id=lesson.id,
                role="assistant",
                content_md=answer,
                turn_index=turn,
                agent_kind="tutor",
                meta_json=meta,
            )
        )
        lesson.status = "complete"
        await session.commit()
        output = answer
        logger.info(
            "[light] lesson_id=%s turn=%s answered in %.1fs (first token %.1fs, %d passages)",
            lid,
            turn,
            meta["seconds"],
            first_token or 0,
            len(items),
        )
        yield StreamEvent(kind="light_done", meta=meta)
    except Exception as exc:  # noqa: BLE001
        error = type(exc).__name__
        message = str(_report_failure(exc))
        try:
            # The lesson is as it was: a failed answer does not break it.
            lesson.status = "complete"
            await session.commit()
        except Exception:  # noqa: BLE001
            logger.exception("[light] could not restore lesson_id=%s", lid)
            await session.rollback()
        yield StreamEvent(kind="light_error", error=message)
    finally:
        bus.put_nowait(None)
        write_stream.deregister(lid)
        trace.end(
            output=output,
            metadata={"seconds": round(time.monotonic() - started, 1)},
            error=error or ("" if output is not None else "interrupted"),
        )
