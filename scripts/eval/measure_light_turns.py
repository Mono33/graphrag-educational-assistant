#!/usr/bin/env python3
"""What a light follow-up turn costs, on a real lesson.

``measure_followup_turns.py`` measured the full chain on follow-ups: 50–70 s,
65–85 % of it the Writer rewriting the whole lesson. This asks a finished
lesson five things a teacher asks after the first answer, through the light
path the WebUI uses, and prints for each: first token, total time, passages
from the documents, citations kept, and whether it was handed to the full
chain. The last request is a new lesson, and should be.

The targets (docs/product/Piano_Turni_Veloci.md): first token under 3 s, the
whole answer under 12 s.

It needs the WebUI database, so run it in the app container. The turns it adds
are removed at the end unless ``--keep``. It calls the real model.

Usage (in the app container)::

    python scripts/eval/measure_light_turns.py                  # latest finished lesson
    python scripts/eval/measure_light_turns.py --lesson <uuid> --keep
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

#: Generic on purpose: they fit any lesson, with or without attachments.
REQUESTS = [
    ("domanda", "Perché hai scelto la prima attività? Che cosa deve ottenere?"),
    ("chiarimento", "Spiegami meglio il concetto principale della lezione, con un esempio."),
    ("documenti", "Che cosa dicono i documenti allegati sull'argomento della lezione?"),
    ("variante", "Proponi una variante dell'ultima attività per uno studente con DSA."),
    ("lezione nuova", "Fammi una lezione nuova sulla rivoluzione francese per una terza media."),
]


async def run(lesson_id: str | None, keep: bool) -> None:
    os.environ.setdefault("AIX_LIGHT_TURNS_ENABLED", "true")
    from sqlalchemy import delete, func, select

    from aix.webui.agent.light import chat_model, run_light_turn
    from aix.webui.db import async_session_maker
    from aix.webui.lessons.models import Lesson, LessonMessage

    async with async_session_maker() as session:
        query = select(Lesson).where(
            Lesson.status == "complete", Lesson.lesson_plan_md.is_not(None)
        )
        if lesson_id:
            query = query.where(Lesson.id == uuid.UUID(lesson_id))
        lesson = (
            await session.execute(query.order_by(Lesson.updated_at.desc()).limit(1))
        ).scalar_one_or_none()
        if lesson is None:
            print("Nessuna lezione completa trovata.")
            return
        first_turn = (
            await session.scalar(
                select(func.max(LessonMessage.turn_index)).where(
                    LessonMessage.lesson_id == lesson.id
                )
            )
            or 0
        ) + 1

        rows = []
        for offset, (label, text) in enumerate(REQUESTS):
            turn = first_turn + offset
            session.add(
                LessonMessage(lesson_id=lesson.id, role="user", content_md=text, turn_index=turn)
            )
            lesson.status = "running"
            await session.commit()
            started = time.monotonic()
            events = [e async for e in run_light_turn(lesson, session)]
            total = time.monotonic() - started
            end = events[-1]
            answer = await session.scalar(
                select(LessonMessage.content_md).where(
                    LessonMessage.lesson_id == lesson.id,
                    LessonMessage.turn_index == turn,
                    LessonMessage.agent_kind == "tutor",
                )
            )
            rows.append(
                (
                    label,
                    end.kind.removeprefix("light_"),
                    end.meta.get("first_token_seconds"),
                    total,
                    end.meta.get("passages", 0),
                    (answer or "").count("[[material:"),
                    len(answer or ""),
                )
            )

        if not keep:
            await session.execute(
                delete(LessonMessage).where(
                    LessonMessage.lesson_id == lesson.id, LessonMessage.turn_index >= first_turn
                )
            )
            await session.commit()

    print(f"# Turni veloci — lezione {lesson.id}, modello {chat_model()}\n")
    print("| richiesta | esito | primo token | totale | passaggi | citazioni | caratteri |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for label, outcome, first, total, passages, cited, chars in rows:
        first_cell = f"{first:.1f}s" if first else "—"
        print(
            f"| {label} | {outcome} | {first_cell} | {total:.1f}s | {passages} | {cited} | {chars} |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lesson", help="the lesson id; default the latest finished one")
    parser.add_argument("--keep", action="store_true", help="keep the turns in the lesson")
    args = parser.parse_args()
    asyncio.run(run(args.lesson, args.keep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
