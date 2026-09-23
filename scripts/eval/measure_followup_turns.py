#!/usr/bin/env python3
"""Measure what a follow-up turn costs, with and without reusing the corpus.

Why this exists
---------------
After the first answer, a teacher keeps talking: "shorten the introduction",
"add a closing question". Those turns bring no new subject, and searching the
Knowledge Graph again for them costs time and changes nothing. The reuse path
has existed since 2026-09-09 behind ``AIX_CONDITIONAL_RETRIEVAL_ENABLED``, but
it was off by default and nobody had measured it on a real thread: on
2026-09-22 a follow-up searched the graph again (14 s) and took 101.8 s.

This drives one real conversation through the compiled graph, turn by turn, on
one thread -- the same way the WebUI does, conversation history included -- and
prints for every turn which way the router went and how long each node took.
It calls the real models: a three-turn run costs a few minutes and a few
lesson generations' worth of tokens.

Usage::

    python scripts/eval/measure_followup_turns.py            # reuse on
    python scripts/eval/measure_followup_turns.py --off      # reuse off, for the baseline
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

#: A first request, then two that only reshape it, then one that brings a new
#: subject. The last one is the check in the other direction: reuse is only
#: right if the router still searches when a turn needs it.
TURNS = [
    ("first", "Crea una lezione di 50 minuti sulla fotosintesi per una classe seconda media."),
    ("reshape", "Accorcia l'introduzione a 5 minuti e aggiungi una domanda di verifica alla fine."),
    ("reshape", "Riscrivi le consegne delle attività con frasi più brevi e un linguaggio più semplice."),
    ("new subject", "Aggiungi una seconda parte sulla respirazione cellulare e sul ciclo di Krebs."),
]


class RouterLog(logging.Handler):
    """Keeps the router's own words, so the report says *why* it chose."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "[Router]" in message or "ReuseRetrieval" in message:
            self.lines.append(message)


async def run(reuse: bool) -> None:
    os.environ["AIX_CONDITIONAL_RETRIEVAL_ENABLED"] = "true" if reuse else "false"

    from aix.agent.graph.checkpointer import thread_config
    from aix.agent.graph.state import create_initial_state
    from aix.agent.orchestrator import AgentOrchestrator

    router = RouterLog()
    logging.getLogger("aix.agent.graph.nodes").addHandler(router)
    logging.getLogger("aix.agent.graph.nodes").setLevel(logging.INFO)

    pipeline = AgentOrchestrator(domain="udl", language="it").get_pipeline()
    graph = await pipeline.get_graph()
    thread = f"measure-followup-{uuid.uuid4()}"
    config = thread_config(thread)

    history: list[dict[str, str]] = []
    rows = []
    for label, query in TURNS:
        router.lines.clear()
        state = create_initial_state(
            query=query,
            domain=pipeline.domain,
            language=pipeline.language,
            session_id=thread,
            max_revisions=pipeline.max_revisions,
            conversation_history=list(history) or None,
            raw_user_turn=query,
        )

        started = time.monotonic()
        last = started
        nodes: list[tuple[str, float]] = []
        final: dict = {}
        async for update in graph.astream(state, config, stream_mode="updates"):
            now = time.monotonic()
            for node, delta in update.items():
                nodes.append((node, now - last))
                if isinstance(delta, dict):
                    final.update(delta)
            last = now
        total = time.monotonic() - started

        answer = final.get("lesson_plan_draft") or ""
        history += [{"role": "user", "content": query}, {"role": "assistant", "content": answer}]

        searched = any(name == "retrieve" for name, _ in nodes)
        reused = any(name == "reuse" for name, _ in nodes)
        rows.append((label, query, total, searched, reused, nodes, list(router.lines)))

    print(f"# Follow-up turns — reuse {'ON' if reuse else 'OFF'}\n")
    print("| # | turno | KG | totale | nodi più lenti |")
    print("| ---: | --- | --- | ---: | --- |")
    for index, (label, _q, total, searched, reused, nodes, _r) in enumerate(rows, start=1):
        kg = "riusato" if reused else ("cercato" if searched else "—")
        slow = ", ".join(f"{n} {s:.0f}s" for n, s in sorted(nodes, key=lambda x: -x[1])[:3])
        print(f"| {index} | {label} | {kg} | {total:.1f}s | {slow} |")
    print()
    for index, (label, query, *_rest, lines) in enumerate(rows, start=1):
        print(f"**Turno {index} ({label})** — {query}")
        for line in lines or ["(nessuna decisione del router registrata)"]:
            print(f"  - {line}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--off", action="store_true", help="measure with reuse switched off")
    args = parser.parse_args()
    asyncio.run(run(reuse=not args.off))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
