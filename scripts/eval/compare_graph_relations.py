#!/usr/bin/env python3
"""Lessons with and without the graph's relationships in the Writer's prompt.

The relationships the retriever returns (ADHD SUGGESTS movement breaks, ...)
never reached the Writer until ``AIX_WRITER_GRAPH_RELATIONS``. This runs the
whole chain -- planner, graph, Writer, Critic -- on requests the graph knows,
once with the switch off and once on, and writes for each run: the lesson, the
Critic's score, how many relationships there were, and the time window, so the
cost can be read from Langfuse. Plus a page to read the pairs blind.

It needs Neo4j and the model keys: run it in the app container. The lessons
are written to ``--out`` (outside the repository).

    python scripts/eval/compare_graph_relations.py --out /tmp/relations
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import os
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

REQUESTS = [
    (
        "udl",
        "Crea una lezione di 50 minuti sulle frazioni per una quarta elementare con due alunni con ADHD",
    ),
    (
        "udl",
        "Prepara una lezione di comprensione del testo per una prima media con studenti con dislessia",
    ),
    ("neuro", "Crea una lezione di 45 minuti sulla memoria di lavoro per una seconda media"),
    ("neuro", "Lezione per aumentare la motivazione intrinseca in una terza media demotivata"),
]


async def one(domain: str, query: str, relations: bool) -> dict:
    os.environ["AIX_WRITER_GRAPH_RELATIONS"] = "true" if relations else "false"
    from aix.agent.graph.checkpointer import thread_config
    from aix.agent.graph.state import create_initial_state
    from aix.agent.orchestrator import AgentOrchestrator

    pipeline = AgentOrchestrator(domain=domain, language="it").get_pipeline()
    graph = await pipeline.get_graph()
    thread = f"relations-{uuid.uuid4()}"
    state = create_initial_state(
        query=query,
        domain=domain,
        language="it",
        session_id=thread,
        max_revisions=pipeline.max_revisions,
        raw_user_turn=query,
    )
    final: dict = {}
    began = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    async for update in graph.astream(state, thread_config(thread), stream_mode="updates"):
        for delta in update.values():
            if isinstance(delta, dict):
                final.update(delta)
    return {
        "domain": domain,
        "query": query,
        "relations": relations,
        "began": began,
        "ended": datetime.now(timezone.utc).isoformat(),
        "seconds": round(time.monotonic() - started, 1),
        "relationships": len(final.get("retrieved_relationships") or []),
        "critic": final.get("critique_score"),
        "lesson": final.get("final_lesson_plan") or final.get("lesson_plan_draft") or "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    async def all_runs() -> list[dict]:
        # One event loop for every run: clients and locks are bound to the loop
        # that made them, and a new loop per run breaks the second one.
        done = []
        for domain, query in REQUESTS:
            for relations in (False, True):
                run = await one(domain, query, relations)
                done.append(run)
                print(
                    f"{'con' if relations else 'senza'} relazioni  {run['seconds']:5.0f}s  "
                    f"relazioni {run['relationships']:3}  Critic {run['critic']}  «{query[:50]}»",
                    flush=True,
                )
        return done

    runs = asyncio.run(all_runs())
    (out / "runs.json").write_text(json.dumps(runs, ensure_ascii=False, indent=1), encoding="utf-8")

    import markdown

    rng = random.Random(9)
    key, sections = {}, []
    for n, (_domain, query) in enumerate(REQUESTS, 1):
        pair = [r for r in runs if r["query"] == query]
        rng.shuffle(pair)
        key[f"lezione {n}"] = {
            chr(65 + i): ("con" if r["relations"] else "senza") + " relazioni"
            for i, r in enumerate(pair)
        }
        cols = "".join(
            f"<section><h3>Versione {chr(65 + i)}</h3>{markdown.markdown(r['lesson'], extensions=['tables'])}</section>"
            for i, r in enumerate(pair)
        )
        sections.append(
            f"<article><h2>Lezione {n}</h2><p class='q'><b>Richiesta:</b> {html.escape(query)}</p>"
            f"<div class='cols'>{cols}</div></article>"
        )
    page = (
        "<!doctype html><html lang='it'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'><title>Relazioni del grafo</title><style>"
        "body{font-family:system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}"
        "article{margin:32px 0;border-top:3px solid #333;padding-top:8px}"
        ".cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:12px}"
        "section{background:#fff;border:1px solid #ddd;border-radius:6px;padding:12px;overflow-x:auto;font-size:14px}"
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:2px 6px}.q{background:#eef;padding:8px}"
        "</style></head><body><h1>Quale lezione è più fondata?</h1>"
        "<p>Stessa richiesta, due lezioni in ordine nascosto. Scrivi quale preferisci (A, B o «uguali») e perché, "
        "guardando soprattutto se le strategie sono adatte alla classe descritta.</p>"
        + "".join(sections)
        + "</body></html>"
    )
    (out / "lettura_alla_cieca.html").write_text(page, encoding="utf-8")
    (out / "chiave.json").write_text(
        json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
