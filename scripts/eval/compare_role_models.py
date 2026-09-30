#!/usr/bin/env python3
"""Which fast model for the Critic and for text2cypher.

Both run on ``google/gemini-2.5-flash``, which is being retired. This compares
candidates on the work each role actually does, inside the running stack (it
needs the WebUI database and Neo4j, so run it in the app container).

critic
    Real lessons from the database, plus three copies of some of them with a
    defect put in on purpose: the differentiation section removed, section
    minutes that no longer add up, a request the lesson does not answer. Each
    case is judged twice per model. Reported: approvals of the real lessons,
    defects caught (REVISE, and the feedback names the problem), agreement with
    the reference model, verdicts that change between the two runs, invalid
    JSON, median seconds.

text2cypher
    Typical planner queries, Italian and English, on both domains. Reported:
    queries executed without error, queries with results, median seconds, and
    how much the results overlap with the reference model's.

Usage (in the app container)::

    python scripts/eval/compare_role_models.py critic
    python scripts/eval/compare_role_models.py text2cypher \\
        --models google/gemini-2.5-flash,google/gemini-3.5-flash-lite
"""

from __future__ import annotations

import argparse
import asyncio
import re
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_MODELS = [
    "google/gemini-2.5-flash",  # reference: today's model
    "google/gemini-3.1-flash-lite",
    "google/gemini-3.5-flash-lite",
    "google/gemini-3.7-flash",
]

# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

#: Lessons generated from a creation request, newest first.
LESSONS_SQL = """
select l.domain, m.content_md,
  (select u.content_md from lesson_message u where u.lesson_id = m.lesson_id
     and u.role = 'user' and u.turn_index = m.turn_index limit 1) as query
from lesson_message m join lesson l on l.id = m.lesson_id
where m.role = 'assistant' and length(m.content_md) > 5000
order by m.created_at desc limit 40
"""

OFF_TOPIC = "Crea una lezione di 50 minuti sulla fotosintesi clorofilliana per una seconda media"


def _no_differentiation(lesson: str) -> str:
    return re.sub(
        r"\n#+\s*[^\n]*Differenzia[^\n]*\n.*?(?=\n#+\s|\n\*Fonti|\Z)",
        "\n",
        lesson,
        flags=re.S | re.I,
    )


def _wrong_minutes(lesson: str) -> str:
    return re.sub(
        r"\((\d{1,3}) (min|minuti)\)", lambda m: f"({int(m.group(1)) * 3} {m.group(2)})", lesson
    )


#: name -> (how to break the lesson, how to break the request, words the feedback should use)
DEFECTS = {
    "senza differenziazione": (_no_differentiation, None, ("differenzi", "bes", "dsa", "inclusi")),
    "minuti sbagliati": (_wrong_minutes, None, ("minut", "durata", "tempo", "tempi")),
    "fuori richiesta": (
        None,
        OFF_TOPIC,
        ("fotosintesi", "richiesta", "argomento", "pertinen", "coeren"),
    ),
}


async def _cases(limit: int, broken: int):
    from sqlalchemy import text

    from aix.webui.db import async_session_maker

    async with async_session_maker() as session:
        rows = (await session.execute(text(LESSONS_SQL))).all()
    creation = [
        r
        for r in rows
        if r.query
        and re.search(r"\b(crea|create|lezione|lesson|migliorare|approfondimento)\b", r.query, re.I)
    ][:limit]
    cases = [
        {
            "name": f"lezione {i + 1}",
            "domain": r.domain,
            "query": r.query,
            "lesson": r.content_md,
            "defect": None,
        }
        for i, r in enumerate(creation)
    ]
    for i, r in enumerate(creation[:broken]):
        for defect, (break_lesson, other_query, _words) in DEFECTS.items():
            lesson = break_lesson(r.content_md) if break_lesson else r.content_md
            if lesson == r.content_md and break_lesson:
                continue  # nothing to break in this lesson
            cases.append(
                {
                    "name": f"lezione {i + 1} · {defect}",
                    "domain": r.domain,
                    "query": other_query or r.query,
                    "lesson": lesson,
                    "defect": defect,
                }
            )
    return cases


async def _judge(model: str, case: dict):
    from aix.agent.agents import critic_agent
    from aix.agent.agents.retriever_agent import RetrievalResult

    critic_agent._CRITIC_MODEL = model
    started = time.monotonic()
    try:
        result = await critic_agent.CriticAgent().critique(
            lesson_plan=case["lesson"],
            teacher_query=case["query"],
            retrieval_result=RetrievalResult(
                nodes=[], relationships=[], recommendations=[], confidence="MEDIUM"
            ),
            revision_count=0,
            max_revisions=1,
            domain=case["domain"],
        )
    except Exception as exc:  # noqa: BLE001 — a failure is a result
        return {"ok": False, "seconds": time.monotonic() - started, "error": type(exc).__name__}
    feedback = " ".join(
        [
            result.revision_instructions or "",
            result.summary or "",
            *map(str, result.weaknesses or []),
        ]
    ).lower()
    return {
        # The Critic reports a JSON it could not read in its summary, not as an error.
        "ok": not any(
            marker in (result.summary or "")
            for marker in ("[parse_error]", "parsing error", "Auto-approved")
        ),
        "seconds": time.monotonic() - started,
        "approved": result.approved,
        "score": result.average_score,
        "feedback": feedback,
    }


async def run_critic(models: list[str], limit: int, broken: int, runs: int) -> None:
    cases = await _cases(limit, broken)
    print(
        f"# Critic — {len(cases)} casi ({sum(1 for c in cases if not c['defect'])} lezioni vere), "
        f"{runs} giri per modello\n"
    )
    verdicts: dict[str, list[list[dict]]] = {}
    for model in models:  # one model at a time: the Critic reads its model from a module global
        gate = asyncio.Semaphore(6)

        async def one(case, model=model, gate=gate):
            async with gate:
                return [await _judge(model, case) for _ in range(runs)]

        verdicts[model] = await asyncio.gather(*(one(c) for c in cases))

    reference = models[0]
    print(
        "| modello | lezioni vere approvate | difetti colti | accordo con il riferimento | "
        "verdetti instabili | JSON non validi | s (mediana) |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for model in models:
        good = bad = caught = agree = unstable = invalid = 0
        approved_good = 0
        seconds = []
        for case, judged in zip(cases, verdicts[model], strict=True):
            first = judged[0]
            seconds += [j["seconds"] for j in judged]
            invalid += sum(not j["ok"] for j in judged)
            if len({j.get("approved") for j in judged}) > 1:
                unstable += 1
            ref = verdicts[reference][cases.index(case)][0]
            agree += first.get("approved") == ref.get("approved")
            if case["defect"]:
                bad += 1
                words = DEFECTS[case["defect"]][2]
                caught += (first.get("approved") is False) and any(
                    w in first.get("feedback", "") for w in words
                )
            else:
                good += 1
                approved_good += bool(first.get("approved"))
        print(
            f"| {model.split('/')[-1]} | {approved_good}/{good} | {caught}/{bad} | "
            f"{agree}/{len(cases)} | {unstable} | {invalid} | {statistics.median(seconds):.1f} |"
        )

    print(
        "\nPer caso (verdetto del primo giro: A approva, R revisione; tra parentesi il punteggio):\n"
    )
    print("| caso | " + " | ".join(m.split("/")[-1] for m in models) + " |")
    print("| --- |" + " --- |" * len(models))
    for i, case in enumerate(cases):
        cells = []
        for model in models:
            j = verdicts[model][i][0]
            if not j["ok"]:
                cells.append("errore")
                continue
            cells.append(f"{'A' if j['approved'] else 'R'} ({j['score']:.1f})")
        print(f"| {case['name']} | " + " | ".join(cells) + " |")


# ---------------------------------------------------------------------------
# text2cypher
# ---------------------------------------------------------------------------

QUERIES = [
    ("neuro", "strategie per ridurre il carico sulla memoria di lavoro"),
    ("neuro", "attention regulation techniques for students with ADHD"),
    ("neuro", "come trasformare le emozioni negative in concentrazione"),
    ("neuro", "retrieval practice e memoria a lungo termine"),
    ("neuro", "feedback formativo e metacognizione"),
    ("neuro", "attività per studenti con difficoltà di attenzione"),
    ("neuro", "cooperative learning and the social brain"),
    ("neuro", "dual coding per studenti con DSA"),
    ("udl", "molteplici mezzi di coinvolgimento per studenti demotivati"),
    ("udl", "strumenti digitali per simulazioni in classe"),
    ("udl", "UDL checkpoints for reading comprehension"),
    ("udl", "rappresentazioni multiple per studenti con dislessia"),
    ("udl", "scaffolding in matematica"),
    ("udl", "flexible assessment options"),
    ("udl", "supporto alle funzioni esecutive"),
    ("udl", "lavoro di gruppo e collaborazione"),
]


def _names(records) -> set[str]:
    names: set[str] = set()

    def walk(value):
        if isinstance(value, dict):
            for k, v in value.items():
                if k in ("name", "n.name", "title") and isinstance(v, str):
                    names.add(v.lower())
                walk(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                walk(v)
        elif hasattr(value, "items"):
            walk(dict(value.items()))

    walk(records or [])
    return names


def run_text2cypher(models: list[str]) -> None:
    from aix.core.config import config
    from aix.retrieval.multilingual_text2cypher import MultilingualText2Cypher

    print(f"# text2cypher — {len(QUERIES)} domande, udl e neuro, italiano e inglese\n")
    found: dict[str, list[set[str]]] = {}
    rows = []
    for model in models:
        config.text2cypher.model = model
        pipeline = MultilingualText2Cypher()
        ok = with_results = 0
        seconds = []
        names = []
        try:
            for domain, query in QUERIES:
                started = time.monotonic()
                try:
                    result = pipeline.process_query(query, domain=domain, execute=True)
                except Exception:  # noqa: BLE001
                    result = {"execution_error": "exception", "results": []}
                seconds.append(time.monotonic() - started)
                ok += not result.get("execution_error")
                with_results += bool(result.get("results"))
                names.append(_names(result.get("results")))
        finally:
            pipeline.close()
        found[model] = names
        rows.append((model, ok, with_results, statistics.median(seconds)))

    reference = found[models[0]]
    print(
        "| modello | eseguite senza errori | con risultati | sovrapposizione con il riferimento | s (mediana) |"
    )
    print("| --- | ---: | ---: | ---: | ---: |")
    for model, ok, with_results, median in rows:
        overlaps = [
            len(a & b) / len(a | b) for a, b in zip(found[model], reference, strict=True) if a or b
        ]
        overlap = statistics.mean(overlaps) if overlaps else 0
        print(
            f"| {model.split('/')[-1]} | {ok}/{len(QUERIES)} | {with_results}/{len(QUERIES)} | "
            f"{overlap * 100:.0f}% | {median:.1f} |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("role", choices=["critic", "text2cypher"])
    parser.add_argument(
        "--models", default=",".join(DEFAULT_MODELS), help="the first is the reference"
    )
    parser.add_argument("--lessons", type=int, default=8, help="real lessons to judge")
    parser.add_argument("--broken", type=int, default=3, help="lessons to break on purpose")
    parser.add_argument("--runs", type=int, default=2, help="judgements per case and model")
    args = parser.parse_args()
    models = [m for m in args.models.split(",") if m]
    if args.role == "critic":
        asyncio.run(run_critic(models, args.lessons, args.broken, args.runs))
    else:
        run_text2cypher(models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
