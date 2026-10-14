#!/usr/bin/env python3
"""Which fast model for the Critic, for text2cypher and for the Planner.

The first two run on ``google/gemini-2.5-flash``, which is being retired. This compares
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

planner
    The plan is a classification (language, intent, scope, lesson type) and an
    extraction (concepts, search queries). 30 teacher requests, Italian first, with
    out-of-scope ones; each model plans each one with escalation OFF, so what is
    measured is the model, not the safety net. The first model is the reference
    (``default`` = the model the Writer uses today). Reported against it:
    agreement on language, intent, scope and lesson type; out-of-scope recall and
    false alarms; overlap of the key concepts; how many plans the safety net would
    send back (``weak_plan_reason``); unparseable plans; median seconds. With
    ``--retrieval`` (needs Neo4j) also the overlap of what the graph returns for the
    plan's search queries, which is what matters in the end. The reference is not
    ground truth: every disagreement is listed, to be read by a person.

Usage (in the app container)::

    python scripts/eval/compare_role_models.py critic
    python scripts/eval/compare_role_models.py planner --retrieval \\
        --models default,google/gemini-3.5-flash-lite,google/gemini-3.1-flash-lite
    python scripts/eval/compare_role_models.py text2cypher \\
        --models google/gemini-2.5-flash,google/gemini-3.5-flash-lite
    # the same model twice: how much it differs from itself (the noise floor)
    python scripts/eval/compare_role_models.py text2cypher \\
        --models google/gemini-3.1-flash-lite,google/gemini-3.1-flash-lite
"""

from __future__ import annotations

import argparse
import asyncio
import os
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

#: As the graph received them from real lessons (Langfuse, 2026-09-30 → 10-07):
#: short, often a bare keyword or a subject outside the graph.
REAL_QUERIES = [
    ("udl", "ADHD"),
    ("neuro", "ADHD"),
    ("udl", "DLDS"),
    ("neuro", "DOP"),
    ("neuro", "guerra fredda"),
    ("udl", "storia e geografia"),
    ("udl", "crisi israelo-palestinese"),
    ("udl", "intelligenza artificiale"),
    ("udl", "authentic task design for practical application"),
    ("neuro", "no-device classroom attention activities"),
    ("neuro", "attention span and fatigue in students"),
    ("neuro", "selective attention mechanisms"),
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

    queries = QUERIES + REAL_QUERIES
    print(
        f"# text2cypher — {len(queries)} domande, udl e neuro, italiano e inglese "
        f"({len(REAL_QUERIES)} prese da lezioni vere)\n"
    )
    found: dict[str, list[set[str]]] = {}
    rows = []
    for model in models:
        config.text2cypher.model = model
        # The same model twice measures its own noise: a candidate is only worse
        # than the reference if it overlaps less than the reference with itself.
        label = model if model not in found else f"{model} (ripetuto)"
        pipeline = MultilingualText2Cypher()
        ok = with_results = 0
        seconds = []
        names = []
        try:
            for domain, query in queries:
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
        found[label] = names
        rows.append((label, ok, with_results, statistics.median(seconds)))

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
            f"| {model.split('/')[-1]} | {ok}/{len(queries)} | {with_results}/{len(queries)} | "
            f"{overlap * 100:.0f}% | {median:.1f} |"
        )


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

#: (domain, request). Italian first, as the teachers write; out-of-scope on purpose.
PLANNER_CASES = [
    ("udl", "Crea una lezione di 50 minuti sulla fotosintesi clorofilliana per una seconda media"),
    (
        "udl",
        "Prepara una lezione sulla Prima guerra mondiale per una terza superiore "
        "con due studenti con dislessia",
    ),
    (
        "udl",
        "Vorrei un'attività di 30 minuti sulle frazioni per una quarta elementare "
        "con un alunno con ADHD",
    ),
    (
        "udl",
        "Progetta una verifica sul sistema solare per una prima media con tempi "
        "aggiuntivi per studenti con DSA",
    ),
    ("udl", "Crea un piano di unità di 4 settimane sulla Costituzione italiana per il biennio"),
    (
        "udl",
        "Lezione sulla Resistenza italiana con strategie UDL per studenti con "
        "difficoltà di comprensione del testo",
    ),
    ("neuro", "Cos'è la neuroplasticità?"),
    ("udl", "Cosa significa Universal Design for Learning?"),
    ("neuro", "Spiega come funziona la memoria di lavoro"),
    ("neuro", "Perché la metacognizione aiuta l'apprendimento?"),
    ("neuro", "What is retrieval practice?"),
    ("neuro", "Differenza tra memoria dichiarativa e procedurale"),
    ("neuro", "Confronta motivazione intrinseca ed estrinseca"),
    ("udl", "Compare scaffolding and differentiation in UDL"),
    ("udl", "Quali strategie usare con uno studente con ADHD durante le verifiche?"),
    ("neuro", "Suggerisci attività per migliorare l'attenzione in classe"),
    ("neuro", "Elenca strategie per ridurre l'ansia da prestazione"),
    ("neuro", "Come posso motivare uno studente demotivato?"),
    ("udl", "Strategie per studenti con discalculia"),
    ("udl", "Come adattare un compito di scrittura per uno studente con disgrafia?"),
    ("neuro", "Qual è la ricetta della carbonara?"),
    ("neuro", "Chi ha vinto i mondiali di calcio del 2006?"),
    ("udl", "Scrivimi una poesia sul mare"),
    ("udl", "Come si risolve un'equazione di secondo grado?"),
    ("udl", "Quanto costa un biglietto del treno Roma-Milano?"),
    ("neuro", "Crée une leçon sur la mémoire de travail pour des élèves de 10 ans"),
    ("udl", "Diseña una actividad sobre la atención para estudiantes con TDAH"),
    ("udl", "Create a 45-minute lesson on fractions for fifth grade students with dyscalculia"),
    ("neuro", "attività pratica sulle emozioni"),
    ("neuro", "ADHD"),
]


async def _plan_all(model: str):
    from aix.agent.agents.planner_agent import PlannerAgent

    if model == "default":
        os.environ.pop("AIX_PLANNER_MODEL", None)
    else:
        os.environ["AIX_PLANNER_MODEL"] = model
    agent = PlannerAgent()
    plans = []
    for domain, query in PLANNER_CASES:
        started = time.monotonic()
        plan = await agent.plan(query, domain=domain, language="it")
        plans.append((plan, time.monotonic() - started))
    return plans


def _as_data(plan) -> dict:
    return {
        "query_intent": plan.query_intent,
        "intent_confidence": plan.intent_confidence,
        "language_confidence": plan.language_confidence,
        "response_language": plan.response_language,
        "scope_confidence": plan.scope_confidence,
    }


def _concepts(plan) -> set[str]:
    return {c.strip().lower() for c in (plan.key_concepts or []) if isinstance(c, str)}


def _graph_names(domain: str, plan, processor) -> set[str]:
    names: set[str] = set()
    for query in (plan.search_queries or [])[:3]:
        try:
            result = processor.process_query(query, domain=domain, execute=True)
        except Exception:  # noqa: BLE001
            continue
        names |= _names(result.get("results"))
    return names


def run_planner(models: list[str], retrieval: bool) -> None:
    from aix.agent.agents.planner_agent import weak_plan_reason

    os.environ["AIX_PLANNER_ESCALATE"] = "false"  # measure the model, not the safety net
    print(f"# planner — {len(PLANNER_CASES)} richieste, riferimento: {models[0]}\n")

    plans: dict[str, list] = {}
    for model in models:
        print(f"... {model}", file=sys.stderr)
        plans[model] = asyncio.run(_plan_all(model))
    reference = plans[models[0]]

    graph: dict[str, list[set[str]]] = {}
    if retrieval:
        from aix.retrieval.multilingual_text2cypher import MultilingualText2Cypher

        processor = MultilingualText2Cypher()
        try:
            for model in models:
                graph[model] = [
                    _graph_names(case[0], plan, processor)
                    for case, (plan, _) in zip(PLANNER_CASES, plans[model], strict=True)
                ]
        finally:
            processor.close()

    def agree(model: str, field: str) -> str:
        pairs = [
            (getattr(m, field), getattr(r, field))
            for (m, _), (r, _) in zip(plans[model], reference, strict=True)
            if getattr(r, field) is not None
        ]
        same = sum(str(a).lower() == str(b).lower() for a, b in pairs)
        return f"{same}/{len(pairs)}"

    header = (
        "| modello | lingua | intento | scope | tipo lezione | fuori tema colti | falsi allarmi "
        "| concetti (sovrapp.) | rimandati da rete | non leggibili | s (mediana) |"
    )
    rule = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    if retrieval:
        header += " grafo (sovrapp.) |"
        rule += " ---: |"
    print(header)
    print(rule)
    for model in models:
        rows = list(zip(plans[model], reference, strict=True))
        oos_ref = [(m, r) for (m, _), (r, _) in rows if r.scope_status == "out_of_scope"]
        caught = sum(m.scope_status == "out_of_scope" for m, _ in oos_ref)
        false_alarms = sum(
            m.scope_status == "out_of_scope" and r.scope_status != "out_of_scope"
            for (m, _), (r, _) in rows
        )
        jaccard = [
            len(_concepts(m) & _concepts(r)) / len(_concepts(m) | _concepts(r))
            for (m, _), (r, _) in rows
            if _concepts(m) | _concepts(r)
        ]
        weak = sum(bool(weak_plan_reason(_as_data(m))) for m, _ in plans[model])
        broken = sum((m.reasoning or "").startswith("Fallback plan") for m, _ in plans[model])
        median = statistics.median(s for _, s in plans[model])
        line = (
            f"| {model.split('/')[-1]} | {agree(model, 'response_language')} "
            f"| {agree(model, 'query_intent')} | {agree(model, 'scope_status')} "
            f"| {agree(model, 'lesson_type')} | {caught}/{len(oos_ref)} | {false_alarms} "
            f"| {statistics.mean(jaccard) * 100:.0f}% | {weak} | {broken} | {median:.1f} |"
        )
        if retrieval:
            overlaps = [
                len(a & b) / len(a | b)
                for a, b in zip(graph[model], graph[models[0]], strict=True)
                if a or b
            ]
            line += f" {statistics.mean(overlaps) * 100 if overlaps else 0:.0f}% |"
        print(line)

    print(
        "\n## Disaccordi con il riferimento (da leggere a mano: il riferimento non è la verità)\n"
    )
    for model in models[1:]:
        print(f"### {model}")
        for (domain, query), (m, _), (r, _) in zip(
            PLANNER_CASES, plans[model], reference, strict=True
        ):
            diffs = [
                f"{f}: {getattr(r, f)} → {getattr(m, f)}"
                for f in ("response_language", "query_intent", "scope_status", "lesson_type")
                if str(getattr(r, f)).lower() != str(getattr(m, f)).lower()
            ]
            if diffs:
                print(f"- «{query[:70]}» ({domain}): " + "; ".join(diffs))
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("role", choices=["critic", "text2cypher", "planner"])
    parser.add_argument(
        "--models", default=",".join(DEFAULT_MODELS), help="the first is the reference"
    )
    parser.add_argument("--lessons", type=int, default=8, help="real lessons to judge")
    parser.add_argument("--broken", type=int, default=3, help="lessons to break on purpose")
    parser.add_argument("--runs", type=int, default=2, help="judgements per case and model")
    parser.add_argument(
        "--retrieval", action="store_true", help="planner: also compare what the graph returns"
    )
    args = parser.parse_args()
    models = [m for m in args.models.split(",") if m]
    if args.role == "critic":
        asyncio.run(run_critic(models, args.lessons, args.broken, args.runs))
    elif args.role == "planner":
        run_planner(models, args.retrieval)
    else:
        run_text2cypher(models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
