"""Does the judge find what a teacher would have to correct, and only that? (2026-10-19)

The same discipline as ``check_citation_support.py``: the harness comes first, so
the judge (``aix.agent.validation.judge``) is wired into the chain on evidence,
and a model is chosen for it on numbers.

The cases
---------
Lessons from ``scripts/eval/quality_ab.py`` runs (``--runs``, outside the
repository: they are lessons). Into copies of them go errors of the kinds the two
blind reads found (:data:`INJECTIONS`), one per copy, each in the lesson it would
plausibly be in: «ga-to» in the reading lesson, a wrong answer key in the fractions
quiz, the Bastille in 1792 in the history one. The lessons as they are, without
anything put in, are run too.

What it reports, per model
--------------------------
``found``          errors put in that the judge reported, quoting the sentence;
``right check``    of those, under the question they belong to;
``blocking``       of those, marked «bloccante» (all of them should be but the
                   unclear example and the tiring activity);
``flags/lesson``   problems reported on the lessons as they are. Not all are false
                   alarms: the lessons have real errors (the blind reads found
                   some). They are written to ``segnalazioni.md`` to be read;
``unquoted``       problems reported with a quote that is not in the lesson;
``errors``, ``s``, ``$``   failures, median seconds, total cost.

    python scripts/eval/check_judge.py --runs C:/axl/qab/giro3 C:/axl/qab/staging \\
        --out C:/axl/judge
    python scripts/eval/check_judge.py ... --models anthropic/claude-sonnet-5.5 anthropic/claude-haiku-5.5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "eval"))

#: (id, the question it belongs to, the quality_ab cases it goes into, the line
#: put in, a piece of it a report must quote to count as found).
INJECTIONS: list[tuple[str, str, list[int], str, str]] = [
    (
        "sillabe",
        "esempi_lingua",
        [7],
        "- Alla lavagna, la divisione in sillabe di «gatto»: «ga-to».",
        "ga-to",
    ),
    (
        "pancia_pb",
        "esempi_lingua",
        [7],
        "- Per distinguere la «p» dalla «b»: la «b» ha la pancia a destra, la «p» ce l'ha a sinistra.",
        "pancia a sinistra",
    ),
    (
        "quiz_frazioni",
        "esercizi",
        [2],
        "- Domanda di verifica: «Quale frazione è equivalente a 2/3? a) 3/4 b) 4/6 c) 6/8». "
        "Risposta corretta: c).",
        "risposta corretta: c)",
    ),
    (
        "frazioni_errate",
        "fatti",
        [2],
        "- Far notare che 1/2 e 2/3 sono equivalenti, perché numeratore e denominatore aumentano "
        "entrambi di uno.",
        "1/2 e 2/3 sono equivalenti",
    ),
    (
        "pizza_confusa",
        "esempi_chiari",
        [2],
        "- Esempio: 1 pizza divisa in 2 parti contro 2 pizze divise in 4 parti, per mostrare che "
        "1/2 è uguale a 2/4.",
        "2 pizze divise in 4 parti",
    ),
    (
        "acqua_aria",
        "fatti",
        [1, 8],
        "- Spiegare che con l'evaporazione l'acqua si trasforma in aria, che poi torna acqua "
        "nelle nuvole.",
        "si trasforma in aria",
    ),
    (
        "acqua_bollente",
        "sicurezza",
        [1, 8],
        "- Ogni gruppo riceve un pentolino di acqua bollente da versare sul ghiaccio.",
        "acqua bollente",
    ),
    (
        "video_docente",
        "risorse",
        [1, 2],
        "- Proiettare alla classe il video «What is Universal Design for Learning?» per spiegare "
        "agli alunni come è costruita la lezione.",
        "proiettare alla classe il video",
    ),
    (
        "perimetro",
        "perimetro",
        [1],
        "- Introdurre anche il calore latente di vaporizzazione e l'equazione di "
        "Clausius-Clapeyron.",
        "clausius",
    ),
    (
        "bastiglia",
        "fatti",
        [4],
        "- Ricordare che la presa della Bastiglia avvenne il 14 luglio 1792.",
        "14 luglio 1792",
    ),
    (
        "narratore",
        "fatti",
        [3],
        "- Spiegare che il narratore onnisciente è sempre il protagonista, che racconta in prima "
        "persona.",
        "narratore onnisciente è sempre il protagonista",
    ),
    (
        "cervello_10",
        "fatti",
        [5],
        "- Spiegare che usiamo solo il 10% del cervello, e che gli esercizi di memoria attivano "
        "il resto.",
        "10% del cervello",
    ),
    (
        "memoria_20",
        "fatti",
        [5],
        "- Ricordare che la memoria di lavoro di un adolescente contiene in media 20 elementi "
        "alla volta.",
        "20 elementi",
    ),
    (
        "scrittura_lunga",
        "classe",
        [6],
        "- Segue mezz'ora di riflessione scritta individuale, in silenzio, su una scheda di due "
        "pagine.",
        "mezz'ora di riflessione scritta",
    ),
]
#: Errors whose report should be «minore», not «bloccante».
MINOR = {"pizza_confusa", "scrittura_lunga"}

_ANCHOR = re.compile(r"^\s*(?:#{2,4}|\*\*)\s*(?:Attivit|Introduzione|Svolgimento)", re.I)


def inject(lesson: str, line: str) -> str | None:
    """``lesson`` with ``line`` among the first activity's bullets; ``None`` when
    the lesson has no activity to put it in."""
    lines = lesson.splitlines()
    for i, current in enumerate(lines):
        if _ANCHOR.match(current):
            for j in range(i + 1, min(i + 12, len(lines))):
                if lines[j].lstrip().startswith("- "):
                    return "\n".join(lines[: j + 1] + [line] + lines[j + 1 :])
            return "\n".join(lines[: i + 1] + [line] + lines[i + 1 :])
    return None


def _found(report, needle: str, line: str):
    from aix.agent.validation.judge import _flat

    for issue in report.issues:
        quote = _flat(issue.quote)
        if _flat(needle) in quote or (len(quote) >= 5 and quote in _flat(line)):
            return issue
    return None


async def run_model(model: str, cases: list[dict], concurrency: int) -> list[dict]:
    from aix.agent.validation.judge import judge_lesson

    gate = asyncio.Semaphore(concurrency)

    async def one(case: dict) -> dict:
        async with gate:
            report = await judge_lesson(
                case["lesson"], request=case["request"], profile=case["profile"], model=model
            )
        row = {**{k: v for k, v in case.items() if k != "lesson"}, "model": model}
        row["report"] = {
            **{k: v for k, v in asdict(report).items() if k not in ("issues", "unquoted")},
            "issues": [asdict(i) for i in report.issues],
            "unquoted": [asdict(i) for i in report.unquoted],
        }
        if case.get("injection"):
            hit = _found(report, case["needle"], case["line"])
            row["found"] = hit is not None
            row["right_check"] = bool(hit and hit.check == case["check"])
            row["blocking"] = bool(hit and hit.severity == "bloccante")
        return row

    return await asyncio.gather(*(one(c) for c in cases))


def build_cases(runs: list[str]) -> list[dict]:
    import quality_ab

    by_case = {i + 1: c for i, c in enumerate(quality_ab.CASES)}
    cases: list[dict] = []
    for run in runs:
        label = Path(run).name
        for row in json.loads((Path(run) / "results.json").read_text("utf-8")):
            _domain, request, profile = by_case[row["case"]]
            base = {"run": label, "case": row["case"], "request": request, "profile": profile}
            cases.append({**base, "lesson": row["lesson"], "injection": None})
            if run != runs[0]:
                continue  # errors go into the first run's lessons only
            for name, check, targets, line, needle in INJECTIONS:
                if row["case"] in targets and (lesson := inject(row["lesson"], line)):
                    cases.append(
                        {
                            **base,
                            "lesson": lesson,
                            "injection": name,
                            "check": check,
                            "line": line,
                            "needle": needle,
                        }
                    )
    return cases


def summarise(rows: list[dict]) -> dict:
    injected = [r for r in rows if r.get("injection")]
    clean = [r for r in rows if not r.get("injection")]
    found = [r for r in injected if r["found"]]
    blocking_ok = [r for r in found if r["blocking"] == (r["injection"] not in MINOR)]
    reports = [r["report"] for r in rows]
    costs = [r["cost"] for r in reports if r.get("cost") is not None]
    return {
        "found": f"{len(found)}/{len(injected)}",
        "right check": f"{sum(1 for r in found if r['right_check'])}/{len(found)}",
        "severity ok": f"{len(blocking_ok)}/{len(found)}",
        "flags/lesson (as is)": round(
            sum(len(r["report"]["issues"]) for r in clean) / max(1, len(clean)), 1
        ),
        "blocking flags (as is)": sum(
            1 for r in clean for i in r["report"]["issues"] if i["severity"] == "bloccante"
        ),
        "unquoted": sum(len(r["unquoted"]) for r in reports),
        "errors": sum(1 for r in reports if r["error"]),
        "median s": statistics.median([r["seconds"] for r in reports]) if reports else 0,
        "$ total": round(sum(costs), 3) if costs else None,
        "missed": sorted({r["injection"] for r in injected if not r["found"]}),
    }


def write_flags(rows: list[dict], path: Path) -> None:
    out = ["# Segnalazioni sulle lezioni senza errori inseriti\n"]
    for r in rows:
        if r.get("injection") or not r["report"]["issues"]:
            continue
        out.append(f"## {r['model']} · {r['run']} · caso {r['case']}\n")
        for i in r["report"]["issues"]:
            out.append(
                f"- **{i['check']} / {i['severity']}** — «{i['quote']}»  \n"
                f"  perché: {i['why']}  \n  correzione: {i['fix']}"
            )
        out.append("")
    path.write_text("\n".join(out), encoding="utf-8")


def _changed_elsewhere(before: str, after: str, line: str = "") -> float:
    """How much of the lesson changed besides the error put in: 0 = nothing else.

    Measured on the lines, the injected one left out on both sides: a repair
    that fixed the error and touched nothing else scores 0.
    """
    import difflib

    def keep(text: str) -> list[str]:
        return [row for row in text.splitlines() if row.strip() and row.strip() != line.strip()]

    return round(1 - difflib.SequenceMatcher(None, keep(before), keep(after)).ratio(), 3)


async def run_repair(cases: list[dict], concurrency: int) -> list[dict]:
    """The judge node on every case: what it fixed, what else it changed."""
    from aix.agent.graph.nodes import judge_node

    gate = asyncio.Semaphore(concurrency)

    async def one(case: dict) -> dict:
        state = {
            "final_lesson_plan": case["lesson"],
            "teacher_query": case["request"],
            "educational_profile": case["profile"],
        }
        async with gate:
            out = await judge_node(state)
        after = out.get("final_lesson_plan") or case["lesson"]
        row = {k: v for k, v in case.items() if k != "lesson"}
        row.update(
            summary=out.get("judge_summary") or {},
            notes=out.get("review_notes") or [],
            changed_elsewhere=_changed_elsewhere(case["lesson"], after, case.get("line", "")),
            before=case["lesson"],
            after=after,
        )
        if case.get("injection"):
            from aix.agent.validation.judge import _flat

            row["resolved"] = _flat(case["needle"]) not in _flat(after)
        return row

    return await asyncio.gather(*(one(c) for c in cases))


def summarise_repair(rows: list[dict]) -> dict:
    injected = [r for r in rows if r.get("injection")]
    clean = [r for r in rows if not r.get("injection")]
    summaries = [r["summary"] for r in rows]
    return {
        "errors put in, fixed in the text": f"{sum(1 for r in injected if r['resolved'])}/{len(injected)}",
        "replaced (code)": sum(s.get("replaced", 0) for s in summaries),
        "revised (block)": sum(s.get("revised", 0) for s in summaries),
        "changes put back (judged wrong)": sum(s.get("reverted", 0) for s in summaries),
        "changed besides the fix, median": statistics.median(
            [r["changed_elsewhere"] for r in injected]
        )
        if injected
        else None,
        "changed besides the fix, max": max(
            (r["changed_elsewhere"] for r in injected), default=None
        ),
        "lessons as they are, changed": f"{sum(1 for r in clean if r['after'] != r['before'])}/{len(clean)}",
        "notes per lesson": round(sum(len(r["notes"]) for r in rows) / max(1, len(rows)), 1),
        "$ total": round(sum(s.get("cost") or 0 for s in summaries), 3),
        "median s": statistics.median([s.get("seconds") or 0 for s in summaries]),
        "not fixed": sorted({r["injection"] for r in injected if not r["resolved"]}),
    }


def write_repairs(rows: list[dict], path: Path) -> None:
    """Every change, before and after, to be read."""
    import difflib

    out = ["# Che cosa ha cambiato la riparazione\n"]
    for r in rows:
        if r["after"] == r["before"]:
            continue
        title = f"{r['run']} · caso {r['case']}" + (
            f" · errore «{r['injection']}»" if r.get("injection") else ""
        )
        out.append(f"## {title}\n\n```diff")
        diff = difflib.unified_diff(
            r["before"].splitlines(), r["after"].splitlines(), lineterm="", n=0
        )
        out.extend(line for line in diff if not line.startswith(("---", "+++")))
        out.append("```\n")
    path.write_text("\n".join(out), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["anthropic/claude-sonnet-5.5", "anthropic/claude-haiku-5.5"],
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="build the cases, call nothing")
    parser.add_argument(
        "--repair",
        action="store_true",
        help="run the whole judge node (judge, repair, judge again) instead of the judge alone",
    )
    args = parser.parse_args()
    out = Path(args.out)
    if REPO_ROOT in out.resolve().parents:
        parser.error("--out must be outside the repository: the cases are lessons")
    out.mkdir(parents=True, exist_ok=True)

    cases = build_cases(args.runs)
    injected = sum(1 for c in cases if c["injection"])
    print(
        f"{len(cases)} cases: {injected} with an error put in, {len(cases) - injected} as they are"
    )
    if args.dry_run:
        for c in cases:
            if c["injection"]:
                print(f"  case {c['case']}: {c['injection']}")
        return 0

    if args.repair:
        rows = asyncio.run(run_repair(cases, args.concurrency))
        (out / "riparazioni.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), "utf-8"
        )
        write_repairs(rows, out / "riparazioni.md")
        result = summarise_repair(rows)
        (out / "summary_repair.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), "utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    # One event loop for every model: the HTTP client is cached and bound to the
    # loop that created it, and a second ``asyncio.run`` found it closed.
    async def all_models() -> dict:
        return {m: await run_model(m, cases, args.concurrency) for m in args.models}

    summary = {}
    for model, rows in asyncio.run(all_models()).items():
        slug = re.sub(r"[^a-z0-9]+", "-", model.lower())
        (out / f"{slug}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")
        write_flags(rows, out / f"segnalazioni-{slug}.md")
        summary[model] = summarise(rows)
        print(model, json.dumps(summary[model], ensure_ascii=False))
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
