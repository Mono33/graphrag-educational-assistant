"""Two versions of the agent on the same teachers' requests, through the API (2026-10-16).

``compare_graph_relations.py`` runs the chain inside the app container, so it can
only compare switches of one build. A change to the agent itself -- the
response-quality branch against what staging runs -- needs the same requests sent
to two builds. This drives a running stack through the sessions API, as a client
would: a lesson with a class profile, then one follow-up about the graph; and
measures what can be measured without a person:

* ``minutes``  the section timings add up to the requested duration (not counted
               when the output has no timed section: another intent, not a lesson);
* ``kg``       Knowledge Graph nodes the lesson cites in its sources;
* ``room``     a learning-environment bullet when the class needs one (ADHD,
               autism);
* ``leaks``    prompt or reasoning text reaching the teacher («non ammesso»,
               «vincolo», an empty «``», a raw ``[[material:`` marker);
* ``raw_profile`` a class-profile node's raw name («Group till 30») shown to the
               teacher, in the lesson or the follow-up;
* ``myths``    learning styles or Brain Gym proposed as practice;
* ``domain``   the closing note names the lesson's own graph;
* ``no_graph`` the follow-up says it has no access to the graph;
* the Critic's verdict and score, seconds, characters.

Then ``report`` puts two runs side by side and writes a page to read the lessons
blind (the key in ``chiave.json``). The lessons are teacher-facing text: ``--out``
must be outside the repository.

    python scripts/eval/quality_ab.py run --label nuovo --out C:/tmp/qab
    # switch the stack to the other build, then
    python scripts/eval/quality_ab.py run --label staging --out C:/tmp/qab
    python scripts/eval/quality_ab.py report --a staging --b nuovo --out C:/tmp/qab

The service account is read from ``deploy/.env.local`` and never printed.
"""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import sys
import time
import uuid
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

FOLLOW_UP = (
    "Quali informazioni hai trovato nel grafo per la mia classe? Fammi un elenco e dimmi "
    "quali hai usato nella lezione e quali no."
)

#: (domain, request, profile). Bisogni, gradi e durate diversi; il primo è il caso
#: della lezione di staging del 16/10 («ciclo dell'acqua»).
CASES: list[tuple[str, str, dict]] = [
    (
        "udl",
        "Ho una classe della seconda primaria e vorrei organizzare una lezione che ha come "
        "obiettivo elencare le fasi del ciclo dell'acqua. La classe è eterogenea: 2 bambini non "
        "italofoni di madre lingua spagnola e 2 bambini con ADHD. Mi dai qualche idea?",
        {
            "group": {"students_number": 22, "grade": "PRIMARIA", "disabilities": ["ADHD", "CL"]},
            "subject_area": "scienze",
            "time_available_minutes": 60,
        },
    ),
    (
        "udl",
        "Lezione sulle frazioni equivalenti per una quarta primaria con un alunno con DSA "
        "(discalculia) e uno con alto potenziale.",
        {
            "group": {"students_number": 24, "grade": "PRIMARIA", "disabilities": ["DSA", "PD"]},
            "subject_area": "matematica",
            "time_available_minutes": 50,
        },
    ),
    (
        "udl",
        "Comprensione del testo narrativo in una prima media, classe divisa in gruppi, con uno "
        "studente nello spettro autistico.",
        {
            "group": {
                "students_number": 26,
                "grade": "SECONDARIA_I_GRADO",
                "disabilities": ["SA"],
                "class_features": ["DIVISA_IN_GRUPPI"],
            },
            "subject_area": "italiano",
            "time_available_minutes": 60,
        },
    ),
    (
        "udl",
        "Lezione di storia sulla Rivoluzione francese per una terza superiore: voglio che sia "
        "accessibile a tutti.",
        {
            "group": {"students_number": 25, "grade": "SECONDARIA_II_GRADO"},
            "subject_area": "storia",
            "time_available_minutes": 90,
        },
    ),
    (
        "neuro",
        "Crea una lezione di 45 minuti sulla memoria di lavoro per una seconda media con due "
        "studenti con ADHD.",
        {
            "group": {
                "students_number": 23,
                "grade": "SECONDARIA_I_GRADO",
                "disabilities": ["ADHD"],
            },
            "subject_area": "scienze",
            "time_available_minutes": 45,
        },
    ),
    (
        "neuro",
        "Lezione per aumentare la motivazione intrinseca in una terza media demotivata.",
        {
            "group": {
                "students_number": 21,
                "grade": "SECONDARIA_I_GRADO",
                "student_attributes": ["MANCANZA_DI_MOTIVAZIONE"],
            },
            "time_available_minutes": 60,
        },
    ),
    (
        "neuro",
        "Ho una seconda primaria con un bambino dislessico: come imposto una lezione di lettura "
        "che tenga conto di come funziona il cervello?",
        {
            "group": {"students_number": 20, "grade": "PRIMARIA", "disabilities": ["DSA"]},
            "subject_area": "italiano",
            "time_available_minutes": 60,
        },
    ),
    (
        "neuro",
        "Lezione sul ciclo dell'acqua per una seconda primaria con 2 bambini con ADHD e 2 non "
        "italofoni.",
        {
            "group": {"students_number": 22, "grade": "PRIMARIA", "disabilities": ["ADHD", "CL"]},
            "subject_area": "scienze",
            "time_available_minutes": 60,
        },
    ),
]

_LEAKS = re.compile(r"non ammesso|vincolo rigido|``|\[\[material:|\(KG\)", re.IGNORECASE)
_MYTHS = re.compile(
    r"alunn\w* (?:visiv|uditiv|cinestetic)|stil\w* di apprendimento (?:visiv|uditiv|cinestetic)"
    r"|brain gym",
    re.IGNORECASE,
)
_NO_GRAPH = re.compile(r"non ho accesso (?:diretto )?al grafo", re.IGNORECASE)
_ROOM = re.compile(r"ambiente di apprendimento", re.IGNORECASE)
#: A class-profile node's raw name («Group till 30»): means nothing to a teacher.
_RAW_PROFILE = re.compile(r"group till \d+", re.IGNORECASE)
# The checks of docs/product/Specifica_Risposta_Mediana.md §4 (2026-10-19).
_CONFIDENCE = re.compile(
    r"confidenza (?:alta|media|bassa)|very_high|\(high\)|\(medium\)|\(low\)", re.IGNORECASE
)
_BAD_SPLIT = re.compile(
    r"\bgrupp[oi] da 1\b|\(0\s*(?:min|′)|\b0 minuti\b|\b1 gruppo da 1\b", re.IGNORECASE
)
_REPEAT = re.compile(r"\b(\w{3,})/\1\b", re.IGNORECASE)
_JARGON = re.compile(r"\bnod[oi]\b|\bKG\b|knowledge graph", re.IGNORECASE)
FOLLOW_UP_MAX_LINES = 25


def _env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _kg_sources(lesson: str) -> int:
    """Names listed under the Knowledge Graph in the sources (0 when none)."""
    for line in lesson.splitlines():
        if "knowledge graph" in line.lower() and ":" in line and "nota" not in line.lower():
            names = line.split(":", 1)[1]
            return len([n for n in re.split(r"[;,]", names) if n.strip(" .*")])
    return 0


def _body(lesson: str) -> str:
    """The lesson without its sources and closing note, where the graph is named."""
    return re.split(
        r"\*Fonti:?\*|^## Fonti|nota sulle fonti", lesson, maxsplit=1, flags=re.I | re.M
    )[0]


def _sources(lesson: str) -> list[str]:
    parts = re.split(r"\*Fonti:?\*|^## Fonti", lesson, maxsplit=1, flags=re.I | re.M)
    return [line for line in (parts[1] if len(parts) > 1 else "").splitlines() if line.strip()]


def measure(domain: str, profile: dict, lesson: str, answer: str, request: str = "") -> dict:
    from aix.agent.kg_evidence import _NEEDS_IN_TEXT, class_need_groups
    from aix.agent.reference_filter import off_topic
    from aix.agent.validation.rules import section_minutes

    minutes = sum(m for _title, m in section_minutes(lesson))
    needs = set((profile.get("group") or {}).get("disabilities") or [])
    note = lesson.lower().rsplit("nota sulle fonti", 1)[-1]
    return {
        "minutes": minutes,
        # No timed section: not a lesson plan (the Planner chose another intent).
        "minutes_ok": (minutes == profile.get("time_available_minutes")) if minutes else None,
        "kg": _kg_sources(lesson),
        "room": bool(_ROOM.search(lesson)) if needs & {"ADHD", "SA"} else None,
        "leaks": sorted({m.group(0).lower() for m in _LEAKS.finditer(lesson + "\n" + answer)}),
        "raw_profile": bool(_RAW_PROFILE.search(lesson + "\n" + answer)),
        "myths": sorted({m.group(0).lower() for m in _MYTHS.finditer(lesson)}),
        "domain_ok": ("neuroscien" not in note) if domain == "udl" else ("udl" not in note),
        "no_graph": bool(_NO_GRAPH.search(answer)),
        "confidence": bool(_CONFIDENCE.search(lesson + "\n" + answer)),
        "bad_split": bool(_BAD_SPLIT.search(lesson)),
        "repeats": sorted({m.group(0) for m in _REPEAT.finditer(lesson)}),
        "jargon": len(_JARGON.findall(_body(lesson))) + len(_JARGON.findall(answer)),
        # Needs the follow-up talks about that the class does not have.
        "foreign_needs": sorted(
            need
            for need, pattern in _NEEDS_IN_TEXT.items()
            if pattern.search(answer) and need not in class_need_groups(profile, request)
        ),
        "followup_lines": len([line for line in answer.splitlines() if line.strip()]),
        "offtopic_refs": sum(1 for line in _sources(lesson) if off_topic({"title": line}, profile)),
    }


def run(args: argparse.Namespace) -> int:
    env = _env(Path(args.env))
    out = Path(args.out) / args.label
    out.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(
        base_url=args.base,
        auth=(env["GRAPH_API_USERNAME"], env["GRAPH_API_PWD"]),
        headers={"X-End-User-Id": f"quality-ab-{args.label}-{uuid.uuid4().hex[:6]}"},
        timeout=600,
    )
    health = client.get("/api/v1/health").json()
    print(f"[{args.label}] build {health.get('code_version')} on {args.base}")
    results = []
    for n, (domain, request, profile) in enumerate(CASES, 1):
        if args.only and n not in args.only:
            continue
        sid = client.post(
            "/api/v1/sessions",
            json={"domain": domain, "title": f"quality-ab {n}", "educational_profile": profile},
        ).json()["session_id"]
        started = time.monotonic()
        first = client.post(
            f"/api/v1/sessions/{sid}/turns", json={"message": request, "mode": "lesson"}
        )
        seconds = round(time.monotonic() - started, 1)
        if first.status_code != 200:
            print(f"  {n}: lesson failed {first.status_code} {first.text[:200]}")
            continue
        lesson_turn = first.json()
        second = client.post(f"/api/v1/sessions/{sid}/turns", json={"message": FOLLOW_UP})
        answer = second.json().get("markdown", "") if second.status_code == 200 else ""
        lesson = lesson_turn.get("markdown") or ""
        critic = lesson_turn.get("critic") or {}
        row = {
            "case": n,
            "domain": domain,
            "request": request,
            "seconds": seconds,
            "chars": len(lesson),
            "critic_approved": critic.get("approved"),
            "critic_score": critic.get("score"),
            "nodes": (lesson_turn.get("retriever") or {}).get("nodes_count"),
            **measure(domain, profile, lesson, answer, request),
            "lesson": lesson,
            "answer": answer,
        }
        results.append(row)
        print(
            f"  {n} {domain}: {seconds}s min={row['minutes']}/{profile.get('time_available_minutes')}"
            f" kg={row['kg']} room={row['room']} leaks={row['leaks']} myths={row['myths']}"
            f" domain_ok={row['domain_ok']} no_graph={row['no_graph']} critic={row['critic_score']}"
        )
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), "utf-8")
    print(f"[{args.label}] {len(results)} case(s) -> {out / 'results.json'}")
    return 0


def _summary(rows: list[dict]) -> dict:
    def share(key: str, value=True) -> str:
        relevant = [r for r in rows if r.get(key) is not None]
        return f"{sum(1 for r in relevant if r[key] == value)}/{len(relevant)}" if relevant else "—"

    return {
        "minuti esatti": share("minutes_ok"),
        "nodi KG citati (media)": round(sum(r["kg"] for r in rows) / max(1, len(rows)), 1),
        "ambiente quando serve": share("room"),
        "testo del prompt che trapela": f"{sum(1 for r in rows if r['leaks'])}/{len(rows)}",
        "«Group till N» alla docente": f"{sum(1 for r in rows if r.get('raw_profile'))}/{len(rows)}",
        "neuromiti proposti": f"{sum(1 for r in rows if r['myths'])}/{len(rows)}",
        "grafo giusto nella nota": share("domain_ok"),
        "«non ho accesso al grafo»": f"{sum(1 for r in rows if r['no_graph'])}/{len(rows)}",
        "etichette di confidenza": f"{sum(1 for r in rows if r.get('confidence'))}/{len(rows)}",
        "gruppo da 1 / fase da 0 minuti": f"{sum(1 for r in rows if r.get('bad_split'))}/{len(rows)}",
        "ripetizioni «x/x»": f"{sum(1 for r in rows if r.get('repeats'))}/{len(rows)}",
        "gergo (nodo, KG) per caso": round(
            sum(r.get("jargon", 0) for r in rows) / max(1, len(rows)), 1
        ),
        "follow-up: bisogni non della classe": (
            f"{sum(1 for r in rows if r.get('foreign_needs'))}/{len(rows)}"
        ),
        f"follow-up oltre {FOLLOW_UP_MAX_LINES} righe": (
            f"{sum(1 for r in rows if r.get('followup_lines', 0) > FOLLOW_UP_MAX_LINES)}/{len(rows)}"
        ),
        "riferimenti fuori tema nelle fonti": sum(r.get("offtopic_refs", 0) for r in rows),
        "Critic: approvate": share("critic_approved"),
        "secondi (media)": round(sum(r["seconds"] for r in rows) / max(1, len(rows)), 1),
    }


def report(args: argparse.Namespace) -> int:

    base = Path(args.out)
    a = {r["case"]: r for r in json.loads((base / args.a / "results.json").read_text("utf-8"))}
    b = {r["case"]: r for r in json.loads((base / args.b / "results.json").read_text("utf-8"))}
    cases = sorted(set(a) & set(b))
    sa, sb = _summary([a[c] for c in cases]), _summary([b[c] for c in cases])
    print(f"{'':32} {args.a:>12} {args.b:>12}")
    for key in sa:
        print(f"{key:32} {str(sa[key]):>12} {str(sb[key]):>12}")
    (base / "summary.json").write_text(
        json.dumps({args.a: sa, args.b: sb}, ensure_ascii=False, indent=2), "utf-8"
    )

    rng = random.Random(args.seed)
    key, articles = {}, []
    for c in cases:
        pair = [(args.a, a[c]), (args.b, b[c])]
        rng.shuffle(pair)
        key[c] = {"X": pair[0][0], "Y": pair[1][0]}
        blocks = "".join(
            f"<section><h3>Versione {tag}</h3><pre>{html.escape(row['lesson'])}</pre>"
            f"<h4>Risposta al follow-up</h4><pre>{html.escape(row['answer'])}</pre></section>"
            for tag, (_label, row) in zip("XY", pair, strict=True)
        )
        articles.append(
            f"<article><h2>Caso {c} ({a[c]['domain']})</h2>"
            f"<p class='q'><b>Richiesta:</b> {html.escape(a[c]['request'])}</p>"
            f"<p class='q'><b>Follow-up:</b> {html.escape(FOLLOW_UP)}</p>{blocks}</article>"
        )
    page = (
        "<!doctype html><html lang='it'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Lettura alla cieca</title><style>body{font-family:system-ui;max-width:1100px;"
        "margin:auto;padding:16px}section{display:inline-block;vertical-align:top;width:49%}"
        "pre{white-space:pre-wrap;font-size:13px;background:#f6f6f6;padding:8px}"
        "@media(max-width:800px){section{width:100%}}.q{color:#444}</style></head><body>"
        "<h1>Due versioni, stesse richieste</h1><p>Per ogni caso: quale lezione useresti, e "
        "perché? La chiave è in <code>chiave.json</code>.</p>"
        + "".join(articles)
        + "</body></html>"
    )
    (base / "lettura_alla_cieca.html").write_text(page, encoding="utf-8")
    (base / "chiave.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    print(f"page -> {base / 'lettura_alla_cieca.html'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--label", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--base", default="http://127.0.0.1:8766")
    r.add_argument("--env", default=str(REPO_ROOT / "deploy" / ".env.local"))
    r.add_argument("--only", type=int, nargs="*", help="case numbers, 1-based")
    p = sub.add_parser("report")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=16)
    args = parser.parse_args()
    if REPO_ROOT in Path(args.out).resolve().parents:
        parser.error("--out must be outside the repository: the lessons are teachers' text")
    return run(args) if args.command == "run" else report(args)


if __name__ == "__main__":
    raise SystemExit(main())
