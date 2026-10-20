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

_LEAKS = re.compile(r"non ammesso|vincolo rigido|``|\[\[material:", re.IGNORECASE)
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
# Second blind read (2026-10-19): an activity with no time, and follow-ups whose
# counts («ne usa 25») did not match what they listed (about 15).
_ACTIVITY_TITLE = re.compile(
    r"^\s*(?:#{2,4}\s*|\*\*)\s*Attivit[àa]\s*\d+[^\n]*$", re.IGNORECASE | re.MULTILINE
)
_TIMED = re.compile(r"\d{1,3}\s*(?:min|′)", re.IGNORECASE)
_DECLARED_COUNT = re.compile(r"usat\w*[^\n(]{0,40}\((\d+)\)", re.IGNORECASE)
#: «… e la lezione ne usa 25»: counted against the bullets before the unused ones.
_PROSE_COUNT = re.compile(r"(?:ne usa|ne utilizza|usate sono)\s+(\d+)", re.IGNORECASE)
_UNUSED_HEADING = re.compile(r"non usat|sconsigliat|il resto|non pertinent", re.IGNORECASE)


def untimed_activities(lesson: str) -> int:
    """Activity titles that state no minutes."""
    return sum(1 for m in _ACTIVITY_TITLE.finditer(lesson) if not _TIMED.search(m.group(0)))


def _names_in(bullet: str) -> int:
    """The strategies a bullet names: its bold names (a bullet may group several),
    or one when it has none."""
    return max(1, len(re.findall(r"\*\*[^*]+\*\*", bullet)))


def count_mismatch(answer: str) -> bool:
    """A follow-up's «Usate (N)» whose list under it does not have N items.

    The items are the bullets after the line, up to a blank line or the next
    bold heading; a bullet naming several («A e B») counts once -- the check is
    for gross mismatches (25 declared, 15 listed), not for that.
    """
    lines = answer.splitlines()
    for i, line in enumerate(lines):
        match = _DECLARED_COUNT.search(line)
        if not match:
            continue
        items = 0
        for below in lines[i + 1 :]:
            stripped = below.strip()
            if not stripped or (stripped.startswith("**") and not stripped.startswith("- ")):
                break
            if stripped.startswith(("-", "*", "•")):
                items += _names_in(stripped)
        if items and abs(items - int(match.group(1))) > max(2, int(match.group(1)) // 5):
            return True
    prose = _PROSE_COUNT.search(answer)
    if prose:
        before_unused = _UNUSED_HEADING.split(answer[prose.end() :], maxsplit=1)[0]
        items = sum(
            _names_in(line.strip())
            for line in before_unused.splitlines()
            if line.strip().startswith(("-", "*", "•"))
        )
        declared = int(prose.group(1))
        if items and abs(items - declared) > max(2, declared // 5):
            return True
    return False


#: Third blind read (2026-10-19): what the first two runs of the judge missed.
_TOTAL = re.compile(r"restituit\w*\s+(\d+)\s+(?:element|concett|nod)", re.IGNORECASE)
_HEADING_COUNT = re.compile(r"\((\d+)(?:\s+su\s+\d+)?(?:\s+element\w*)?\)|\b(\d+)\s+element", re.I)
_NAMES = re.compile(r"\*\*([^*]+)\*\*|(?<!\*)\*([^*\n]+)\*(?!\*)")
#: The closing note claiming documents the teacher never attached.
_FALSE_PROVENANCE = re.compile(
    r"allegat\w* dall?a? docente|fornit\w* dall?a? docente|fornit\w* dal docente", re.I
)


def _follow_up_sections(answer: str) -> list[tuple[str, list[str]]]:
    """The follow-up as ``(heading, bullets)``: a heading is a line opening in bold."""
    sections: list[tuple[str, list[str]]] = []
    for line in answer.splitlines():
        stripped = line.strip()
        if stripped.startswith(("**", "#")):
            sections.append((stripped, []))
        elif stripped.startswith(("-", "*", "•")) and sections:
            sections[-1][1].append(stripped)
    return sections


#: «- **ADHD:** Structured Routines [in «…»], Visual Aids»: a group, then its list.
_GROUPED = re.compile(r"^[-*•]\s*\*\*[^*]+?:?\*\*:?\s+(?=\S)(?!\()")


def _grouped_items(bullet: str) -> list[str]:
    """The list after a bold group label, or [] when the bullet is not one."""
    if not _GROUPED.match(bullet):
        return []
    tail = re.sub(r"\[[^\]]*\]", "", _GROUPED.sub("", bullet, count=1))
    items = [i.strip(" .*") for i in re.split(r",|;|\be\b", tail) if i.strip(" .*")]
    # The graph's names, «Structured Routines»: a sentence after a bold label is not a list.
    names = all(i[:1].isupper() and len(i.split()) <= 6 for i in items)
    return items if items and names else []


def _count_names(text: str) -> int:
    """The graph's elements a bullet names: its bold names, or a list after a colon."""
    grouped = _grouped_items(text)
    if grouped:
        return len(grouped)
    bold = len(re.findall(r"\*\*[^*]+\*\*", text))
    if bold:
        return bold
    tail = text.split(":", 1)[1] if ":" in text else ""
    items = [i for i in re.split(r",|;", tail) if i.strip()]
    return len(items) if len(items) >= 2 and all(len(i.split()) <= 6 for i in items) else 1


def total_mismatch(answer: str) -> bool:
    """The follow-up's parts do not add up to what it says the graph returned.

    «53 elementi»: 2 used + 5 unused + «il resto (51 elementi)» is 58. Each part
    is the number in its heading, or the names in its bullets; the rest is the
    last number of its line, or «gli altri N elementi».
    """
    total = _TOTAL.search(answer)
    if not total:
        return False
    parts, rest_seen = 0, False
    for heading, bullets in _follow_up_sections(answer):
        low = heading.lower()
        if any(word in low for word in ("lacuna", "proposta", "ambiente")):
            continue
        if "resto" in low:
            numbers = [int(n) for n in re.findall(r"\d+", heading) if int(n) != int(total.group(1))]
            parts += numbers[-1] if numbers else 0
            rest_seen = True
            continue
        number = _HEADING_COUNT.search(heading)
        if number:
            parts += int(number.group(1) or number.group(2))
        elif bullets:
            parts += sum(_count_names(b) for b in bullets)
        else:  # «**Neuromiti evitati:** a, b, c» on the heading's line
            parts += _count_names(re.sub(r"^\*\*[^*]+\*\*", "x:", heading))
    rest = re.search(r"altri (\d+) (?:element|concett|nod)", answer, re.I)
    if rest and not rest_seen:
        parts += int(rest.group(1))
        rest_seen = True
    # Without a stated rest nothing claims to add up; avoided items told in prose
    # are counted roughly, hence the margin.
    declared = int(total.group(1))
    return rest_seen and abs(parts - declared) > max(2, declared // 20)


def usage_contradictions(answer: str, lesson: str) -> list[str]:
    """Names the follow-up places in the wrong list, read against the lesson body.

    «Usati»: one of its names (English or Italian) is in the body, or it is
    wrong -- a myth avoided is not used. «Non usati»: none of its names may be
    a label of the lesson, in brackets or in italics, as the Writer marks what
    it applies (Adaptive Strategies, labelled in the lesson and listed as
    unused). Only labels count there: «ritmo di comprensione» in a sentence is
    not the strategy applied.
    """
    from aix.agent.kg_evidence import _norm, lesson_body

    raw = lesson_body(lesson)
    body = _norm(raw)
    labels = _norm(
        " | ".join(
            re.findall(r"\(([^()]+)\)", raw) + [a or b for a, b in _NAMES.findall(raw)]
        ).replace("*", " ")
    )
    wrong: list[str] = []
    for heading, bullets in _follow_up_sections(answer):
        low = heading.lower()
        unused = bool(re.search(r"non (?:\w+ )?(?:usat|utilizzat|applicat|previst)", low))
        used = not unused and ("usat" in low or "utilizzat" in low)
        if not (used or unused):
            continue
        for bullet in bullets:
            grouped = _grouped_items(bullet)
            # A group of names: each is checked on its own.
            for names in [[n] for n in grouped] if grouped else [None]:
                if names is None:
                    label = re.split(r":\s", bullet, maxsplit=1)[0]
                    names = [a or b for a, b in _NAMES.findall(label)]
                    names += [
                        p for group in re.findall(r"\(([^()]+)\)", label) for p in group.split(",")
                    ]
                wrong.extend(_misplaced(names, used=used, body=body, labels=labels))
    return wrong


def _misplaced(names: list[str], *, used: bool, body: str, labels: str) -> list[str]:
    """One item of a follow-up list against the lesson: see ``usage_contradictions``."""
    from aix.agent.kg_evidence import _NEEDS_IN_TEXT, _in_lesson

    names = [n.strip(" *:") for n in names if n.strip(" *:")]
    # «(ADHD)», «(plusdotazione)» say which need, not which strategy.
    needs = {need.lower() for need in _NEEDS_IN_TEXT}
    names = [
        n
        for n in names
        if n.lower() not in needs and not any(p.search(n) for p in _NEEDS_IN_TEXT.values())
    ]
    if not names:
        return []
    if used:
        return [] if any(_in_lesson(n, body) for n in names) else [f"usato, assente: {names[0]}"]
    present = [n for n in names if _in_lesson(n, labels)]
    return [f"non usato, presente: {present[0]}"] if present else []


def _env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _kg_sources(lesson: str) -> int:
    """Names listed under the Knowledge Graph in the sources (0 when none)."""
    lines = lesson.splitlines()
    for i, line in enumerate(lines):
        if "knowledge graph" in line.lower() and ":" in line and "nota" not in line.lower():
            names = line.split(":", 1)[1]
            count = len([n for n in re.split(r"[;,]", names) if n.strip(" .*")])
            # Or a heading with the names as bullets under it, up to the next heading.
            for below in lines[i + 1 :]:
                stripped = below.strip()
                if not stripped.startswith(("-", "*", "•")) or stripped.startswith("**"):
                    break
                # One name per bullet, or several separated by «;» (commas are prose).
                count += stripped.count(";") + 1
            return count
    return 0


def _body(lesson: str) -> str:
    """The lesson without its sources and closing note, where the graph is named.

    The system's own boundary: «*Fonti: Knowledge Graph — …*» on one line was
    counted as jargon in the body (third blind read)."""
    from aix.agent.kg_evidence import _SOURCES

    return _SOURCES.split(lesson, maxsplit=1)[0]


def _sources(lesson: str) -> list[str]:
    parts = re.split(r"\*Fonti:?\*|^## Fonti", lesson, maxsplit=1, flags=re.I | re.M)
    return [line for line in (parts[1] if len(parts) > 1 else "").splitlines() if line.strip()]


def measure(domain: str, profile: dict, lesson: str, answer: str, request: str = "") -> dict:
    from aix.agent.kg_evidence import _NEEDS_IN_TEXT, class_need_groups
    from aix.agent.reference_filter import off_topic
    from aix.agent.validation.rules import section_minutes

    minutes = sum(m for _title, m in section_minutes(lesson))
    needs = set((profile.get("group") or {}).get("disabilities") or [])
    note = lesson.rsplit("ota sulle fonti", 1)[-1] if "ota sulle fonti" in lesson else ""
    return {
        "minutes": minutes,
        # No timed section: not a lesson plan (the Planner chose another intent).
        "minutes_ok": (minutes == profile.get("time_available_minutes")) if minutes else None,
        "kg": _kg_sources(lesson),
        "room": bool(_ROOM.search(lesson)) if needs & {"ADHD", "SA"} else None,
        "leaks": sorted({m.group(0).lower() for m in _LEAKS.finditer(lesson + "\n" + answer)}),
        "raw_profile": bool(_RAW_PROFILE.search(lesson + "\n" + answer)),
        # The body only: «nessuna pratica di tipo Brain Gym» in the sources is a myth avoided.
        "myths": sorted({m.group(0).lower() for m in _MYTHS.finditer(_body(lesson))}),
        "domain_ok": ("neuroscien" not in note.lower())
        if domain == "udl"
        else ("udl" not in note.lower()),
        "no_graph": bool(_NO_GRAPH.search(answer)),
        "confidence": bool(_CONFIDENCE.search(lesson + "\n" + answer)),
        "bad_split": bool(_BAD_SPLIT.search(lesson)),
        "repeats": sorted({m.group(0) for m in _REPEAT.finditer(lesson)}),
        # «(KG)» in the sources is jargon too (third blind read), not a prompt leak.
        "jargon": len(_JARGON.findall(_body(lesson)))
        + len(_JARGON.findall(answer))
        + lesson.count("(KG)"),
        # Needs the follow-up talks about that the class does not have.
        "foreign_needs": sorted(
            need
            for need, pattern in _NEEDS_IN_TEXT.items()
            # «DSA» alone names no specific disorder: only the words count here.
            if re.search(pattern.pattern.replace(r"|\bdsa\b", ""), answer, re.I)
            and need not in class_need_groups(profile, request)
        ),
        "followup_lines": len([line for line in answer.splitlines() if line.strip()]),
        "untimed_activities": untimed_activities(lesson),
        "count_mismatch": count_mismatch(answer) or total_mismatch(answer),
        "usage_wrong": usage_contradictions(answer, lesson),
        # Every case is sent without attachments.
        "false_provenance": bool(_FALSE_PROVENANCE.search(note)),
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
        # What the lesson uses of the graph, computed with the lesson and kept in
        # its turn's meta (``kg_evidence.usage``); absent on builds without it.
        detail = client.get(f"/api/v1/sessions/{sid}").json()
        kept = next(
            (
                (t.get("meta") or {}).get("kg_digest") or {}
                for t in detail.get("turns", [])
                if t.get("kind") == "lesson"
            ),
            {},
        )
        usage = kept.get("usage") if isinstance(kept, dict) else None
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
            "advised_used": (usage or {}).get("advised_against_used") if usage else None,
            # The graph's nodes the lesson applies, as the system computes them: the
            # sources list them whether or not they are used (third blind read).
            "kg_used": len((usage or {}).get("used") or []) if usage else None,
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
        "nodi KG nelle fonti (media)": round(sum(r["kg"] for r in rows) / max(1, len(rows)), 1),
        "nodi del grafo usati nel testo (media)": (
            round(sum(r["kg_used"] for r in used) / len(used), 1)
            if (used := [r for r in rows if r.get("kg_used") is not None])
            else "—"
        ),
        "nota: allegati che non ci sono": share("false_provenance"),
        "follow-up: usato/non usato contro il testo": (
            f"{sum(1 for r in rows if r.get('usage_wrong'))}/{len(rows)}"
        ),
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
        "attività senza minuti": sum(r.get("untimed_activities", 0) for r in rows),
        "follow-up: conteggi che non tornano": (
            f"{sum(1 for r in rows if r.get('count_mismatch'))}/{len(rows)}"
        ),
        "sconsigliate dal grafo ma usate": (
            f"{sum(1 for r in rows if r.get('advised_used'))}/"
            f"{sum(1 for r in rows if r.get('advised_used') is not None)}"
        ),
        "Critic: approvate": share("critic_approved"),
        "secondi (media)": round(sum(r["seconds"] for r in rows) / max(1, len(rows)), 1),
    }


def report(args: argparse.Namespace) -> int:

    base = Path(args.out)

    def load(labels: str) -> list[dict]:
        """One run, or several separated by commas: the same cases sent again."""
        return [
            row
            for label in labels.split(",")
            for row in json.loads((base / label / "results.json").read_text("utf-8"))
        ]

    runs_a, runs_b = load(args.a), load(args.b)
    a = {r["case"]: r for r in reversed(runs_a)}  # the first run's lesson, for the page
    b = {r["case"]: r for r in reversed(runs_b)}
    cases = sorted(set(a) & set(b))
    sa = _summary([r for r in runs_a if r["case"] in cases])
    sb = _summary([r for r in runs_b if r["case"] in cases])
    name_a, name_b = args.a.split(",")[0].rstrip("0123456789"), args.b.split(",")[0]
    name_b = name_b.rstrip("0123456789")
    print(f"{'':42} {name_a:>12} {name_b:>12}")
    for key in sa:
        print(f"{key:42} {str(sa[key]):>12} {str(sb[key]):>12}")
    # Case by case, every run: «7/7 against 8/8» was one case answered with
    # strategies instead of a lesson (third blind read).
    print("\nper caso (una colonna per esecuzione): minuti ok · conti · liste · nota falsa · s")
    for c in cases:
        for name, rows in ((name_a, runs_a), (name_b, runs_b)):
            cells = [
                f"{'✓' if r.get('minutes_ok') else ('—' if r.get('minutes_ok') is None else '✗')}"
                f"{'C' if r.get('count_mismatch') else '.'}"
                f"{len(r.get('usage_wrong') or [])}"
                f"{'N' if r.get('false_provenance') else '.'} {r['seconds']:>5}"
                for r in rows
                if r["case"] == c
            ]
            print(f"  {c} {name:>8}: " + " | ".join(cells))
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
