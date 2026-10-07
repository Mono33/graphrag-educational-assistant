#!/usr/bin/env python3
"""Sonnet 5 and 5.5, with and without the reasoning effort, on real lessons.

The Writer, revision, light-turn and Planner calls recorded in Langfuse are sent
again, with exactly their messages and parameters, under each configuration. What
changes is only the model and the reasoning setting. Reported per kind of call:

* cost (as billed by OpenRouter), seconds, output and reasoning tokens, and how
  many answers were cut off by ``max_tokens``;
* lessons (Writer, revision): the Critic's score and verdict -- the same fixed
  Critic as ``compare_role_models.py critic`` -- and the citations: markers the
  input gave that the lesson kept, and markers it invented;
* light turns: cut-offs and invented citations;
* Planner: valid JSON, and agreement on language, intent and scope with the first
  configuration (today's).

It also writes ``lettura_alla_cieca.html``: each lesson in every configuration,
in a hidden order, for a person to read; the key is in ``chiave.json``.

The outputs are teachers' lessons: ``--out`` must be outside the repository.
Run from the repo root, with the ``.env`` keys (OpenRouter, Langfuse)::

    python scripts/eval/replay_sonnet_settings.py --out /tmp/aix_replay
    python scripts/eval/replay_sonnet_settings.py --out /tmp/x --dry-run   # what would be replayed
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import html
import json
import os
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("AIX_TRACING_ENABLED", "false")  # do not trace the replay itself

#: (label, model, effort); effort None = no reasoning setting, as before 2026-10-07.
CONFIGS = [
    ("Sonnet 5 · come oggi", "anthropic/claude-sonnet-5", None),
    ("Sonnet 5 · low", "anthropic/claude-sonnet-5", "low"),
    ("Sonnet 5.5 · come oggi", "anthropic/claude-sonnet-5.5", None),
    ("Sonnet 5.5 · low", "anthropic/claude-sonnet-5.5", "low"),
]

KINDS = {  # marker in the system prompt -> kind
    "Content Writer specialized": "writer",
    "Content Writer creating lessons": "revisione",
    "Sei l'assistente didattico": "leggera",
    "Educational Planning": "planner",
}

MARKER = re.compile(r"\[\[material:[^\]]*\]\]")


# ---------------------------------------------------------------------------
# Langfuse
# ---------------------------------------------------------------------------


def _langfuse_get(path: str, **query) -> dict:
    base = (os.getenv("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com").strip('"').rstrip("/")
    token = base64.b64encode(
        f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()
    ).decode()
    url = f"{base}/api/public/{path}" + ("?" + urllib.parse.urlencode(query) if query else "")
    request = urllib.request.Request(url, headers={"Authorization": "Basic " + token})
    for attempt in range(5):
        try:
            return json.load(urllib.request.urlopen(request, timeout=60))
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))  # rate limited: wait and try again
    raise RuntimeError("unreachable")


def _all(path: str) -> list[dict]:
    rows, page = [], 1
    while True:
        data = _langfuse_get(
            path, page=page, limit=100, **({"type": "GENERATION"} if path == "observations" else {})
        )
        rows += data["data"]
        if page >= data["meta"]["totalPages"]:
            return rows
        page += 1


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n\n".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content or "")


def _kind(messages: list) -> str | None:
    system = _text(next((m.get("content") for m in messages if m.get("role") == "system"), ""))
    return next((kind for marker, kind in KINDS.items() if marker in system), None)


def _domain(system: str) -> str:
    """The Writer names its domain extension; a revision does not, and defaults to udl."""
    found = re.search(r"Domain Expert Knowledge \((UDL|NEURO)\)", system)
    return found.group(1).lower() if found else "udl"


def fetch_cases(source_model: str, wanted: dict[str, int]) -> list[dict]:
    generations = _all("observations")
    queries = {t["id"]: t.get("input") for t in _all("traces")}

    cases, taken = [], {k: 0 for k in wanted}
    for g in sorted(generations, key=lambda g: g["startTime"], reverse=True):
        messages = g.get("input")
        if g.get("model") != source_model or not isinstance(messages, list):
            continue
        if any(m.get("role") == "assistant" for m in messages):
            continue  # a continuation: replayed as part of its first call, not alone
        kind = _kind(messages)
        if kind not in wanted or taken[kind] >= wanted[kind]:
            continue
        params = g.get("modelParameters") or {}
        query = queries.get(g["traceId"]) or ""
        writer_system = _text(messages[0].get("content")) if messages else ""
        cases.append(
            {
                "id": g["id"],
                "kind": kind,
                "when": g["startTime"][:16],
                "messages": messages,
                "max_tokens": int(params.get("max_tokens") or 4000),
                "temperature": float(params.get("temperature") or 0.5),
                "query": query if isinstance(query, str) else json.dumps(query, ensure_ascii=False),
                "domain": _domain(writer_system),
            }
        )
        taken[kind] += 1
        if taken == wanted:
            break
    return cases


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


async def replay(client, case: dict, model: str, effort: str | None) -> dict:
    kwargs = {
        "model": model,
        "messages": case["messages"],
        "max_tokens": case["max_tokens"],
        "temperature": case["temperature"],
    }
    if case["kind"] == "planner":
        kwargs["response_format"] = {"type": "json_object"}
    if effort:
        kwargs["extra_body"] = {"reasoning": {"effort": effort}}
    started = time.monotonic()
    try:
        response = await client.chat.completions.create(**kwargs)
    except Exception as exc:  # noqa: BLE001 — a failure is a result
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:120]}"}
    usage = response.usage
    details = getattr(usage, "completion_tokens_details", None)
    return {
        "ok": True,
        "seconds": time.monotonic() - started,
        "cost": getattr(usage, "cost", None) or 0.0,
        "output_tokens": usage.completion_tokens or 0,
        "reasoning_tokens": (getattr(details, "reasoning_tokens", 0) or 0) if details else 0,
        "finish": response.choices[0].finish_reason,
        "content": response.choices[0].message.content or "",
    }


async def critique(case: dict, lesson: str) -> dict:
    from aix.agent.agents import critic_agent
    from aix.agent.agents.retriever_agent import RetrievalResult

    try:
        result = await critic_agent.CriticAgent().critique(
            lesson_plan=lesson,
            teacher_query=case["query"],
            retrieval_result=RetrievalResult(
                nodes=[], relationships=[], recommendations=[], confidence="MEDIUM"
            ),
            revision_count=0,
            max_revisions=1,
            domain=case["domain"],
        )
    except Exception as exc:  # noqa: BLE001
        return {"critic_ok": False, "critic_error": type(exc).__name__}
    unreadable = any(
        marker in (result.summary or "")
        for marker in ("[parse_error]", "parsing error", "Auto-approved")
    )
    return {
        "critic_ok": not unreadable,
        "score": result.average_score,
        "approved": result.approved,
    }


def citations(case: dict, content: str) -> dict:
    given = set(MARKER.findall("\n".join(_text(m.get("content")) for m in case["messages"])))
    used = set(MARKER.findall(content))
    return {"cited": len(used & given), "invented": len(used - given)}


def plan_fields(content: str) -> dict | None:
    from aix.agent.agents.planner_agent import _extract_json

    try:
        data = _extract_json(content)
    except Exception:  # noqa: BLE001
        return None
    return {
        k: str(data.get(k)).lower() for k in ("response_language", "query_intent", "scope_status")
    }


async def run(cases: list[dict], concurrency: int) -> dict:
    from aix.core.config import config

    client = config.openai.get_async_client()
    gate = asyncio.Semaphore(concurrency)
    results: dict[tuple[str, str], dict] = {}

    async def one(case, label, model, effort):
        async with gate:
            r = await replay(client, case, model, effort)
            if r["ok"] and case["kind"] in ("writer", "revisione", "leggera"):
                r.update(citations(case, r["content"]))
            if r["ok"] and case["kind"] in ("writer", "revisione"):
                r.update(await critique(case, r["content"]))
            if r["ok"] and case["kind"] == "planner":
                r["plan"] = plan_fields(r["content"])
            results[(case["id"], label)] = r
            print(
                f"  {case['kind']:9} {label:24} {'ok' if r['ok'] else r['error']}", file=sys.stderr
            )

    await asyncio.gather(*(one(c, *cfg) for c in cases for cfg in CONFIGS))
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def report(cases: list[dict], results: dict) -> str:
    reference = CONFIGS[0][0]
    lines = [f"# Sonnet: modello e sforzo di ragionamento su {len(cases)} chiamate vere\n"]
    for kind in ("writer", "revisione", "leggera", "planner"):
        group = [c for c in cases if c["kind"] == kind]
        if not group:
            continue
        lines.append(f"\n## {kind} ({len(group)} chiamate vere)\n")
        head = "| configurazione | costo medio | secondi (mediana) | output medio | ragionamento medio | troncate | errori |"
        rule = "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
        if kind in ("writer", "revisione"):
            head += " voto Critic medio | approvate | citazioni tenute | inventate |"
            rule += " ---: | ---: | ---: | ---: |"
        if kind == "leggera":
            head += " citazioni inventate |"
            rule += " ---: |"
        if kind == "planner":
            head += " JSON valido | accordo lingua/intento/scope con «come oggi» |"
            rule += " ---: | ---: |"
        lines += [head, rule]
        for label, _, _ in CONFIGS:
            rs = [results.get((c["id"], label), {"ok": False}) for c in group]
            ok = [r for r in rs if r["ok"]]
            if not ok:
                lines.append(f"| {label} | — | — | — | — | — | {len(rs)} |")
                continue
            row = (
                f"| {label} | ${statistics.mean(r['cost'] for r in ok):.4f} "
                f"| {statistics.median(r['seconds'] for r in ok):.0f} "
                f"| {statistics.mean(r['output_tokens'] for r in ok):.0f} "
                f"| {statistics.mean(r['reasoning_tokens'] for r in ok):.0f} "
                f"| {sum(r['finish'] in ('length', 'max_tokens') for r in ok)}/{len(ok)} "
                f"| {len(rs) - len(ok)} |"
            )
            if kind in ("writer", "revisione"):
                scored = [r for r in ok if r.get("critic_ok")]
                row += (
                    f" {statistics.mean(r['score'] for r in scored):.2f} |" if scored else " — |"
                ) + (
                    f" {sum(r['approved'] for r in scored)}/{len(scored)} "
                    f"| {sum(r.get('cited', 0) for r in ok)} | {sum(r.get('invented', 0) for r in ok)} |"
                )
            if kind == "leggera":
                row += f" {sum(r.get('invented', 0) for r in ok)} |"
            if kind == "planner":
                valid = sum(r.get("plan") is not None for r in ok)
                same = total = 0
                for c in group:
                    mine = results.get((c["id"], label), {}).get("plan")
                    ref = results.get((c["id"], reference), {}).get("plan")
                    if mine and ref:
                        for field in ref:
                            total += 1
                            same += mine[field] == ref[field]
                row += f" {valid}/{len(ok)} | {same}/{total} |"
            lines.append(row)
    return "\n".join(lines) + "\n"


def blind_page(cases: list[dict], results: dict, seed: int) -> tuple[str, dict]:
    import markdown

    rng = random.Random(seed)
    key, sections = {}, []
    lessons = [c for c in cases if c["kind"] in ("writer", "revisione")]
    for n, case in enumerate(lessons, 1):
        labels = [
            label for label, _, _ in CONFIGS if results.get((case["id"], label), {}).get("ok")
        ]
        rng.shuffle(labels)
        key[f"lezione {n}"] = {chr(65 + i): label for i, label in enumerate(labels)}
        columns = "".join(
            f"<section><h3>Versione {chr(65 + i)}</h3>"
            f"{markdown.markdown(results[(case['id'], label)]['content'], extensions=['tables'])}</section>"
            for i, label in enumerate(labels)
        )
        sections.append(
            f"<article><h2>Lezione {n} — {html.escape(case['kind'])}</h2>"
            f"<p class='q'><b>Richiesta della docente:</b> {html.escape(case['query'][:600])}</p>"
            f"<div class='cols'>{columns}</div></article>"
        )
    page = (
        "<!doctype html><html lang='it'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Lettura alla cieca</title><style>"
        "body{font-family:system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}"
        "article{margin:32px 0;border-top:3px solid #333;padding-top:8px}"
        ".cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px}"
        "section{background:#fff;border:1px solid #ddd;border-radius:6px;padding:12px;overflow-x:auto;font-size:14px}"
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:2px 6px}.q{background:#eef;padding:8px}"
        "</style></head><body><h1>Lettura alla cieca</h1>"
        "<p>Per ogni lezione, le versioni A–D sono la stessa richiesta con modelli o impostazioni "
        "diverse, in ordine nascosto. Annota per ognuna l'ordine di preferenza (es. «B &gt; D &gt; A &gt; C») "
        "e cosa ti ha fatto scegliere. La chiave è in <code>chiave.json</code>: aprila solo dopo.</p>"
        + "".join(sections)
        + "</body></html>"
    )
    return page, key


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True, help="a folder OUTSIDE the repository")
    parser.add_argument("--source-model", default="anthropic/claude-sonnet-5")
    parser.add_argument("--writer", type=int, default=3)
    parser.add_argument("--revisions", type=int, default=3)
    parser.add_argument("--light", type=int, default=6)
    parser.add_argument("--planner", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    if REPO_ROOT in out.parents or out == REPO_ROOT:
        parser.error("--out is inside the repository: the lessons must not end up in git")
    out.mkdir(parents=True, exist_ok=True)

    import aix.core.config  # noqa: F401 — loads .env

    wanted = {
        "writer": args.writer,
        "revisione": args.revisions,
        "leggera": args.light,
        "planner": args.planner,
    }
    cases = fetch_cases(args.source_model, {k: v for k, v in wanted.items() if v})
    print(
        f"{len(cases)} chiamate vere: "
        + ", ".join(f"{k} {sum(c['kind'] == k for c in cases)}" for k in wanted),
        file=sys.stderr,
    )
    if args.dry_run:
        for c in cases:
            print(
                f"{c['when']} {c['kind']:9} {c['domain']:5} max_tokens={c['max_tokens']} «{c['query'][:70]}»"
            )
        return 0

    results = asyncio.run(run(cases, args.concurrency))
    text = report(cases, results)
    (out / "risultati.md").write_text(text, encoding="utf-8")
    page, key = blind_page(cases, results, args.seed)
    (out / "lettura_alla_cieca.html").write_text(page, encoding="utf-8")
    (out / "chiave.json").write_text(
        json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    raw = [
        {
            "case": {k: v for k, v in c.items() if k != "messages"},
            "config": label,
            **results.get((c["id"], label), {}),
        }
        for c in cases
        for label, _, _ in CONFIGS
    ]
    (out / "grezzi.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(text)
    print(f"Scritti in {out}: risultati.md, lettura_alla_cieca.html, chiave.json, grezzi.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
