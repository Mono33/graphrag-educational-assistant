#!/usr/bin/env python3
"""Outline first, sections on request: what it would cost, against the whole lesson.

A prototype of the incremental output, run on the same real Writer calls as
``replay_sonnet_settings.py`` and outside the WebUI. For each recorded call:

1. **outline**: the same messages, told to write only the outline -- title,
   objectives, one line per section with its minutes -- and to offer three or
   four concrete next steps ("le attività nel dettaglio", "una verifica", ...);
2. **section on request**: the first offered step, written in full, as the
   teacher would get it after one click. The conversation so far is resent with
   the provider-cache mark, so the long input is billed at a tenth.

Compared with the whole lesson from a replay run (``--baseline`` = its
``grezzi.json``): cost, seconds to the first useful text, characters to read.
It also writes a page for a person to read the two side by side.

The outputs are teachers' lessons: ``--out`` must be outside the repository::

    python scripts/eval/prototype_outline_first.py --out /tmp/x --baseline /tmp/replay/grezzi.json
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("AIX_TRACING_ENABLED", "false")
os.environ["AIX_PROMPT_CACHE_ENABLED"] = "true"

import replay_sonnet_settings as replay  # noqa: E402

MODEL = "anthropic/claude-sonnet-5.5"

OUTLINE_RULE = (
    "\n\n**OUTPUT FOR THIS TURN (overrides the format above)**: do not write the full lesson. "
    "Write only its outline, in Italian: the title, level and duration; the objectives (at most 4 "
    "bullets); then one line per section or activity with its minutes and what happens in it. At the "
    "end, under the heading «Cosa sviluppo adesso?», offer 3 or 4 concrete next steps the teacher may "
    "want, numbered, the most useful first (for example: le attività nel dettaglio, una verifica "
    "pronta, gli adattamenti per gli studenti con BES, una scheda per gli studenti). At most 250 words."
)

SECTION_REQUEST = (
    "Sviluppa ora nel dettaglio il punto 1 di «Cosa sviluppo adesso?», coerente con la scaletta: "
    "solo quella parte, pronta da usare in classe, con le citazioni dei materiali dove servono."
)


async def call(client, messages, max_tokens):
    started = time.monotonic()
    r = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.7,
        extra_body={"reasoning": {"effort": "low"}},
    )
    u = r.usage
    return {
        "content": r.choices[0].message.content or "",
        "cost": getattr(u, "cost", 0) or 0,
        "seconds": time.monotonic() - started,
        "cached": (u.prompt_tokens_details.cached_tokens or 0) if u.prompt_tokens_details else 0,
    }


async def prototype(client, case):
    from aix.core import prompt_cache

    base = [{**m, "content": replay._text(m.get("content"))} for m in case["messages"]]
    first = [*base]
    first[0] = {**first[0], "content": first[0]["content"] + OUTLINE_RULE}
    outline = await call(
        client,
        prompt_cache.cacheable(first, model=MODEL, stable_user_prefix=first[1]["content"]),
        1200,
    )
    second = [
        *first,
        {"role": "assistant", "content": outline["content"]},
        {"role": "user", "content": SECTION_REQUEST},
    ]
    section = await call(
        client,
        prompt_cache.cacheable(second, model=MODEL, stable_user_prefix=first[1]["content"]),
        4000,
    )
    return {"outline": outline, "section": section}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--baseline", required=True, help="grezzi.json of a replay run with the whole lessons"
    )
    parser.add_argument("--baseline-config", default="claude-sonnet-5.5 · low")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    if REPO_ROOT in out.parents:
        parser.error("--out is inside the repository: the lessons must not end up in git")
    out.mkdir(parents=True, exist_ok=True)

    from aix.core.config import config

    baseline = {
        r["case"]["id"]: r
        for r in json.load(open(args.baseline, encoding="utf-8"))
        if r["config"] == args.baseline_config and r.get("ok")
    }
    cases = [
        c
        for c in replay.fetch_cases("anthropic/claude-sonnet-5", {"writer": 3, "ibrida": 6})
        if c["id"] in baseline
    ]
    print(f"{len(cases)} richieste vere con la lezione intera di riferimento", file=sys.stderr)

    async def run_all():
        client = config.openai.get_async_client()
        return await asyncio.gather(*(prototype(client, c) for c in cases))

    results = asyncio.run(run_all())

    rows = []
    for case, res in zip(cases, results, strict=True):
        full = baseline[case["id"]]
        rows.append(
            {
                "kind": case["kind"],
                "full_cost": full["cost"],
                "full_seconds": full["seconds"],
                "full_chars": len(full["content"]),
                "outline_cost": res["outline"]["cost"],
                "outline_seconds": res["outline"]["seconds"],
                "outline_chars": len(res["outline"]["content"]),
                "section_cost": res["section"]["cost"],
                "section_seconds": res["section"]["seconds"],
                "section_chars": len(res["section"]["content"]),
                "section_cached": res["section"]["cached"],
            }
        )

    m = lambda k: statistics.mean(r[k] for r in rows)  # noqa: E731
    text = (
        f"# Prima la scaletta, poi la sezione — {len(rows)} richieste vere (Sonnet 5.5 · low)\n\n"
        "| | costo medio | secondi (media) | caratteri da leggere |\n| --- | ---: | ---: | ---: |\n"
        f"| lezione intera (oggi) | ${m('full_cost'):.4f} | {m('full_seconds'):.0f} | {m('full_chars'):.0f} |\n"
        f"| solo scaletta | ${m('outline_cost'):.4f} | {m('outline_seconds'):.0f} | {m('outline_chars'):.0f} |\n"
        f"| scaletta + una sezione su richiesta | ${m('outline_cost') + m('section_cost'):.4f} | "
        f"{m('outline_seconds'):.0f} + {m('section_seconds'):.0f} | {m('outline_chars') + m('section_chars'):.0f} |\n\n"
        f"Token letti dalla cache nella seconda chiamata (media): {m('section_cached'):.0f}.\n"
    )
    (out / "risultati.md").write_text(text, encoding="utf-8")

    import markdown

    rng = random.Random(3)
    key, sections = {}, []
    for n, (case, res) in enumerate(zip(cases, results, strict=True), 1):
        versions = [
            (
                "scaletta + sezione",
                res["outline"]["content"] + "\n\n---\n\n" + res["section"]["content"],
            ),
            ("lezione intera", baseline[case["id"]]["content"]),
        ]
        rng.shuffle(versions)
        key[f"richiesta {n}"] = {chr(65 + i): name for i, (name, _) in enumerate(versions)}
        cols = "".join(
            f"<section><h3>Versione {chr(65 + i)}</h3>{markdown.markdown(body, extensions=['tables'])}</section>"
            for i, (_, body) in enumerate(versions)
        )
        sections.append(
            f"<article><h2>Richiesta {n}</h2><p class='q'><b>Richiesta:</b> {html.escape(case['query'][:500])}</p>"
            f"<div class='cols'>{cols}</div></article>"
        )
    page = (
        "<!doctype html><html lang='it'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'><title>Scaletta o lezione intera</title><style>"
        "body{font-family:system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}"
        "article{margin:32px 0;border-top:3px solid #333;padding-top:8px}"
        ".cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:12px}"
        "section{background:#fff;border:1px solid #ddd;border-radius:6px;padding:12px;overflow-x:auto;font-size:14px}"
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:2px 6px}.q{background:#eef;padding:8px}"
        "</style></head><body><h1>Per partire, che cosa ti serve di più?</h1>"
        "<p>Per ogni richiesta, una versione è la lezione intera di oggi, l'altra una scaletta con proposte "
        "seguita dalla prima proposta sviluppata. Scrivi quale ti è più utile per cominciare (A o B) e perché.</p>"
        + "".join(sections)
        + "</body></html>"
    )
    (out / "lettura_alla_cieca.html").write_text(page, encoding="utf-8")
    (out / "chiave.json").write_text(
        json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
