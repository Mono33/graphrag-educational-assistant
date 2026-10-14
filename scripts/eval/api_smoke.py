#!/usr/bin/env python3
"""The chatbot API end to end, against a running stack (2026-10-14).

Uploads a document, waits until it is ready, opens a session on it and runs
three turns -- a lesson, a question, a request for a new lesson -- printing what
each returned and how long it took. Real model calls: a run costs what one
lesson and one answer cost in the WebUI.

Usage::

    python scripts/eval/api_smoke.py                       # http://127.0.0.1:8766, deploy/.env.local
    python scripts/eval/api_smoke.py --base http://host:port --env path/to/.env
    python scripts/eval/api_smoke.py --file dispensa.pdf --end-user prova-1

The service account is read from ``GRAPH_API_USERNAME`` / ``GRAPH_API_PWD`` in the
env file; nothing is printed of it.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TEXT = """La Guerra fredda (1947-1991)

Dopo la Seconda guerra mondiale il mondo si divise in due blocchi: gli Stati Uniti
con i paesi della NATO e l'Unione Sovietica con il Patto di Varsavia.
Il Muro di Berlino fu costruito nel 1961 e abbattuto il 9 novembre 1989.
La crisi dei missili di Cuba, nell'ottobre 1962, portò le due superpotenze
vicinissime a una guerra nucleare.
"""


def _env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"')
    return values


def _docx(text: str) -> bytes:
    import docx

    document = docx.Document()
    for paragraph in text.strip().split("\n"):
        document.add_paragraph(paragraph)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _show(label: str, started: float, body: dict) -> None:
    kind = body.get("kind")
    text = body.get("markdown") or body.get("handoff_query") or ""
    print(f"\n== {label}: kind={kind} in {time.monotonic() - started:.1f}s")
    print("   " + text[:400].replace("\n", "\n   "))
    if body.get("phases"):
        print(f"   phases: {sorted(body['phases'])}")
    if "Fonti dai tuoi materiali" in text or "(" in text and ".docx" in text:
        print("   citations: resolved to the document name")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", default="http://127.0.0.1:8766")
    parser.add_argument("--env", default=str(REPO_ROOT / "deploy" / ".env.local"))
    parser.add_argument(
        "--file", help="a document to upload (default: a short DOCX on the Cold War)"
    )
    parser.add_argument("--end-user", default="smoke-test")
    args = parser.parse_args()

    env = _env(Path(args.env))
    auth = (env.get("GRAPH_API_USERNAME", ""), env.get("GRAPH_API_PWD", ""))
    if not all(auth):
        print("GRAPH_API_USERNAME / GRAPH_API_PWD missing in", args.env)
        return 2
    headers = {"X-End-User-Id": args.end_user}

    with httpx.Client(base_url=args.base, auth=auth, headers=headers, timeout=300) as client:
        if args.file:
            name, content = Path(args.file).name, Path(args.file).read_bytes()
        else:
            name, content = "guerra-fredda.docx", _docx(DEFAULT_TEXT)
        upload = client.post("/api/v1/materials", files={"file": (name, content)})
        upload.raise_for_status()
        material = upload.json()
        print(f"uploaded {name}: {material['material_id']} {material['status']}")

        for _ in range(120):
            state = client.get(f"/api/v1/materials/{material['material_id']}").json()
            if state["status"] != "processing":
                break
            time.sleep(2)
        print(
            f"document {state['status']}: {state['passages']} passages, quality={state['quality']}"
        )
        if state["status"] != "ready":
            return 1

        created = client.post(
            "/api/v1/sessions",
            json={
                "domain": "udl",
                "title": "Prova API",
                "educational_profile": {"pedagogical_intent": "intro"},
                "material_ids": [material["material_id"]],
            },
        )
        created.raise_for_status()
        session_id = created.json()["session_id"]
        print(f"session {session_id}")

        def turn(label: str, **body) -> dict:
            started = time.monotonic()
            response = client.post(f"/api/v1/sessions/{session_id}/turns", json=body)
            if response.status_code != 200:
                print(f"\n== {label}: HTTP {response.status_code} {response.text[:300]}")
                return {}
            result = response.json()
            _show(label, started, result)
            return result

        turn("lesson", message="Crea una lezione di 45 minuti sulla Guerra fredda dal documento")
        turn("question", message="Perché il Muro di Berlino è importante per la lezione?")
        handoff = turn("new lesson?", message="Ora fammi una lezione completa sui vulcani")
        if handoff.get("kind") == "handoff":
            print("   (handoff: the client would now resend with mode=lesson)")

        detail = client.get(f"/api/v1/sessions/{session_id}").json()
        print(f"\nsession turns: {[(t['role'], t['kind']) for t in detail['turns']]}")
        print(f"materials: {[(m['name'], m['status']) for m in detail['materials']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
