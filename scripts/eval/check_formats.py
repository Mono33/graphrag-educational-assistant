#!/usr/bin/env python3
"""Does a Word file, a deck or a photo survive the trip to a citation?

For every file in ``data/eval/corpus/derivati/manifest.yaml`` (built by
``build_derived_corpus.py``) this runs the same code a teacher's upload runs --
intake, parser, chunker -- and then renders the citation she would read. For
each probe sentence it answers three questions:

* **found** -- is the sentence in the extracted text at all?
* **where** -- does its chunk carry the right location: the slide it was on,
  the section heading it sat under?
* **what she reads** -- the citation as rendered in the lesson. A slide cited
  as "p. 3" sends a teacher to a page that does not exist.

Matching ignores whitespace, hyphens and case: the source PDF's text layer
splits words ("discipli na"), and OCR reads them whole. For OCR files a probe
counts as found when 80% of its words are in one chunk, since OCR errors are
expected and measured separately by their confidence.

No database, no embeddings, no network: it measures the parsing half, which is
where the format matters. The search half is format-blind once the chunk and
its locator exist.

Usage::

    python scripts/eval/check_formats.py
"""

from __future__ import annotations

import re
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

CORPUS = REPO_ROOT / "data" / "eval" / "corpus" / "derivati"


def squash(text: str) -> str:
    return re.sub(r"[\s\-­]+", "", text).lower()


def words(text: str) -> list[str]:
    return re.findall(r"\w{3,}", text.lower())


def locate(probe: str, chunks, ocr: bool):
    """The first chunk that carries the probe, or None."""
    target = squash(probe)
    for chunk in chunks:
        if target and target in squash(chunk.text):
            return chunk
    if ocr:
        wanted = words(probe)
        for chunk in chunks:
            have = set(words(chunk.text))
            if wanted and sum(w in have for w in wanted) / len(wanted) >= 0.8:
                return chunk
    return None


def rendered_citation(chunk, name: str) -> str:
    from aix.webui.materials.citations import render_citations
    from aix.webui.materials.retrieval import MaterialContextItem

    source = str(uuid.uuid4())
    item = MaterialContextItem(
        source_id=source, text=chunk.text, score=1.0, locator=dict(chunk.locator)
    )
    return render_citations(item.citation, {source: name})


def main() -> int:
    import yaml

    from aix.webui.materials.chunking import chunk_document
    from aix.webui.materials.parsing import IntakeRejected, check_upload, parse_document

    manifest = yaml.safe_load((CORPUS / "manifest.yaml").read_text(encoding="utf-8"))

    print("# Formati — dal file alla citazione\n")
    print(
        "| file | lettura | passaggi | sonda | trovata | posizione attesa | nel passaggio | la docente legge |"
    )
    print("| --- | --- | ---: | --- | --- | --- | --- | --- |")

    failures = 0
    for name, entry in manifest.items():
        content = (CORPUS / name).read_bytes()
        try:
            accepted = check_upload(name, content)
        except IntakeRejected as exc:
            print(f"| {name} | rifiutato: {exc.code} | | | | | | |")
            failures += 1
            continue

        ir = parse_document(content, accepted.media_type)
        quality = ir.quality()
        chunks = chunk_document(ir)
        reading = f"{quality['pages']} pag./slide"
        if quality["ocr_pages"]:
            reading += f", OCR {quality['ocr_pages']}"
        if quality["unread_pages"]:
            reading += f", **{quality['unread_pages']} non lette**"
        if quality["warnings"]:
            reading += f" ({', '.join(quality['warnings'])})"

        probes = entry.get("probes") or [{"text": ""}]
        for index, probe in enumerate(probes):
            chunk = locate(probe["text"], chunks, bool(entry.get("ocr"))) if probe["text"] else None
            if "slide" in probe:
                expected = f"slide {probe['slide']}" + (" (note)" if probe.get("notes") else "")
            elif "section" in probe:
                expected = f"sezione «{probe['section']}»"
            else:
                expected = "—"

            if not probe["text"]:
                found, where, cite = "(senza sonda)", "", ""
            elif chunk is None:
                found, where, cite = "**no**", "", ""
                failures += 1
            else:
                found = "sì"
                bits = []
                if chunk.locator.get("page") is not None:
                    bits.append(f"page={chunk.locator['page']}")
                if chunk.locator.get("heading"):
                    bits.append(f"heading=«{chunk.locator['heading'][:30]}»")
                where = ", ".join(bits) or "(nessuna)"
                cite = rendered_citation(chunk, name)

            label = name if index == 0 else ""
            summary = reading if index == 0 else ""
            count = str(len(chunks)) if index == 0 else ""
            print(
                f"| {label} | {summary} | {count} | {probe['text'][:40]}… | {found} | {expected} | {where} | {cite} |"
            )

    print(f"\nsonde non trovate: {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
