#!/usr/bin/env python3
"""Build Word, PowerPoint and image versions of real pages, with known content.

Why this exists
---------------
Until 2026-09-25 only PDFs had been tried by a teacher. The parsers for DOCX,
PPTX and images existed and had unit tests on fixtures, but nobody had checked
that a real lesson's text survives the trip: upload, parse, chunk, search,
cite. A fixture proves the parser runs; it does not prove the citation points
somewhere a teacher can find.

So the content is not invented. Four pages of the Italian ministerial curriculum
(``indicazioni_nazionali_dm254_2012.pdf`` -- the pages where history,
geography, mathematics and music open) are re-laid out as the files a teacher actually attaches:

* a **Word** document, one section per subject under its heading, and a table;
* a **PowerPoint** deck, one slide per subject, with part of the text only in
  the **speaker notes** -- teachers put real content there;
* **images** of one page: a clean scan, and a phone-photo imitation (tilted,
  soft, heavily compressed);
* one page of the real scanned grammar, as a photograph would arrive.

For every file the manifest records *probe* sentences taken from the source
page and where a correct parse must put them: which slide, which section, or
simply "in the text". ``check_formats.py`` then measures against it.

The output folder is git-ignored; this script is the versioned part.

Usage::

    python scripts/eval/build_derived_corpus.py
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ORIGINALS = REPO_ROOT / "data" / "eval" / "corpus" / "originali"
OUT = REPO_ROOT / "data" / "eval" / "corpus" / "derivati"

CURRICULUM = ORIGINALS / "indicazioni_nazionali_dm254_2012.pdf"
SCAN = ORIGINALS / "grammatica_vergani_scansione.pdf"

#: Pages where a subject *opens*, each starting with its own heading -- read
#: page by page on 2026-09-25, after a keyword search had labelled a page about
#: infant-school routines as "music".
SUBJECTS = [("Storia", 41), ("Geografia", 46), ("Matematica", 49), ("Musica", 58)]


def page_text(pdf: Path, number: int) -> str:
    from pypdf import PdfReader

    return PdfReader(str(pdf)).pages[number - 1].extract_text() or ""


def sentences(text: str) -> list[str]:
    flat = " ".join(text.split())
    return [s.strip() for s in re.split(r"(?<=[.;:])\s+", flat) if len(s.strip()) > 25]


def probe(candidates: list[str]) -> str:
    """A sentence distinctive enough to search for: long, but not a whole page."""
    # Not the first sentence: it carries the page number and the running
    # heading ("46 GEOGRAFIA La geografia…"), which no teacher's file would.
    # Not one with a hyphenation the PDF split ("geomet riche").
    usable = [
        s
        for s in candidates[1:]
        if 60 <= len(s) <= 220 and not s[0].isdigit() and not re.search(r"\w -\s|\w - \w", s)
    ]
    return (usable or candidates)[0]


def build_docx(sections: dict[str, list[str]], path: Path) -> list[dict]:
    from docx import Document

    doc = Document()
    doc.add_heading("Estratto dalle Indicazioni nazionali", level=0)
    probes = []
    for subject, body in sections.items():
        doc.add_heading(subject, level=1)
        for start in range(0, len(body), 3):
            doc.add_paragraph(" ".join(body[start : start + 3]))
        probes.append({"text": probe(body), "section": subject})
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text, table.rows[0].cells[1].text = "Disciplina", "Pagina originale"
    for subject, number in SUBJECTS:
        row = table.add_row().cells
        row[0].text, row[1].text = subject, str(number)
    doc.save(path)
    return probes


def build_pptx(sections: dict[str, list[str]], path: Path) -> list[dict]:
    from pptx import Presentation

    deck = Presentation()
    title = deck.slides.add_slide(deck.slide_layouts[0])
    title.shapes.title.text = "Indicazioni nazionali"
    title.placeholders[1].text = "Estratto per la lezione"

    probes = []
    for slide_number, (subject, body) in enumerate(sections.items(), start=2):
        slide = deck.slides.add_slide(deck.slide_layouts[1])
        slide.shapes.title.text = subject
        on_slide, in_notes = body[:3], body[3:]
        frame = slide.placeholders[1].text_frame
        frame.text = on_slide[0]
        for line in on_slide[1:]:
            frame.add_paragraph().text = line
        slide.notes_slide.notes_text_frame.text = " ".join(in_notes)
        probes.append({"text": probe(on_slide), "slide": slide_number})
        if in_notes:
            probes.append({"text": probe(in_notes), "slide": slide_number, "notes": True})
    deck.save(path)
    return probes


def render(pdf: Path, number: int, scale: float):
    import pypdfium2 as pdfium

    return pdfium.PdfDocument(str(pdf))[number - 1].render(scale=scale).to_pil().convert("RGB")


def phone_photo(image):
    """Tilted, soft, small and compressed: what a phone on a desk produces."""
    from PIL import Image, ImageFilter

    tilted = image.rotate(2.5, expand=True, fillcolor=(235, 232, 225), resample=Image.BICUBIC)
    small = tilted.resize((1100, int(1100 * tilted.height / tilted.width)))
    return small.filter(ImageFilter.GaussianBlur(0.8))


def main() -> int:
    if not CURRICULUM.exists():
        print(f"missing source: {CURRICULUM}", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)

    sections = {
        subject: sentences(page_text(CURRICULUM, number))[:8] for subject, number in SUBJECTS
    }
    manifest: dict[str, dict] = {}

    manifest["indicazioni_estratto.docx"] = {
        "source": f"{CURRICULUM.name}, pagine {', '.join(str(n) for _, n in SUBJECTS)}",
        "probes": build_docx(sections, OUT / "indicazioni_estratto.docx"),
    }
    manifest["indicazioni_slide.pptx"] = {
        "source": f"{CURRICULUM.name}, pagine {', '.join(str(n) for _, n in SUBJECTS)}",
        "probes": build_pptx(sections, OUT / "indicazioni_slide.pptx"),
    }

    geography = render(CURRICULUM, 46, scale=150 / 72)
    geography.save(OUT / "geografia_scansione.png")
    buffer = io.BytesIO()
    phone_photo(geography).save(buffer, format="JPEG", quality=55)
    (OUT / "geografia_foto.jpg").write_bytes(buffer.getvalue())
    geo_probe = probe(sentences(page_text(CURRICULUM, 46)))
    for name, kind in (
        ("geografia_scansione.png", "scansione pulita"),
        ("geografia_foto.jpg", "foto da telefono"),
    ):
        manifest[name] = {
            "source": f"{CURRICULUM.name}, pagina 46 ({kind})",
            "ocr": True,
            "probes": [{"text": geo_probe}],
        }

    if SCAN.exists():
        render(SCAN, 21, scale=1.5).save(OUT / "grammatica_pagina.jpg", quality=70)
        manifest["grammatica_pagina.jpg"] = {
            "source": f"{SCAN.name}, pagina 21 (scansione vera, senza testo di riferimento)",
            "ocr": True,
            "probes": [],
        }

    (OUT / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False, width=100), encoding="utf-8"
    )
    for name in manifest:
        print(f"  {name:<28} {(OUT / name).stat().st_size / 1024:>7.0f} KB")
    print(f"manifest: {OUT / 'manifest.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
