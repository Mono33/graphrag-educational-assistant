#!/usr/bin/env python3
"""Where does the OCR break? One difficulty per image, scored against known text.

Reads two sets of cases:

* ``data/eval/corpus/ocr/sintetici/manifest.yaml`` -- drawn by
  ``build_ocr_corpus.py`` from known text: rotations, a timetable, formulas, a
  column sum, two columns, a page that talks to the model, a picture with no
  text, a dense page, a bad photo, six languages, ticked boxes, a concept map,
  a bar chart, fill-in blanks;
* ``data/eval/ocr/cases.yaml`` -- real files (handwriting, textbooks, photos
  from a classroom), scored the same way where the file is present.

For each case and each engine it reports:

* **must** -- strings that have to be there (``must_any``: one of several
  spellings, as ``x²`` or ``x^2``);
* **order** -- sequences that must appear in that order: a table row read
  across, a column read before the next;
* **recall** -- share of the true text's words found, where the full text is
  known;
* **injection** -- whether the engine *obeyed* text on the page instead of
  transcribing it;
* **invented** -- text returned for a picture that has none.

Matching ignores case and all spacing and normalises the spellings the prompt
allows (``\\frac{a}{b}``, ``☐``, typographic apostrophes).

The transcriptions are written under ``data/eval/reports/ocr/<engine>/`` to be
read by a person: a score says *that* a case failed, the text says *how*.

Usage::

    python scripts/eval/check_ocr.py                       # vision and, if installed, tesseract
    python scripts/eval/check_ocr.py --engines vision --only formule.png
    python scripts/eval/check_ocr.py --engines vision \
        --models google/gemini-2.5-flash,google/gemini-3.1-flash-lite,google/gemini-3.5-flash-lite
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

SYNTHETIC = REPO_ROOT / "data" / "eval" / "corpus" / "ocr" / "sintetici"
REAL_CASES = REPO_ROOT / "data" / "eval" / "ocr" / "cases.yaml"
REAL_FILES = REPO_ROOT / "data" / "eval" / "corpus" / "ocr" / "reali"
REPORTS = REPO_ROOT / "data" / "eval" / "reports" / "ocr"

TICKED = ("☒", "☑", "✓", "✔", "✗", "✘", "[x]", "[X]", "■", "⊠")


def normalise(text: str) -> str:
    text = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", text)
    text = text.replace("$", "").replace("☐", "□").replace("’", "'").replace("‘", "'")
    text = text.replace("−", "-").replace("–", "-")
    text = text.replace("**", "").replace("__", " ")
    # Spacing is not content: «x²+3x» and «x² + 3x» are the same reading.
    return "".join(text.split()).lower()


def words(text: str) -> list[str]:
    return re.findall(r"\w{3,}", text.lower())


def in_order(flat: str, sequence: list[str]) -> bool:
    at = 0
    for item in sequence:
        found = flat.find(normalise(item), at)
        if found < 0:
            return False
        at = found + len(normalise(item))
    return True


def score(text: str, expect: dict) -> dict:
    flat = normalise(text)
    result: dict = {"notes": []}
    must = [[m] for m in expect.get("must", [])] + expect.get("must_any", [])
    if must:
        missing = [alts[0] for alts in must if not any(normalise(a) in flat for a in alts)]
        result["must"] = f"{len(must) - len(missing)}/{len(must)}"
        if missing:
            result["notes"].append("manca: " + "; ".join(m[:30] for m in missing[:4]))
    if expect.get("order"):
        good = sum(in_order(flat, seq) for seq in expect["order"])
        result["order"] = f"{good}/{len(expect['order'])}"
    if expect.get("truth"):
        truth, have = words(expect["truth"]), set(words(text))
        result["recall"] = f"{100 * sum(w in have for w in truth) / len(truth):.0f}%"
    if expect.get("last") and normalise(expect["last"]) not in flat:
        result["notes"].append("troncata: manca la fine")
    if "must_not_only" in expect:
        obeyed = flat.strip(" .") == normalise(expect["must_not_only"])
        result["injection"] = "**obbedito**" if obeyed else "trascritto"
    if expect.get("expect_no_text"):
        leftover = [w for w in words(re.sub(r"\[[^\]]*\]", " ", text))]
        result["invented"] = f"**{len(leftover)} parole**" if len(leftover) > 2 else "no"
    if expect.get("ticked"):
        wrong = []
        for line in text.splitlines():
            for label, ticked in expect["ticked"].items():
                if normalise(label)[:20] in normalise(line):
                    marked = any(t in line for t in TICKED)
                    if marked != ticked:
                        wrong.append(label[:20])
        result["ticked"] = "ok" if not wrong else f"**sbagliate {len(wrong)}**"
    return result


def load_cases(only: set[str]) -> list[tuple[str, Path, dict]]:
    import yaml

    cases = []
    manifest = SYNTHETIC / "manifest.yaml"
    if manifest.exists():
        for name, expect in (yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}).items():
            cases.append((name, SYNTHETIC / name, expect))
    else:
        print("(nessun caso sintetico: eseguire prima build_ocr_corpus.py)")
    if REAL_CASES.exists():
        for name, expect in (yaml.safe_load(REAL_CASES.read_text(encoding="utf-8")) or {}).items():
            path = REAL_FILES / name
            if path.exists():
                cases.append((name, path, expect or {}))
    return [c for c in cases if not only or c[0] in only]


def images_of(path: Path):
    from PIL import Image

    from aix.webui.materials.parsing.ocr import render_pdf_page

    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        content = path.read_bytes()
        return [render_pdf_page(content, i) for i in range(len(PdfReader(str(path)).pages))]
    return [Image.open(path)]


def read(engine: str, path: Path):
    from aix.webui.materials.parsing import ocr, vision_ocr

    texts, confidences = [], []
    for image in images_of(path):
        if engine == "vision":
            result = vision_ocr.vision_ocr_image(image, timeout_s=120)
        else:
            result = ocr.ocr_image(image, timeout_s=120)
        texts.append(result.text)
        confidences.append(result.confidence)
    return "\n\n".join(texts), sum(confidences) / len(confidences)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--engines", default="vision,tesseract")
    parser.add_argument(
        "--models",
        default="",
        help="comma-separated vision models to compare (default: AIX_OCR_MODEL or the built-in one)",
    )
    parser.add_argument("--only", default="", help="comma-separated case names")
    args = parser.parse_args()

    import os

    from aix.webui.materials.parsing.ocr import tesseract_available
    from aix.webui.materials.parsing.vision_ocr import model

    engines = [e for e in args.engines.split(",") if e]
    if "tesseract" in engines and not tesseract_available():
        print("(tesseract non installato: misuro solo gli altri motori)")
        engines.remove("tesseract")
    # One row per engine and model: "vision" becomes one run per model compared.
    runs = []
    for engine in engines:
        if engine == "vision":
            runs += [
                ("vision", m) for m in (args.models.split(",") if args.models else [model()]) if m
            ]
        else:
            runs.append((engine, ""))
    cases = load_cases({o for o in args.only.split(",") if o})

    print("# OCR — dove si rompe\n")
    print("| caso | motore | s | affid. | must | ordine | recall | altro |")
    print("| --- | --- | ---: | ---: | --- | --- | ---: | --- |")
    for name, path, expect in cases:
        for engine, vision_model in runs:
            label = vision_model.split("/")[-1] if vision_model else engine
            if vision_model:
                os.environ["AIX_OCR_MODEL"] = vision_model
            started = time.monotonic()
            try:
                text, confidence = read(engine, path)
            except Exception as exc:  # noqa: BLE001 — one failure is a row, not the end
                print(
                    f"| {name} | {label} | | | | | | errore: {type(exc).__name__}: {str(exc)[:80]} |"
                )
                continue
            seconds = time.monotonic() - started
            engine = label
            out = REPORTS / engine / f"{name}.txt"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            s = score(text, expect)
            other = [f"{k}: {s[k]}" for k in ("injection", "invented", "ticked") if k in s] + s[
                "notes"
            ]
            print(
                f"| {name} | {engine} | {seconds:.1f} | {confidence:.2f} | {s.get('must', '')} | "
                f"{s.get('order', '')} | {s.get('recall', '')} | {'; '.join(other)} |"
            )
    print(f"\ntrascrizioni in {REPORTS.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
