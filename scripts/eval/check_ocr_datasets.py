#!/usr/bin/env python3
"""OCR on public form datasets: FUNSD (English) and XFUND (Italian).

``check_ocr.py`` tests one difficulty per image, drawn by us. These are real
scans annotated word by word by other people, so they answer a different
question: on pages nobody here chose, how much of the text comes back, how much
is invented, and does a label stay next to its value?

* **FUNSD** -- 199 noisy scanned business forms from the 1980s-90s, typed and
  partly handwritten, 754x1000 grey. The 50 test forms are used.
* **XFUND (it)** -- Italian forms (ministries, banks, schools) filled in by hand
  or typed with invented data and scanned at 300 dpi; the 50 validation forms.
  The only public set with Italian handwriting and its true text.

Both are for research and non-commercial use (FUNSD; XFUND is CC BY-NC-SA
4.0): they live in ``data/eval/dataset/``, git-ignored, and are downloaded by
hand -- see ``data/eval/ocr/cases.yaml``.

For each page:

* **ritrovate** -- share of the annotated words found in the transcription
  (counted with repetition, case and punctuation ignored; a word also counts
  when it is there with its letters split, as in boxed fields);
* **inventate** -- share of the transcription's words that appear nowhere on
  the page's annotation. Some are real text the annotators skipped (stamps,
  page numbers, words cut by the annotation), so this is an upper bound;
* **coppie** -- for every question->answer link («Data di nascita» -> «12/07/1996»),
  whether the answer follows the question within ``PAIR_CHARS`` characters.
  A transcription that lists all labels and then all values loses the form's
  meaning even with every word right.

Usage::

    python scripts/eval/check_ocr_datasets.py --dataset xfund-it --limit 10
    python scripts/eval/check_ocr_datasets.py --dataset funsd \\
        --models google/gemini-3.5-flash-lite,google/gemini-3.7-flash
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

DATASETS = REPO_ROOT / "data" / "eval" / "dataset"
REPORTS = REPO_ROOT / "data" / "eval" / "reports" / "ocr_datasets"

#: Characters (spaces removed) between a question's end and its answer's start
#: still counted as "next to it". Wide enough for a table read as a table --
#: the header row, then the row of values -- not for all labels, then all values.
PAIR_CHARS = 400


def tokens(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.lower())


def load(dataset: str) -> list[dict]:
    """Pages as {name, image, words, pairs}."""
    pages = []
    if dataset == "funsd":
        root = DATASETS / "dataset" / "testing_data"
        for ann in sorted((root / "annotations").glob("*.json")):
            form = json.loads(ann.read_text(encoding="utf-8"))["form"]
            pages.append(_page(ann.stem, root / "images" / f"{ann.stem}.png", form))
    elif dataset == "xfund-it":
        root = DATASETS / "xfund_it"
        data = json.loads((root / "it.val.json").read_text(encoding="utf-8"))
        for doc in data["documents"]:
            fname = doc["img"]["fname"]
            pages.append(_page(Path(fname).stem, root / fname, doc["document"]))
    else:
        raise SystemExit(f"dataset sconosciuto: {dataset}")
    missing = [p for p in pages if not p["image"].exists()]
    if not pages or missing:
        raise SystemExit(f"{dataset}: mancano {len(missing) or 'tutte le'} immagini in {DATASETS}")
    return pages


def _page(name: str, image: Path, entities: list[dict]) -> dict:
    by_id = {e["id"]: e for e in entities}
    words = [t for e in entities for w in e.get("words", []) for t in tokens(w.get("text", ""))]
    pairs = set()
    for e in entities:
        for a, b in e.get("linking", []):
            q, ans = by_id.get(a), by_id.get(b)
            if q and ans and q.get("label") == "question" and ans.get("label") == "answer":
                if tokens(q["text"]) and tokens(ans["text"]):
                    pairs.add((a, b))
    return {
        "name": name,
        "image": image,
        "words": words,
        "pairs": [(tokens(by_id[a]["text"]), tokens(by_id[b]["text"])) for a, b in sorted(pairs)],
    }


def score(page: dict, text: str) -> dict:
    out = tokens(text)
    truth = Counter(page["words"])
    # Boxed fields are read digit by digit («M 3 | 4 | 2») and annotated in
    # groups («M34 244»), or the reverse: compare the text with every space and
    # separator removed as well, or right digits count as missing and invented.
    squashed_out, squashed_truth = "".join(out), "".join(page["words"])
    exact = truth & Counter(out)
    left = truth - exact
    found = sum(exact.values()) + sum(
        n for t, n in left.items() if len(t) > 1 and t in squashed_out
    )
    invented = sum(1 for t in out if t not in truth and t not in squashed_truth)
    near = 0
    for question, answer in page["pairs"]:
        q, a = "".join(question), "".join(answer)[:12]
        at = squashed_out.find(q)
        if at >= 0 and squashed_out.find(a, at + len(q), at + len(q) + PAIR_CHARS + len(a)) >= 0:
            near += 1
    return {
        "recall": found / max(1, sum(truth.values())),
        "invented": invented / max(1, len(out)),
        "pairs": near / len(page["pairs"]) if page["pairs"] else None,
    }


def transcribe(engine: str, image_path: Path) -> str:
    from PIL import Image

    from aix.webui.materials.parsing import ocr, vision_ocr

    with Image.open(image_path) as image:
        image.load()
        if engine == "vision":
            return vision_ocr.vision_ocr_image(image, timeout_s=120).text
        return ocr.ocr_image(image, timeout_s=120).text


def run(engine: str, label: str, pages: list[dict], dataset: str, workers: int) -> dict:
    out_dir = REPORTS / dataset / label
    out_dir.mkdir(parents=True, exist_ok=True)

    def one(page):
        started = time.monotonic()
        try:
            text = transcribe(engine, page["image"])
        except Exception as exc:  # noqa: BLE001 — a failed page is a result
            return (
                page["name"],
                None,
                time.monotonic() - started,
                f"{type(exc).__name__}: {str(exc)[:80]}",
            )
        (out_dir / f"{page['name']}.txt").write_text(text, encoding="utf-8")
        return page["name"], score(page, text), time.monotonic() - started, ""

    # Tesseract is CPU-bound and sized for one page at a time; the model is not.
    with ThreadPoolExecutor(max_workers=workers if engine == "vision" else 1) as pool:
        results = list(pool.map(one, pages))

    scored = [(n, s, sec) for n, s, sec, _ in results if s]
    failed = [(n, err) for n, s, _, err in results if not s]
    pairs = [s["pairs"] for _, s, _ in scored if s["pairs"] is not None]
    worst = sorted(scored, key=lambda r: r[1]["recall"])[:3]
    return {
        "label": label,
        "pages": len(pages),
        "failed": failed,
        "recall": statistics.mean(s["recall"] for _, s, _ in scored) if scored else 0,
        "recall_min": min((s["recall"] for _, s, _ in scored), default=0),
        "invented": statistics.mean(s["invented"] for _, s, _ in scored) if scored else 0,
        "pairs": statistics.mean(pairs) if pairs else None,
        "seconds": statistics.median(sec for _, _, sec in scored) if scored else 0,
        "worst": [(n, s["recall"]) for n, s, _ in worst],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", choices=["funsd", "xfund-it"], required=True)
    parser.add_argument("--engines", default="vision")
    parser.add_argument("--models", default="", help="comma-separated vision models")
    parser.add_argument("--limit", type=int, default=0, help="first N pages only")
    parser.add_argument("--workers", type=int, default=4, help="concurrent calls to the model")
    args = parser.parse_args()

    from aix.webui.materials.parsing.ocr import tesseract_available
    from aix.webui.materials.parsing.vision_ocr import model

    pages = load(args.dataset)
    if args.limit:
        pages = pages[: args.limit]

    runs = []
    for engine in [e for e in args.engines.split(",") if e]:
        if engine == "vision":
            runs += [
                ("vision", m) for m in (args.models.split(",") if args.models else [model()]) if m
            ]
        elif engine == "tesseract" and not tesseract_available():
            print("(tesseract non installato: salto)")
        else:
            runs.append((engine, ""))

    print(
        f"# {args.dataset}: {len(pages)} pagine, {sum(len(p['pairs']) for p in pages)} coppie etichetta→valore\n"
    )
    print(
        "| motore | ritrovate (media) | peggiore | inventate | coppie vicine | s/pagina (mediana) | fallite |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    summaries = []
    for engine, vision_model in runs:
        if vision_model:
            os.environ["AIX_OCR_MODEL"] = vision_model
        label = vision_model.split("/")[-1] if vision_model else engine
        r = run(engine, label, pages, args.dataset, args.workers)
        summaries.append(r)
        pairs = f"{r['pairs'] * 100:.0f}%" if r["pairs"] is not None else "—"
        print(
            f"| {label} | {r['recall'] * 100:.1f}% | {r['recall_min'] * 100:.0f}% | "
            f"{r['invented'] * 100:.1f}% | {pairs} | {r['seconds']:.1f} | {len(r['failed'])} |"
        )
    for r in summaries:
        print(
            f"\n{r['label']}: pagine peggiori "
            + ", ".join(f"{n} ({v * 100:.0f}%)" for n, v in r["worst"])
        )
        for name, err in r["failed"]:
            print(f"  fallita {name}: {err}")
    print(f"\ntrascrizioni in {REPORTS.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
