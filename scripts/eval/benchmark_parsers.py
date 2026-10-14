#!/usr/bin/env python3
"""Measure how well the parsers do on a folder of real documents.

Put the documents in a directory, run this, get a table. No WebUI, no database,
no upload — the same checks and the same parsing the upload and the ingestion
worker run (``aix.webui.materials.parsing``), over files on disk.

The first version counted *empty* extracts. The corpus showed why that is the
wrong measure: a 355-page scan with text on its cover is not empty and is almost
entirely unread. The table therefore counts pages — with a text layer, read by
OCR, not read at all — and the OCR confidence.

A file the intake refuses is listed with the refusal code rather than measured:
that is what the teacher would see on upload.

OCR needs the Tesseract binary. Without it, scanned pages show up as unread and
the row carries ``ocr_unavailable`` — run inside the Linux image for real numbers.

Usage::

    python scripts/eval/benchmark_parsers.py path/to/corpus
    python scripts/eval/benchmark_parsers.py path/to/corpus --out report.md
    python scripts/eval/benchmark_parsers.py path/to/corpus --format csv --out report.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Row:
    file: str
    extension: str
    bytes: int
    rejected: Optional[str] = None
    chars: Optional[int] = None
    pages: Optional[int] = None
    pages_with_text_layer: Optional[int] = None
    ocr_pages: Optional[int] = None
    unread_pages: Optional[int] = None
    ocr_confidence: Optional[float] = None
    unreadable: Optional[bool] = None
    warnings: str = ""
    seconds: Optional[float] = None


def benchmark(folder: Path) -> list[Row]:
    """Check and parse every file under ``folder``, recursively, in a stable order."""
    from aix.webui.materials.handlers import measure_extraction
    from aix.webui.materials.parsing import IntakeRejected, check_upload

    rows: list[Row] = []
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        content = path.read_bytes()
        row = Row(
            file=str(path.relative_to(folder)),
            extension=path.suffix.lower() or "(nessuna)",
            bytes=len(content),
        )
        rows.append(row)

        try:
            accepted = check_upload(path.name, content)
        except IntakeRejected as exc:
            row.rejected = exc.code
            continue

        started = time.perf_counter()
        _ir, quality = measure_extraction(content, accepted.media_type)
        row.seconds = round(time.perf_counter() - started, 3)
        row.chars = quality.get("chars")
        row.pages = quality.get("pages")
        row.pages_with_text_layer = quality.get("pages_with_text_layer")
        row.ocr_pages = quality.get("ocr_pages")
        row.unread_pages = quality.get("unread_pages")
        row.ocr_confidence = quality.get("ocr_mean_confidence")
        row.unreadable = bool(quality.get("unreadable"))
        row.warnings = " ".join(quality.get("warnings") or [])
    return rows


def summarize(rows: list[Row]) -> dict[str, dict[str, int]]:
    """Per extension: files, refused, and where the pages went."""
    summary: dict[str, dict[str, int]] = defaultdict(
        lambda: {"files": 0, "rejected": 0, "pages": 0, "text": 0, "ocr": 0, "unread": 0}
    )
    for row in rows:
        bucket = summary[row.extension]
        bucket["files"] += 1
        if row.rejected:
            bucket["rejected"] += 1
            continue
        bucket["pages"] += row.pages or 0
        bucket["text"] += row.pages_with_text_layer or 0
        bucket["ocr"] += row.ocr_pages or 0
        bucket["unread"] += row.unread_pages or 0
    return dict(summary)


def _cell(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "sì" if value else "no"
    return str(value)


def render_markdown(rows: list[Row]) -> str:
    lines = [
        "# Benchmark parsing — profilo ir@1",
        "",
        "| File | Tipo | Byte | Caratteri | Pagine | Text layer | OCR | Non lette "
        "| Conf. OCR | Avvisi | Secondi |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for r in rows:
        if r.rejected:
            lines.append(
                f"| {r.file} | {r.extension} | {r.bytes} | *rifiutato: {r.rejected}* "
                "| | | | | | | |"
            )
            continue
        lines.append(
            f"| {r.file} | {r.extension} | {r.bytes} | {_cell(r.chars)} | {_cell(r.pages)} "
            f"| {_cell(r.pages_with_text_layer)} | {_cell(r.ocr_pages)} "
            f"| {_cell(r.unread_pages)} | {_cell(r.ocr_confidence)} | {r.warnings} "
            f"| {_cell(r.seconds)} |"
        )

    lines += [
        "",
        "## Per tipo",
        "",
        "| Tipo | File | Rifiutati | Pagine | Text layer | OCR | Non lette |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for extension, s in sorted(summarize(rows).items()):
        lines.append(
            f"| {extension} | {s['files']} | {s['rejected']} | {s['pages']} "
            f"| {s['text']} | {s['ocr']} | {s['unread']} |"
        )
    return "\n".join(lines) + "\n"


def render_csv(rows: list[Row]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(asdict(rows[0]).keys()) if rows else ["file"])
    writer.writeheader()
    for row in rows:
        writer.writerow(asdict(row))
    return buffer.getvalue()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path, help="cartella con i documenti del corpus")
    parser.add_argument("--format", choices=("md", "csv"), default="md")
    parser.add_argument("--out", type=Path, help="file di output (default: stdout)")
    args = parser.parse_args(argv)

    if not args.folder.is_dir():
        print(f"Non è una cartella: {args.folder}", file=sys.stderr)
        return 2

    rows = benchmark(args.folder)
    if not rows:
        print(f"Nessun file in {args.folder}", file=sys.stderr)
        return 1

    report = render_csv(rows) if args.format == "csv" else render_markdown(rows)
    if args.out:
        args.out.write_text(report, encoding="utf-8")
        print(f"Scritto {args.out} ({len(rows)} file)")
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
