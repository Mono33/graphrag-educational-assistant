#!/usr/bin/env python3
"""Measure how well the extractors do on a folder of real documents.

Put the documents in a directory, run this, get a table. No WebUI, no database,
no upload — the same measurement the ingestion worker records on every upload
(``aix.webui.materials.handlers.measure_extraction``), run over files on disk.

It exists to answer one question with numbers instead of opinions: **how often
does extraction come back empty?** An empty extract from a PDF is almost always a
scanned document, and that share is what decides how much the OCR work in PR 5
is worth, and whether the lighter parsers are enough on the pilot VM.

Formats the current profile cannot read are listed as *not supported* rather than
measured. Decoding a DOCX as text produces a character count that looks like
success and means nothing; reporting it would make the table lie about exactly
the formats PR 5 is meant to add.

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

#: What ``pypdf@1`` actually reads. Everything else is reported, not measured.
SUPPORTED = frozenset({".pdf", ".txt", ".md", ".markdown"})


@dataclass
class Row:
    file: str
    extension: str
    bytes: int
    supported: bool
    chars: Optional[int] = None
    pages: Optional[int] = None
    pages_with_text: Optional[int] = None
    empty: Optional[bool] = None
    unreadable: Optional[bool] = None
    seconds: Optional[float] = None


def benchmark(folder: Path) -> list[Row]:
    """Measure every file under ``folder``, recursively, in a stable order."""
    from aix.webui.materials.handlers import measure_extraction

    rows: list[Row] = []
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        extension = path.suffix.lower()
        content = path.read_bytes()
        row = Row(
            file=str(path.relative_to(folder)),
            extension=extension or "(nessuna)",
            bytes=len(content),
            supported=extension in SUPPORTED,
        )
        if row.supported:
            started = time.perf_counter()
            quality = measure_extraction(content, path.name)
            row.seconds = round(time.perf_counter() - started, 3)
            row.chars = quality.get("chars")
            row.pages = quality.get("pages")
            row.pages_with_text = quality.get("pages_with_text")
            row.empty = quality.get("empty")
            row.unreadable = bool(quality.get("unreadable"))
        rows.append(row)
    return rows


def summarize(rows: list[Row]) -> dict[str, dict[str, int]]:
    """Per extension: how many files, how many measured, how many empty."""
    summary: dict[str, dict[str, int]] = defaultdict(
        lambda: {"files": 0, "measured": 0, "empty": 0, "unreadable": 0}
    )
    for row in rows:
        bucket = summary[row.extension]
        bucket["files"] += 1
        if row.supported:
            bucket["measured"] += 1
            bucket["empty"] += int(bool(row.empty))
            bucket["unreadable"] += int(bool(row.unreadable))
    return dict(summary)


def _cell(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "sì" if value else "no"
    return str(value)


def render_markdown(rows: list[Row]) -> str:
    lines = [
        "# Benchmark estrazione — profilo pypdf@1",
        "",
        "| File | Tipo | Byte | Caratteri | Pagine | Con testo | Vuoto | Illeggibile | Secondi |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for r in rows:
        if not r.supported:
            lines.append(f"| {r.file} | {r.extension} | {r.bytes} | *non supportato* | | | | | |")
            continue
        lines.append(
            f"| {r.file} | {r.extension} | {r.bytes} | {_cell(r.chars)} | {_cell(r.pages)} "
            f"| {_cell(r.pages_with_text)} | {_cell(r.empty)} | {_cell(r.unreadable)} "
            f"| {_cell(r.seconds)} |"
        )

    lines += [
        "",
        "## Per tipo",
        "",
        "| Tipo | File | Misurati | Vuoti | % vuoti | Illeggibili |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for extension, s in sorted(summarize(rows).items()):
        share = f"{100 * s['empty'] / s['measured']:.0f}%" if s["measured"] else "—"
        lines.append(
            f"| {extension} | {s['files']} | {s['measured']} | {s['empty']} "
            f"| {share} | {s['unreadable']} |"
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
