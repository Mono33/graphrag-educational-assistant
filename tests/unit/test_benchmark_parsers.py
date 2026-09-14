"""The corpus benchmark (B7).

It runs the worker's own measurement over a folder, so these tests check the two
things it adds on top: that formats the profile cannot read are reported as
unsupported instead of producing a meaningless character count, and that the
per-type summary counts what it claims to.
"""

from __future__ import annotations

import io

import pytest
from scripts.eval.benchmark_parsers import benchmark, main, render_markdown, summarize

pytestmark = pytest.mark.unit


def _blank_pdf() -> bytes:
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "programma.txt").write_text("Programma: memoria di lavoro.", encoding="utf-8")
    (tmp_path / "scansioni").mkdir()
    (tmp_path / "scansioni" / "scheda.pdf").write_bytes(_blank_pdf())
    (tmp_path / "verifica.docx").write_bytes(b"PK\x03\x04 non misurabile da pypdf")
    return tmp_path


def test_every_file_is_listed_including_nested_ones(corpus):
    files = {row.file.replace("\\", "/") for row in benchmark(corpus)}
    assert files == {"programma.txt", "scansioni/scheda.pdf", "verifica.docx"}


def test_a_readable_text_is_measured(corpus):
    row = next(r for r in benchmark(corpus) if r.file == "programma.txt")

    assert row.supported is True
    assert row.empty is False
    assert row.chars == len("Programma: memoria di lavoro.")


def test_a_scanned_pdf_is_counted_as_empty(corpus):
    row = next(r for r in benchmark(corpus) if r.extension == ".pdf")

    assert row.empty is True
    assert row.pages == 1
    assert row.pages_with_text == 0


def test_an_unsupported_format_is_reported_not_measured(corpus):
    """A DOCX decoded as text would look like success and mean nothing."""
    row = next(r for r in benchmark(corpus) if r.extension == ".docx")

    assert row.supported is False
    assert row.chars is None
    assert row.empty is None


def test_the_summary_counts_empties_per_type(corpus):
    summary = summarize(benchmark(corpus))

    assert summary[".pdf"] == {"files": 1, "measured": 1, "empty": 1, "unreadable": 0}
    assert summary[".docx"]["measured"] == 0


def test_the_report_marks_unsupported_rows(corpus):
    report = render_markdown(benchmark(corpus))

    assert "*non supportato*" in report
    assert "| .pdf | 1 | 1 | 1 | 100% | 0 |" in report


def test_the_cli_writes_a_report(corpus, tmp_path_factory):
    out = tmp_path_factory.mktemp("out") / "report.md"

    assert main([str(corpus), "--out", str(out)]) == 0
    assert "Benchmark estrazione" in out.read_text(encoding="utf-8")


def test_the_cli_refuses_a_path_that_is_not_a_folder(tmp_path):
    assert main([str(tmp_path / "manca")]) == 2
