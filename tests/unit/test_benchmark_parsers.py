"""The corpus benchmark (B7).

It runs the upload's own intake and the worker's own parsing over a folder, so
these tests check the two things it adds on top: that a file the intake refuses
is reported with its refusal code instead of being measured, and that the
per-type summary counts pages where they went — text layer, OCR, unread.
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
    (tmp_path / "verifica.docx").write_bytes(b"PK\x03\x04 non un vero documento Word")
    return tmp_path


def test_every_file_is_listed_including_nested_ones(corpus):
    files = {row.file.replace("\\", "/") for row in benchmark(corpus)}
    assert files == {"programma.txt", "scansioni/scheda.pdf", "verifica.docx"}


def test_a_readable_text_is_measured(corpus):
    row = next(r for r in benchmark(corpus) if r.file == "programma.txt")

    assert row.rejected is None
    assert row.chars == len("Programma: memoria di lavoro.")
    assert row.unreadable is False


def test_a_scanned_pdf_counts_its_unread_page(corpus):
    """Not «empty yes/no»: the page exists, has no text layer, and was not read."""
    row = next(r for r in benchmark(corpus) if r.extension == ".pdf")

    assert row.pages == 1
    assert row.pages_with_text_layer == 0
    assert row.unread_pages == 1


def test_a_file_the_intake_refuses_is_reported_not_measured(corpus):
    """What the teacher would be told on upload, not a meaningless character count."""
    row = next(r for r in benchmark(corpus) if r.extension == ".docx")

    assert row.rejected == "not_ooxml"
    assert row.chars is None


def test_the_summary_counts_pages_per_type(corpus):
    summary = summarize(benchmark(corpus))

    assert summary[".pdf"] == {
        "files": 1,
        "rejected": 0,
        "pages": 1,
        "text": 0,
        "ocr": 0,
        "unread": 1,
    }
    assert summary[".docx"]["rejected"] == 1


def test_the_report_marks_rejected_rows(corpus):
    report = render_markdown(benchmark(corpus))

    assert "*rifiutato: not_ooxml*" in report
    assert "| .pdf | 1 | 0 | 1 | 0 | 0 | 1 |" in report


def test_the_cli_writes_a_report(corpus, tmp_path_factory):
    out = tmp_path_factory.mktemp("out") / "report.md"

    assert main([str(corpus), "--out", str(out)]) == 0
    assert "Benchmark parsing — profilo ir@1" in out.read_text(encoding="utf-8")


def test_the_cli_refuses_a_path_that_is_not_a_folder(tmp_path):
    assert main([str(tmp_path / "manca")]) == 2
