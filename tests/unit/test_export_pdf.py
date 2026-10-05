"""The PDF export, and the route that serves it (Fase 3 · PR 7).

The PDF is checked by reading it back with pypdf — its text, its links, its
bookmarks — rather than by comparing bytes or pixels. A byte comparison breaks on
every ReportLab upgrade; what a teacher needs is that the words are there, the
links work and the disclosure is on the page.
"""

from __future__ import annotations

import io
import uuid

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from aix.export import adapt_lesson_markdown, build_export_ir
from aix.export.pdf import SUBSTITUTES, render_pdf

pytestmark = pytest.mark.unit

LESSON = """<!-- ai-generated: true, system: agentic-graphrag, trace_id: abc -->

# Memoria di lavoro — perché dimentichiamo

**Materia:** Scienze | **Durata:** 60 minuti

## Introduzione *(10 minuti)*

Città, perché, €5, «virgolette», trattino — lungo, e una freccia → con ≤ e ≥.
Un'emoji 🧠 che il font non ha.

## Attività Principali

### Fase 1 — I Do *(25 minuti)*

- Spiegazione con **schede cartacee**
  - una scheda per coppia
- Video: [How We Make Memories](https://youtu.be/bSycdIx-C48)

### Fase 2 — We Do *(25 minuti)*

1. Lavoro in gruppi da 4
2. Restituzione

## Valutazione

| Strumento | Funzione |
| --- | --- |
| Exit ticket | Verifica **immediata** |

*Fonti:* Cowan (2013); Knowledge Graph.
"""


def _pdf(markdown: str = LESSON):
    return render_pdf(build_export_ir(adapt_lesson_markdown(markdown, lesson_id="L1")))


def _text(content: bytes) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(content)).pages)


@pytest.fixture(scope="module")
def result():
    return _pdf()


# ---------------------------------------------------------------------------
# It is a PDF, with the lesson in it
# ---------------------------------------------------------------------------


def test_the_output_is_a_pdf(result):
    assert result.content.startswith(b"%PDF-")
    assert result.page_count >= 1


def test_the_lesson_text_is_in_the_file(result):
    text = _text(result.content)

    assert "Memoria di lavoro" in text
    assert "Introduzione" in text
    assert "schede cartacee" in text
    assert "Exit ticket" in text, "table cells are rendered"
    assert "Cowan (2013)" in text, "the sources footer is part of the document"


def test_italian_and_typographic_characters_survive(result):
    """Helvetica would lose some of these; the bundled Noto Sans does not."""
    text = _text(result.content)

    for fragment in ("Città", "perché", "€5", "«virgolette»", "—"):
        assert fragment in text, fragment


def test_characters_the_font_lacks_are_substituted_not_boxed(result):
    text = _text(result.content)

    assert "->" in text and "<=" in text and ">=" in text
    assert "→" not in text


def test_what_cannot_be_substituted_is_dropped_and_reported(result):
    assert "🧠" in result.dropped_characters
    assert "🧠" not in _text(result.content)
    assert "→" not in result.dropped_characters, "a substituted character was not dropped"
    assert set(SUBSTITUTES).isdisjoint(result.dropped_characters)


# ---------------------------------------------------------------------------
# The disclosure, links and bookmarks
# ---------------------------------------------------------------------------


def test_the_ai_disclosure_is_on_every_page():
    long_lesson = LESSON + "\n".join(f"\nParagrafo di riempimento numero {i}." for i in range(160))
    content = _pdf(long_lesson).content
    pages = PdfReader(io.BytesIO(content)).pages

    assert len(pages) > 1, "the fixture must actually span pages"
    for page in pages:
        assert "Generato dall'IA" in (page.extract_text() or "")


def test_links_are_clickable(result):
    reader = PdfReader(io.BytesIO(result.content))
    uris = [
        annotation.get_object().get("/A", {}).get("/URI")
        for page in reader.pages
        for annotation in (page.get("/Annots") or [])
    ]

    assert "https://youtu.be/bSycdIx-C48" in uris


def test_an_unsafe_link_does_not_become_an_annotation():
    content = _pdf("# T\n\n[clicca](tel:+390000000) e [anche](ftp://x.org/a)\n").content
    reader = PdfReader(io.BytesIO(content))

    assert not any(page.get("/Annots") for page in reader.pages)


def test_headings_become_bookmarks(result):
    outline = PdfReader(io.BytesIO(result.content)).outline

    def _titles(items):
        for item in items:
            if isinstance(item, list):
                yield from _titles(item)
            else:
                yield item.title

    titles = list(_titles(outline))
    assert titles[0].startswith("Memoria di lavoro")
    assert any("Introduzione" in t for t in titles)
    assert any("Fase 1" in t for t in titles)


def test_a_heading_that_skips_levels_does_not_break_the_outline():
    """## straight to #### is legal Markdown; the outline must still build."""
    content = _pdf("# T\n\n## Sezione\n\n#### Sotto-sotto\n\nTesto.\n").content

    assert content.startswith(b"%PDF-")


def test_the_metadata_identifies_the_revision(result):
    info = PdfReader(io.BytesIO(result.content)).metadata

    assert info.title.startswith("Memoria di lavoro")
    assert "Generato dall'IA" in info.author
    assert "content=" in info.subject


# ---------------------------------------------------------------------------
# Content that must not break the renderer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hostile",
    [
        "Testo con <b>tag</b> e <font name='x'>font</font> e & commerciale",
        "Uno <script>alert(1)</script> nel testo",
        "Minore < maggiore > e &amp; già scappato",
    ],
    ids=["reportlab tags", "script", "xml entities"],
)
def test_markup_in_the_lesson_is_text_not_instructions(hostile):
    """ReportLab paragraphs parse XML-like tags; lesson text must never reach that parser raw."""
    content = _pdf(f"# T\n\n{hostile}\n").content

    assert content.startswith(b"%PDF-")
    assert "tag" in _text(content) or "alert" in _text(content) or "Minore" in _text(content)


def test_a_long_table_splits_across_pages():
    rows = "\n".join(f"| Riga {i} | Descrizione della riga {i} |" for i in range(120))
    content = _pdf(f"# T\n\n| A | B |\n| --- | --- |\n{rows}\n").content

    assert len(PdfReader(io.BytesIO(content)).pages) > 1
    assert "Riga 119" in _text(content)


def test_an_empty_lesson_still_renders():
    assert _pdf("").content.startswith(b"%PDF-")


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------

ANNA = uuid.UUID("11111111-1111-1111-1111-111111111111")
BRUNO = uuid.UUID("22222222-2222-2222-2222-222222222222")
LESSON_ID = uuid.UUID("33333333-3333-3333-3333-333333333333")


@pytest.fixture
async def client(tmp_path, monkeypatch):
    monkeypatch.setenv("WEBUI_CSRF_ENABLED", "false")

    from aix.api.main import app
    from aix.webui.auth.dependencies import optional_current_user
    from aix.webui.auth.models import User
    from aix.webui.db import Base, get_async_session
    from aix.webui.lessons.models import Lesson

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{(tmp_path / 'export.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with maker() as session:
        for user_id, email in ((ANNA, "anna@scuola.it"), (BRUNO, "bruno@scuola.it")):
            session.add(
                User(
                    id=user_id,
                    email=email,
                    hashed_password="x",
                    is_active=True,
                    is_superuser=False,
                    is_verified=True,
                )
            )
        session.add(
            Lesson(
                id=LESSON_ID,
                owner_id=ANNA,
                domain="neuro",
                status="complete",
                title="Memoria di lavoro",
                lesson_plan_md=LESSON,
            )
        )
        await session.commit()

    async def _session():
        async with maker() as s:
            yield s

    state = {"who": ANNA}

    def _user():
        return User(
            id=state["who"],
            email="x@scuola.it",
            hashed_password="x",
            is_active=True,
            is_superuser=False,
            is_verified=True,
        )

    app.dependency_overrides[get_async_session] = _session
    app.dependency_overrides[optional_current_user] = _user
    try:
        test_client = TestClient(app, follow_redirects=False)
        test_client.act_as = lambda user_id: state.__setitem__("who", user_id)
        yield test_client
    finally:
        app.dependency_overrides.clear()
        await engine.dispose()


def _export(client, fmt):
    return client.get(f"/webui/lesson/{LESSON_ID}/export?format={fmt}")


def test_the_route_serves_a_pdf_download(client):
    response = _export(client, "pdf")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"].endswith('.pdf"')
    assert response.headers["x-ai-generated"] == "true"
    assert "Memoria di lavoro" in _text(response.content)


def test_the_route_serves_the_outline(client):
    response = _export(client, "outline")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "scaletta" in response.headers["content-disposition"]
    assert " 10′  Introduzione" in response.text


@pytest.mark.parametrize("fmt", ["md", "txt"])
def test_markdown_and_text_are_unchanged(client, fmt):
    """§10.4: the legacy formats must not move."""
    response = _export(client, fmt)

    assert response.status_code == 200
    if fmt == "md":
        assert response.text == LESSON


def test_an_unknown_format_names_the_supported_ones(client):
    response = _export(client, "docx")

    assert response.status_code == 400
    assert "pdf" in response.json()["detail"]


@pytest.mark.parametrize("fmt", ["pdf", "outline"])
def test_another_teacher_cannot_export_the_lesson(client, fmt):
    client.act_as(BRUNO)

    assert _export(client, fmt).status_code == 404
