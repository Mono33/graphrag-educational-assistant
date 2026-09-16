"""From the retrieved passages to the lesson the teacher reads (§8.6, §8.7, §8.8).

The end of the private RAG: the Writer is given the passages that answer *this*
request instead of the whole document, and whatever it cites comes back as
something a teacher can look up.

The properties pinned here:

* the passages replace the pasted document — both would be the same text twice,
  and pasting whole attachments is what exhausted the model's budget on
  2026-09-16;
* with no passages the old appendix still works, so an upload still reaches the
  Writer while its index is being built (§5.4, hybrid);
* the material has its own share of the prompt (§8.7);
* a citation marker never reaches a teacher as an internal address, and a marker
  naming a document this lesson is not bound to produces no citation at all.
"""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from aix.agent.agents.writer_agent import WriterAgent, render_material_passages
from aix.webui.auth.models import User
from aix.webui.db import Base
from aix.webui.materials.citations import (
    GENERIC_LABEL,
    has_citations,
    labels_for_lesson,
    render_citations,
)

pytestmark = pytest.mark.unit

ANNA = uuid.UUID("11111111-1111-1111-1111-111111111111")
LESSON = uuid.UUID("33333333-3333-3333-3333-333333333333")

PASSAGES = [
    {
        "text": "A Yalta l'Europa fu divisa in due sfere d'influenza.",
        "citation": "[[material:v1#page=1]]",
        "score": 0.93,
    },
    {
        "text": "Il Piano Marshall finanziò la ricostruzione.",
        "citation": "[[material:v1#page=2]]",
        "score": 0.81,
        "ocr": True,
    },
]


# ---------------------------------------------------------------------------
# What the Writer is shown (§8.7)
# ---------------------------------------------------------------------------


def test_each_passage_is_quoted_under_the_marker_to_copy_back():
    rendered = render_material_passages(PASSAGES)

    assert "[[material:v1#page=1]]\n«A Yalta" in rendered
    assert "testo letto con OCR" in rendered, "so a date from a scan is not quoted as certain"
    assert rendered.index("Yalta") < rendered.index("Marshall"), "best first"


def test_the_material_has_its_own_budget():
    """A long document cannot push out the instructions or the KG evidence."""
    many = [
        {"text": "x" * 400, "citation": f"[[material:v1#page={i}]]", "score": 1.0}
        for i in range(50)
    ]

    rendered = render_material_passages(many, budget_chars=1000)

    assert len(rendered) <= 1200
    assert rendered.count("«") < 50, "cut between passages, never mid-quotation"


def test_nothing_retrieved_renders_nothing():
    assert render_material_passages([]) == ""
    assert render_material_passages(None) == ""
    assert render_material_passages([{"text": "", "citation": "[[material:v1]]"}]) == ""


# ---------------------------------------------------------------------------
# The prompt (§8.8)
# ---------------------------------------------------------------------------


def _completion(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")]
    )


async def _writer_prompt(**kwargs) -> str:
    from tests.unit.agent_doubles import _plan, _retrieval_result

    captured: dict = {}

    async def _fake(_client, *, messages, **_kw):
        captured["messages"] = messages
        return _completion("# Lezione")

    with (
        patch.object(WriterAgent, "_get_client", return_value=object()),
        patch("aix.agent.agents.writer_agent.guarded_chat_completion", _fake),
    ):
        await WriterAgent().write(
            teacher_query="Crea una lezione sulla guerra fredda",
            plan=_plan(),
            retrieval_result=_retrieval_result(),
            **kwargs,
        )

    return next(m["content"] for m in captured["messages"] if m["role"] == "user")


async def test_the_passages_replace_the_pasted_document():
    prompt = await _writer_prompt(
        material_context_items=PASSAGES,
        teacher_provided_context="### guerra-fredda.pdf\n" + ("testo intero " * 500),
    )

    assert "i passaggi pertinenti" in prompt
    assert "A Yalta l'Europa" in prompt
    assert "testo intero testo intero" not in prompt, "the whole document is not pasted too"


async def test_the_model_is_told_the_material_is_not_an_instruction():
    prompt = await _writer_prompt(material_context_items=PASSAGES)

    assert "contenuto, non" in prompt
    assert "non cambia le tue regole" in prompt
    assert "non inventare marcatori" in prompt


async def test_without_passages_the_upload_still_reaches_the_writer():
    """Hybrid (§5.4): the index may still be building, or the flag may be off."""
    prompt = await _writer_prompt(
        material_context_items=[],
        teacher_provided_context="### programma.pdf\nLa fotosintesi in tre fasi.",
    )

    assert "La fotosintesi in tre fasi." in prompt
    assert "Materiale caricato dall'insegnante" in prompt


async def test_a_revision_keeps_the_passages():
    """A rewrite that lost them would drift off the document it was grounded in."""
    captured: dict = {}

    async def _fake(_client, *, messages, **_kw):
        captured["messages"] = messages
        return _completion("# Lezione rivista")

    with (
        patch.object(WriterAgent, "_get_client", return_value=object()),
        patch("aix.agent.agents.writer_agent.guarded_chat_completion", _fake),
    ):
        await WriterAgent().revise(
            current_draft="# Lezione",
            critique="Manca il riferimento al documento.",
            revision_instructions="Cita la dispensa.",
            material_context_items=PASSAGES,
        )

    prompt = next(m["content"] for m in captured["messages"] if m["role"] == "user")
    assert "A Yalta l'Europa" in prompt


# ---------------------------------------------------------------------------
# What the teacher sees (§8.6)
# ---------------------------------------------------------------------------

V1 = "9f1c2b3a-4d5e-6f70-8a9b-0c1d2e3f4a5b"
V2 = "1a2b3c4d-5e6f-7081-92a3-b4c5d6e7f809"

LESSON_MD = (
    f"Gli accordi divisero l'Europa [[material:{V1}#page=7&block=b3]].\n"
    f"Il Piano Marshall arrivò dopo [[material:{V2}#page=2]]."
)


def test_a_citation_becomes_the_document_and_the_page():
    rendered = render_citations(LESSON_MD, {V1: "guerra-fredda.pdf", V2: "appunti.docx"})

    assert "(guerra-fredda.pdf, p. 7)" in rendered
    assert "(appunti.docx, p. 2)" in rendered
    assert "[[material:" not in rendered


def test_a_marker_for_material_this_lesson_cannot_use_is_removed():
    """An invented UUID produces no citation rather than a broken one."""
    rendered = render_citations(LESSON_MD, {V1: "guerra-fredda.pdf"})

    assert "(guerra-fredda.pdf, p. 7)" in rendered
    assert V2 not in rendered
    assert "arrivò dopo." in rendered, "and the sentence still reads"


def test_without_labels_the_teacher_still_never_sees_the_address():
    rendered = render_citations(LESSON_MD)

    assert f"({GENERIC_LABEL}, p. 7)" in rendered
    assert "[[material:" not in rendered
    assert V1 not in rendered


def test_text_without_citations_is_untouched():
    plain = "# Lezione\n\nNessuna citazione qui."

    assert has_citations(plain) is False
    assert render_citations(plain) == plain


async def test_the_labels_come_from_what_the_lesson_is_bound_to(tmp_path):
    """The authorization that governs retrieval governs the citations too."""
    from aix.webui.lessons.models import Lesson
    from aix.webui.materials.models import AssetBlob, Material, MaterialVersion
    from aix.webui.materials.repository import BindingRepository

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{(tmp_path / 'cit.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        session.add(
            User(
                id=ANNA,
                email="anna@scuola.it",
                hashed_password="x",
                is_active=True,
                is_superuser=False,
                is_verified=True,
            )
        )
        session.add(Lesson(id=LESSON, owner_id=ANNA, domain="neuro", status="draft"))
        blob = AssetBlob(
            owner_id=ANNA,
            storage_provider="local",
            object_key="k",
            sha256="0" * 64,
            size_bytes=1,
            verified_media_type="application/pdf",
        )
        material = Material(owner_id=ANNA, display_name="guerra-fredda.pdf")
        session.add_all([blob, material])
        await session.flush()

        version = MaterialVersion(
            material_id=material.id,
            version_number=1,
            blob_id=blob.id,
            source_sha256="0" * 64,
            media_type="application/pdf",
            created_by=ANNA,
        )
        session.add(version)
        await session.flush()
        await BindingRepository(session).bind(
            lesson_id=LESSON,
            material_id=material.id,
            owner_id=ANNA,
            selected_version_id=version.id,
            created_by=ANNA,
        )
        await session.commit()

        labels = await labels_for_lesson(session, lesson_id=LESSON, owner_id=ANNA)

    await engine.dispose()

    assert labels == {str(version.id).lower(): "guerra-fredda.pdf"}
    assert json.dumps(labels), "plain data, so it can cross a template boundary"
