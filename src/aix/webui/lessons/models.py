"""
SQLAlchemy ``Lesson`` model — one row per submitted lesson form.

Schema rationale (P1):
    id              UUID, PK
    owner_id        UUID, FK → user.id, indexed (lesson library queries by user)
    title           str, optional human-readable title
    domain          str, "neuro" / "udl" / "all" — captured at submit time so the
                    UI can recreate the right agent context later
    educational_profile_json
                    JSON-encoded EducationalProfile (stored as TEXT in SQLite,
                    JSONB in Postgres in CORE 6 — both work via the ``JSON``
                    column type which delegates to the dialect).
    status          str, lifecycle marker: "draft" → "running" → "complete"
                    / "error". POST /run flips the row straight from "draft"
                    to "running" and the SSE pane drives the full pipeline
                    (planner → retriever → writer → critic) live.
    lesson_plan_md  Optional[str], the final Markdown output of the agent on
                    success. Persisted so reloading /webui/lesson/{id} after the
                    stream finished still shows the result without re-running.
    error_message   Optional[str], short human-readable error trail when
                    status == "error" (truncated to ~500 chars). Used to render
                    the "Riprova" callout. Full traceback goes to the logs.
    created_at      tz-aware UTC timestamp
    updated_at      tz-aware UTC timestamp, auto-updated on row mutation

We intentionally store the EducationalProfile as a JSON blob rather than
normalizing each enum into its own column. Two reasons:
    1. The CORE 1 #2.5 schema is still evolving (Angelo branch may add
       fields). Normalizing now would force a migration on every change.
    2. The agent consumes the profile as a dict anyway, so JSON-in / JSON-out
       skips a useless serialization round-trip.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi_users_db_sqlalchemy.generics import GUID
from sqlalchemy import JSON, DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from aix.webui.db import Base


class Lesson(Base):
    """A single lesson submission with its EducationalProfile snapshot."""

    __tablename__ = "lesson"

    # UUID primary key. We reuse FastAPI-Users' ``GUID`` TypeDecorator (the
    # same one ``SQLAlchemyBaseUserTableUUID`` uses for the user.id column)
    # so Python ``UUID`` objects are serialized correctly on every dialect:
    #   - SQLite: stored as 32-char hex string, bound from str(uuid)
    #   - Postgres: stored as native UUID, bound from UUID
    # Using a raw ``PG_UUID(...).with_variant(String, "sqlite")`` would only
    # change the column type — the bind layer would still hand aiosqlite a
    # Python UUID object and fail with "type 'UUID' is not supported".
    id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        primary_key=True,
        default=uuid.uuid4,
    )

    owner_id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        ForeignKey("user.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    title: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    domain: Mapped[str] = mapped_column(String(20), default="neuro", nullable=False)

    # JSON-typed column → TEXT in SQLite, JSONB in Postgres.
    educational_profile_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )

    # Free-text query the teacher typed in the form (P2 phase 2). When the
    # user leaves it empty, the agent service synthesizes one from the
    # profile (subject + topic + grade) — see ``_query_from_lesson`` in
    # ``aix.webui.agent.service``. Stored as ``Text`` so a multi-paragraph
    # request doesn't truncate, and persisted (rather than re-derived from
    # the profile) so we can render it as the user's first chat bubble even
    # after a reload — and preserve the teacher's exact phrasing for #10
    # Conversation Memory when it lands.
    teacher_query: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # P3 — chat attachments (files the teacher dropped on the chat input).
    # Files live on disk under ``data/webui/uploads/{lesson_id}/``; this
    # column is just the manifest (id, filename, mime, size, text_excerpt,
    # stored_name). The Writer prompt receives a joined excerpt as
    # ``teacher_provided_context``; nothing here goes into the KG.
    uploaded_files_json: Mapped[Optional[list[Any]]] = mapped_column(
        JSON, nullable=True
    )

    status: Mapped[str] = mapped_column(String(24), default="draft", nullable=False)

    # Final agent output. ``Text`` (unbounded TEXT in both SQLite and Postgres)
    # because lesson plans can easily exceed VARCHAR(N) limits.
    lesson_plan_md: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Short human-readable error trail for failed runs (~500 chars).
    error_message: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
