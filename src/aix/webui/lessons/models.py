"""
SQLAlchemy ``Lesson`` and ``LessonMessage`` models.

``Lesson`` (P1) — one row per submitted lesson form:
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
                    Post-#10.3 this mirrors the LATEST assistant turn's
                    content_md — older turns live in ``LessonMessage`` rows.
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

``LessonMessage`` (CORE 2 #10.3b) — one row per chat turn:
    id, lesson_id, role ('user'|'assistant'), content_md, turn_index,
    agent_kind (optional 'writer'|'planner'|… for fine-grained traces),
    meta_json (scores / nodes_count / media_counts / approved / revision_count),
    checkpoint_id (optional LangGraph checkpoint reference for time-travel),
    created_at.

Why this CQRS pattern (separate from LangGraph's checkpointer L1 store):
    The checkpointer stores the *agent's* view (msgpack-serialised AgentState
    snapshots — fast for the agent, opaque for everything else). The UI needs
    the *user's* view (timestamped messages with rendered markdown, scores,
    agent badges). Reconstructing the UI view from msgpack on every page load
    is slow, version-fragile, and SQL-unfriendly (no LIKE-search, no analytics
    joins, no PDF export queries). We keep both in sync via the SSE event
    emitter — the same code path that pushes events to the browser also writes
    a ``LessonMessage`` row. Standard CQRS pattern.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi_users_db_sqlalchemy.generics import GUID
from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

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
    uploaded_files_json: Mapped[Optional[list[Any]]] = mapped_column(JSON, nullable=True)

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

    # Reverse relation populated lazily — convenient for the rendering path
    # to do ``lesson.messages`` instead of an explicit query. Ordered by
    # turn_index so the chat pane gets them in the right order without an
    # extra ``order_by`` on every consumer.
    messages: Mapped[list[LessonMessage]] = relationship(
        "LessonMessage",
        back_populates="lesson",
        cascade="all, delete-orphan",
        order_by="LessonMessage.turn_index, LessonMessage.created_at",
    )


class SavedProfile(Base):
    """
    A named EducationalProfile preset the teacher can reuse across lessons.

    Saved from the lesson creation form via POST /webui/profiles (includes the
    current form values).  Loaded back via GET /webui/lesson/new?profile_id={id}
    which pre-fills the form.  No relationship to Lesson — profiles are
    independent library entries owned by the user.
    """

    __tablename__ = "saved_profile"

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

    # Short human-readable name chosen by the teacher (e.g. "Classe 3A Fisica").
    name: Mapped[str] = mapped_column(String(120), nullable=False)

    # Profile fields in the same nested shape as Lesson.educational_profile_json
    # (i.e. an EducationalProfile-compatible dict).
    profile_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class LessonMessage(Base):
    """
    One chat turn for a lesson (CORE 2 #10.3b).

    Persistence model:
        • ``role="user"`` rows are written by the route layer when the
          teacher submits a query (POST /run, mode=new or mode=follow_up).
        • ``role="assistant"`` rows are written by the service layer
          (run_agent_stream / stream_agent_events) after the LangGraph run
          terminates with ``kind="done"`` — content_md mirrors the lesson
          plan, meta_json carries scores / KG counts / approval flag.

    The ordering invariant is ``(turn_index ASC, created_at ASC)``. Each
    turn is exactly one user row + one assistant row (assistant absent
    on errors). turn_index is 1-based and monotonically increasing per
    lesson; the route layer computes it as ``COALESCE(MAX(turn_index), 0) + 1``
    when persisting a new user message.

    The ``checkpoint_id`` column is optional and reserved for time-travel
    branching (regenerate from a past turn) — populated when the service
    snapshots the LangGraph checkpoint id alongside the assistant write.
    Stays NULL for the V1 ``mode=new`` and ``mode=follow_up`` paths; first
    populated when the regenerate branch lands.
    """

    __tablename__ = "lesson_message"

    id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        primary_key=True,
        default=uuid.uuid4,
    )

    lesson_id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        ForeignKey("lesson.id", ondelete="CASCADE"),
        nullable=False,
    )
    lesson: Mapped[Lesson] = relationship("Lesson", back_populates="messages")

    # ``user`` | ``assistant``. We keep ``system`` reserved for future
    # summary-buffer rows (#10.4) but don't emit them yet — V1 keeps the
    # summary in AgentState only.
    role: Mapped[str] = mapped_column(String(16), nullable=False)

    # Free-text payload. ``Text`` (unbounded) — assistant messages are
    # full lesson plans, easily exceeding any VARCHAR ceiling.
    content_md: Mapped[str] = mapped_column(Text, nullable=False)

    # 1-based turn ordinal, shared by the user message and its assistant
    # reply. Indexed jointly with lesson_id for the rendering path's
    # primary query: ``WHERE lesson_id=? ORDER BY turn_index``.
    turn_index: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    # Fine-grained agent identity for assistant messages. None for user
    # messages. V1 always writes ``writer`` (the lesson plan IS the writer
    # output); future per-agent traces (planner explanations, critic
    # rationales) can use ``planner`` / ``critic`` here.
    agent_kind: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    # Run summary attached to the assistant turn. Shape mirrors what
    # ``_extract_meta`` in the service layer returns:
    #   {approved, revision_count, scores, nodes_count, recommendations_count,
    #    media_counts, search_queries_count, duration_seconds}
    # For user messages this is None.
    meta_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # LangGraph checkpoint id snapshotted at this turn (CORE 2 #10.3 future
    # use — regenerate via aupdate_state branches from this checkpoint).
    # Optional because V1 doesn't populate it for the simple replay path;
    # regenerate-from-history will set it when that branch lands.
    checkpoint_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        # Compound index for the chat pane's primary query. Without this,
        # SQLite scans the whole table and re-sorts on every page render
        # of a lesson with 10+ turns. With this, an index seek + ordered
        # walk delivers messages in O(log N + K).
        Index("ix_lesson_message_lesson_turn", "lesson_id", "turn_index"),
    )
