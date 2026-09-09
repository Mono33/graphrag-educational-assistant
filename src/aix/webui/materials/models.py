"""Metadata for the teacher's material (Fase 0 · PR 2).

Four tables, following Appendix A.2 of the implementation guide. The DDL there
is written for PostgreSQL; here it is expressed in portable SQLAlchemy so the
same schema runs on the SQLite a developer has locally, and the ``vector``
column and its index are absent because they belong to ``material_chunk``,
which the guide places in the private-RAG PR, after ``EMBEDDING_DIMENSION`` is
settled.

Why four tables and not one column on ``lesson``
------------------------------------------------
Today an upload is a filename inside a JSON list on the lesson row. That models
"this lesson has some files" and nothing else. What the teacher actually has is
a *document* — one they may replace with a corrected version, attach to a second
lesson, and expect to keep after the first lesson is deleted.

* ``AssetBlob`` is the bytes: one row per distinct piece of content, addressed
  by digest, pointing at whatever storage backend holds it. It knows nothing
  about lessons.
* ``Material`` is the document as the teacher thinks of it — a name and a
  purpose, owned by a person, surviving every lesson that used it.
* ``MaterialVersion`` is one revision of that document. Replacing a file adds a
  version; it does not overwrite one. A lesson generated last month keeps
  pointing at the version it was actually written from, which is the whole
  reason the export can still be trusted later.
* ``LessonMaterialBinding`` joins a lesson to a *specific version* of a
  material, with the mode it was attached under.

``owner_id`` is on every table, including the two that could reach it through a
join. That is deliberate: it means every query in the repository layer can
filter on the owner directly, and a missing filter is a missing WHERE clause
rather than a missing join — which is far easier to see in review and to pin in
a test.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi_users_db_sqlalchemy.generics import GUID
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    true,
)
from sqlalchemy.orm import Mapped, mapped_column

from aix.webui.db import Base

# ---------------------------------------------------------------------------
# Enumerations, kept as CHECK constraints rather than native enum types
# ---------------------------------------------------------------------------
#
# A native ``ENUM`` on PostgreSQL needs its own migration to add a value, and
# SQLite has none at all, so the same schema would diverge between the two.
# A CHECK over a VARCHAR is portable, migrates with a plain constraint change,
# and reads identically on both.

MALWARE_STATUSES = ("pending", "clean", "infected", "not_scanned", "error")

MATERIAL_PURPOSES = ("reference", "must_follow", "review", "revise", "resource")

VERSION_STATUSES = (
    "quarantined",
    "scanning",
    "processing",
    "ready",
    "ready_with_warnings",
    "failed",
    "deleted",
)


def _in(column: str, values: tuple[str, ...]) -> str:
    joined = ", ".join(f"'{v}'" for v in values)
    return f"{column} IN ({joined})"


class AssetBlob(Base):
    """One distinct piece of content at rest, wherever the backend put it.

    Separate from ``MaterialVersion`` because the same bytes can be referenced
    more than once — the same syllabus uploaded to two materials, or a version
    restored from an earlier one — and because the row that records *where* the
    bytes are has a different lifecycle from the row that records what they
    mean. ``material_version.blob_id`` is ``ON DELETE RESTRICT`` for that
    reason: metadata must never be able to orphan content.
    """

    __tablename__ = "asset_blob"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)

    owner_id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        ForeignKey("user.id", ondelete="CASCADE"),
        nullable=False,
    )

    #: Which backend can resolve ``object_key`` — "local", "s3", …
    storage_provider: Mapped[str] = mapped_column(String(32), nullable=False)

    #: Derived from owner and digest, never from the filename or the email.
    #: See ``storage.object_key_for``.
    object_key: Mapped[str] = mapped_column(Text, nullable=False)

    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)

    #: What the bytes were *sniffed* to be, not what the upload claimed. A
    #: browser-supplied content type is a hint from the client and nothing more.
    verified_media_type: Mapped[str] = mapped_column(String(127), nullable=False)

    encryption_key_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    #: ``not_scanned`` is a legitimate recorded state, not a placeholder: the
    #: 2026-09-08 decision was that ClamAV is not required for the pilot, and
    #: that choice has to be visible on the row rather than assumed.
    malware_status: Mapped[str] = mapped_column(
        String(24), nullable=False, default="pending", server_default="pending"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    #: Soft delete: the row survives so an audit can still explain what a lesson
    #: was written from, after the bytes themselves are gone.
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint("storage_provider", "object_key", name="uq_asset_blob_provider_key"),
        CheckConstraint("size_bytes >= 0", name="ck_asset_blob_size_non_negative"),
        CheckConstraint(_in("malware_status", MALWARE_STATUSES), name="ck_asset_blob_malware"),
        Index("ix_asset_blob_owner_created", "owner_id", "created_at"),
        Index("ix_asset_blob_sha256", "sha256"),
    )


class Material(Base):
    """The document as the teacher thinks of it, across all its versions.

    Owned by a person and not by a lesson. Deleting the lesson a file was first
    uploaded to must not take the file with it — that is the difference between
    an attachment and a library.
    """

    __tablename__ = "material"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)

    owner_id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        ForeignKey("user.id", ondelete="CASCADE"),
        nullable=False,
    )

    display_name: Mapped[str] = mapped_column(String(240), nullable=False)

    #: How the material is meant to be used. It is on the material as a default
    #: and on the binding as the value actually in force, so attaching the same
    #: syllabus as ``reference`` to one lesson and ``must_follow`` to another
    #: does not require two materials.
    purpose: Mapped[str] = mapped_column(
        String(24), nullable=False, default="reference", server_default="reference"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        CheckConstraint(_in("purpose", MATERIAL_PURPOSES), name="ck_material_purpose"),
        Index("ix_material_owner_created", "owner_id", "created_at"),
    )


class MaterialVersion(Base):
    """One revision of a material. Replacing a file adds a row; it overwrites none.

    A lesson written in March keeps pointing at the March version even after
    the teacher uploads a corrected file in May. Without that, an export could
    cite a source that no longer says what the lesson claims it said.
    """

    __tablename__ = "material_version"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)

    material_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("material.id", ondelete="CASCADE"), nullable=False
    )

    version_number: Mapped[int] = mapped_column(Integer, nullable=False)

    #: RESTRICT, not CASCADE: metadata must never be able to orphan content.
    blob_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("asset_blob.id", ondelete="RESTRICT"), nullable=False
    )

    #: The digest as uploaded. Duplicated from the blob on purpose — it is what
    #: lets a later integrity check compare two independent records instead of
    #: reading one value twice.
    source_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    media_type: Mapped[str] = mapped_column(String(127), nullable=False)

    #: Starts at ``quarantined``, not ``ready``. Nothing may reach the agent
    #: before the pipeline has said it is safe to.
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="quarantined", server_default="quarantined"
    )

    #: What the parser could and could not do — pages read, OCR fallbacks,
    #: tables dropped. It is how ``ready_with_warnings`` becomes explainable to
    #: the teacher instead of being a silent quality drop.
    parse_quality: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)

    created_by: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("user.id", ondelete="RESTRICT"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    supersedes_version_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID, ForeignKey("material_version.id", ondelete="SET NULL"), nullable=True
    )

    __table_args__ = (
        UniqueConstraint("material_id", "version_number", name="uq_material_version_number"),
        # Redundant on its own — ``id`` is already unique — but it is the target
        # the binding's composite foreign key needs, and it is what makes the
        # database refuse a binding whose version belongs to another material.
        UniqueConstraint("id", "material_id", name="uq_material_version_id_material"),
        CheckConstraint("version_number > 0", name="ck_material_version_number_positive"),
        CheckConstraint(_in("status", VERSION_STATUSES), name="ck_material_version_status"),
        Index("ix_material_version_material_created", "material_id", "created_at"),
    )


class LessonMaterialBinding(Base):
    """Which version of which material a lesson is using, and how.

    The composite foreign key onto ``(id, material_id)`` is the point of the
    table. A plain reference to ``material_version.id`` would let a lesson bind
    material A while selecting a version of material B — a bug that would look
    fine in every list view and produce a lesson grounded in the wrong document.
    Here the database refuses it.
    """

    __tablename__ = "lesson_material_binding"

    lesson_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("lesson.id", ondelete="CASCADE"), primary_key=True
    )
    material_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("material.id", ondelete="CASCADE"), primary_key=True
    )

    owner_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("user.id", ondelete="CASCADE"), nullable=False
    )

    selected_version_id: Mapped[uuid.UUID] = mapped_column(GUID, nullable=False)

    mode: Mapped[str] = mapped_column(
        String(24), nullable=False, default="reference", server_default="reference"
    )

    #: Detaching without forgetting: the teacher can switch a material off for
    #: one lesson and back on, and the run history still explains what was in
    #: force when each turn was generated.
    enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default=true()
    )

    created_by: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("user.id", ondelete="RESTRICT"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["selected_version_id", "material_id"],
            ["material_version.id", "material_version.material_id"],
            name="fk_binding_selected_material_version",
            ondelete="RESTRICT",
        ),
        CheckConstraint(_in("mode", MATERIAL_PURPOSES), name="ck_binding_mode"),
        Index("ix_binding_owner_lesson_enabled", "owner_id", "lesson_id", "enabled"),
    )
