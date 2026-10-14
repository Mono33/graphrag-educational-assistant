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
from aix.webui.materials.vectors import EmbeddingVector

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


class SavedResource(Base):
    """An external media item a teacher explicitly bookmarked.

    This is intentionally separate from ``Material``.  A material is private
    source content with immutable versions and retrieval bindings; a saved
    resource is a pointer to third-party content whose availability and
    provenance can change.
    """

    __tablename__ = "saved_resource"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        ForeignKey("user.id", ondelete="CASCADE"),
        nullable=False,
    )
    kind: Mapped[str] = mapped_column(String(24), nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    stable_key: Mapped[str] = mapped_column(String(320), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("owner_id", "stable_key", name="uq_saved_resource_owner_key"),
        Index("ix_saved_resource_owner_created", "owner_id", "created_at"),
    )


#: Where a job can be, from accepted to finished. ``quarantined`` and
#: ``scanning`` exist even though the pilot does not scan: the 2026-09-08
#: decision was that ClamAV is not required *yet*, and a status set that cannot
#: express scanning would have to be migrated the day it is.
JOB_STATUSES = (
    "pending",
    "quarantined",
    "scanning",
    "parsing",
    "embedding",
    "ready",
    "ready_with_warnings",
    "failed",
    "cancelled",
)

#: Statuses from which nothing more will happen on its own.
JOB_TERMINAL_STATUSES = ("ready", "ready_with_warnings", "failed", "cancelled")


class IngestionJob(Base):
    """One unit of processing for one version of one material.

    A table rather than a queue product. The runs of the agent are already
    coordinated this way — ``agent_run`` does atomic claim, heartbeat and TTL
    against the same database — and that pattern has tests pinning its
    semantics on both SQLite and PostgreSQL. Reusing it means the pilot gains
    durable background work without gaining Redis, Celery, and a second thing
    that can be down at three in the morning.

    Durability is the whole point. A job is a committed row before any worker
    touches it, so a restart mid-parse loses the attempt and not the work: the
    lease expires and the next worker picks the row up where it was.

    The lease is what makes that safe. A worker writes its own id and an expiry
    into the row; another worker may take it only once that expiry has passed.
    Two workers therefore cannot hold the same job, and a worker that dies
    holding one blocks it for at most one lease.
    """

    __tablename__ = "ingestion_job"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)

    owner_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("user.id", ondelete="CASCADE"), nullable=False
    )
    material_version_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("material_version.id", ondelete="CASCADE"), nullable=False
    )

    #: Which parser configuration produced, or is producing, this result. Part
    #: of the unique key so re-parsing the same file under a *new* profile is a
    #: new job rather than a conflict — that is how a parser upgrade re-runs
    #: over the corpus without anyone deleting rows by hand.
    parser_profile_version: Mapped[str] = mapped_column(String(64), nullable=False)

    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending", server_default="pending"
    )
    #: Where inside the pipeline the work got to. Separate from ``status`` so a
    #: retry can resume rather than restart: the status says whether the job is
    #: runnable, the step says what still has to happen.
    current_step: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending", server_default="pending"
    )

    attempt_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    #: Not before this. A failure pushes it into the future by a growing
    #: interval, so a job that fails for a reason that will not fix itself in a
    #: second — a service down, a rate limit — does not spin.
    next_attempt_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    lease_owner: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    lease_expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    #: A code, never a message. What went wrong belongs in the log with the
    #: incident id (see ``aix.webui.errors``); this column exists so the UI can
    #: say something specific without a failure describing the deployment.
    safe_error_code: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "material_version_id",
            "parser_profile_version",
            name="uq_ingestion_version_profile",
        ),
        CheckConstraint("attempt_count >= 0", name="ck_ingestion_job_attempts_non_negative"),
        CheckConstraint(_in("status", JOB_STATUSES), name="ck_ingestion_job_status"),
        # The claim query orders by these three. Unlike the guide's DDL this is
        # not a partial index: PostgreSQL supports ``WHERE status = 'pending'``
        # and SQLite does too, but Alembic's batch mode on SQLite does not
        # render one, and an index that exists on one dialect only is a
        # performance difference nobody would notice until production.
        Index("ix_ingestion_job_claim", "status", "next_attempt_at", "created_at"),
    )


#: The dimension of ``baai/bge-m3``, measured against OpenRouter on 2026-09-16.
#: It is in the DDL, so changing it is a migration plus a re-embedding of every
#: chunk — not a setting. ``AIX_PRIVATE_EMBEDDING_DIMENSION`` exists to *verify*
#: this number at runtime, not to change it behind the schema's back.
EMBEDDING_DIMENSION = 1024

CHUNK_STATUSES = ("active", "deleted")


class MaterialChunk(Base):
    """One retrievable piece of one version of a material (§8.3, A.2).

    The unit the private retriever searches. Each row keeps the text, its
    vector, and the locator that lets a citation open the right page — without
    which a quotation is a claim the teacher cannot check (§8.6).

    ``owner_id`` is here, not reached through a join, because the authorization
    belongs *inside* the retrieval query (§8.5). A missing owner filter must be
    a missing WHERE clause, visible in review, rather than a missing join.

    ``embedding_model`` and ``chunker_version`` are on every row so that two
    different chunkings, or two different embedding models, can never be mixed
    in one search without anyone noticing: they are part of what a row is.
    """

    __tablename__ = "material_chunk"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)

    owner_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("user.id", ondelete="CASCADE"), nullable=False
    )
    material_version_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("material_version.id", ondelete="CASCADE"), nullable=False
    )
    #: §8.3 allows a parent/child hierarchy; the first chunker is flat and
    #: leaves this null rather than inventing a parent that means nothing.
    parent_chunk_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID, ForeignKey("material_chunk.id", ondelete="SET NULL"), nullable=True
    )

    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)

    text: Mapped[str] = mapped_column(Text, nullable=False)
    text_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    #: Page or slide, block ids, heading. What a citation resolves against.
    locator_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    embedding: Mapped[list[float]] = mapped_column(
        EmbeddingVector(EMBEDDING_DIMENSION), nullable=False
    )
    embedding_model: Mapped[str] = mapped_column(String(128), nullable=False)
    chunker_version: Mapped[str] = mapped_column(String(64), nullable=False)

    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="active", server_default="active"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "material_version_id",
            "chunker_version",
            "ordinal",
            name="uq_material_chunk_ordinal",
        ),
        CheckConstraint("ordinal >= 0", name="ck_material_chunk_ordinal_non_negative"),
        CheckConstraint(_in("status", CHUNK_STATUSES), name="ck_material_chunk_status"),
        # The retrieval filter, which runs before the similarity ordering.
        Index("ix_material_chunk_owner_version", "owner_id", "material_version_id"),
    )


class DocumentIR(Base):
    """What a parser made of one version, under one parser profile (§7.7).

    The structured result — blocks with their page or slide, their confidence,
    the pages OCR read and the ones nobody could — as ``DocumentIRV1.to_json``.
    It is what the private retriever of PR 6 will chunk and what a citation will
    point into; ``material_version.parse_quality`` stays the summary.

    Keyed like the job that produced it: one row per version and profile, so a
    new parser profile adds a second reading of the same document rather than
    overwriting the first, and the two can be compared.
    """

    __tablename__ = "document_ir"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)

    owner_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("user.id", ondelete="CASCADE"), nullable=False
    )
    material_version_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("material_version.id", ondelete="CASCADE"), nullable=False
    )

    parser_profile_version: Mapped[str] = mapped_column(String(64), nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)

    #: Blocks and text only. §7.7: no base64 assets in here — assets are blobs.
    ir_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "material_version_id", "parser_profile_version", name="uq_document_ir_version_profile"
        ),
        Index("ix_document_ir_owner", "owner_id"),
    )


class DocumentMap(Base):
    """The overview of one version: its parts, their summaries, its themes.

    Built by ``document_map.build_map`` when the version is indexed. A model's
    summary, not the document's words: the Writer reads it to know what the
    document holds, and cites the passages, never the map. Keyed by map version
    so a new way of mapping adds a row instead of silently replacing the old one.
    """

    __tablename__ = "document_map"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)

    owner_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("user.id", ondelete="CASCADE"), nullable=False
    )
    material_version_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("material_version.id", ondelete="CASCADE"), nullable=False
    )

    map_version: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    map_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("material_version_id", "map_version", name="uq_document_map_version"),
        Index("ix_document_map_owner", "owner_id"),
    )
