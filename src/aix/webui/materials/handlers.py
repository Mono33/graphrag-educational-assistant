"""What the ingestion worker actually does to a job (Fase 2 · PR 5).

PR 4 ran pypdf and recorded only how well it went, to learn how often teacher
uploads come back empty. That question is answered — scans are real, and
``empty`` was the wrong measure of them — and this handler now runs the parser
that replaced pypdf: ``aix.webui.materials.parsing``, profile ``ir@1``.

It records two things:

* ``document_ir`` — the structured result, blocks with page or slide locators
  and confidences. It is what the private retriever of PR 6 will read;
* ``material_version.parse_quality`` — the summary: pages with a text layer,
  pages read by OCR, pages nobody could read.

``ir@1`` is a ``parser_profile_version``, and that string is half of
``ingestion_job``'s unique key. Versions queued under ``pypdf@1`` keep their
results; a new profile is a second job over the same material, not a migration.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from sqlalchemy import select, update

from aix.webui.materials.jobs import JobDispatcher
from aix.webui.materials.models import DocumentIR, IngestionJob, MaterialVersion
from aix.webui.materials.parsing import PARSER_PROFILE, DocumentIRV1, parse_document
from aix.webui.materials.repository import AssetBlobRepository
from aix.webui.materials.storage import StorageBackend

logger = logging.getLogger(__name__)

__all__ = ["PARSER_PROFILE", "ExtractionHandler", "measure_extraction", "status_for"]


def measure_extraction(
    content: bytes, media_type: str
) -> tuple[Optional[DocumentIRV1], dict[str, Any]]:
    """Parse, and report what came out: ``(ir or None, quality)``.

    A document the parser cannot read is a result, not an exception: the bytes
    will not become readable on the third attempt, and retrying five times
    before failing would spend a worker slot on a conclusion already known.
    """
    try:
        ir = parse_document(content, media_type)
    except Exception:  # noqa: BLE001 — a document that cannot be read is a result
        logger.info("[ingestion] a %s could not be read at all", media_type, exc_info=True)
        return None, {"parser": PARSER_PROFILE, "chars": 0, "empty": True, "unreadable": True}
    return ir, ir.quality()


def status_for(quality: dict[str, Any]) -> str:
    """What the version's status becomes once the parser has spoken.

    A document nobody could open is ``failed``, not ``ready_with_warnings``:
    "ready" says a lesson may be written from it, and there is nothing to write
    from. It stays a terminal state — the bytes will not become readable on a
    retry — but an honest one (Louis, 2026-09-15).
    """
    if quality.get("unreadable"):
        return "failed"
    if quality.get("empty") or quality.get("unread_pages") or quality.get("warnings"):
        return "ready_with_warnings"
    return "ready"


class ExtractionHandler:
    """Reads a job's bytes back out of storage, parses, records, finishes.

    Reads rather than receives: the worker claims a row, and the bytes live
    behind a ``StorageBackend`` that verifies them against their digest on the
    way out. Passing the content along from the upload request instead would
    mean the pipeline never exercises the path it will actually use.
    """

    def __init__(self, dispatcher: JobDispatcher, storage: StorageBackend) -> None:
        self._dispatcher = dispatcher
        self._storage = storage

    async def __call__(self, job: IngestionJob) -> Optional[str]:
        """Process one job. Returns ``None`` — there is only one step here.

        Raising is how a failure is reported: the worker catches it, records a
        failed attempt with a code, and moves to the next job.
        """
        content, media_type = await self._load(job)

        # In a thread: OCR is CPU-bound and seconds long, and the worker shares
        # its event loop with every request the app is serving.
        ir, quality = await asyncio.to_thread(measure_extraction, content, media_type)

        status = status_for(quality)
        await self._record(job, ir=ir, quality=quality, status=status)

        # Fase 2 · PR 6 — the private index, and only THEN the job's completion.
        #
        # The first version completed the job inside ``_record``, before this
        # line. When embedding failed on the first real upload (2026-09-16) the
        # job was already "ready": the worker's failure could not be recorded,
        # nothing was retried, and the material sat unsearchable behind a status
        # that said all was well. Completing last means a raise here leaves the
        # job leased and failing, so the worker retries it with backoff.
        if ir is not None:
            await self._index_privately(job, ir)

        await self._complete(job, status)

        if status != "ready":
            logger.info(
                "[ingestion] job=%s finished with warnings: unread=%s ocr=%s",
                job.id,
                quality.get("unread_pages"),
                quality.get("ocr_pages"),
            )
        return None

    async def _index_privately(self, job: IngestionJob, ir: DocumentIRV1) -> None:
        """Chunk the IR and embed it, so the material can be retrieved (§8.3–8.4).

        Off unless ``AIX_PRIVATE_RAG_ENABLED`` is on: with the flag down nothing
        is chunked, nothing is embedded, and no text leaves the machine.

        Failures are raised, not swallowed. A half-embedded document answers
        half the questions it should while looking perfectly healthy, which is
        the kind of quality problem nobody traces back to an ingestion job. The
        job retries with its own backoff instead.
        """
        from aix.webui.materials.chunking import CHUNKER_VERSION, chunk_document
        from aix.webui.materials.embedding import (
            embed_texts,
            embedding_model,
            private_rag_enabled,
        )
        from aix.webui.materials.repository import ChunkRepository

        if not private_rag_enabled():
            return

        chunks = chunk_document(ir)
        if not chunks:
            logger.info("[private-rag] job=%s produced no chunks", job.id)
            return

        vectors = await embed_texts([chunk.text for chunk in chunks])

        async with self._dispatcher.session_maker() as session:
            written = await ChunkRepository(session).replace_for_version(
                owner_id=job.owner_id,
                material_version_id=job.material_version_id,
                chunker_version=CHUNKER_VERSION,
                embedding_model=embedding_model(),
                chunks=list(zip(chunks, vectors, strict=True)),
            )
            await session.commit()

        logger.info(
            "[private-rag] job=%s indexed %d chunks (%s, %s)",
            job.id,
            written,
            CHUNKER_VERSION,
            embedding_model(),
        )

    async def _load(self, job: IngestionJob) -> tuple[bytes, str]:
        async with self._dispatcher.session_maker() as session:
            version = await session.get(MaterialVersion, job.material_version_id)
            if version is None:
                raise LookupError(f"material_version {job.material_version_id} is gone")

            blob = await AssetBlobRepository(session).get(version.blob_id, owner_id=job.owner_id)
            if blob is None:
                raise LookupError(f"asset_blob {version.blob_id} is gone")

            content = await self._storage.get(blob.object_key)
            return content, version.media_type

    async def _record(
        self,
        job: IngestionJob,
        *,
        ir: Optional[DocumentIRV1],
        quality: dict[str, Any],
        status: str,
    ) -> None:
        """Write the IR and the summary. The job is completed later, by the caller.

        The IR is upserted by hand because a retry may find the row its previous
        attempt already wrote.
        """
        async with self._dispatcher.session_maker() as session:
            if ir is not None:
                existing = (
                    await session.execute(
                        select(DocumentIR).where(
                            DocumentIR.material_version_id == job.material_version_id,
                            DocumentIR.parser_profile_version == PARSER_PROFILE,
                        )
                    )
                ).scalar_one_or_none()
                if existing is None:
                    session.add(
                        DocumentIR(
                            owner_id=job.owner_id,
                            material_version_id=job.material_version_id,
                            parser_profile_version=PARSER_PROFILE,
                            schema_version=ir.schema_version,
                            ir_json=ir.to_json(),
                        )
                    )
                else:
                    existing.schema_version = ir.schema_version
                    existing.ir_json = ir.to_json()

            await session.execute(
                update(MaterialVersion)
                .where(MaterialVersion.id == job.material_version_id)
                .values(parse_quality=quality, status=status)
            )
            await session.commit()

    async def _complete(self, job: IngestionJob, status: str) -> None:
        """Mark the job done — the last thing the handler does, never earlier.

        ``complete`` only writes while the lease is still ours, and says so.
        False means another worker took the job while this one was working —
        worth a line in the log, because it is the visible trace of work done
        twice (Louis, 2026-09-15).
        """
        if not await self._dispatcher.complete(job.id, lease_owner=job.lease_owner, status=status):
            logger.warning("[ingestion] job=%s finished but the lease was no longer ours", job.id)
