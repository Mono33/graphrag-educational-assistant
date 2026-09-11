"""What the ingestion worker actually does to a job (Fase 1 · PR 4).

The real parser is PR 5. This handler runs the extraction that already runs
today — ``aix.webui.lessons.uploads.extract_text``, which is pypdf for a PDF and
a decode for anything else — and records **how well it went** rather than what
it produced.

Not storing the text is the point, not a shortcut. In shadow mode the content
that reaches the Writer is still the legacy manifest excerpt; this pipeline is
being proved, not yet used. The place for structured content is ``document_ir``,
which arrives with the parser that can produce it. Writing the extract into
``parse_quality`` instead would be putting content in a metadata column because
it was the column that happened to be there.

What it does produce is the one measurement nobody has: **how often pypdf comes
back empty on real teacher uploads.** An empty extract from a PDF almost always
means a scanned document, and that number — not an opinion about parsers — is
what should decide whether Docling is worth its weight.

``pypdf@1`` is a ``parser_profile_version``, and that string is half of
``ingestion_job``'s unique key. Adding ``docling@1`` later is therefore a second
job over the same material rather than a migration: the corpus gets re-parsed,
and the two results can be compared instead of one overwriting the other.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from sqlalchemy import update

from aix.webui.materials.jobs import JobDispatcher
from aix.webui.materials.models import IngestionJob, MaterialVersion
from aix.webui.materials.repository import AssetBlobRepository
from aix.webui.materials.storage import StorageBackend

logger = logging.getLogger(__name__)

#: The extractor this handler is. Recorded on every row it touches, so a later
#: profile's results are distinguishable from these rather than mixed in.
PARSER_PROFILE = "pypdf@1"


def measure_extraction(content: bytes, filename: str) -> dict[str, Any]:
    """Extract exactly as the legacy path does, and report what came out.

    The text comes from ``extract_text`` rather than from a second
    implementation, so the measurement describes what the teacher's lesson is
    actually being written from and not something close to it.

    The page counts are a separate pypdf pass, because ``extract_text`` joins
    pages into one string and the seam is gone by the time it returns. They are
    worth the second read: a PDF where pypdf finds text on 3 pages out of 40 is
    a partly scanned document, and that is a different problem from one where
    it finds nothing at all — a distinction a character count hides.
    """
    from aix.webui.lessons.uploads import extract_text

    try:
        text = extract_text(content, filename) or ""
    except Exception:  # noqa: BLE001 — a document that cannot be read is a result
        # pypdf raises on a truncated or malformed file. That is an outcome
        # worth recording once, not a reason to retry the job five times and
        # then fail it: the bytes will not become readable on the third attempt.
        logger.info("[ingestion] %s could not be read at all", filename, exc_info=True)
        return {"extractor": PARSER_PROFILE, "chars": 0, "empty": True, "unreadable": True}

    stripped = text.strip()

    quality: dict[str, Any] = {
        "extractor": PARSER_PROFILE,
        "chars": len(stripped),
        # The measurement this handler exists for. True on a scanned PDF, which
        # pypdf reports as success with nothing in it.
        "empty": not stripped,
    }
    quality.update(_page_counts(content, filename))
    return quality


def _page_counts(content: bytes, filename: str) -> dict[str, Any]:
    """Pages, and pages pypdf got text out of. Empty dict for non-PDFs.

    Best effort by design: this is a diagnostic, and a document whose page
    structure cannot be read is still a document that extracted some text.
    Failing the job over a count nobody is waiting for would be the wrong
    trade.
    """
    if not filename.lower().endswith(".pdf"):
        return {}

    try:
        import io

        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        pages = reader.pages[:50]  # the same ceiling extract_text applies
        return {
            "pages": len(pages),
            "pages_with_text": sum(1 for page in pages if (page.extract_text() or "").strip()),
        }
    except Exception:  # noqa: BLE001 — a diagnostic must not fail the job
        logger.debug("[ingestion] could not count pages for a PDF", exc_info=True)
        return {}


class ExtractionHandler:
    """Reads a job's bytes back out of storage, measures, records, finishes.

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
        failed attempt with a code, and moves to the next job. Nothing in here
        should swallow anything.
        """
        quality = await self._measure(job)

        status = "ready_with_warnings" if quality["empty"] else "ready"
        await self._record(job, quality=quality, status=status)

        if quality["empty"]:
            logger.info(
                "[ingestion] job=%s extracted nothing — likely a scanned document",
                job.id,
            )
        return None

    async def _measure(self, job: IngestionJob) -> dict[str, Any]:
        async with self._dispatcher.session_maker() as session:
            version = await session.get(MaterialVersion, job.material_version_id)
            if version is None:
                raise LookupError(f"material_version {job.material_version_id} is gone")

            blob = await AssetBlobRepository(session).get(version.blob_id, owner_id=job.owner_id)
            if blob is None:
                raise LookupError(f"asset_blob {version.blob_id} is gone")

            content = await self._storage.get(blob.object_key)

        # ``extract_text`` dispatches on the extension, and the object key
        # carries none — it is a digest by design. The recorded media type is
        # what the version knows about itself.
        return measure_extraction(content, _filename_for(version.media_type))

    async def _record(self, job: IngestionJob, *, quality: dict[str, Any], status: str) -> None:
        """Write the measurement onto the version, then finish the job.

        The version is updated before the job is completed so that a crash in
        between leaves a job that will be retried, rather than a job marked
        ready with nothing recorded.
        """
        async with self._dispatcher.session_maker() as session:
            await session.execute(
                update(MaterialVersion)
                .where(MaterialVersion.id == job.material_version_id)
                .values(parse_quality=quality, status=status)
            )
            await session.commit()

        await self._dispatcher.complete(job.id, lease_owner=job.lease_owner, status=status)


def _filename_for(media_type: str) -> str:
    """A name whose only job is to carry an extension into ``extract_text``."""
    return {
        "application/pdf": "material.pdf",
        "text/markdown": "material.md",
    }.get(media_type, "material.txt")
