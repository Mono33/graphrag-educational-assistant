#!/usr/bin/env python3
"""Put already-uploaded documents back through ingestion.

Why this exists
---------------
The private search reads only chunks written by the current ``CHUNKER_VERSION``
and embedding model. Change either -- or the parser, as on 2026-09-25 when
PowerPoint speaker notes started being read -- and every document uploaded
before stays as it was: under a new chunker version it vanishes from the search
entirely, silently; under a new parser it keeps missing what the parser now
reads. Until this script the only procedure was ad-hoc SQL against the
database, which is not a procedure anyone should run on the pilot VM.

What it does
------------
It picks versions that need re-processing and puts their existing ingestion
job back in the queue. The worker does the rest, exactly as for a new upload:
re-reads the bytes from storage, parses, replaces the version's chunks. Nothing
is deleted up front, so a document stays searchable -- on its old chunks --
until its new ones are written.

A job is *reset*, not created: enqueueing is idempotent per version and parser
profile, so a second job for the same version would be refused anyway.

Selection, any of:

* ``stale`` (default) -- chunks from another chunker version or embedding
  model, a version marked ready with no chunks at all, or -- with the document
  map on -- an indexed version without a map of the current version (every
  document uploaded before the maps existed, 2026-09-29);
* ``--media-type`` -- every version of that type, e.g. PowerPoint decks for
  the speaker notes;
* ``--all`` -- everything not deleted.

Dry run unless ``--apply``. While a version is re-processed its status is
"processing": a lesson bound to it waits, as for a fresh upload, so run it
outside school hours.

Usage::

    python scripts/ops/reindex_materials.py                       # what is stale
    python scripts/ops/reindex_materials.py --media-type pptx     # decks, for the notes
    python scripts/ops/reindex_materials.py --all --apply
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

#: Short names for ``--media-type``, so an operator types "pptx", not a MIME type.
MEDIA_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}

#: Versions in these states have nothing to re-process, or must not be touched.
SKIP_STATUSES = ("deleted", "quarantined", "scanning")


async def select_versions(session, *, media_type: str | None, everything: bool) -> list[tuple]:
    """``(version, material_name, reason)`` for every version to re-process."""
    from sqlalchemy import case, func, or_, select

    from aix.webui.materials.chunking import CHUNKER_VERSION
    from aix.webui.materials.document_map import MAP_VERSION, map_enabled
    from aix.webui.materials.embedding import embedding_model
    from aix.webui.materials.models import DocumentMap, Material, MaterialChunk, MaterialVersion

    model = embedding_model()
    outdated = or_(MaterialChunk.chunker_version != CHUNKER_VERSION, MaterialChunk.embedding_model != model)
    stats = (
        select(
            MaterialChunk.material_version_id,
            func.count().label("chunks"),
            func.sum(case((outdated, 1), else_=0)).label("stale"),
        )
        .group_by(MaterialChunk.material_version_id)
        .subquery()
    )
    rows = (
        await session.execute(
            select(MaterialVersion, Material.display_name, stats.c.chunks, stats.c.stale)
            .join(Material, Material.id == MaterialVersion.material_id)
            .outerjoin(stats, stats.c.material_version_id == MaterialVersion.id)
            .where(Material.deleted_at.is_(None))
            .where(MaterialVersion.status.not_in(SKIP_STATUSES))
            .order_by(MaterialVersion.created_at)
        )
    ).all()

    mapped = set()
    if map_enabled():
        mapped = set(
            (
                await session.execute(
                    select(DocumentMap.material_version_id).where(
                        DocumentMap.map_version == MAP_VERSION
                    )
                )
            ).scalars()
        )

    chosen = []
    for version, name, chunks, stale in rows:
        chunks, stale = chunks or 0, stale or 0
        if everything:
            reason = "--all"
        elif media_type and version.media_type == media_type:
            reason = f"tipo {version.media_type.rsplit('.', 1)[-1]}"
        elif media_type:
            continue
        elif stale:
            reason = f"{stale} passaggi di un'altra versione di chunker o embedding"
        elif chunks == 0 and version.status in ("ready", "ready_with_warnings"):
            reason = "pronto ma senza passaggi"
        elif map_enabled() and chunks and version.id not in mapped:
            reason = "indicizzato ma senza mappa del documento"
        else:
            continue
        chosen.append((version, name, reason))
    return chosen


async def requeue(session, version) -> str:
    """Reset this version's ingestion job, or create one if it has none."""
    from sqlalchemy import select

    from aix.webui.materials.models import IngestionJob, Material
    from aix.webui.materials.parsing.parsers import PARSER_PROFILE

    now = datetime.now(timezone.utc)
    job = (
        await session.execute(
            select(IngestionJob)
            .where(IngestionJob.material_version_id == version.id)
            .where(IngestionJob.parser_profile_version == PARSER_PROFILE)
        )
    ).scalar_one_or_none()

    if job is None:
        owner = (
            await session.execute(select(Material.owner_id).where(Material.id == version.material_id))
        ).scalar_one()
        session.add(
            IngestionJob(
                owner_id=owner,
                material_version_id=version.id,
                parser_profile_version=PARSER_PROFILE,
                status="pending",
                current_step="pending",
                next_attempt_at=now,
            )
        )
        return "creato"

    if job.lease_owner and job.lease_expires_at and job.lease_expires_at > now:
        return "in lavorazione adesso: lasciato stare"

    job.status = "pending"
    job.current_step = "pending"
    job.attempt_count = 0
    job.next_attempt_at = now
    job.lease_owner = None
    job.lease_expires_at = None
    job.safe_error_code = None
    job.finished_at = None
    return "rimesso in coda"


async def main_async(args: argparse.Namespace) -> int:
    from aix.webui.db import async_session_maker

    media_type = MEDIA_TYPES.get(args.media_type, args.media_type) if args.media_type else None

    async with async_session_maker() as session:
        chosen = await select_versions(session, media_type=media_type, everything=args.all)
        if not chosen:
            print("Niente da reindicizzare.")
            return 0

        print(f"{'Applico' if args.apply else 'Prova a vuoto'} — {len(chosen)} versione/i:\n")
        for version, name, reason in chosen:
            outcome = await requeue(session, version) if args.apply else "da rimettere in coda"
            print(f"  {name[:48]:<48}  {reason:<52}  {outcome}")

        if args.apply:
            await session.commit()
            print("\nIl worker le riprenderà al prossimo giro; lo stato torna «pronto» a lavoro finito.")
        else:
            print("\nNessuna modifica. Aggiungi --apply per rimetterle in coda.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--media-type", help="pdf, docx, pptx, or a full MIME type")
    parser.add_argument("--all", action="store_true", help="every version not deleted")
    parser.add_argument("--apply", action="store_true", help="actually requeue (default: dry run)")
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
