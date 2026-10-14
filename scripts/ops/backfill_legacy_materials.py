#!/usr/bin/env python3
"""Backfill pre-library lesson attachments into the durable material pipeline.

Dry-run by default. Run inside the app container after the storage audit:

    python scripts/ops/backfill_legacy_materials.py
    python scripts/ops/backfill_legacy_materials.py --apply

Only manifest entries without ``material_id`` are considered. Existing durable
materials and annotations are never overwritten, so the command is resumable.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm.attributes import flag_modified

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


async def backfill(*, apply: bool) -> tuple[int, int]:
    from aix.webui.db import async_session_maker
    from aix.webui.lessons.models import Lesson
    from aix.webui.lessons.uploads import UPLOAD_ROOT
    from aix.webui.materials.handlers import PARSER_PROFILE
    from aix.webui.materials.models import IngestionJob
    from aix.webui.materials.parsing.intake import IntakeRejected, check_upload
    from aix.webui.materials.repository import BindingRepository, MaterialService
    from aix.webui.materials.storage import get_storage_backend

    candidates = 0
    missing = 0
    async with async_session_maker() as session:
        lessons = list((await session.execute(select(Lesson))).scalars())
        service = MaterialService(session, get_storage_backend())

        for lesson in lessons:
            manifest = list(lesson.uploaded_files_json or [])
            changed = False
            for entry in manifest:
                if not isinstance(entry, dict) or entry.get("material_id"):
                    continue
                stored_name = entry.get("stored_name")
                if not isinstance(stored_name, str) or not stored_name:
                    continue
                path = UPLOAD_ROOT / str(lesson.id) / stored_name
                candidates += 1
                if not path.is_file():
                    missing += 1
                    print(f"missing lesson={lesson.id} attachment={entry.get('id')}")
                    continue
                print(f"candidate lesson={lesson.id} attachment={entry.get('id')}")
                if not apply:
                    continue

                content = await asyncio.to_thread(path.read_bytes)
                filename = str(entry.get("filename") or "materiale")[:240]
                try:
                    accepted = check_upload(filename, content)
                except IntakeRejected as exc:
                    missing += 1
                    print(f"rejected lesson={lesson.id} attachment={entry.get('id')} reason={exc}")
                    continue
                material, version = await service.store_new_material(
                    content,
                    owner_id=lesson.owner_id,
                    display_name=filename,
                    media_type=accepted.media_type,
                )
                await BindingRepository(session).bind(
                    lesson_id=lesson.id,
                    material_id=material.id,
                    owner_id=lesson.owner_id,
                    selected_version_id=version.id,
                    created_by=lesson.owner_id,
                )
                entry.update(
                    {
                        "material_id": str(material.id),
                        "material_version_id": str(version.id),
                        "processing_status": "processing",
                    }
                )
                # Keep manifest annotation and durable work atomic. If the
                # process stops after commit, there is still a pending job; a
                # second backfill run cannot silently skip an unqueued version.
                session.add(
                    IngestionJob(
                        owner_id=lesson.owner_id,
                        material_version_id=version.id,
                        parser_profile_version=PARSER_PROFILE,
                    )
                )
                changed = True

            if changed:
                lesson.uploaded_files_json = manifest
                flag_modified(lesson, "uploaded_files_json")

        if apply:
            await session.commit()

    print(
        f"{'backfilled' if apply else 'dry run'}: "
        f"{candidates - missing} available, {missing} missing"
    )
    return candidates, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    _candidates, missing = asyncio.run(backfill(apply=args.apply))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
