#!/usr/bin/env python3
"""Read-only audit of lesson attachment manifests and durable material blobs.

Run inside the deployed application container after a migration or redeploy:

    python scripts/ops/audit_material_storage.py

The command prints identifiers, never filenames, object keys, document text or
owner data.  Exit 0 means every referenced local object exists; exit 1 means the
database contains at least one dangling reference.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


@dataclass
class AuditResult:
    lessons: int = 0
    legacy_files_checked: int = 0
    durable_blobs_checked: int = 0
    skipped_provider_blobs: int = 0
    missing_legacy_files: list[str] = field(default_factory=list)
    missing_durable_blobs: list[str] = field(default_factory=list)
    dangling_manifest_materials: list[str] = field(default_factory=list)
    dangling_manifest_versions: list[str] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return not (
            self.missing_legacy_files
            or self.missing_durable_blobs
            or self.dangling_manifest_materials
            or self.dangling_manifest_versions
        )


async def audit(
    session: AsyncSession,
    *,
    upload_root: Path,
    storage,
) -> AuditResult:
    from aix.webui.lessons.models import Lesson
    from aix.webui.materials.models import AssetBlob, Material, MaterialVersion

    result = AuditResult()
    lessons = list((await session.execute(select(Lesson))).scalars())
    result.lessons = len(lessons)

    known_materials = {
        str(item)
        for item in (
            await session.execute(select(Material.id).where(Material.deleted_at.is_(None)))
        ).scalars()
    }
    known_versions = {
        str(version_id): str(material_id)
        for version_id, material_id in (
            await session.execute(
                select(MaterialVersion.id, MaterialVersion.material_id).where(
                    MaterialVersion.status != "deleted"
                )
            )
        )
    }
    for lesson in lessons:
        for entry in lesson.uploaded_files_json or []:
            if not isinstance(entry, dict):
                continue
            stored_name = entry.get("stored_name")
            if isinstance(stored_name, str) and stored_name:
                result.legacy_files_checked += 1
                if not (upload_root / str(lesson.id) / stored_name).is_file():
                    result.missing_legacy_files.append(f"{lesson.id}:{entry.get('id')}")

            material_id = entry.get("material_id")
            if material_id and str(material_id) not in known_materials:
                result.dangling_manifest_materials.append(f"{lesson.id}:{entry.get('id')}")
            version_id = entry.get("material_version_id")
            if version_id and (
                str(version_id) not in known_versions
                or (material_id and known_versions[str(version_id)] != str(material_id))
            ):
                result.dangling_manifest_versions.append(f"{lesson.id}:{entry.get('id')}")

    blobs = list(
        (await session.execute(select(AssetBlob).where(AssetBlob.deleted_at.is_(None)))).scalars()
    )
    for blob in blobs:
        if blob.storage_provider != storage.provider:
            result.skipped_provider_blobs += 1
            continue
        result.durable_blobs_checked += 1
        if not await storage.exists(blob.object_key):
            result.missing_durable_blobs.append(str(blob.id))

    return result


async def _run(*, as_json: bool) -> int:
    from aix.webui.db import async_session_maker
    from aix.webui.lessons.uploads import UPLOAD_ROOT
    from aix.webui.materials.storage import get_storage_backend

    async with async_session_maker() as session:
        result = await audit(
            session,
            upload_root=UPLOAD_ROOT,
            storage=get_storage_backend(),
        )

    payload = {"healthy": result.healthy, **asdict(result)}
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"healthy: {result.healthy}")
        print(f"lessons: {result.lessons}")
        print(f"legacy files checked: {result.legacy_files_checked}")
        print(f"durable blobs checked: {result.durable_blobs_checked}")
        print(f"provider-mismatched blobs skipped: {result.skipped_provider_blobs}")
        for key in (
            "missing_legacy_files",
            "missing_durable_blobs",
            "dangling_manifest_materials",
            "dangling_manifest_versions",
        ):
            values = getattr(result, key)
            print(f"{key.replace('_', ' ')}: {len(values)}")
            for value in values:
                print(f"  - {value}")
    return 0 if result.healthy else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args()
    return asyncio.run(_run(as_json=args.json))


if __name__ == "__main__":
    raise SystemExit(main())
