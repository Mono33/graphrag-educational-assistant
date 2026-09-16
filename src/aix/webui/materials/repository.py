"""Owner-scoped access to material (Fase 0 · PR 2).

Every method here takes ``owner_id`` as a required keyword argument, including
the ones that look up a single row by primary key. That is the whole design of
this module, and it is a deliberate constraint rather than a convention.

The alternative — a plain ``get(material_id)`` with the caller expected to check
ownership afterwards — fails in the one direction that matters. The check lives
somewhere else, it is easy to omit, its absence looks like nothing at all in a
diff, and the resulting bug is not a crash: it is one teacher quietly reading
another's material. Here there is no method to call without saying whose data
you are asking for, so forgetting is not a thing a caller can do.

``owner_id`` is keyword-only for the same reason. ``get(a, b)`` invites a
transposition; ``get(material_id, owner_id=user.id)`` does not.

None of this is wired to a route yet. The shadow adapter on the existing upload
endpoint is PR 4; until then the legacy filesystem path is the only one running.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from aix.webui.materials.models import (
    AssetBlob,
    LessonMaterialBinding,
    Material,
    MaterialChunk,
    MaterialVersion,
)
from aix.webui.materials.storage import StorageBackend, StoredObject

logger = logging.getLogger(__name__)


class AssetBlobRepository:
    """Rows recording where bytes live and what they hash to."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        owner_id: uuid.UUID,
        stored: StoredObject,
        verified_media_type: str,
        malware_status: str = "not_scanned",
    ) -> AssetBlob:
        """Record an object the storage backend has already accepted.

        ``stored`` comes from :meth:`StorageBackend.put`, so the digest and the
        size are the backend's own — this never recomputes them from a second
        copy of the bytes, which would defeat the point of verifying them once.

        ``malware_status`` defaults to ``not_scanned`` rather than ``pending``:
        the 2026-09-08 decision was that ClamAV is not required for the pilot,
        and a row that says ``pending`` forever would describe a scan that is
        never coming.
        """
        blob = AssetBlob(
            owner_id=owner_id,
            storage_provider=stored.provider,
            object_key=stored.object_key,
            sha256=stored.sha256,
            size_bytes=stored.size_bytes,
            verified_media_type=verified_media_type,
            malware_status=malware_status,
        )
        self._session.add(blob)
        await self._session.flush()
        return blob

    async def get(self, blob_id: uuid.UUID, *, owner_id: uuid.UUID) -> Optional[AssetBlob]:
        result = await self._session.execute(
            select(AssetBlob).where(
                AssetBlob.id == blob_id,
                AssetBlob.owner_id == owner_id,
                AssetBlob.deleted_at.is_(None),
            )
        )
        return result.scalar_one_or_none()

    async def find_by_digest(self, sha256: str, *, owner_id: uuid.UUID) -> Optional[AssetBlob]:
        """The owner's existing row for this content, if they already have one.

        Storage keys are content-addressed per owner, so re-uploading the same
        file produces the same object. Reusing the row keeps that true in the
        database instead of leaving two rows pointing at one object.
        """
        result = await self._session.execute(
            select(AssetBlob).where(
                AssetBlob.sha256 == sha256,
                AssetBlob.owner_id == owner_id,
                AssetBlob.deleted_at.is_(None),
            )
        )
        return result.scalars().first()


class MaterialRepository:
    """The documents a teacher has, independent of any lesson."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        owner_id: uuid.UUID,
        display_name: str,
        purpose: str = "reference",
    ) -> Material:
        material = Material(owner_id=owner_id, display_name=display_name, purpose=purpose)
        self._session.add(material)
        await self._session.flush()
        return material

    async def get(self, material_id: uuid.UUID, *, owner_id: uuid.UUID) -> Optional[Material]:
        result = await self._session.execute(
            select(Material).where(
                Material.id == material_id,
                Material.owner_id == owner_id,
                Material.deleted_at.is_(None),
            )
        )
        return result.scalar_one_or_none()

    async def list_for_owner(
        self, owner_id: uuid.UUID, *, limit: int = 100, offset: int = 0
    ) -> list[Material]:
        result = await self._session.execute(
            select(Material)
            .where(Material.owner_id == owner_id, Material.deleted_at.is_(None))
            .order_by(Material.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars())

    async def soft_delete(self, material_id: uuid.UUID, *, owner_id: uuid.UUID) -> bool:
        """Mark a material deleted, keeping the row.

        The row survives so a lesson written from it can still be explained
        afterwards. Returns False when the material is not this owner's, which
        is the same answer as "does not exist" — deliberately, so a caller
        cannot use the response to learn that someone else's id is real.
        """
        material = await self.get(material_id, owner_id=owner_id)
        if material is None:
            return False
        material.deleted_at = func.now()
        await self._session.flush()
        return True


class MaterialVersionRepository:
    """Revisions of a material. Adding one never overwrites another."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        material_id: uuid.UUID,
        owner_id: uuid.UUID,
        blob: AssetBlob,
        media_type: str,
        created_by: uuid.UUID,
        status: str = "quarantined",
        supersedes_version_id: Optional[uuid.UUID] = None,
    ) -> Optional[MaterialVersion]:
        """Append a version, or return None if the material is not this owner's.

        The version number is read and then written, so two concurrent uploads
        to the same material can pick the same number. The unique constraint on
        ``(material_id, version_number)`` is what settles it: the loser gets an
        ``IntegrityError`` and retries. Deciding it in the database rather than
        in a lock here is what keeps this correct across workers.
        """
        material = await MaterialRepository(self._session).get(material_id, owner_id=owner_id)
        if material is None:
            return None

        version = MaterialVersion(
            material_id=material_id,
            version_number=await self._next_version_number(material_id),
            blob_id=blob.id,
            source_sha256=blob.sha256,
            media_type=media_type,
            status=status,
            created_by=created_by,
            supersedes_version_id=supersedes_version_id,
        )
        self._session.add(version)
        await self._session.flush()
        return version

    async def _next_version_number(self, material_id: uuid.UUID) -> int:
        result = await self._session.execute(
            select(func.max(MaterialVersion.version_number)).where(
                MaterialVersion.material_id == material_id
            )
        )
        return (result.scalar() or 0) + 1

    async def get(self, version_id: uuid.UUID, *, owner_id: uuid.UUID) -> Optional[MaterialVersion]:
        """Scoped through the material, which is where ownership lives.

        A version has no ``owner_id`` column of its own — the guide's schema
        puts it on the material — so this joins rather than skipping the check.
        """
        result = await self._session.execute(
            select(MaterialVersion)
            .join(Material, Material.id == MaterialVersion.material_id)
            .where(
                MaterialVersion.id == version_id,
                Material.owner_id == owner_id,
                Material.deleted_at.is_(None),
            )
        )
        return result.scalar_one_or_none()

    async def list_for_material(
        self, material_id: uuid.UUID, *, owner_id: uuid.UUID
    ) -> list[MaterialVersion]:
        result = await self._session.execute(
            select(MaterialVersion)
            .join(Material, Material.id == MaterialVersion.material_id)
            .where(
                MaterialVersion.material_id == material_id,
                Material.owner_id == owner_id,
                Material.deleted_at.is_(None),
            )
            .order_by(MaterialVersion.version_number.desc())
        )
        return list(result.scalars())

    async def latest_for_material(
        self, material_id: uuid.UUID, *, owner_id: uuid.UUID
    ) -> Optional[MaterialVersion]:
        versions = await self.list_for_material(material_id, owner_id=owner_id)
        return versions[0] if versions else None


class BindingRepository:
    """Which version of which material a lesson is actually using."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def bind(
        self,
        *,
        lesson_id: uuid.UUID,
        material_id: uuid.UUID,
        owner_id: uuid.UUID,
        selected_version_id: uuid.UUID,
        created_by: uuid.UUID,
        mode: str = "reference",
    ) -> Optional[LessonMaterialBinding]:
        """Attach a material to a lesson at one specific version.

        Returns None when the version is not a version of that material, or the
        material is not this owner's. The database enforces the first through
        the composite foreign key; checking here as well turns a constraint
        violation into an answer the caller can act on.
        """
        version = await MaterialVersionRepository(self._session).get(
            selected_version_id, owner_id=owner_id
        )
        if version is None or version.material_id != material_id:
            return None

        # And the lesson must be this owner's too. Checking only the material
        # would let a caller attach their own document to *someone else's*
        # lesson: every read afterwards filters by owner and would find nothing
        # wrong, because the binding's own ``owner_id`` says what the caller
        # claimed rather than what the lesson is (Louis, 2026-09-15).
        from aix.webui.lessons.models import Lesson

        lesson_owner = await self._session.scalar(
            select(Lesson.owner_id).where(Lesson.id == lesson_id)
        )
        if lesson_owner is None or lesson_owner != owner_id:
            return None

        binding = LessonMaterialBinding(
            lesson_id=lesson_id,
            material_id=material_id,
            owner_id=owner_id,
            selected_version_id=selected_version_id,
            mode=mode,
            created_by=created_by,
        )
        self._session.add(binding)
        await self._session.flush()
        return binding

    async def list_for_lesson(
        self, lesson_id: uuid.UUID, *, owner_id: uuid.UUID, enabled_only: bool = True
    ) -> list[LessonMaterialBinding]:
        stmt = select(LessonMaterialBinding).where(
            LessonMaterialBinding.lesson_id == lesson_id,
            LessonMaterialBinding.owner_id == owner_id,
        )
        if enabled_only:
            stmt = stmt.where(LessonMaterialBinding.enabled.is_(True))
        result = await self._session.execute(stmt)
        return list(result.scalars())

    async def set_enabled(
        self,
        *,
        lesson_id: uuid.UUID,
        material_id: uuid.UUID,
        owner_id: uuid.UUID,
        enabled: bool,
    ) -> bool:
        """Switch a material off for one lesson without forgetting it was there."""
        result = await self._session.execute(
            select(LessonMaterialBinding).where(
                LessonMaterialBinding.lesson_id == lesson_id,
                LessonMaterialBinding.material_id == material_id,
                LessonMaterialBinding.owner_id == owner_id,
            )
        )
        binding = result.scalar_one_or_none()
        if binding is None:
            return False
        binding.enabled = enabled
        await self._session.flush()
        return True


class ChunkRepository:
    """The private index of one version of one material (§8.3, §8.5).

    Owner-scoped like everything else here, and for the sharper reason: these
    rows are what the retriever searches, so an owner filter missing *here*
    would put one teacher's paragraphs into another teacher's lesson.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def replace_for_version(
        self,
        *,
        owner_id: uuid.UUID,
        material_version_id: uuid.UUID,
        chunker_version: str,
        embedding_model: str,
        chunks: list[tuple[Any, list[float]]],
    ) -> int:
        """Replace this version's chunks for this chunker version.

        Replace, not append: a re-parse of the same version under the same
        chunker must leave the index as if it had run once. Appending would
        duplicate every chunk and double its weight in retrieval — a ranking
        bug that looks like a quality problem, not like a bug.
        """
        await self._session.execute(
            delete(MaterialChunk).where(
                MaterialChunk.material_version_id == material_version_id,
                MaterialChunk.owner_id == owner_id,
                MaterialChunk.chunker_version == chunker_version,
            )
        )

        for chunk, vector in chunks:
            self._session.add(
                MaterialChunk(
                    owner_id=owner_id,
                    material_version_id=material_version_id,
                    ordinal=chunk.ordinal,
                    text=chunk.text,
                    text_hash=chunk.text_hash,
                    locator_json=chunk.locator,
                    embedding=vector,
                    embedding_model=embedding_model,
                    chunker_version=chunk.chunker_version,
                )
            )
        return len(chunks)

    async def delete_for_version(
        self, material_version_id: uuid.UUID, *, owner_id: uuid.UUID
    ) -> int:
        """Remove a version from the index — §12.3, deletion means retrievable no more."""
        result = await self._session.execute(
            delete(MaterialChunk).where(
                MaterialChunk.material_version_id == material_version_id,
                MaterialChunk.owner_id == owner_id,
            )
        )
        return int(result.rowcount or 0)

    async def list_for_versions(
        self,
        version_ids: list[uuid.UUID],
        *,
        owner_id: uuid.UUID,
        chunker_version: Optional[str] = None,
    ) -> list[MaterialChunk]:
        """Active chunks of the given versions. The authorization is the query."""
        if not version_ids:
            return []

        stmt = select(MaterialChunk).where(
            MaterialChunk.owner_id == owner_id,
            MaterialChunk.material_version_id.in_(version_ids),
            MaterialChunk.status == "active",
        )
        if chunker_version:
            stmt = stmt.where(MaterialChunk.chunker_version == chunker_version)
        return list((await self._session.execute(stmt)).scalars())


class MaterialService:
    """Storing an upload touches three tables and one backend; this is that sequence.

    It exists so the order is written once. Bytes go to storage first and the
    rows follow, never the reverse: a row pointing at an object that was never
    written is a broken lesson, while an object with no row is an orphan a
    sweeper can collect. Of the two failure modes, only one is recoverable.
    """

    def __init__(self, session: AsyncSession, storage: StorageBackend) -> None:
        self._session = session
        self._storage = storage
        self.blobs = AssetBlobRepository(session)
        self.materials = MaterialRepository(session)
        self.versions = MaterialVersionRepository(session)
        self.bindings = BindingRepository(session)

    async def store_new_material(
        self,
        data: bytes,
        *,
        owner_id: uuid.UUID,
        display_name: str,
        media_type: str,
        purpose: str = "reference",
    ) -> tuple[Material, MaterialVersion]:
        """Store bytes and create the material with its first version."""
        blob = await self._store_blob(data, owner_id=owner_id, media_type=media_type)
        material = await self.materials.create(
            owner_id=owner_id, display_name=display_name, purpose=purpose
        )
        version = await self.versions.create(
            material_id=material.id,
            owner_id=owner_id,
            blob=blob,
            media_type=media_type,
            created_by=owner_id,
        )
        assert version is not None  # the material was just created by this owner
        return material, version

    async def add_version(
        self,
        data: bytes,
        *,
        material_id: uuid.UUID,
        owner_id: uuid.UUID,
        media_type: str,
    ) -> Optional[MaterialVersion]:
        """Append a corrected file to a material the owner already has.

        The previous version is recorded as superseded rather than removed, so
        a lesson written from it still resolves to what it actually used.
        """
        previous = await self.versions.latest_for_material(material_id, owner_id=owner_id)
        if previous is None and await self.materials.get(material_id, owner_id=owner_id) is None:
            return None

        blob = await self._store_blob(data, owner_id=owner_id, media_type=media_type)
        return await self.versions.create(
            material_id=material_id,
            owner_id=owner_id,
            blob=blob,
            media_type=media_type,
            created_by=owner_id,
            supersedes_version_id=previous.id if previous else None,
        )

    async def read(self, version: MaterialVersion, *, owner_id: uuid.UUID) -> Optional[bytes]:
        """The bytes behind a version, checked against the digest on the way out."""
        blob = await self.blobs.get(version.blob_id, owner_id=owner_id)
        if blob is None:
            return None
        return await self._storage.get(blob.object_key)

    async def _store_blob(self, data: bytes, *, owner_id: uuid.UUID, media_type: str) -> AssetBlob:
        stored = await self._storage.put(data, owner_id=owner_id)

        # Content-addressed keys make ``put`` idempotent, so a re-upload must
        # reuse the row as well as the object — otherwise the unique constraint
        # on (provider, key) would reject the second insert.
        existing = await self.blobs.find_by_digest(stored.sha256, owner_id=owner_id)
        if existing is not None:
            return existing

        return await self.blobs.create(
            owner_id=owner_id, stored=stored, verified_media_type=media_type
        )
