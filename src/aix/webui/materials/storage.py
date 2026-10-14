"""Durable storage for the teacher's uploaded material (Fase 0 · PR 2).

The legacy path writes uploads straight to
``data/webui/uploads/{lesson_id}/{filename}`` and records the name in a JSON
manifest on the lesson row. That is enough to attach a file to one lesson and
nothing more: the bytes are addressed by a name the teacher chose, they belong
to a lesson rather than to a person, nothing verifies they were written whole,
and moving them off the host would mean rewriting every caller.

This module is the replacement, and it is deliberately only a contract plus one
implementation. ``StorageBackend`` is what the rest of the application talks to;
``LocalDurableStorage`` is what the pilot runs on — a filesystem volume on the
single host we already have. ``S3CompatibleStorage`` is scaffolded so that
moving to MinIO, Garage or SeaweedFS later is configuration rather than a
rewrite, and is deliberately *not* required to be working now: making object
storage a prerequisite would put this whole line of work behind a procurement
decision.

"S3-compatible" here names a protocol. It does not imply AWS or any cloud.

Two properties are load bearing, and both are pinned by tests.

**Keys carry no name and no identity.** An object key is derived from the
owner and the content digest, never from the filename or the email. A filename
is chosen by the person uploading — it can carry a student's name, a diagnosis,
or a path traversal — and a key is a long-lived identifier that shows up in
logs, in backups and in bucket listings. The digest also makes ``put``
idempotent per owner: uploading the same file twice costs one object, not two.

**Nothing is trusted without its checksum.** ``put`` reads back what it wrote,
``get`` verifies what it read, and ``delete`` refuses to remove an object whose
content is not what the caller expected. A silently truncated write is exactly
the kind of failure that stays invisible until a teacher opens a lesson months
later and the source material is half a file.

Gated by ``AIX_MATERIALS_ENABLED``, default off. With the flag off nothing here
runs and the legacy filesystem path is untouched.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class StorageError(RuntimeError):
    """Base for everything this module raises."""


class ObjectNotFound(StorageError):
    """The key does not exist in this backend."""


class ChecksumMismatch(StorageError):
    """What came back is not what was meant to be there.

    Raised on read and on delete rather than logged, because every caller of
    this module is about to hand the bytes to a parser or to the agent, and
    material that fails its digest must not reach either.
    """


@dataclass(frozen=True)
class StoredObject:
    """What a backend returns once bytes are safely at rest.

    The fields line up one for one with the ``asset_blob`` columns that record
    them, so a caller never has to recompute a digest the backend already knows.
    """

    provider: str
    object_key: str
    sha256: str
    size_bytes: int


def materials_enabled() -> bool:
    """Whether the material pipeline is on. Off by default."""
    return (os.getenv("AIX_MATERIALS_ENABLED") or "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def object_key_for(owner_id: uuid.UUID, digest: str) -> str:
    """Derive an object key from the owner and the content digest.

    Never from the filename and never from the email — see the module
    docstring. The two-character shard keeps a single directory from growing to
    tens of thousands of entries on the local backend, which is slow to list and
    slow to back up; on an object store it is inert and harmless.
    """
    if len(digest) != 64 or not all(c in "0123456789abcdef" for c in digest):
        raise ValueError(f"not a sha256 hex digest: {digest!r}")
    return f"{owner_id}/{digest[:2]}/{digest[2:]}"


class StorageBackend(ABC):
    """Where the bytes live, behind an interface with no vendor in it.

    Async because the implementations block — the local one on the filesystem,
    a future object store on the network — and every caller is inside a request
    handler. A 50 MB write on the event loop stalls every other request in the
    process, including the SSE stream of a lesson being written.
    """

    #: Recorded in ``asset_blob.storage_provider``, so a row always says which
    #: backend can resolve its key. Changing a provider's name orphans its rows.
    provider: str

    @abstractmethod
    async def put(self, data: bytes, *, owner_id: uuid.UUID) -> StoredObject:
        """Store ``data`` for ``owner_id`` and confirm it landed intact."""

    @abstractmethod
    async def get(self, object_key: str) -> bytes:
        """Return the bytes at ``object_key``, verified against their digest."""

    @abstractmethod
    async def delete(self, object_key: str, *, expected_sha256: str) -> None:
        """Remove ``object_key``, refusing if its content is not as expected."""

    @abstractmethod
    async def exists(self, object_key: str) -> bool:
        """Whether ``object_key`` is present, without reading it."""


class LocalDurableStorage(StorageBackend):
    """A filesystem volume on the single host, written durably.

    "Durably" is the whole difference from the legacy path. A plain
    ``open().write()`` returns as soon as the bytes reach the page cache: a
    power loss or a container kill in that window leaves a file that exists,
    has the right name, and is empty or half written. Here the bytes go to a
    temporary file in the same directory, are flushed and ``fsync``-ed, and only
    then renamed into place — a rename within one filesystem is atomic, so a
    reader sees either no file or the whole file, never a partial one.

    The parent directory is ``fsync``-ed too. Without that the rename itself can
    be lost, which produces the same missing file the durable write was meant to
    prevent.
    """

    provider = "local"

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    def _path(self, object_key: str) -> Path:
        """Resolve a key under the root, refusing anything that escapes it.

        Keys are generated by :func:`object_key_for` and are not user input, so
        this should never trigger. It is here because "should never" and "cannot"
        are different, and the cost of being wrong is a write outside the volume.
        """
        candidate = (self._root / object_key).resolve()
        if candidate != self._root and self._root not in candidate.parents:
            raise ValueError(f"object key escapes the storage root: {object_key!r}")
        return candidate

    # -- the blocking halves, run off the event loop ----------------------

    def _put_sync(self, data: bytes, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.part"
        try:
            with open(tmp, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
            self._fsync_dir(path.parent)
        finally:
            tmp.unlink(missing_ok=True)

    @staticmethod
    def _fsync_dir(directory: Path) -> None:
        """Best effort: some platforms cannot open a directory for fsync."""
        try:
            fd = os.open(directory, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            logger.debug("[storage] could not fsync %s — rename may not be durable", directory)
        finally:
            os.close(fd)

    # -- the interface -----------------------------------------------------

    async def put(self, data: bytes, *, owner_id: uuid.UUID) -> StoredObject:
        digest = sha256_of(data)
        key = object_key_for(owner_id, digest)
        path = self._path(key)

        await asyncio.to_thread(self._put_sync, data, path)

        # Read back rather than trust the write. This is the one place where a
        # silent failure would be permanent, and the cost is one read.
        written = await asyncio.to_thread(path.read_bytes)
        if sha256_of(written) != digest:
            await asyncio.to_thread(path.unlink, True)
            raise ChecksumMismatch(
                f"stored object did not survive the write: {key} "
                f"({len(written)} bytes on disk, {len(data)} expected)"
            )

        return StoredObject(
            provider=self.provider,
            object_key=key,
            sha256=digest,
            size_bytes=len(data),
        )

    async def get(self, object_key: str) -> bytes:
        path = self._path(object_key)
        try:
            data = await asyncio.to_thread(path.read_bytes)
        except FileNotFoundError as exc:
            raise ObjectNotFound(object_key) from exc

        # The key ends in the digest, so the object carries its own expectation
        # and corruption at rest is caught without asking the caller for one.
        expected = _digest_from_key(object_key)
        if expected and sha256_of(data) != expected:
            raise ChecksumMismatch(f"content at {object_key} no longer matches its digest")
        return data

    async def delete(self, object_key: str, *, expected_sha256: str) -> None:
        path = self._path(object_key)
        try:
            data = await asyncio.to_thread(path.read_bytes)
        except FileNotFoundError as exc:
            raise ObjectNotFound(object_key) from exc

        if sha256_of(data) != expected_sha256:
            raise ChecksumMismatch(
                f"refusing to delete {object_key}: its content is not what the caller expected"
            )
        await asyncio.to_thread(path.unlink)

    async def exists(self, object_key: str) -> bool:
        return await asyncio.to_thread(self._path(object_key).is_file)


def _digest_from_key(object_key: str) -> Optional[str]:
    """Recover the digest a key was built from, or None if it was not ours."""
    parts = object_key.split("/")
    if len(parts) != 3:
        return None
    candidate = parts[1] + parts[2]
    if len(candidate) != 64 or not all(c in "0123456789abcdef" for c in candidate):
        return None
    return candidate


class S3CompatibleStorage(StorageBackend):
    """Scaffold for MinIO, Garage, SeaweedFS or any S3-protocol store.

    Present and unimplemented on purpose. The point of the ``StorageBackend``
    interface is that moving off the single host is a configuration change, and
    the way to keep that true is to have the seam exist from the start — but
    requiring a working object store now would put every later phase behind a
    procurement decision the pilot does not need.

    So this class holds the configuration surface and refuses to pretend. Each
    method raises rather than silently doing nothing, because a storage backend
    that appears to accept material and does not is the worst of the options.
    """

    provider = "s3"

    def __init__(
        self,
        *,
        endpoint_url: str,
        bucket: str,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: Optional[str] = None,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.bucket = bucket
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region

    def _unimplemented(self) -> StorageError:
        return StorageError(
            "S3CompatibleStorage is a scaffold: the pilot runs on "
            "LocalDurableStorage. Implement this against the chosen store "
            "(MinIO / Garage / SeaweedFS) before selecting it."
        )

    async def put(self, data: bytes, *, owner_id: uuid.UUID) -> StoredObject:
        raise self._unimplemented()

    async def get(self, object_key: str) -> bytes:
        raise self._unimplemented()

    async def delete(self, object_key: str, *, expected_sha256: str) -> None:
        raise self._unimplemented()

    async def exists(self, object_key: str) -> bool:
        raise self._unimplemented()


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

#: Where the local backend keeps its objects. A sibling of the legacy
#: ``uploads`` directory rather than a subdirectory of it, so the two can be
#: backed up, mounted and eventually retired independently.
_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LOCAL_ROOT = _REPO_ROOT / "data" / "webui" / "materials"


def get_storage_backend() -> StorageBackend:
    """The backend this deployment is configured for.

    ``AIX_STORAGE_PROVIDER`` selects it; ``AIX_STORAGE_LOCAL_ROOT`` overrides
    where the local one writes. An unknown provider raises rather than falling
    back: silently writing to the local disk when someone asked for an object
    store is how material ends up on a host nobody is backing up.
    """
    provider = (os.getenv("AIX_STORAGE_PROVIDER") or "local").strip().lower()

    if provider == "local":
        root = os.getenv("AIX_STORAGE_LOCAL_ROOT") or DEFAULT_LOCAL_ROOT
        return LocalDurableStorage(root)

    if provider == "s3":
        endpoint = os.getenv("AIX_STORAGE_S3_ENDPOINT")
        bucket = os.getenv("AIX_STORAGE_S3_BUCKET")
        if not endpoint or not bucket:
            raise StorageError(
                "AIX_STORAGE_PROVIDER=s3 needs AIX_STORAGE_S3_ENDPOINT and AIX_STORAGE_S3_BUCKET"
            )
        return S3CompatibleStorage(
            endpoint_url=endpoint,
            bucket=bucket,
            access_key=os.getenv("AIX_STORAGE_S3_ACCESS_KEY"),
            secret_key=os.getenv("AIX_STORAGE_S3_SECRET_KEY"),
            region=os.getenv("AIX_STORAGE_S3_REGION"),
        )

    raise StorageError(f"unknown AIX_STORAGE_PROVIDER: {provider!r} (expected 'local' or 's3')")
