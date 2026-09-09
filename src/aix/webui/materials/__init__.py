"""The material pipeline: durable storage, versioned metadata, owner-scoped access.

Fase 0 · PR 2. Nothing here is wired into a route yet — the shadow adapter on
``POST /webui/lesson/{id}/upload`` is PR 4. Until then this package exists, is
tested, and is inert: with ``AIX_MATERIALS_ENABLED`` off the legacy filesystem
path is the only one that runs.
"""

from aix.webui.materials.jobs import JobDispatcher
from aix.webui.materials.storage import (
    ChecksumMismatch,
    LocalDurableStorage,
    ObjectNotFound,
    S3CompatibleStorage,
    StorageBackend,
    StorageError,
    StoredObject,
    get_storage_backend,
    materials_enabled,
    object_key_for,
    sha256_of,
)
from aix.webui.materials.worker import (
    IngestionWorker,
    start_ingestion_worker,
    stop_ingestion_worker,
)

__all__ = [
    "ChecksumMismatch",
    "IngestionWorker",
    "JobDispatcher",
    "LocalDurableStorage",
    "ObjectNotFound",
    "S3CompatibleStorage",
    "StorageBackend",
    "StorageError",
    "StoredObject",
    "get_storage_backend",
    "materials_enabled",
    "object_key_for",
    "sha256_of",
    "start_ingestion_worker",
    "stop_ingestion_worker",
]
