"""Durable storage for uploaded material (Fase 0 · PR 2).

The legacy path writes to ``data/webui/uploads/{lesson_id}/{filename}`` with a
plain ``write()``. Three things about that are load bearing here, and each has
its own group below.

**The key must carry no name and no identity.** A filename is chosen by the
person uploading: it can carry a student's name, a diagnosis, or a path that
climbs out of the directory. And a key outlives the upload — it appears in
logs, in backups, in bucket listings.

**Nothing is trusted without its checksum.** A write that returns before the
bytes are on disk produces a file that exists, has the right name, and is half
there. That failure stays invisible until a teacher opens a lesson months later.

**The vendor must not leak into the callers.** ``S3CompatibleStorage`` is a
scaffold on purpose — the seam has to exist from the start, but requiring a
working object store now would put every later phase behind a procurement
decision. What it must never do is quietly accept material and drop it.
"""

from __future__ import annotations

import uuid

import pytest

from aix.webui.materials.storage import (
    ChecksumMismatch,
    LocalDurableStorage,
    ObjectNotFound,
    S3CompatibleStorage,
    StorageError,
    get_storage_backend,
    materials_enabled,
    object_key_for,
    sha256_of,
)

pytestmark = pytest.mark.unit

OWNER = uuid.UUID("11111111-1111-1111-1111-111111111111")
OTHER = uuid.UUID("22222222-2222-2222-2222-222222222222")
CONTENT = b"Programma di classe: si affronta la memoria di lavoro."


@pytest.fixture
def storage(tmp_path):
    return LocalDurableStorage(tmp_path / "materials")


# ---------------------------------------------------------------------------
# Keys: no filename, no identity
# ---------------------------------------------------------------------------


def test_the_key_never_contains_the_filename():
    key = object_key_for(OWNER, sha256_of(CONTENT))

    assert "Programma" not in key
    assert ".pdf" not in key and ".docx" not in key


def test_the_key_never_contains_an_email():
    """The owner appears as an opaque id, which is not a personal identifier."""
    key = object_key_for(OWNER, sha256_of(CONTENT))

    assert "@" not in key
    assert str(OWNER) in key


def test_the_key_is_derived_from_the_content():
    digest = sha256_of(CONTENT)
    key = object_key_for(OWNER, digest)

    assert key == f"{OWNER}/{digest[:2]}/{digest[2:]}"


def test_the_same_file_from_two_teachers_gets_two_keys():
    """Blobs are owner-scoped, so one teacher's key must never resolve another's."""
    digest = sha256_of(CONTENT)

    assert object_key_for(OWNER, digest) != object_key_for(OTHER, digest)


@pytest.mark.parametrize(
    "digest",
    ["", "abc", "z" * 64, "A" * 64, "0" * 63],
    ids=["empty", "too short", "not hex", "uppercase", "off by one"],
)
def test_a_key_cannot_be_built_from_a_non_digest(digest):
    with pytest.raises(ValueError):
        object_key_for(OWNER, digest)


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


async def test_put_then_get_returns_the_same_bytes(storage):
    stored = await storage.put(CONTENT, owner_id=OWNER)

    assert await storage.get(stored.object_key) == CONTENT


async def test_put_reports_what_the_blob_row_has_to_record(storage):
    stored = await storage.put(CONTENT, owner_id=OWNER)

    assert stored.provider == "local"
    assert stored.sha256 == sha256_of(CONTENT)
    assert stored.size_bytes == len(CONTENT)
    assert stored.object_key.startswith(f"{OWNER}/")


async def test_uploading_the_same_file_twice_costs_one_object(storage):
    first = await storage.put(CONTENT, owner_id=OWNER)
    second = await storage.put(CONTENT, owner_id=OWNER)

    assert first.object_key == second.object_key
    assert await storage.get(first.object_key) == CONTENT


async def test_an_empty_file_round_trips(storage):
    """Zero bytes is a legitimate upload and must not read as 'missing'."""
    stored = await storage.put(b"", owner_id=OWNER)

    assert stored.size_bytes == 0
    assert await storage.get(stored.object_key) == b""
    assert await storage.exists(stored.object_key)


async def test_exists_answers_without_reading(storage):
    stored = await storage.put(CONTENT, owner_id=OWNER)

    assert await storage.exists(stored.object_key)
    assert not await storage.exists(object_key_for(OTHER, sha256_of(CONTENT)))


async def test_reading_a_missing_object_is_not_a_crash(storage):
    with pytest.raises(ObjectNotFound):
        await storage.get(object_key_for(OWNER, sha256_of(CONTENT)))


# ---------------------------------------------------------------------------
# Checksums
# ---------------------------------------------------------------------------


async def test_corruption_at_rest_is_caught_on_read(storage):
    """The key ends in the digest, so an object carries its own expectation."""
    stored = await storage.put(CONTENT, owner_id=OWNER)
    (storage.root / stored.object_key).write_bytes(b"altered")

    with pytest.raises(ChecksumMismatch):
        await storage.get(stored.object_key)


async def test_delete_refuses_when_the_content_is_not_what_was_expected(storage):
    stored = await storage.put(CONTENT, owner_id=OWNER)

    with pytest.raises(ChecksumMismatch):
        await storage.delete(stored.object_key, expected_sha256=sha256_of(b"something else"))

    assert await storage.exists(stored.object_key), "the object must survive a refused delete"


async def test_delete_removes_the_object_when_it_matches(storage):
    stored = await storage.put(CONTENT, owner_id=OWNER)

    await storage.delete(stored.object_key, expected_sha256=stored.sha256)

    assert not await storage.exists(stored.object_key)


async def test_deleting_a_missing_object_is_not_a_crash(storage):
    with pytest.raises(ObjectNotFound):
        await storage.delete(
            object_key_for(OWNER, sha256_of(CONTENT)), expected_sha256=sha256_of(CONTENT)
        )


# ---------------------------------------------------------------------------
# Durability and containment
# ---------------------------------------------------------------------------


async def test_no_partial_file_is_left_behind(storage):
    """The bytes land through a temporary file; none of it may survive."""
    await storage.put(CONTENT, owner_id=OWNER)

    leftovers = [p.name for p in storage.root.rglob("*") if p.name.endswith(".part")]
    assert leftovers == []


@pytest.mark.parametrize(
    "escaping",
    ["../outside", f"{OWNER}/../../outside", "/etc/passwd"],
    ids=["relative", "climbing", "absolute"],
)
async def test_a_key_cannot_reach_outside_the_root(storage, escaping):
    """Keys are generated, not user input — but 'should never' is not 'cannot'."""
    with pytest.raises(ValueError):
        await storage.exists(escaping)


# ---------------------------------------------------------------------------
# The vendor seam
# ---------------------------------------------------------------------------


async def test_the_s3_scaffold_refuses_rather_than_pretending():
    """A backend that appears to accept material and does not is the worst case."""
    backend = S3CompatibleStorage(endpoint_url="https://minio.local", bucket="aix")

    for call in (
        backend.put(CONTENT, owner_id=OWNER),
        backend.get("k"),
        backend.delete("k", expected_sha256="0" * 64),
        backend.exists("k"),
    ):
        with pytest.raises(StorageError):
            await call


def test_the_s3_scaffold_holds_its_configuration():
    backend = S3CompatibleStorage(endpoint_url="https://minio.local", bucket="aix", region="fr-par")

    assert backend.provider == "s3"
    assert (backend.endpoint_url, backend.bucket, backend.region) == (
        "https://minio.local",
        "aix",
        "fr-par",
    )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_local_is_the_default(monkeypatch, tmp_path):
    monkeypatch.delenv("AIX_STORAGE_PROVIDER", raising=False)
    monkeypatch.setenv("AIX_STORAGE_LOCAL_ROOT", str(tmp_path))

    backend = get_storage_backend()
    assert isinstance(backend, LocalDurableStorage)
    assert backend.root == tmp_path.resolve()


def test_an_unknown_provider_raises_instead_of_falling_back(monkeypatch):
    """Writing to local disk when someone asked for an object store loses data."""
    monkeypatch.setenv("AIX_STORAGE_PROVIDER", "gcs")

    with pytest.raises(StorageError, match="unknown"):
        get_storage_backend()


def test_s3_without_its_configuration_raises(monkeypatch):
    monkeypatch.setenv("AIX_STORAGE_PROVIDER", "s3")
    monkeypatch.delenv("AIX_STORAGE_S3_ENDPOINT", raising=False)
    monkeypatch.delenv("AIX_STORAGE_S3_BUCKET", raising=False)

    with pytest.raises(StorageError, match="ENDPOINT"):
        get_storage_backend()


@pytest.mark.parametrize(
    "value, expected",
    [(None, False), ("false", False), ("", False), ("true", True), ("1", True), ("ON", True)],
)
def test_the_pipeline_is_off_by_default(value, expected, monkeypatch):
    monkeypatch.delenv("AIX_MATERIALS_ENABLED", raising=False)
    if value is not None:
        monkeypatch.setenv("AIX_MATERIALS_ENABLED", value)

    assert materials_enabled() is expected
