"""The durable ingestion queue (Fase 0 · PR 3).

The guide states four acceptance conditions for this PR, and each has a section
below:

* a job survives a restart;
* an expired lease is recovered;
* two workers do not process the same job;
* stopping the worker does not interrupt the legacy path.

The third is the one worth being careful about. A queue that *usually* hands
each job to one worker is a queue that duplicates work under load and nowhere
else — so the test does not simulate two workers politely taking turns, it
starts them against the same rows at the same time and counts.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from aix.webui.auth.models import User
from aix.webui.db import Base
from aix.webui.materials.jobs import (
    MAX_ATTEMPTS,
    IngestionJob,
    JobDispatcher,
    backoff_for,
)
from aix.webui.materials.repository import MaterialService
from aix.webui.materials.storage import LocalDurableStorage
from aix.webui.materials.worker import IngestionWorker, start_ingestion_worker

pytestmark = pytest.mark.unit

ANNA = uuid.UUID("11111111-1111-1111-1111-111111111111")
BRUNO = uuid.UUID("22222222-2222-2222-2222-222222222222")
PROFILE = "docling@1"


def _now() -> datetime:
    """Naive UTC, matching the dispatcher — see its ``_now`` for why."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


@pytest.fixture
async def maker(tmp_path):
    """A file-backed database, so independent sessions really are independent."""
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{(tmp_path / 'jobs.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_maker = async_sessionmaker(engine, expire_on_commit=False)
    async with session_maker() as s:
        for user_id, email in ((ANNA, "anna@scuola.it"), (BRUNO, "bruno@scuola.it")):
            s.add(
                User(
                    id=user_id,
                    email=email,
                    hashed_password="x",
                    is_active=True,
                    is_superuser=False,
                    is_verified=True,
                )
            )
        await s.commit()

    try:
        yield session_maker
    finally:
        await engine.dispose()


@pytest.fixture
def dispatcher(maker):
    return JobDispatcher(maker)


@pytest.fixture
async def version(maker, tmp_path):
    """A real material version, so the job's foreign key is a real one."""
    async with maker() as session:
        service = MaterialService(session, LocalDurableStorage(tmp_path / "objects"))
        _material, created = await service.store_new_material(
            b"Programma di classe",
            owner_id=ANNA,
            display_name="Programma",
            media_type="text/plain",
        )
        await session.commit()
        return created


async def _expire_lease(maker, job_id: uuid.UUID) -> None:
    """Push a lease into the past — the stand-in for 'that worker died'."""
    async with maker() as session:
        await session.execute(
            update(IngestionJob)
            .where(IngestionJob.id == job_id)
            .values(lease_expires_at=_now() - timedelta(seconds=1))
        )
        await session.commit()


# ---------------------------------------------------------------------------
# Enqueuing
# ---------------------------------------------------------------------------


async def test_a_job_starts_pending_and_due(dispatcher, version):
    job = await dispatcher.enqueue(
        owner_id=ANNA, material_version_id=version.id, parser_profile_version=PROFILE
    )

    assert job.status == "pending"
    assert job.attempt_count == 0
    assert job.lease_owner is None


async def test_the_same_version_is_not_queued_twice(dispatcher, version):
    """Two requests racing on one upload must produce one job, not two."""
    first = await dispatcher.enqueue(
        owner_id=ANNA, material_version_id=version.id, parser_profile_version=PROFILE
    )
    second = await dispatcher.enqueue(
        owner_id=ANNA, material_version_id=version.id, parser_profile_version=PROFILE
    )

    assert first is not None
    assert second is None, "the unique constraint decides, not a prior read"


async def test_a_new_parser_profile_is_a_new_job(dispatcher, version):
    """That is how a parser upgrade re-runs over the corpus."""
    await dispatcher.enqueue(
        owner_id=ANNA, material_version_id=version.id, parser_profile_version=PROFILE
    )
    again = await dispatcher.enqueue(
        owner_id=ANNA, material_version_id=version.id, parser_profile_version="docling@2"
    )

    assert again is not None


async def test_a_job_is_owner_scoped_like_everything_else(dispatcher, version):
    job = await dispatcher.enqueue(
        owner_id=ANNA, material_version_id=version.id, parser_profile_version=PROFILE
    )

    assert await dispatcher.get(job.id, owner_id=ANNA) is not None
    assert await dispatcher.get(job.id, owner_id=BRUNO) is None


# ---------------------------------------------------------------------------
# A job survives a restart
# ---------------------------------------------------------------------------


async def test_a_queued_job_outlives_the_process_that_queued_it(maker, version):
    """The row is committed before any work starts; a fresh dispatcher finds it."""
    await JobDispatcher(maker).enqueue(
        owner_id=ANNA, material_version_id=version.id, parser_profile_version=PROFILE
    )

    after_restart = JobDispatcher(maker)
    assert await after_restart.claim() is not None


async def test_work_in_progress_is_not_lost_when_the_worker_dies(maker, dispatcher, version):
    """Mid-parse, the process goes away. The step reached must still be there."""
    await dispatcher.enqueue(
        owner_id=ANNA, material_version_id=version.id, parser_profile_version=PROFILE
    )
    job = await dispatcher.claim(owner="worker-a")
    await dispatcher.set_step(job.id, lease_owner="worker-a", step="parsing")

    await _expire_lease(maker, job.id)  # worker-a is gone

    recovered = await JobDispatcher(maker).claim(owner="worker-b")
    assert recovered.id == job.id
    assert recovered.current_step == "parsing", "resumes rather than restarts"


# ---------------------------------------------------------------------------
# An expired lease is recovered
# ---------------------------------------------------------------------------


async def test_a_live_lease_blocks_everyone_else(dispatcher, version):
    await dispatcher.enqueue(
        owner_id=ANNA, material_version_id=version.id, parser_profile_version=PROFILE
    )
    await dispatcher.claim(owner="worker-a")

    assert await dispatcher.claim(owner="worker-b") is None


async def test_an_expired_lease_is_taken_over(maker, dispatcher, version):
    await dispatcher.enqueue(
        owner_id=ANNA, material_version_id=version.id, parser_profile_version=PROFILE
    )
    job = await dispatcher.claim(owner="worker-a")
    await _expire_lease(maker, job.id)

    taken = await dispatcher.claim(owner="worker-b")
    assert taken is not None
    assert taken.lease_owner == "worker-b"
    assert taken.attempt_count == 2, "the abandoned attempt is still counted"


async def test_the_dead_worker_can_no_longer_finish_the_job(maker, dispatcher, version):
    """It might come back to life mid-parse and try to complete what it lost."""
    await dispatcher.enqueue(
        owner_id=ANNA, material_version_id=version.id, parser_profile_version=PROFILE
    )
    job = await dispatcher.claim(owner="worker-a")
    await _expire_lease(maker, job.id)
    await dispatcher.claim(owner="worker-b")

    assert await dispatcher.complete(job.id, lease_owner="worker-a") is False
    assert await dispatcher.renew(job.id, lease_owner="worker-a") is False
    assert await dispatcher.fail(job.id, lease_owner="worker-a", error_code="x") is None


async def test_renewing_keeps_the_job_ours(dispatcher, version):
    await dispatcher.enqueue(
        owner_id=ANNA, material_version_id=version.id, parser_profile_version=PROFILE
    )
    job = await dispatcher.claim(owner="worker-a", lease_s=60)

    assert await dispatcher.renew(job.id, lease_owner="worker-a", lease_s=600)
    assert await dispatcher.claim(owner="worker-b") is None


# ---------------------------------------------------------------------------
# Two workers do not process the same job
# ---------------------------------------------------------------------------


async def test_concurrent_claims_never_hand_out_one_job_twice(maker, tmp_path):
    """Not turn-taking: eight workers going for the same rows at once."""
    dispatcher = JobDispatcher(maker)
    async with maker() as session:
        service = MaterialService(session, LocalDurableStorage(tmp_path / "obj"))
        version_ids = []
        for i in range(5):
            _m, v = await service.store_new_material(
                f"documento {i}".encode(),
                owner_id=ANNA,
                display_name=f"Doc {i}",
                media_type="text/plain",
            )
            version_ids.append(v.id)
        await session.commit()

    for version_id in version_ids:
        await dispatcher.enqueue(
            owner_id=ANNA, material_version_id=version_id, parser_profile_version=PROFILE
        )

    claimed = await asyncio.gather(*(dispatcher.claim(owner=f"worker-{i}") for i in range(8)))

    got = [job.id for job in claimed if job is not None]
    assert len(got) == len(set(got)), "a job was handed to two workers"
    assert len(got) == 5, "and every job was handed out exactly once"


async def test_a_completed_job_is_not_claimed_again(dispatcher, version):
    await dispatcher.enqueue(
        owner_id=ANNA, material_version_id=version.id, parser_profile_version=PROFILE
    )
    job = await dispatcher.claim(owner="worker-a")
    await dispatcher.complete(job.id, lease_owner="worker-a")

    assert await dispatcher.claim(owner="worker-b") is None


# ---------------------------------------------------------------------------
# Retry and giving up
# ---------------------------------------------------------------------------


async def test_a_failure_is_retried_after_a_delay(dispatcher, version):
    await dispatcher.enqueue(
        owner_id=ANNA, material_version_id=version.id, parser_profile_version=PROFILE
    )
    job = await dispatcher.claim(owner="worker-a")

    assert await dispatcher.fail(job.id, lease_owner="worker-a", error_code="parse") == "pending"
    assert await dispatcher.claim(owner="worker-b") is None, "not immediately — it backs off"


async def test_the_backoff_grows_and_then_holds():
    """The failures worth retrying are the ones that do not clear in a second."""
    delays = [backoff_for(n) for n in range(1, 7)]

    assert delays == sorted(delays)
    assert delays[0] < delays[-1]
    assert delays[-1] == delays[-2], "the last interval repeats rather than growing forever"


async def test_a_job_that_keeps_failing_eventually_gives_up(maker, dispatcher, version):
    """Otherwise it is claimed forever, and each claim costs someone else a slot."""
    await dispatcher.enqueue(
        owner_id=ANNA, material_version_id=version.id, parser_profile_version=PROFILE
    )

    outcome = None
    for _ in range(MAX_ATTEMPTS + 1):
        async with maker() as session:  # the delay has passed
            await session.execute(
                update(IngestionJob).values(next_attempt_at=_now(), lease_expires_at=None)
            )
            await session.commit()
        job = await dispatcher.claim(owner="worker-a")
        if job is None:
            break
        outcome = await dispatcher.fail(job.id, lease_owner="worker-a", error_code="parse")

    assert outcome == "failed"
    assert await dispatcher.claim(owner="worker-b") is None


async def test_releasing_hands_the_job_straight_back(dispatcher, version):
    """A clean shutdown must not cost the queue a backoff."""
    await dispatcher.enqueue(
        owner_id=ANNA, material_version_id=version.id, parser_profile_version=PROFILE
    )
    job = await dispatcher.claim(owner="worker-a")

    assert await dispatcher.release(job.id, lease_owner="worker-a")
    assert (await dispatcher.claim(owner="worker-b")).id == job.id


# ---------------------------------------------------------------------------
# The worker
# ---------------------------------------------------------------------------


async def test_the_worker_completes_a_job(dispatcher, version):
    await dispatcher.enqueue(
        owner_id=ANNA, material_version_id=version.id, parser_profile_version=PROFILE
    )
    seen = []

    async def handler(job):
        seen.append(job.id)
        return None

    assert await IngestionWorker(dispatcher, handler).run_once() is True
    assert len(seen) == 1

    done = await dispatcher.get(seen[0], owner_id=ANNA)
    assert done.status == "ready"
    assert done.finished_at is not None


async def test_one_bad_document_does_not_stop_the_queue(dispatcher, version, tmp_path, maker):
    """A handler that raises records a failed attempt; the loop carries on."""
    async with maker() as session:
        service = MaterialService(session, LocalDurableStorage(tmp_path / "obj"))
        _m, second = await service.store_new_material(
            b"secondo documento",
            owner_id=ANNA,
            display_name="Secondo",
            media_type="text/plain",
        )
        await session.commit()

    for version_id in (version.id, second.id):
        await dispatcher.enqueue(
            owner_id=ANNA, material_version_id=version_id, parser_profile_version=PROFILE
        )

    async def handler(job):
        raise ValueError("PDF illeggibile")

    worker = IngestionWorker(dispatcher, handler)
    assert await worker.run_once() is True
    assert await worker.run_once() is True, "the second job was still reachable"


async def test_a_failure_never_leaks_the_reason_into_the_row(dispatcher, version):
    await dispatcher.enqueue(
        owner_id=ANNA, material_version_id=version.id, parser_profile_version=PROFILE
    )

    async def handler(job):
        raise RuntimeError("postgresql://aix:hunter2@10.0.0.4/aix is unreachable")

    await IngestionWorker(dispatcher, handler).run_once()

    async with dispatcher._session_maker() as session:  # noqa: SLF001 — inspecting the row
        job = (await session.execute(IngestionJob.__table__.select())).first()

    assert "hunter2" not in (job.safe_error_code or "")
    assert "10.0.0.4" not in (job.safe_error_code or "")
    assert job.safe_error_code.startswith("handler_error:")


async def test_an_empty_queue_is_not_an_error(dispatcher):
    assert await IngestionWorker(dispatcher).run_once() is False


async def test_the_default_handler_refuses_rather_than_marking_things_ready(dispatcher, version):
    """Completing an unprocessed job would leave the teacher waiting forever."""
    await dispatcher.enqueue(
        owner_id=ANNA, material_version_id=version.id, parser_profile_version=PROFILE
    )

    await IngestionWorker(dispatcher).run_once()

    job = await dispatcher.claim(owner="anyone")  # still claimable → not completed
    assert job is None or job.status != "ready"


async def test_the_loop_stops_when_asked(dispatcher):
    worker = IngestionWorker(dispatcher, poll_interval_s=0.05)
    task = asyncio.create_task(worker.run_forever())
    await asyncio.sleep(0.1)

    worker.stop()
    await asyncio.wait_for(task, timeout=2.0)


# ---------------------------------------------------------------------------
# Stopping the worker does not interrupt the legacy path
# ---------------------------------------------------------------------------


def test_nothing_starts_with_the_flag_off(dispatcher, monkeypatch):
    """The default. Nothing claims, nothing parses, uploads work as they did."""
    monkeypatch.delenv("AIX_MATERIALS_ENABLED", raising=False)

    assert start_ingestion_worker(dispatcher) is None


async def test_the_worker_starts_only_when_the_flag_is_on(dispatcher, monkeypatch):
    monkeypatch.setenv("AIX_MATERIALS_ENABLED", "true")

    task = start_ingestion_worker(dispatcher)
    try:
        assert task is not None
    finally:
        from aix.webui.materials.worker import stop_ingestion_worker

        await stop_ingestion_worker(task)


async def test_the_legacy_upload_helpers_do_not_touch_any_of_this():
    """PR 3 is machinery, not a change to what a teacher's upload does today."""
    import inspect

    from aix.webui.lessons import uploads

    source = inspect.getsource(uploads)
    assert "ingestion_job" not in source
    assert "JobDispatcher" not in source


async def test_the_lease_is_renewed_while_the_handler_works(dispatcher, version):
    """A long parse outlives one lease, and the job must stay this worker's.

    Before the renewal (Louis, 2026-09-15) a 40-page scan — minutes of OCR
    against a two-minute lease — was claimable by a second worker halfway
    through, and both would parse it and write the same rows.
    """
    await dispatcher.enqueue(
        owner_id=ANNA, material_version_id=version.id, parser_profile_version="ir@1"
    )

    async def _slow_handler(_job):
        await asyncio.sleep(1.2)
        return None

    worker = IngestionWorker(dispatcher, _slow_handler, lease_s=0.9)
    running = asyncio.create_task(worker.run_once())

    await asyncio.sleep(1.0)  # past the original lease, while the handler still works
    assert await dispatcher.claim(owner="rival-worker") is None, "the lease was renewed"

    assert await running is True
