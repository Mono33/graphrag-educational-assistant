"""A durable queue for material processing, in the database (Fase 0 · PR 3).

Parsing a document is slow, fails for boring reasons, and must not be lost when
the container is replaced mid-way. That is a queue — and the cheapest correct
queue available here is a table, because one already exists and works.

``agent_run`` coordinates the agent's runs with an atomic claim, a heartbeat and
a TTL against this same database, and ``tests/unit/test_run_registry.py`` pins
its semantics on SQLite and PostgreSQL both. This module is that pattern applied
to ingestion. The pilot gets durable background work without gaining Redis,
Celery, and a second service that can be down at three in the morning.

How a job is claimed
--------------------
Two steps, and the second is the one that makes it safe:

1. read a few candidate ids — rows that are not finished, whose retry time has
   come, and whose lease is absent or expired;
2. ``UPDATE ... WHERE id = :id AND <that same condition still holds>``.

The row lock serialises the second step, so under any number of concurrent
workers exactly one gets ``rowcount == 1`` and the losers get zero and move to
the next candidate. This is the same argument ``agent_run.claim`` makes, and it
is portable: ``SELECT ... FOR UPDATE SKIP LOCKED`` exists on PostgreSQL and not
on SQLite, so it is used here only as an optimisation on top of a scheme that is
already correct without it — it saves the losers a wasted round trip, it is not
what prevents the double claim.

Why a lease and not a heartbeat
-------------------------------
``agent_run`` refreshes a heartbeat because a run is one continuous coroutine.
A job is a sequence of steps with a database write between them, so the natural
unit is a lease the worker renews at each step. A worker that dies holding one
blocks its job for at most one lease, and then any worker may take it — with
``attempt_count`` already incremented, so a job that reliably kills its worker
gives up rather than eating the queue forever.
"""

from __future__ import annotations

import logging
import os
import socket
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import or_, select, update
from sqlalchemy.exc import IntegrityError

from aix.webui.materials.models import (
    JOB_TERMINAL_STATUSES,
    IngestionJob,
)

logger = logging.getLogger(__name__)

#: How long a claim is good for. Long enough that a slow parse does not lose its
#: own job mid-step, short enough that a crashed worker's job is not stuck for a
#: shift. Renewed at every step, so it bounds one step and not the whole job.
DEFAULT_LEASE_S = 120.0

#: After this many attempts a job is failed rather than retried. A job that
#: kills its worker every time would otherwise be claimed forever, and each
#: claim costs a worker slot that a job which *could* succeed is waiting for.
MAX_ATTEMPTS = 5

#: Backoff between attempts, in seconds, by attempt number. The last value
#: repeats. Growing, because the failures worth retrying — a service down, a
#: rate limit — are the ones that do not clear in a second.
BACKOFF_S = (5, 30, 120, 600)


def worker_id() -> str:
    """Who holds a lease. Host plus pid, so a log line points at a process."""
    return f"{socket.gethostname()}:{os.getpid()}"


def _now() -> datetime:
    """UTC, and deliberately naive.

    SQLite hands ``DateTime(timezone=True)`` back without a tzinfo, so an
    aware value written here comes back naive — and SQLAlchemy evaluates an
    UPDATE's WHERE clause in Python against rows already in the identity map,
    where the two then meet and raise. ``agent_run`` has used naive UTC since
    it was written; matching it is what keeps the two from disagreeing.

    Not ``datetime.utcnow()``, which is deprecated in 3.12 — same value,
    without the warning.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def backoff_for(attempt: int) -> float:
    return float(BACKOFF_S[min(max(attempt, 1) - 1, len(BACKOFF_S) - 1)])


class JobDispatcher:
    """Enqueue, claim, renew and finish jobs.

    Takes a session *maker* rather than a session: a worker outlives any one
    request, and holding a request's session past its response is how a pool
    gets exhausted quietly.
    """

    def __init__(self, session_maker: Any) -> None:
        self._session_maker = session_maker

    # -- producing -------------------------------------------------------

    async def enqueue(
        self,
        *,
        owner_id: uuid.UUID,
        material_version_id: uuid.UUID,
        parser_profile_version: str,
    ) -> Optional[IngestionJob]:
        """Queue a version for processing, once per parser profile.

        Returns ``None`` when this version already has a job for this profile —
        the unique constraint decides, not a prior read, so two requests racing
        to enqueue the same upload produce one job rather than two.
        """
        async with self._session_maker() as session:
            job = IngestionJob(
                owner_id=owner_id,
                material_version_id=material_version_id,
                parser_profile_version=parser_profile_version,
                status="pending",
                current_step="pending",
                next_attempt_at=_now(),
            )
            session.add(job)
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                return None
            await session.refresh(job)
            return job

    # -- consuming -------------------------------------------------------

    async def claim(
        self, *, lease_s: float = DEFAULT_LEASE_S, owner: Optional[str] = None, batch: int = 8
    ) -> Optional[IngestionJob]:
        """Take the next runnable job, or ``None`` when there is nothing to do.

        ``batch`` is how many candidates to consider before giving up for this
        round. Losing a race is not a failure — it means another worker took
        that job — so a loser tries the next candidate rather than the caller
        having to loop.
        """
        holder = owner or worker_id()
        async with self._session_maker() as session:
            for job_id in await self._candidates(session, batch):
                now = _now()
                result = await session.execute(
                    update(IngestionJob)
                    .where(IngestionJob.id == job_id, *self._claimable(now))
                    .values(
                        lease_owner=holder,
                        lease_expires_at=now + timedelta(seconds=lease_s),
                        attempt_count=IngestionJob.attempt_count + 1,
                        updated_at=now,
                    )
                )
                await session.commit()

                if result.rowcount == 1:
                    claimed = await session.get(IngestionJob, job_id)
                    logger.info(
                        "[jobs] claimed job=%s attempt=%s by=%s",
                        job_id,
                        claimed.attempt_count if claimed else "?",
                        holder,
                    )
                    return claimed
            return None

    @staticmethod
    def _claimable(now: datetime) -> tuple:
        """A job nobody is working on, that is due, and that is not finished."""
        return (
            IngestionJob.status.notin_(JOB_TERMINAL_STATUSES),
            IngestionJob.next_attempt_at <= now,
            or_(
                IngestionJob.lease_expires_at.is_(None),
                IngestionJob.lease_expires_at <= now,
            ),
        )

    async def _candidates(self, session: Any, batch: int) -> list[uuid.UUID]:
        stmt = (
            select(IngestionJob.id)
            .where(*self._claimable(_now()))
            .order_by(IngestionJob.next_attempt_at, IngestionJob.created_at)
            .limit(batch)
        )
        # PostgreSQL only, and only to save the losers a wasted round trip:
        # correctness comes from the conditional UPDATE below, not from here.
        if session.bind.dialect.name == "postgresql":
            stmt = stmt.with_for_update(skip_locked=True)
        return list((await session.execute(stmt)).scalars())

    async def renew(
        self, job_id: uuid.UUID, *, lease_owner: str, lease_s: float = DEFAULT_LEASE_S
    ) -> bool:
        """Extend the lease, if it is still ours.

        Returns False when it is not — the lease expired and someone else took
        the job. A worker that sees False must stop working on it: continuing
        would mean two workers on one job, which is exactly what the lease is
        for.
        """
        now = _now()
        async with self._session_maker() as session:
            result = await session.execute(
                update(IngestionJob)
                .where(*self._held_by(job_id, lease_owner, now))
                .values(lease_expires_at=now + timedelta(seconds=lease_s), updated_at=now)
            )
            await session.commit()
            return result.rowcount == 1

    async def set_step(self, job_id: uuid.UUID, *, lease_owner: str, step: str) -> bool:
        """Record progress, so a retry resumes instead of restarting."""
        now = _now()
        async with self._session_maker() as session:
            result = await session.execute(
                update(IngestionJob)
                .where(*self._held_by(job_id, lease_owner, now))
                .values(current_step=step, status=step, updated_at=now)
            )
            await session.commit()
            return result.rowcount == 1

    async def complete(self, job_id: uuid.UUID, *, lease_owner: str, status: str = "ready") -> bool:
        now = _now()
        async with self._session_maker() as session:
            result = await session.execute(
                update(IngestionJob)
                .where(*self._held_by(job_id, lease_owner, now))
                .values(
                    status=status,
                    current_step=status,
                    finished_at=now,
                    updated_at=now,
                    lease_owner=None,
                    lease_expires_at=None,
                    safe_error_code=None,
                )
            )
            await session.commit()
            return result.rowcount == 1

    async def fail(self, job_id: uuid.UUID, *, lease_owner: str, error_code: str) -> Optional[str]:
        """Record a failed attempt; retry it or give up.

        Returns the status the job ended in — ``"pending"`` when it will be
        tried again, ``"failed"`` when the budget is spent — or ``None`` if the
        lease was no longer ours, in which case nothing was written.

        ``error_code`` is a code and never a message: what actually went wrong
        goes to the log with an incident id (see ``aix.webui.errors``), because
        this column is read by the interface.
        """
        now = _now()
        async with self._session_maker() as session:
            job = await session.get(IngestionJob, job_id)
            if job is None or job.lease_owner != lease_owner:
                return None

            give_up = job.attempt_count >= MAX_ATTEMPTS
            values: dict[str, Any] = {
                "safe_error_code": error_code,
                "lease_owner": None,
                "lease_expires_at": None,
                "updated_at": now,
            }
            if give_up:
                values.update(status="failed", finished_at=now)
            else:
                values.update(
                    status="pending",
                    next_attempt_at=now + timedelta(seconds=backoff_for(job.attempt_count)),
                )

            result = await session.execute(
                update(IngestionJob)
                .where(*self._held_by(job_id, lease_owner, now))
                .values(**values)
            )
            await session.commit()
            if result.rowcount != 1:
                return None

            outcome = "failed" if give_up else "pending"
            logger.warning(
                "[jobs] job=%s attempt=%s → %s (%s)",
                job_id,
                job.attempt_count,
                outcome,
                error_code,
            )
            return outcome

    async def release(self, job_id: uuid.UUID, *, lease_owner: str) -> bool:
        """Hand a job back untouched — a clean shutdown, not a failure.

        The attempt is not charged back: it was spent. But the job becomes
        immediately claimable rather than waiting out the lease, so stopping a
        worker on purpose costs nothing.
        """
        now = _now()
        async with self._session_maker() as session:
            result = await session.execute(
                update(IngestionJob)
                .where(*self._held_by(job_id, lease_owner, now))
                .values(
                    lease_owner=None,
                    lease_expires_at=None,
                    next_attempt_at=now,
                    updated_at=now,
                )
            )
            await session.commit()
            return result.rowcount == 1

    @staticmethod
    def _held_by(job_id: uuid.UUID, lease_owner: str, now: datetime) -> tuple:
        """Ours, and still ours: the lease must not have lapsed underneath us."""
        return (
            IngestionJob.id == job_id,
            IngestionJob.lease_owner == lease_owner,
            IngestionJob.lease_expires_at > now,
        )

    # -- looking ---------------------------------------------------------

    async def get(self, job_id: uuid.UUID, *, owner_id: uuid.UUID) -> Optional[IngestionJob]:
        """Owner-scoped, like everything else a teacher can reach."""
        async with self._session_maker() as session:
            result = await session.execute(
                select(IngestionJob).where(
                    IngestionJob.id == job_id, IngestionJob.owner_id == owner_id
                )
            )
            return result.scalar_one_or_none()

    async def pending_count(self) -> int:
        async with self._session_maker() as session:
            result = await session.execute(
                select(IngestionJob.id).where(IngestionJob.status.notin_(JOB_TERMINAL_STATUSES))
            )
            return len(list(result.scalars()))
