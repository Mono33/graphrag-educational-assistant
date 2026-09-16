"""The worker that drains the ingestion queue (Fase 0 · PR 3).

A loop, inside the application process, that claims one job at a time and runs a
handler over it. Not a separate service: the pilot is one container on one host,
and a second deployable would double what can be misconfigured while doubling
nothing that matters yet. The seam is the ``JobDispatcher``, so moving this loop
into its own process later means running the same class somewhere else.

What it does *not* do is decide what a job means. The handler is injected,
because parsing arrives in Fase 1 and this PR is the machinery that will carry
it. Until then the worker is not started at all: :func:`start_ingestion_worker`
returns ``None`` unless ``AIX_MATERIALS_ENABLED`` is on, so the legacy upload
path is the only one running and stopping a worker that was never started
cannot break it.

Three properties matter, and each is tested:

* **a job survives a restart** — it is a committed row before any work starts,
  so a process that dies mid-parse loses the attempt, not the work;
* **an abandoned job comes back** — the lease expires and the next worker takes
  it, with the attempt already counted against it;
* **two workers never run one job** — the claim is a conditional UPDATE whose
  row lock lets exactly one win.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Optional

from aix.webui.errors import report
from aix.webui.materials.jobs import DEFAULT_LEASE_S, JobDispatcher, worker_id
from aix.webui.materials.models import IngestionJob
from aix.webui.materials.storage import materials_enabled

logger = logging.getLogger(__name__)

#: How long to wait after finding nothing before asking again. The queue is fed
#: by teachers uploading files, so it is empty most of the time and there is no
#: point spinning on it; a few seconds is well inside what someone waiting for
#: their upload to be processed would notice.
DEFAULT_POLL_INTERVAL_S = 3.0

#: What the handler is given, and what it may return: a job, and the step name
#: it reached. Returning ``None`` means "done" and completes the job.
JobHandler = Callable[[IngestionJob], Awaitable[Optional[str]]]


async def _unconfigured(job: IngestionJob) -> Optional[str]:
    """The default handler, which refuses rather than pretending.

    Completing a job that nothing processed would mark a document ready and
    leave the teacher waiting for content that is never coming. Failing loudly
    is the honest behaviour until Fase 1 supplies a parser.
    """
    raise NotImplementedError(f"no ingestion handler configured; job {job.id} cannot be processed")


class IngestionWorker:
    """Claims jobs and runs them, one at a time, until told to stop."""

    def __init__(
        self,
        dispatcher: JobDispatcher,
        handler: JobHandler = _unconfigured,
        *,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        lease_s: float = DEFAULT_LEASE_S,
        name: Optional[str] = None,
    ) -> None:
        self._dispatcher = dispatcher
        self._handler = handler
        self._poll_interval_s = poll_interval_s
        self._lease_s = lease_s
        self.name = name or worker_id()
        self._stopping = asyncio.Event()

    async def run_once(self) -> bool:
        """Process at most one job. Returns whether there was one.

        Never raises on a job's behalf: a handler that fails records a failed
        attempt and the loop continues, because one unparseable PDF must not
        stop every other teacher's uploads.
        """
        job = await self._dispatcher.claim(lease_s=self._lease_s, owner=self.name)
        if job is None:
            return False

        # Hold the lease for as long as the work takes. A 40-page scan is
        # minutes of OCR and the lease is two, so without this the lease expires
        # mid-parse, another worker claims the same job, and both write the same
        # rows (Louis, 2026-09-15).
        heartbeat = asyncio.create_task(self._keep_the_lease(job))
        try:
            step = await self._handler(job)
        except Exception as exc:  # noqa: BLE001 — every failure is one job's failure
            reported = report(exc, context=f"ingestion job={job.id}")
            await self._dispatcher.fail(
                job.id, lease_owner=self.name, error_code=f"handler_error:{reported.incident}"
            )
            return True
        finally:
            heartbeat.cancel()

        if step is None:
            await self._dispatcher.complete(job.id, lease_owner=self.name)
            return True

        # More to do. Record where we got to and extend the lease, so the next
        # step starts from here rather than from the beginning.
        await self._dispatcher.set_step(job.id, lease_owner=self.name, step=step)
        await self._dispatcher.renew(job.id, lease_owner=self.name, lease_s=self._lease_s)
        return True

    async def _keep_the_lease(self, job: IngestionJob) -> None:
        """Renew this job's lease while the handler works on it.

        Renews at a third of the lease, so two renewals can be missed before it
        lapses. If a renewal comes back False the lease is already someone
        else's: there is nothing to do about it from here — the handler's own
        writes are refused for the same reason — so it stops renewing and says
        so in the log, where it is the trace of a job processed twice.
        """
        interval = max(0.5, self._lease_s / 3)
        while True:
            await asyncio.sleep(interval)
            if not await self._dispatcher.renew(
                job.id, lease_owner=self.name, lease_s=self._lease_s
            ):
                logger.warning(
                    "[jobs] worker %s lost the lease on job=%s while working on it",
                    self.name,
                    job.id,
                )
                return

    async def run_forever(self) -> None:
        """Drain the queue until :meth:`stop` is called.

        Sleeps only when the queue was empty: a backlog is worked through
        without a poll interval between each item.
        """
        logger.info("[jobs] worker %s started", self.name)
        try:
            while not self._stopping.is_set():
                try:
                    did_work = await self.run_once()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 — the loop must outlive the queue
                    report(exc, context="ingestion worker loop")
                    did_work = False

                if did_work:
                    continue
                try:
                    await asyncio.wait_for(self._stopping.wait(), timeout=self._poll_interval_s)
                except asyncio.TimeoutError:
                    pass
        finally:
            logger.info("[jobs] worker %s stopped", self.name)

    def stop(self) -> None:
        """Ask the loop to finish the job in hand and then exit."""
        self._stopping.set()


def _poll_interval_s() -> float:
    try:
        return max(0.1, float(os.getenv("AIX_INGESTION_POLL_S") or DEFAULT_POLL_INTERVAL_S))
    except ValueError:
        return DEFAULT_POLL_INTERVAL_S


def start_ingestion_worker(
    dispatcher: JobDispatcher, handler: JobHandler = _unconfigured
) -> Optional[asyncio.Task]:
    """Start the worker, if the material pipeline is switched on.

    Returns ``None`` when it is off, which is the default. That is what keeps
    this whole PR inert: with the flag down nothing claims, nothing parses, and
    the legacy upload path is untouched.
    """
    if not materials_enabled():
        logger.info("[jobs] AIX_MATERIALS_ENABLED is off — no ingestion worker started")
        return None

    worker = IngestionWorker(dispatcher, handler, poll_interval_s=_poll_interval_s())
    task = asyncio.create_task(worker.run_forever(), name="aix-ingestion-worker")
    task.aix_worker = worker  # type: ignore[attr-defined]  # so the lifespan can stop it
    return task


async def stop_ingestion_worker(task: Optional[asyncio.Task]) -> None:
    """Ask the worker to finish, then wait briefly for it.

    Cancelled rather than waited on forever: a shutdown that hangs because one
    parse is slow is worse than a lost attempt, and a lost attempt is exactly
    what the lease is designed to recover.
    """
    if task is None:
        return

    worker = getattr(task, "aix_worker", None)
    if worker is not None:
        worker.stop()
    try:
        await asyncio.wait_for(task, timeout=5.0)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        task.cancel()
