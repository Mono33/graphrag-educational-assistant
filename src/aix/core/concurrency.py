"""Process-wide concurrency controls for agent generation (CORE 6 — Phase A).

Single-worker hardening. Three knobs, all env-configurable, all PER-PROCESS
(``asyncio.Semaphore``); cross-worker coordination is Phase B (#37):

  * #31 — bound how many full agent pipelines run at once
          (``AIX_MAX_CONCURRENT_RUNS``), with a short queue window
          (``AIX_RUN_QUEUE_TIMEOUT_S``) before shedding.
  * #34 — shed load gracefully when at capacity: callers get
          :class:`AtCapacity` and surface a friendly "sistema occupato"
          message instead of piling on more concurrent pipelines.
  * #32 — cap concurrent **LLM calls** process-wide
          (``AIX_MAX_CONCURRENT_LLM_CALLS``) to protect against provider
          429s / cost spikes independent of the run cap.

Semaphores are created lazily on first use so they bind to the running
event loop (Python ≥3.10 binds asyncio primitives to the loop on first
await, not at construction). The module keeps best-effort active-count
gauges for the future observability dashboard (#40).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class AtCapacity(RuntimeError):
    """Raised by :func:`acquire_run_slot` when no generation slot becomes
    available within the configured queue window (#34 load-shedding)."""


class RateLimited(RuntimeError):
    """Raised by :func:`check_user_rate_limit` when a caller exceeds their
    per-window run budget (#33). ``retry_after`` is the suggested wait (s)."""

    def __init__(self, retry_after: float, limit: int, window_s: float):
        self.retry_after = max(0.0, retry_after)
        self.limit = limit
        self.window_s = window_s
        super().__init__(
            f"Rate limit exceeded: {limit} runs per {window_s:.0f}s "
            f"(retry in {self.retry_after:.0f}s)."
        )


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

_DEFAULT_MAX_CONCURRENT_RUNS = 6
_DEFAULT_RUN_QUEUE_TIMEOUT_S = 5.0
_DEFAULT_MAX_CONCURRENT_LLM_CALLS = 12


def _int_env(name: str, default: int, *, lo: int, hi: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return max(lo, min(hi, int(raw)))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float, *, lo: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return max(lo, float(raw))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# #31 — global generation-run cap
# ---------------------------------------------------------------------------

_run_semaphore: asyncio.Semaphore | None = None
_run_capacity = 0
_run_active = 0  # best-effort gauge (single event loop → safe int ops)
_run_loop: asyncio.AbstractEventLoop | None = None


def _get_run_semaphore() -> asyncio.Semaphore:
    # Lazily (re)create the semaphore bound to the *running* loop. In
    # production there is one long-lived loop so this creates once; in tests
    # each case runs in its own loop, so we rebind to avoid "got Future
    # attached to a different loop" errors.
    global _run_semaphore, _run_capacity, _run_active, _run_loop
    loop = asyncio.get_running_loop()
    if _run_semaphore is None or _run_loop is not loop:
        _run_capacity = _int_env(
            "AIX_MAX_CONCURRENT_RUNS", _DEFAULT_MAX_CONCURRENT_RUNS, lo=1, hi=128
        )
        _run_semaphore = asyncio.Semaphore(_run_capacity)
        _run_loop = loop
        _run_active = 0
    return _run_semaphore


async def _acquire_with_timeout(sem: asyncio.Semaphore, timeout: float) -> bool:
    """Acquire ``sem`` within ``timeout`` seconds. Returns True on success.

    Robust against the classic ``wait_for`` race where the permit is acquired
    exactly as the timeout fires: in that case we release it back so capacity
    never silently shrinks.
    """
    task = asyncio.ensure_future(sem.acquire())
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout if timeout > 0 else 0.001)
        return True
    except asyncio.TimeoutError:
        if task.done() and not task.cancelled() and task.exception() is None:
            # Acquired during the race — hand the permit back.
            sem.release()
        else:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        return False


async def acquire_run_slot(label: str = "") -> None:
    """Acquire one global generation slot (#31) or raise :class:`AtCapacity` (#34).

    Waits up to ``AIX_RUN_QUEUE_TIMEOUT_S`` for a slot (the "in coda" window)
    before shedding. Pair every successful call with :func:`release_run_slot`
    in a ``finally`` block.
    """
    global _run_active
    sem = _get_run_semaphore()
    timeout = _float_env("AIX_RUN_QUEUE_TIMEOUT_S", _DEFAULT_RUN_QUEUE_TIMEOUT_S, lo=0.0)
    acquired = await _acquire_with_timeout(sem, timeout)
    if not acquired:
        logger.warning(
            "[concurrency] at capacity — shedding run%s (cap=%d, waited=%.1fs)",
            f" [{label}]" if label else "",
            _run_capacity,
            timeout,
        )
        raise AtCapacity(
            f"All {_run_capacity} generation slots are busy (waited {timeout:.1f}s)."
        )
    _run_active += 1
    logger.info(
        "[concurrency] run slot acquired%s (active=%d/%d)",
        f" [{label}]" if label else "",
        _run_active,
        _run_capacity,
    )


def release_run_slot() -> None:
    """Release a slot previously acquired with :func:`acquire_run_slot`."""
    global _run_active
    if _run_semaphore is None:
        return
    _run_active = max(0, _run_active - 1)
    _run_semaphore.release()


@asynccontextmanager
async def run_slot(label: str = ""):
    """Async context-manager form of :func:`acquire_run_slot` /
    :func:`release_run_slot`. Raises :class:`AtCapacity` on entry when the
    queue window elapses."""
    await acquire_run_slot(label=label)
    try:
        yield
    finally:
        release_run_slot()


# ---------------------------------------------------------------------------
# #32 — process-wide concurrent-LLM-call cap
# ---------------------------------------------------------------------------

_llm_semaphore: asyncio.Semaphore | None = None
_llm_capacity = 0
_llm_active = 0
_llm_loop: asyncio.AbstractEventLoop | None = None


def _get_llm_semaphore() -> asyncio.Semaphore:
    global _llm_semaphore, _llm_capacity, _llm_active, _llm_loop
    loop = asyncio.get_running_loop()
    if _llm_semaphore is None or _llm_loop is not loop:
        _llm_capacity = _int_env(
            "AIX_MAX_CONCURRENT_LLM_CALLS", _DEFAULT_MAX_CONCURRENT_LLM_CALLS, lo=1, hi=512
        )
        _llm_semaphore = asyncio.Semaphore(_llm_capacity)
        _llm_loop = loop
        _llm_active = 0
    return _llm_semaphore


@asynccontextmanager
async def llm_slot():
    """Bound process-wide concurrent LLM calls (#32).

    Unlike :func:`run_slot` this **blocks** rather than shedding: callers are
    already inside a run slot, so this simply smooths bursts of concurrent
    provider calls to avoid 429s / cost spikes. Wrap each
    ``await client.chat.completions.create(...)`` call site.
    """
    global _llm_active
    sem = _get_llm_semaphore()
    await sem.acquire()
    _llm_active += 1
    try:
        yield
    finally:
        _llm_active = max(0, _llm_active - 1)
        sem.release()


async def guarded_chat_completion(client, /, **kwargs):
    """Run ``client.chat.completions.create(**kwargs)`` inside an LLM slot (#32).

    For **non-streaming** calls only — the slot is released as soon as the
    response returns. For streaming (``stream=True``) hold :func:`llm_slot`
    across the whole token-consumption loop instead, because the slot must
    cover the entire in-flight request, not just the call that opens the
    stream.
    """
    async with llm_slot():
        return await client.chat.completions.create(**kwargs)


# ---------------------------------------------------------------------------
# #33 — per-user rate limiting (in-house; no external dependency)
# ---------------------------------------------------------------------------

_DEFAULT_USER_RATE_MAX = 10  # runs allowed …
_DEFAULT_USER_RATE_WINDOW_S = 60.0  # … per this rolling window (seconds)
_RL_SWEEP_EVERY = 512  # opportunistic stale-key cleanup cadence

# Per-user sliding-window log of run-start timestamps (monotonic seconds).
_user_hits: dict[str, deque] = {}
_rl_calls = 0


def _user_rate_config() -> tuple[int, float]:
    return (
        _int_env("AIX_USER_RATE_LIMIT", _DEFAULT_USER_RATE_MAX, lo=0, hi=100000),
        _float_env("AIX_USER_RATE_WINDOW_S", _DEFAULT_USER_RATE_WINDOW_S, lo=1.0),
    )


def _sweep_user_hits(now: float, window: float) -> None:
    """Drop keys whose window is fully expired so idle users don't accumulate."""
    cutoff = now - window
    stale = []
    for key, dq in _user_hits.items():
        while dq and dq[0] <= cutoff:
            dq.popleft()
        if not dq:
            stale.append(key)
    for key in stale:
        _user_hits.pop(key, None)


def check_user_rate_limit(user_key: str) -> None:
    """Enforce a per-user rolling-window run budget (#33).

    No-op when ``AIX_USER_RATE_LIMIT`` is 0 (disabled). Raises
    :class:`RateLimited` when the caller has already started ``limit`` runs
    within the trailing ``AIX_USER_RATE_WINDOW_S`` seconds. A hit is recorded
    only when the call is **allowed**, so rejected attempts don't extend the
    window. In-memory + per-process (like the Phase A semaphores); a shared
    store (Redis) would be needed for a cross-worker guarantee (Phase B).
    """
    global _rl_calls
    limit, window = _user_rate_config()
    if limit <= 0:
        return

    now = time.monotonic()
    cutoff = now - window

    dq = _user_hits.get(user_key)
    if dq is None:
        dq = deque()
        _user_hits[user_key] = dq
    while dq and dq[0] <= cutoff:
        dq.popleft()

    _rl_calls += 1
    if _rl_calls % _RL_SWEEP_EVERY == 0:
        _sweep_user_hits(now, window)

    if len(dq) >= limit:
        retry_after = dq[0] + window - now
        logger.warning(
            "[concurrency] rate limit hit for %s (%d/%d in %.0fs window)",
            user_key,
            len(dq),
            limit,
            window,
        )
        raise RateLimited(retry_after=retry_after, limit=limit, window_s=window)

    dq.append(now)


# ---------------------------------------------------------------------------
# Observability (feeds the Phase B #40 concurrency dashboard)
# ---------------------------------------------------------------------------


def concurrency_stats() -> dict[str, int]:
    """Best-effort snapshot of current concurrency usage."""
    return {
        "active_runs": _run_active,
        "max_runs": _run_capacity
        or _int_env("AIX_MAX_CONCURRENT_RUNS", _DEFAULT_MAX_CONCURRENT_RUNS, lo=1, hi=128),
        "active_llm_calls": _llm_active,
        "max_llm_calls": _llm_capacity
        or _int_env(
            "AIX_MAX_CONCURRENT_LLM_CALLS", _DEFAULT_MAX_CONCURRENT_LLM_CALLS, lo=1, hi=512
        ),
        "rate_tracked_users": len(_user_hits),
    }
