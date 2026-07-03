"""CORE 6 — Phase A concurrency controls regression coverage.

Pure unit tests for ``aix.core.concurrency`` (no LLM, no network). They lock
in the three Phase-A guarantees:

1. #31 — the global run cap bounds simultaneous pipelines.
2. #34 — once the cap (and the queue window) is exhausted, the next
   acquirer is shed with :class:`AtCapacity` rather than piling on.
3. The queue window lets a waiter pick up a slot the instant one frees, and
   both the explicit and context-manager APIs always release — even on error.
4. #32 — ``llm_slot`` blocks (does not shed) past the LLM-call ceiling.
5. #33 — ``check_user_rate_limit`` enforces a per-user rolling-window budget,
   records a hit only when allowed, is per-key, and recovers as the window
   slides.

The semaphores are loop-aware, so each test's fresh event loop gets a fresh
limiter; we only need to set the env knobs before the first acquire. The #33
limiter is a plain module-global dict (not loop-bound), so those tests clear
``conc._user_hits`` up front and drive a fake clock for determinism.
"""

from __future__ import annotations

import asyncio

import pytest

import aix.core.concurrency as conc

pytestmark = pytest.mark.unit


async def test_run_cap_sheds_when_full(monkeypatch):
    monkeypatch.setenv("AIX_MAX_CONCURRENT_RUNS", "2")
    monkeypatch.setenv("AIX_RUN_QUEUE_TIMEOUT_S", "0")

    await conc.acquire_run_slot(label="a")
    await conc.acquire_run_slot(label="b")
    assert conc.concurrency_stats()["active_runs"] == 2

    # Third concurrent run is shed immediately (queue window = 0).
    with pytest.raises(conc.AtCapacity):
        await conc.acquire_run_slot(label="c")

    # Releasing frees exactly one slot, which the next acquire can take.
    conc.release_run_slot()
    assert conc.concurrency_stats()["active_runs"] == 1
    await conc.acquire_run_slot(label="d")
    assert conc.concurrency_stats()["active_runs"] == 2

    conc.release_run_slot()
    conc.release_run_slot()
    assert conc.concurrency_stats()["active_runs"] == 0


async def test_queue_window_lets_waiter_acquire_on_release(monkeypatch):
    monkeypatch.setenv("AIX_MAX_CONCURRENT_RUNS", "1")
    monkeypatch.setenv("AIX_RUN_QUEUE_TIMEOUT_S", "2")

    await conc.acquire_run_slot(label="holder")

    async def _waiter() -> bool:
        await conc.acquire_run_slot(label="waiter")
        return True

    task = asyncio.ensure_future(_waiter())
    await asyncio.sleep(0.05)
    assert not task.done(), "waiter should still be queued while the slot is held"

    conc.release_run_slot()  # frees the slot → waiter wakes within the window
    assert await asyncio.wait_for(task, timeout=1.0) is True

    conc.release_run_slot()
    assert conc.concurrency_stats()["active_runs"] == 0


async def test_run_slot_context_manager_releases_on_error(monkeypatch):
    monkeypatch.setenv("AIX_MAX_CONCURRENT_RUNS", "1")
    monkeypatch.setenv("AIX_RUN_QUEUE_TIMEOUT_S", "0")

    with pytest.raises(ValueError):
        async with conc.run_slot(label="boom"):
            assert conc.concurrency_stats()["active_runs"] == 1
            raise ValueError("simulated pipeline failure")

    # Slot is released despite the exception, so the next run can proceed.
    assert conc.concurrency_stats()["active_runs"] == 0
    async with conc.run_slot(label="next"):
        assert conc.concurrency_stats()["active_runs"] == 1
    assert conc.concurrency_stats()["active_runs"] == 0


async def test_llm_slot_blocks_past_ceiling_then_proceeds(monkeypatch):
    monkeypatch.setenv("AIX_MAX_CONCURRENT_LLM_CALLS", "1")

    waiter_started = asyncio.Event()
    waiter_done = asyncio.Event()

    async def _second_call() -> None:
        waiter_started.set()
        async with conc.llm_slot():
            waiter_done.set()

    async with conc.llm_slot():
        assert conc.concurrency_stats()["active_llm_calls"] == 1
        task = asyncio.ensure_future(_second_call())
        await waiter_started.wait()
        await asyncio.sleep(0.05)
        # Second call must block (not shed) while the only slot is held.
        assert not waiter_done.is_set()

    await asyncio.wait_for(task, timeout=1.0)
    assert waiter_done.is_set()
    assert conc.concurrency_stats()["active_llm_calls"] == 0


async def test_guarded_chat_completion_holds_slot_and_passes_kwargs(monkeypatch):
    monkeypatch.setenv("AIX_MAX_CONCURRENT_LLM_CALLS", "5")

    seen: dict = {}

    class _Completions:
        async def create(self, **kwargs):
            # The slot must be held for the duration of the call.
            seen["active_during"] = conc.concurrency_stats()["active_llm_calls"]
            seen["kwargs"] = kwargs
            return "ok"

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    result = await conc.guarded_chat_completion(_Client(), model="m", messages=[{"role": "user"}])

    assert result == "ok"
    assert seen["active_during"] == 1
    # ``client`` is consumed by the helper; only the create() kwargs pass through.
    assert seen["kwargs"] == {"model": "m", "messages": [{"role": "user"}]}
    # Slot released once the call returns.
    assert conc.concurrency_stats()["active_llm_calls"] == 0


# ---------------------------------------------------------------------------
# #33 — per-user rate limiting (sync; module-global dict, fake-clock driven)
# ---------------------------------------------------------------------------


def test_rate_limit_allows_up_to_budget_then_rejects(monkeypatch):
    conc._user_hits.clear()
    monkeypatch.setenv("AIX_USER_RATE_LIMIT", "3")
    monkeypatch.setenv("AIX_USER_RATE_WINDOW_S", "60")

    for _ in range(3):
        conc.check_user_rate_limit("webui:42")  # within budget

    with pytest.raises(conc.RateLimited) as excinfo:
        conc.check_user_rate_limit("webui:42")
    assert excinfo.value.limit == 3
    assert excinfo.value.window_s == 60.0
    assert excinfo.value.retry_after > 0


def test_rate_limit_disabled_when_zero(monkeypatch):
    conc._user_hits.clear()
    monkeypatch.setenv("AIX_USER_RATE_LIMIT", "0")

    for _ in range(50):
        conc.check_user_rate_limit("webui:1")  # never raises, never tracked
    assert conc.concurrency_stats()["rate_tracked_users"] == 0


def test_rate_limit_is_per_user(monkeypatch):
    conc._user_hits.clear()
    monkeypatch.setenv("AIX_USER_RATE_LIMIT", "1")

    conc.check_user_rate_limit("webui:1")
    conc.check_user_rate_limit("api:other")  # distinct key — unaffected
    with pytest.raises(conc.RateLimited):
        conc.check_user_rate_limit("webui:1")


def test_rate_limit_window_expiry_recovers(monkeypatch):
    conc._user_hits.clear()
    monkeypatch.setenv("AIX_USER_RATE_LIMIT", "1")
    monkeypatch.setenv("AIX_USER_RATE_WINDOW_S", "10")

    clock = {"t": 1000.0}
    monkeypatch.setattr(conc.time, "monotonic", lambda: clock["t"])

    conc.check_user_rate_limit("webui:7")
    with pytest.raises(conc.RateLimited):
        conc.check_user_rate_limit("webui:7")

    clock["t"] += 11.0  # whole window elapsed → key recovers
    conc.check_user_rate_limit("webui:7")


def test_rate_limit_rejected_attempt_does_not_extend_window(monkeypatch):
    conc._user_hits.clear()
    monkeypatch.setenv("AIX_USER_RATE_LIMIT", "1")
    monkeypatch.setenv("AIX_USER_RATE_WINDOW_S", "10")

    clock = {"t": 0.0}
    monkeypatch.setattr(conc.time, "monotonic", lambda: clock["t"])

    conc.check_user_rate_limit("u")  # t=0 recorded
    clock["t"] = 5.0
    with pytest.raises(conc.RateLimited):
        conc.check_user_rate_limit("u")  # rejected — must NOT be recorded
    clock["t"] = 10.5  # only the t=0 hit existed, now expired
    conc.check_user_rate_limit("u")  # allowed (proves reject@5 wasn't logged)
