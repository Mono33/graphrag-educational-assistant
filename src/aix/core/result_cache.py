"""Answers worth keeping between turns: the graph's reply to a question, a verdict.

A model call that is deterministic in its inputs need not be paid twice. The first
two users are the questions a lesson asks the Knowledge Graph (short keywords
like "ADHD" that come back turn after turn: 32 of 100 repeated in the 2026-10
rehearsals) and the citation judge's verdict on a (sentence, passage) pair.

It sits on :class:`~aix.agent.media.media_cache.MediaCache`, so the backend is the
one already configured (``AIX_MEDIA_CACHE_BACKEND``: diskcache by default, Redis
when there is a URL), under namespaces of its own. Same contract: it never raises
into the caller -- any failure is a miss, and the work is simply done.

Off with ``AIX_RESULT_CACHE_ENABLED=false``. The graph only changes through
``scripts/ingest``, which stores no version, so a cached answer would outlive a
re-ingest: bump ``AIX_KG_VERSION`` after every one (it is part of the key).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
from typing import Any, Awaitable, Callable, Optional

from aix.agent.media.media_cache import MediaCache
from aix.agent.media.media_config import MediaConfig

logger = logging.getLogger(__name__)

DEFAULT_TTL_S = 86_400

_caches: dict[str, MediaCache] = {}


def enabled() -> bool:
    raw = os.getenv("AIX_RESULT_CACHE_ENABLED")
    if raw is None or not raw.strip():
        return True
    return raw.strip().lower() in ("1", "true", "yes", "on")


def kg_version() -> str:
    """The label of the data in the graph; part of every graph-derived key."""
    return (os.getenv("AIX_KG_VERSION") or "").strip()


def ttl_seconds() -> int:
    try:
        return max(60, int(os.getenv("AIX_RESULT_CACHE_TTL_S") or DEFAULT_TTL_S))
    except ValueError:
        return DEFAULT_TTL_S


def normalize(text: Optional[str]) -> str:
    """Case and spacing do not change a question: «ADHD » and «adhd» are one key."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _cache_for(namespace: str) -> MediaCache:
    cache = _caches.get(namespace)
    if cache is None:
        cfg = dataclasses.replace(
            MediaConfig.from_env(), cache_namespace=namespace, cache_ttl=ttl_seconds()
        )
        cache = _caches[namespace] = MediaCache.from_config(cfg)
    return cache


def _json_safe(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


async def remember(
    namespace: str,
    parts: tuple[Any, ...],
    compute: Callable[[], Awaitable[Any]],
    *,
    keep: Callable[[Any], bool] = lambda value: True,
    ttl: Optional[Callable[[Any], int]] = None,
    cache: Optional[MediaCache] = None,
) -> Any:
    """The cached value for ``parts``, or ``await compute()`` stored when ``keep`` says so.

    ``keep`` filters what is worth storing (never an error, never a failed call);
    ``ttl`` may shorten the life of a doubtful value, such as an empty answer.
    Values that are not plain JSON are returned but not stored.
    """
    if not enabled():
        return await compute()

    try:
        store = cache or _cache_for(namespace)
        key = store.key(*parts)
        hit = await store.get(key)
    except Exception:  # noqa: BLE001 — a broken cache is a miss
        logger.warning("[result-cache] lookup failed in %s", namespace, exc_info=True)
        return await compute()

    if isinstance(hit, dict) and "value" in hit:
        logger.info("[result-cache] hit in %s", namespace)
        return hit["value"]

    value = await compute()
    try:
        if keep(value) and _json_safe(value):
            await store.set(key, {"value": value}, ttl(value) if ttl else None)
    except Exception:  # noqa: BLE001
        logger.warning("[result-cache] could not store in %s", namespace, exc_info=True)
    return value
