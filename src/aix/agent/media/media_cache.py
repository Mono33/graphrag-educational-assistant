"""Off-critical-path cache for the live media layer.

Phase 0 of the Dynamic Media Retrieval plan
(see ``docs/product/Dynamic_Media_Retrieval_Plan.md``).

Backends (selected from :class:`MediaConfig`):
  * ``redis``     → ``redis.asyncio`` — production; reuse the shared instance,
                    isolate entries under a versioned key namespace
                    (e.g. ``aix:media:v1``) so they can be flushed independently.
  * ``diskcache`` → local on-disk cache — dev default (no Redis required).
  * ``none``/``null`` → always-miss no-op (used when disabled or a dependency
                    is missing).

Design contract (critical for the "zero added latency" guarantee):
  * This cache lives **off** the lesson-generation critical path. The agent
    pipeline never calls it; only the future live media worker will.
  * It NEVER raises into callers. Any backend/connection/serialization error
    degrades to a cache *miss* (for reads) or a silent skip (for writes), so
    the live layer simply falls back to an off-path fetch and nothing breaks.
  * The API is async so it can be awaited from the async media worker without
    blocking the event loop (synchronous ``diskcache`` calls are offloaded to a
    worker thread via :func:`asyncio.to_thread`).
  * ``redis`` and ``diskcache`` are imported lazily, so importing this module is
    free and safe even when neither package is installed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any, Optional

from .media_config import MediaConfig

logger = logging.getLogger(__name__)


def make_cache_key(namespace: str, *parts: Any) -> str:
    """Build a stable, namespaced cache key from arbitrary parts.

    Parts are normalized to strings and hashed (sha256, truncated) so the key
    is fixed-length and safe regardless of the input (query text, concept lists,
    level, language, duration bucket, …).
    """
    raw = "|".join("" if p is None else str(p) for p in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return f"{namespace}:{digest}"


class _Backend:
    """Informal backend interface (documentation only — not enforced).

    Concrete backends below implement ``get``/``set``/``close`` and expose a
    ``backend_name``. Kept as a plain base (rather than ``typing.Protocol``) so
    the module imports on any supported Python without extra constraints.
    """

    backend_name: str = "base"


class _NullBackend:
    """Always-miss backend. Used when caching is disabled or deps are absent."""

    backend_name = "null"

    async def get(self, key: str) -> Optional[Any]:
        return None

    async def set(self, key: str, value: Any, ttl: int) -> None:
        return None

    async def close(self) -> None:
        return None


class _DiskCacheBackend:
    """Local on-disk backend (dev default). Sync calls offloaded to a thread."""

    backend_name = "diskcache"

    def __init__(self, directory: str) -> None:
        import diskcache  # lazy import — optional dependency

        self._cache = diskcache.Cache(directory)

    async def get(self, key: str) -> Optional[Any]:
        return await asyncio.to_thread(self._cache.get, key)

    async def set(self, key: str, value: Any, ttl: int) -> None:
        await asyncio.to_thread(self._cache.set, key, value, ttl)

    async def close(self) -> None:
        await asyncio.to_thread(self._cache.close)


class _RedisBackend:
    """Shared-Redis backend (prod). Values are JSON-serialized strings."""

    backend_name = "redis"

    def __init__(self, url: str) -> None:
        import redis.asyncio as aioredis  # lazy import — optional dependency

        self._redis = aioredis.from_url(url, encoding="utf-8", decode_responses=True)

    async def get(self, key: str) -> Optional[Any]:
        raw = await self._redis.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    async def set(self, key: str, value: Any, ttl: int) -> None:
        await self._redis.set(key, json.dumps(value), ex=ttl)

    async def close(self) -> None:
        aclose = getattr(self._redis, "aclose", None)
        if aclose is not None:
            await aclose()
        else:  # redis-py < 5 compatibility
            await self._redis.close()


def _build_backend(cfg: MediaConfig) -> _Backend:
    """Resolve a backend from config, degrading to null on any problem."""
    backend = (cfg.cache_backend or "").strip().lower()

    if backend in ("none", "null", "off", ""):
        return _NullBackend()

    if backend == "redis":
        if not cfg.redis_url:
            logger.warning(
                "[media-cache] backend=redis but no AIX_MEDIA_REDIS_URL/REDIS_URL "
                "is set — using null cache (off-path fetch will still work)."
            )
            return _NullBackend()
        try:
            return _RedisBackend(cfg.redis_url)
        except Exception as exc:  # ImportError or connection-config error
            logger.warning("[media-cache] redis init failed — using null cache: %s", exc)
            return _NullBackend()

    # Default / "diskcache".
    try:
        return _DiskCacheBackend(cfg.cache_dir)
    except Exception as exc:  # ImportError or filesystem error
        logger.warning("[media-cache] diskcache init failed — using null cache: %s", exc)
        return _NullBackend()


class MediaCache:
    """Thin async, fail-safe wrapper around a pluggable cache backend."""

    def __init__(self, backend: _Backend, namespace: str, default_ttl: int) -> None:
        self._backend = backend
        self._namespace = namespace
        self._default_ttl = default_ttl

    @property
    def backend_name(self) -> str:
        return getattr(self._backend, "backend_name", "unknown")

    @classmethod
    def from_config(cls, cfg: Optional[MediaConfig] = None) -> "MediaCache":
        cfg = cfg or MediaConfig.from_env()
        backend = _build_backend(cfg)
        cache = cls(backend, cfg.cache_namespace, cfg.cache_ttl)
        logger.debug("[media-cache] initialized backend=%s ns=%s", cache.backend_name, cfg.cache_namespace)
        return cache

    def key(self, *parts: Any) -> str:
        """Build a namespaced key for the given parts."""
        return make_cache_key(self._namespace, *parts)

    async def get(self, key: str) -> Optional[Any]:
        """Return the cached value or ``None`` on miss/any error."""
        try:
            return await self._backend.get(key)
        except Exception as exc:
            logger.warning(
                "[media-cache] get failed (backend=%s) — treating as miss: %s",
                self.backend_name,
                exc,
            )
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store a JSON-serializable value; silently skip on any error."""
        try:
            await self._backend.set(key, value, ttl if ttl is not None else self._default_ttl)
        except Exception as exc:
            logger.warning(
                "[media-cache] set failed (backend=%s) — skipping write: %s",
                self.backend_name,
                exc,
            )

    async def close(self) -> None:
        try:
            await self._backend.close()
        except Exception:
            pass
