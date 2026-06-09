"""Configuration for the dynamic (live) media layer.

Phase 0 of the Dynamic Media Retrieval plan
(see ``docs/product/Dynamic_Media_Retrieval_Plan.md``).

This is **pure scaffolding**: it only reads environment variables with safe
defaults. Nothing in this module changes runtime behavior on its own — the
master switch ``AIX_MEDIA_LIVE_ENABLED`` defaults to ``False`` so, until a later
phase wires the live layer into the retrieval pipeline, the system behaves
exactly as before (curated pool only).

All values are resolved from the environment via :meth:`MediaConfig.from_env`,
mirroring the convention used elsewhere (``os.getenv`` with defaults), so
omitting every variable preserves today's behavior.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


@dataclass(frozen=True)
class MediaConfig:
    """Resolved settings for the live media layer.

    Defaults are chosen so that an unconfigured environment reproduces the
    current (curated-only) behavior:
      * ``live_enabled=False``      → live layer is off
      * per-bucket caps match today's panel density (5 videos / 3 papers / 6 web)
      * ``cache_backend`` resolves to ``diskcache`` in dev (no Redis required)
      * ``refresh_enabled=False``   → no background refresh job
    """

    # Master switch — when False, the live layer must be a complete no-op.
    live_enabled: bool = False

    # Per-bucket caps (Q4 — resolved: keep today's counts, env-configurable).
    max_videos: int = 5
    max_papers: int = 3
    max_web: int = 6

    # Cache (Q3 — resolved: reuse Redis in prod via a versioned namespace; dev = diskcache).
    cache_backend: str = "diskcache"  # "redis" | "diskcache" | "none"
    cache_namespace: str = "aix:media:v1"
    cache_ttl: int = 1_209_600  # 14 days
    cache_dir: str = "artifacts/media_cache"
    redis_url: Optional[str] = None

    # Background refresh job (Q5 — resolved: separate scheduled process, off by default).
    refresh_enabled: bool = False

    # Phase 3 — semantic re-ranking of the live items against query + lesson
    # content. Off by default (flag-gated like every phase): with it off the live
    # items keep their fetch order, exactly as Phase 1/2 shipped. The blended
    # score is ``w_semantic·cosine(item, query+content) + w_quality·signal`` where
    # the quality signal uses fields we already store (citations/recency for
    # papers; semantic-only for videos/Wikipedia in this phase).
    rerank_enabled: bool = False
    rerank_weight_semantic: float = 0.7
    rerank_weight_quality: float = 0.3

    # YouTube Data API key (optional — fallback to curated + DDGS when absent).
    youtube_api_key: Optional[str] = None

    @classmethod
    def from_env(cls) -> "MediaConfig":
        """Build a :class:`MediaConfig` from the process environment."""
        redis_url = os.getenv("AIX_MEDIA_REDIS_URL") or os.getenv("REDIS_URL")

        # Default backend: redis when a URL is configured (typically prod),
        # otherwise diskcache (dev). An explicit AIX_MEDIA_CACHE_BACKEND wins.
        default_backend = "redis" if redis_url else "diskcache"
        backend = (os.getenv("AIX_MEDIA_CACHE_BACKEND") or default_backend).strip().lower()

        namespace = (os.getenv("AIX_MEDIA_CACHE_NAMESPACE") or "aix:media:v1").strip()

        return cls(
            live_enabled=_env_bool("AIX_MEDIA_LIVE_ENABLED", False),
            max_videos=_env_int("AIX_MEDIA_MAX_VIDEOS", 5),
            max_papers=_env_int("AIX_MEDIA_MAX_PAPERS", 3),
            max_web=_env_int("AIX_MEDIA_MAX_WEB", 6),
            cache_backend=backend,
            cache_namespace=namespace,
            cache_ttl=_env_int("AIX_MEDIA_CACHE_TTL", 1_209_600),
            cache_dir=(os.getenv("AIX_MEDIA_CACHE_DIR") or "artifacts/media_cache").strip(),
            redis_url=redis_url,
            refresh_enabled=_env_bool("AIX_MEDIA_REFRESH_ENABLED", False),
            rerank_enabled=_env_bool("AIX_MEDIA_RERANK_ENABLED", False),
            rerank_weight_semantic=_env_float("AIX_MEDIA_RERANK_W_SEMANTIC", 0.7),
            rerank_weight_quality=_env_float("AIX_MEDIA_RERANK_W_QUALITY", 0.3),
            youtube_api_key=os.getenv("YOUTUBE_API_KEY"),
        )

    def summary(self) -> str:
        """One-line, secret-free description for diagnostics/logging."""
        return (
            f"live={self.live_enabled} backend={self.cache_backend} "
            f"ns={self.cache_namespace} ttl={self.cache_ttl}s "
            f"caps(v/p/w)={self.max_videos}/{self.max_papers}/{self.max_web} "
            f"rerank={self.rerank_enabled}(w={self.rerank_weight_semantic}/{self.rerank_weight_quality}) "
            f"refresh={self.refresh_enabled} youtube_key={'set' if self.youtube_api_key else 'unset'}"
        )
