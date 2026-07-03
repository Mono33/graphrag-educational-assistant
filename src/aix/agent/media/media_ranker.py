"""Phase 3 — semantic re-ranking of live media items.

Part of the Dynamic Media Retrieval plan
(see ``docs/product/Dynamic_Media_Retrieval_Plan.md`` §5.3 / Phase 3).

Given the live candidate pool produced by :class:`~aix.agent.media.live_media.
LiveMediaService` (the broad, concept-keyed set *before* the per-bucket caps),
this module **re-orders each bucket by relevance to the teacher's query + the
generated lesson content**, so the downstream top-N cap keeps the *most
relevant* items rather than whatever the upstream APIs happened to return first.

Design properties (matching the plan's zero-latency / fail-safe posture):
  * **Off the critical path.** Only invoked by the off-path live-media worker,
    never by planner → retriever → writer → critic. Lesson latency is unchanged.
  * **Re-order only.** This function never adds, removes, or caps items — it just
    sorts each bucket best-first. The caller applies the per-bucket caps, so the
    panel density (Q4: 5 videos / 3 papers) is preserved exactly.
  * **Fail-safe.** Any failure (no embedder, API error, empty input) returns the
    pool **unchanged** in fetch order — never raises into the caller.
  * **Blended score.** ``w_semantic·cosine(item, query+content) +
    w_quality·signal`` where the quality signal reuses fields we already store
    (citations + recency for papers; semantic-only for videos/Wikipedia).

The embedding stack is the project's existing ``SemanticEmbedder``
(OpenAI ``text-embedding-3-small``, multilingual — good for Italian). It is
synchronous; callers should invoke :func:`rank_live_media` via
``asyncio.to_thread`` so the event loop is never blocked.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Optional

from .media_config import MediaConfig

logger = logging.getLogger(__name__)

# Cap the text we embed per item / per query so token cost stays bounded.
_ITEM_TEXT_MAX = 600
_QUERY_TEXT_MAX = 1600

# Buckets we re-rank and the (title, body) field names used to build item text.
# Phase 3.1: videos embed title + snippet description (and the channel is
# appended in _item_text) so they rank on real topical text, not a terse title.
_BUCKET_TEXT_FIELDS: dict[str, tuple[str, str]] = {
    "papers": ("title", "abstract"),
    "wikipedia": ("title", "summary"),
    "videos": ("title", "description"),
}

# Module-level lazy embedder. Building it is cheap (the OpenAI client is
# lazy-loaded on first embed call, and domain="all" has no node-cache file to
# load), but we still reuse one instance across calls to amortize that.
_embedder: Any = None
_embedder_failed = False


def _get_embedder() -> Optional[Any]:
    """Return a shared :class:`SemanticEmbedder`, or ``None`` if unavailable."""
    global _embedder, _embedder_failed
    if _embedder is not None:
        return _embedder
    if _embedder_failed:
        return None
    try:
        from aix.retrieval.graph_retriever import SemanticEmbedder

        _embedder = SemanticEmbedder(domain="all")
        return _embedder
    except Exception as exc:  # pragma: no cover - import/env guard
        logger.warning("[live-media] rerank embedder unavailable: %s", exc)
        _embedder_failed = True
        return None


def _item_text(bucket: str, item: dict[str, Any]) -> str:
    title_field, body_field = _BUCKET_TEXT_FIELDS[bucket]
    title = str(item.get(title_field) or "").strip()
    body = str(item.get(body_field) or "").strip()
    parts = [title, body]
    # Phase 3.1: the channel often signals topical authority (e.g. an educational
    # publisher), so append it for videos after the title + description.
    if bucket == "videos":
        channel = str(item.get("channel") or "").strip()
        if channel:
            parts.append(channel)
    text = ". ".join(p for p in parts if p).strip(". ").strip()
    return text[:_ITEM_TEXT_MAX]


def _quality_signal(bucket: str, item: dict[str, Any]) -> float:
    """A 0..1 quality prior from fields we already store.

    Papers blend citation impact (log-scaled) and recency. Videos and Wikipedia
    expose no comparable numeric signal in this phase, so they fall back to a
    neutral 0.0 — i.e. they are ranked semantically (``w_quality`` is unused for
    them, which is the intended Phase 3 behavior).
    """
    if bucket != "papers":
        return 0.0

    # Citations: log10(1 + n) / 3 saturates at ~1000 citations → 1.0.
    try:
        citations = int(item.get("citation_count") or 0)
    except (TypeError, ValueError):
        citations = 0
    citation_score = min(1.0, math.log10(1 + max(0, citations)) / 3.0)

    # Recency: linear over the last ~30 years, clamped to 0..1.
    recency_score = 0.0
    year = item.get("year")
    try:
        if year:
            this_year = datetime.utcnow().year
            recency_score = max(0.0, min(1.0, (int(year) - (this_year - 30)) / 30.0))
    except (TypeError, ValueError):
        recency_score = 0.0

    return 0.6 * citation_score + 0.4 * recency_score


def _rank_bucket(
    embedder: Any,
    query_embedding: Any,
    bucket: str,
    items: list[dict[str, Any]],
    *,
    w_semantic: float,
    w_quality: float,
) -> list[dict[str, Any]]:
    """Return ``items`` sorted best-first; falls back to the input on failure."""
    if len(items) <= 1:
        return items

    texts = [_item_text(bucket, it) for it in items]
    embeddings = embedder.embed_texts([t for t in texts if t])
    if not embeddings:
        return items

    scored: list[tuple[float, int, dict[str, Any]]] = []
    for idx, (item, text) in enumerate(zip(items, texts)):
        emb = embeddings.get(text)
        sim = embedder.compute_similarity(query_embedding, emb) if emb is not None else 0.0
        quality = _quality_signal(bucket, item)
        score = w_semantic * sim + w_quality * quality
        # ``idx`` is a stable tie-breaker so equal scores keep fetch order.
        scored.append((score, idx, item))

    scored.sort(key=lambda t: (-t[0], t[1]))
    return [item for _, _, item in scored]


def rank_live_media(
    pool: dict[str, Any],
    *,
    query: str = "",
    content: str = "",
    config: Optional[MediaConfig] = None,
) -> dict[str, Any]:
    """Re-order each live bucket by relevance to ``query`` + ``content``.

    Returns a new dict with the same keys/items but each ranked bucket sorted
    best-first. Items are never added/removed/capped here (the caller caps).
    Returns ``pool`` unchanged when there is nothing to rank or anything fails.
    """
    if not pool:
        return pool

    cfg = config or MediaConfig.from_env()
    query_text = f"{query or ''} {content or ''}".strip()
    if not query_text:
        return pool
    query_text = query_text[:_QUERY_TEXT_MAX]

    has_rankable = any((pool.get(b) or []) for b in _BUCKET_TEXT_FIELDS)
    if not has_rankable:
        return pool

    embedder = _get_embedder()
    if embedder is None:
        return pool

    try:
        query_embedding = embedder.embed_query(query_text)
        if query_embedding is None:
            return pool

        ranked = dict(pool)
        for bucket in _BUCKET_TEXT_FIELDS:
            items = pool.get(bucket) or []
            ranked[bucket] = _rank_bucket(
                embedder,
                query_embedding,
                bucket,
                items,
                w_semantic=cfg.rerank_weight_semantic,
                w_quality=cfg.rerank_weight_quality,
            )
        logger.info(
            "[live-media] reranked buckets (papers=%d, videos=%d, wikipedia=%d) against query+content",
            len(ranked.get("papers") or []),
            len(ranked.get("videos") or []),
            len(ranked.get("wikipedia") or []),
        )
        return ranked
    except Exception as exc:
        logger.warning("[live-media] rerank failed — keeping fetch order: %s", exc)
        return pool
