"""Live media layer service — Phase 1 (papers + Wikipedia) + Phase 2 (videos)
+ Phase 3 (semantic re-ranking against query + lesson content).

Part of the Dynamic Media Retrieval plan
(see ``docs/product/Dynamic_Media_Retrieval_Plan.md``).

This service performs the *dynamic* media retrieval that today's curated pool
cannot: it queries live sources (Phase 1 = OpenAlex papers + Wikipedia; Phase 2
adds live YouTube videos) based on the lesson's concepts/query and returns
results in the **exact same dict shape** that the retriever already produces for
``external_resources`` — so the existing media panel renders them with no
template changes.

Videos (Phase 2) are only fetched when a ``YOUTUBE_API_KEY`` is configured; with
no key the curated video pool remains the floor (no fake search-link items are
emitted). YouTube results are embeddable-only (the API call already requests
``videoEmbeddable=true`` + ``safeSearch=strict``).

Critical design properties (matching the approved plan):
  * **Off the critical path.** Nothing here is called by the planner → retriever
    → writer → critic pipeline. It is intended to be invoked by an off-path
    worker (Phase 1b web wiring), so lesson-generation latency is unchanged.
  * **Flag-gated.** When ``AIX_MEDIA_LIVE_ENABLED`` is false (default), every
    entry point is a no-op returning ``{}`` — i.e. today's behavior.
  * **Cache-first.** The BROAD candidate pool is read from / written to
    :class:`MediaCache` (Redis prod / diskcache dev / null fallback) keyed by
    concepts, so repeated concept-sets are served instantly and external quota
    is amortized. Phase 3 re-ranking runs *after* the cache read, so ordering is
    query-specific without sacrificing the concept-level cache hit-rate.
  * **Bounded & fail-safe.** A global per-call timeout caps the work, each
    source failure is isolated, and any error degrades to ``{}`` — never raises
    into the caller.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

from .media_cache import MediaCache
from .media_config import MediaConfig

logger = logging.getLogger(__name__)

# Off-path overall budget. Generous (we are NOT on the critical path) but bounded
# so a slow/hanging upstream can never wedge the worker.
_OVERALL_TIMEOUT_S = 12.0

# Off-path budget for the (embedding-backed) Phase 3 re-ranking step.
_RANK_TIMEOUT_S = 8.0

# Hard cap on how many concepts we expand into live queries per call (cost guard).
_MAX_CONCEPTS = 3

# Phase 3: the (concept-keyed, cacheable) candidate pool kept BEFORE the per-bucket
# caps so the semantic re-ranker has room to select the best top-N. The natural
# pool is already bounded (≤ _MAX_CONCEPTS × per-source max); these only guard
# against unexpectedly large caps blowing up embedding cost.
_CANDIDATE_CEILING_PAPERS = 15
_CANDIDATE_CEILING_VIDEOS = 20

# Trim lengths mirror the retriever's existing normalization so payloads match.
_WIKI_SUMMARY_MAX = 500
_PAPER_ABSTRACT_MAX = 300


class LiveMediaService:
    """Fetch live subject resources (papers + Wikipedia), cached and off-path."""

    def __init__(
        self,
        config: Optional[MediaConfig] = None,
        api: Optional[Any] = None,
        cache: Optional[MediaCache] = None,
    ) -> None:
        self._config = config or MediaConfig.from_env()
        self._cache = cache
        self._api = api
        # We only close an API/session we created ourselves; an injected one is
        # owned by the caller.
        self._owns_api = api is None

    # -- lazy collaborators ------------------------------------------------

    def _get_cache(self) -> MediaCache:
        if self._cache is None:
            self._cache = MediaCache.from_config(self._config)
        return self._cache

    def _build_api(self) -> Optional[Any]:
        """Create an ExternalMediaAPI, or return None if unavailable."""
        try:
            from .external_apis import ExternalMediaAPI
        except Exception as exc:  # pragma: no cover - import guard
            logger.warning("[live-media] ExternalMediaAPI unavailable: %s", exc)
            return None
        try:
            return ExternalMediaAPI(
                semantic_scholar_api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY")
            )
        except Exception as exc:
            logger.warning("[live-media] ExternalMediaAPI init failed: %s", exc)
            return None

    # -- public API --------------------------------------------------------

    async def fetch_subject_resources(
        self,
        *,
        concepts: list[str],
        language: str = "it",
        query: Optional[str] = None,
        content: Optional[str] = None,
    ) -> dict[str, Any]:
        """Return live papers + Wikipedia for ``concepts``.

        Shape mirrors the retriever's ``external_resources``::

            {
              "wikipedia": [{title, summary, url, thumbnail_url, concept}],
              "papers":    [{title, authors, year, abstract, url,
                             citation_count, concept}],
              "videos":    [{title, video_id, channel, url, thumbnail_url,
                             duration, concept}],
              "source_attribution": "live",
              "_meta": {concepts, language, cache, counts},
            }

        Returns ``{}`` when the live layer is disabled, no concepts are given,
        or every source fails/times out.
        """
        if not self._config.live_enabled:
            return {}

        concepts = [c.strip() for c in (concepts or []) if c and c.strip()][:_MAX_CONCEPTS]
        if not concepts:
            return {}

        cache = self._get_cache()
        cache_key = cache.key(
            "live_subject",
            language,
            sorted(c.lower() for c in concepts),
            self._config.max_papers,
            # ``max_videos`` participates so enabling/raising the video cap (and
            # the Phase 1 → Phase 2 transition) yields a fresh key, never a stale
            # papers-only entry.
            self._config.max_videos if self._config.youtube_api_key else 0,
            # Phase 3: the candidate ceilings participate so the cached value is
            # the BROAD (ranking-ready) pool — forcing a fresh miss past the
            # Phase 2 entries that were stored already-capped.
            _CANDIDATE_CEILING_PAPERS,
            _CANDIDATE_CEILING_VIDEOS,
        )

        cached = await cache.get(cache_key)
        if cached is not None:
            logger.info("[live-media] cache HIT (%s) key=%s", cache.backend_name, cache_key)
            pool = cached
            cache_state = "hit"
        else:
            logger.info("[live-media] cache MISS (%s) — fetching %d concept(s)", cache.backend_name, len(concepts))
            try:
                pool = await asyncio.wait_for(
                    self._fetch(concepts, language), timeout=_OVERALL_TIMEOUT_S
                )
            except asyncio.TimeoutError:
                logger.warning("[live-media] overall timeout (%.0fs) — returning empty", _OVERALL_TIMEOUT_S)
                return {}
            except Exception as exc:
                logger.warning("[live-media] fetch failed — returning empty: %s", exc)
                return {}

            # Cache the BROAD candidate pool (concept-keyed, query-independent) so
            # external quota is amortized across queries while re-ranking below
            # stays query-specific. Skip caching a transient empty fetch.
            if pool.get("wikipedia") or pool.get("papers") or pool.get("videos"):
                await cache.set(cache_key, pool)
            cache_state = "miss"

        # Phase 3: re-rank the (possibly cached) pool against query + content,
        # then apply the per-bucket caps. Off-path and fully fail-safe.
        result = await self._rank_and_cap(pool, query=query, content=content)
        meta = {**result.get("_meta", {}), "cache": cache_state}
        return {**result, "_meta": meta}

    async def _rank_and_cap(
        self,
        pool: dict[str, Any],
        *,
        query: Optional[str],
        content: Optional[str],
    ) -> dict[str, Any]:
        """Re-rank (Phase 3) then apply per-bucket caps to a candidate pool.

        Ranking is gated on ``AIX_MEDIA_RERANK_ENABLED`` + a non-empty
        query/content; with it off (or on any failure/timeout) the pool keeps its
        fetch order — so this only ever changes *which* items survive the cap and
        in *what order*, never the counts. Returns ``{}`` for an empty pool.
        """
        if not pool:
            return {}

        ranked = pool
        attempted = self._config.rerank_enabled and bool((query or "").strip() or (content or "").strip())
        if attempted:
            try:
                from .media_ranker import rank_live_media

                ranked = await asyncio.wait_for(
                    asyncio.to_thread(
                        rank_live_media,
                        pool,
                        query=query or "",
                        content=content or "",
                        config=self._config,
                    ),
                    timeout=_RANK_TIMEOUT_S,
                )
            except Exception as exc:
                logger.warning("[live-media] rerank step failed/timeout — keeping fetch order: %s", exc)
                ranked = pool

        papers = (ranked.get("papers") or [])[: self._config.max_papers]
        videos = (ranked.get("videos") or [])[: self._config.max_videos]
        wikipedia = ranked.get("wikipedia") or []
        prev_meta = pool.get("_meta") or {}

        return {
            "wikipedia": wikipedia,
            "papers": papers,
            "videos": videos,
            "source_attribution": "live",
            "_meta": {
                "concepts": prev_meta.get("concepts"),
                "language": prev_meta.get("language"),
                "reranked": attempted,
                "counts": {
                    "wikipedia": len(wikipedia),
                    "papers": len(papers),
                    "videos": len(videos),
                },
            },
        }

    # -- internals ---------------------------------------------------------

    async def _fetch(self, concepts: list[str], language: str) -> dict[str, Any]:
        api = self._api or self._build_api()
        if api is None:
            return {}

        # Videos are Phase 2 and only make sense with a real key — without one,
        # ``search_youtube`` would return a fallback search-link (no video_id),
        # which we deliberately do NOT surface as a "live video" (the curated
        # pool is the video floor). So we skip the call entirely when unkeyed.
        want_videos = bool(self._config.youtube_api_key)

        resources: dict[str, Any] = {
            "wikipedia": [],
            "papers": [],
            "videos": [],
            "source_attribution": "live",
        }

        try:
            async def _one(concept: str):
                tasks: list[Any] = [
                    api.get_wikipedia_summary(concept, language=language),
                    api.search_openalex(
                        query=f"{concept} education teaching",
                        max_results=self._config.max_papers,
                    ),
                ]
                if want_videos:
                    tasks.append(
                        api.search_youtube(
                            query=concept,
                            max_results=self._config.max_videos,
                            language=language,
                        )
                    )
                gathered = await asyncio.gather(*tasks, return_exceptions=True)
                wiki, papers = gathered[0], gathered[1]
                videos = gathered[2] if want_videos else None
                return concept, wiki, papers, videos

            per_concept = await asyncio.gather(
                *[_one(c) for c in concepts], return_exceptions=True
            )

            seen_paper_urls: set[str] = set()
            seen_video_ids: set[str] = set()
            # Wikipedia is fetched per concept, but synonymous concepts (e.g.
            # "adhd" and "disturbi da deficit di attenzione") resolve to the SAME
            # canonical page — so de-duplicate by canonical URL (normalized title
            # as fallback), mirroring the paper/video dedup, to avoid identical
            # "Enciclopedia" entries. Dedup happens BEFORE the cache write, so the
            # cached pool is already clean and every cache hit serves it dedup-free.
            seen_wiki: set[str] = set()
            for item in per_concept:
                if isinstance(item, Exception):
                    logger.debug("[live-media] concept fetch raised: %s", item)
                    continue
                concept, wiki, papers, videos = item

                if wiki and not isinstance(wiki, Exception):
                    wiki_key = (
                        (getattr(wiki, "url", "") or "").strip().lower().rstrip("/")
                        or (getattr(wiki, "title", "") or "").strip().lower()
                    )
                    if wiki_key and wiki_key not in seen_wiki:
                        seen_wiki.add(wiki_key)
                        resources["wikipedia"].append(
                            {
                                "title": wiki.title,
                                "summary": (wiki.summary or "")[:_WIKI_SUMMARY_MAX],
                                "url": wiki.url,
                                "thumbnail_url": getattr(wiki, "thumbnail_url", None),
                                "concept": concept,
                            }
                        )
                elif isinstance(wiki, Exception):
                    logger.debug("[live-media] wikipedia failed for '%s': %s", concept, wiki)

                if papers and not isinstance(papers, Exception):
                    for paper in papers:
                        url = getattr(paper, "url", "") or ""
                        if url and url in seen_paper_urls:
                            continue
                        if url:
                            seen_paper_urls.add(url)
                        resources["papers"].append(
                            {
                                "title": paper.title,
                                "authors": (paper.authors or [])[:3],
                                "year": paper.year,
                                "abstract": (paper.abstract or "")[:_PAPER_ABSTRACT_MAX],
                                "url": url,
                                "doi": getattr(paper, "doi", None),
                                "journal": getattr(paper, "venue", None),
                                "citation_count": getattr(paper, "citation_count", 0),
                                "concept": concept,
                            }
                        )
                elif isinstance(papers, Exception):
                    logger.debug("[live-media] openalex failed for '%s': %s", concept, papers)

                if videos and not isinstance(videos, Exception):
                    for video in videos:
                        video_id = getattr(video, "video_id", "") or ""
                        # Embeddable-only: drop the keyless fallback search-link
                        # sentinel (empty video_id) and de-duplicate by id.
                        if not video_id or video_id in seen_video_ids:
                            continue
                        seen_video_ids.add(video_id)
                        resources["videos"].append(
                            {
                                "title": video.title,
                                "video_id": video_id,
                                "channel": getattr(video, "channel", None),
                                "url": video.url,
                                "thumbnail_url": getattr(video, "thumbnail_url", None),
                                "duration": getattr(video, "duration", None),
                                # Phase 3.1: keep the snippet description so the
                                # semantic re-ranker scores videos on real topical
                                # text, not just the (often terse) title.
                                "description": getattr(video, "description", None),
                                "concept": concept,
                            }
                        )
                elif isinstance(videos, Exception):
                    logger.debug("[live-media] youtube failed for '%s': %s", concept, videos)

            # Phase 3: keep a BROAD candidate pool (bounded by the candidate
            # ceilings, not the final per-bucket caps) so the downstream
            # semantic re-ranker can pick the most relevant top-N. The final
            # max_papers/max_videos caps are applied later in _rank_and_cap.
            resources["papers"] = resources["papers"][:_CANDIDATE_CEILING_PAPERS]
            resources["videos"] = resources["videos"][:_CANDIDATE_CEILING_VIDEOS]

            resources["_meta"] = {
                "concepts": concepts,
                "language": language,
                "counts": {
                    "wikipedia": len(resources["wikipedia"]),
                    "papers": len(resources["papers"]),
                    "videos": len(resources["videos"]),
                },
            }
            logger.info(
                "[live-media] fetched %d wikipedia, %d paper candidate(s), %d video candidate(s)",
                len(resources["wikipedia"]),
                len(resources["papers"]),
                len(resources["videos"]),
            )
            return resources
        finally:
            if self._owns_api and api is not None:
                try:
                    await api.close()
                except Exception:
                    pass


def to_panel_media(live: dict[str, Any]) -> dict[str, Any]:
    """Map the live ``external_resources`` shape to the media-panel buckets.

    The media panel (``partials/media_panel.html``) renders ``curated_media``
    buckets. This adapter converts the live result so those same sections
    render the dynamic items:

        live ``papers``    → ``citations``  ("Articoli scientifici")
        live ``wikipedia`` → ``wikipedia``  (a dedicated "Enciclopedia" section)
        live ``videos``    → ``videos``     (merged under "Video")

    Every produced item carries ``source: "live"`` so the template can badge it
    distinctly from curated/verified entries. Returns ``{}`` when empty.
    """
    if not live:
        return {}

    videos: list[dict[str, Any]] = []
    for video in live.get("videos") or []:
        videos.append(
            {
                "title": video.get("title"),
                "url": video.get("url"),
                "channel": video.get("channel"),
                "thumbnail_url": video.get("thumbnail_url"),
                # The panel video item reads ``duration_hint`` for its meta line.
                "duration_hint": video.get("duration"),
                "source": "live",
            }
        )

    citations: list[dict[str, Any]] = []
    for paper in live.get("papers") or []:
        citations.append(
            {
                "title": paper.get("title"),
                "authors": paper.get("authors") or [],
                "year": paper.get("year"),
                "journal": paper.get("journal"),
                "doi": paper.get("doi"),
                "url": paper.get("url"),
                "source": "live",
            }
        )

    wikipedia: list[dict[str, Any]] = []
    for entry in live.get("wikipedia") or []:
        wikipedia.append(
            {
                "title": entry.get("title"),
                "summary": entry.get("summary"),
                "url": entry.get("url"),
                "thumbnail_url": entry.get("thumbnail_url"),
                "source": "live",
            }
        )

    out: dict[str, Any] = {}
    if videos:
        out["videos"] = videos
    if citations:
        out["citations"] = citations
    if wikipedia:
        out["wikipedia"] = wikipedia
    return out


async def fetch_live_subject_resources(
    *,
    concepts: list[str],
    language: str = "it",
    query: Optional[str] = None,
    content: Optional[str] = None,
    config: Optional[MediaConfig] = None,
) -> dict[str, Any]:
    """Convenience one-shot wrapper around :class:`LiveMediaService`.

    Creates a service with a fresh (self-closing) ExternalMediaAPI, fetches, and
    returns the normalized dict (``{}`` when disabled or on failure). ``query``
    and ``content`` drive the Phase 3 semantic re-ranking (when enabled).
    """
    service = LiveMediaService(config=config)
    return await service.fetch_subject_resources(
        concepts=concepts, language=language, query=query, content=content
    )
