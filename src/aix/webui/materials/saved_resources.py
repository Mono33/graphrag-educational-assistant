"""Owner-scoped bookmarks for external media resources."""

from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from aix.webui.materials.models import SavedResource

RESOURCE_KINDS = {"video", "article", "oer", "web", "wikipedia", "textbook"}
_TRACKING_QUERY_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid"}


logger = logging.getLogger(__name__)


def normalize_external_url(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    value = raw.strip()
    if len(value) > 4000:
        raise ValueError("URL troppo lungo")
    parts = urlsplit(value)
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        raise ValueError("URL non valido")

    query = [
        (key, item)
        for key, item in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in _TRACKING_QUERY_KEYS
    ]
    host = (parts.hostname or "").lower()
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    return urlunsplit(
        (
            parts.scheme.lower(),
            host,
            parts.path or "/",
            urlencode(sorted(query)),
            "",
        )
    )


def stable_resource_key(
    *,
    kind: str,
    source: str,
    title: str,
    url: Optional[str],
    doi: Optional[str] = None,
    video_id: Optional[str] = None,
) -> str:
    if video_id and video_id.strip():
        return f"video:{video_id.strip().lower()}"[:320]
    if doi and doi.strip():
        value = doi.strip().lower()
        for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
            if value.startswith(prefix):
                value = value[len(prefix) :]
        return f"doi:{value}"[:320]
    if url:
        return f"url:{url}"[:320]
    fallback = "\x1f".join((kind, source, title.strip().lower()))
    return f"hash:{hashlib.sha256(fallback.encode('utf-8')).hexdigest()}"


def sanitize_resource_metadata(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    allowed = {
        "authors",
        "year",
        "journal",
        "doi",
        "video_id",
        "channel",
        "duration_hint",
        "license",
        "provenance",
        # The lesson the resource was saved from: the library links back to it.
        "lesson_id",
    }
    clean: dict[str, Any] = {}
    for key in allowed:
        value = raw.get(key)
        if value is None or isinstance(value, (bool, int, float)):
            if value is not None:
                clean[key] = value
        elif isinstance(value, str):
            clean[key] = value[:1000]
        elif isinstance(value, (list, tuple)):
            clean[key] = [str(item)[:300] for item in list(value)[:20]]
    return clean


def resource_key(kind: str, source: str, title: str, url: Optional[str], item: Any = None) -> str:
    """The key a resource is saved under, from what the panel renders.

    The same computation ``SavedResourceRepository.save`` makes, so the panel can
    tell a resource already in the library from one that is not.
    """
    metadata = sanitize_resource_metadata(item if isinstance(item, dict) else {})
    return stable_resource_key(
        kind=(kind or "").strip().lower(),
        source=(source or "unknown").strip().lower()[:64],
        title=(title or "").strip()[:500],
        url=normalize_external_url(url),
        doi=str(metadata.get("doi") or "") or None,
        video_id=str(metadata.get("video_id") or "") or None,
    )


async def saved_keys_for_owner(session: AsyncSession, owner_id: uuid.UUID) -> set[str]:
    """Every key this owner has saved: what the panel marks as already saved."""
    rows = await session.execute(
        select(SavedResource.stable_key).where(SavedResource.owner_id == owner_id)
    )
    return set(rows.scalars().all())


async def saved_keys_safe(session: AsyncSession, owner_id: Any) -> set[str]:
    """``saved_keys_for_owner`` for a render path: an empty set rather than an error."""
    try:
        return await saved_keys_for_owner(session, owner_id)
    except Exception:  # noqa: BLE001 — a page is not lost over a bookmark's colour
        logger.warning("[saved-resources] could not read the saved keys", exc_info=True)
        return set()


class SavedResourceRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(
        self,
        *,
        owner_id: uuid.UUID,
        kind: str,
        source: str,
        title: str,
        url: Optional[str],
        metadata: Any = None,
    ) -> SavedResource:
        kind = kind.strip().lower()
        if kind not in RESOURCE_KINDS:
            raise ValueError("Tipo di risorsa non valido")
        source = (source or "unknown").strip().lower()[:64]
        title = title.strip()[:500]
        if not title:
            raise ValueError("Titolo mancante")
        normalized_url = normalize_external_url(url)
        clean_metadata = sanitize_resource_metadata(metadata)
        key = stable_resource_key(
            kind=kind,
            source=source,
            title=title,
            url=normalized_url,
            doi=str(clean_metadata.get("doi") or "") or None,
            video_id=str(clean_metadata.get("video_id") or "") or None,
        )

        existing = await self._session.scalar(
            select(SavedResource).where(
                SavedResource.owner_id == owner_id,
                SavedResource.stable_key == key,
            )
        )
        if existing is not None:
            return existing

        try:
            async with self._session.begin_nested():
                saved = SavedResource(
                    owner_id=owner_id,
                    kind=kind,
                    source=source,
                    stable_key=key,
                    title=title,
                    url=normalized_url,
                    metadata_json=clean_metadata or None,
                )
                self._session.add(saved)
                await self._session.flush()
                return saved
        except IntegrityError:
            # Another request saved the same owner/key between the read and the
            # insert. The savepoint keeps the caller's transaction usable.
            existing = await self._session.scalar(
                select(SavedResource).where(
                    SavedResource.owner_id == owner_id,
                    SavedResource.stable_key == key,
                )
            )
            if existing is None:
                raise
            return existing

    async def list_for_owner(
        self,
        owner_id: uuid.UUID,
        *,
        limit: int = 200,
        offset: int = 0,
    ) -> list[SavedResource]:
        result = await self._session.execute(
            select(SavedResource)
            .where(SavedResource.owner_id == owner_id)
            .order_by(SavedResource.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars())

    async def get(
        self,
        resource_id: uuid.UUID,
        *,
        owner_id: uuid.UUID,
    ) -> SavedResource | None:
        return await self._session.scalar(
            select(SavedResource).where(
                SavedResource.id == resource_id,
                SavedResource.owner_id == owner_id,
            )
        )

    async def delete(
        self,
        resource_id: uuid.UUID,
        *,
        owner_id: uuid.UUID,
    ) -> bool:
        result = await self._session.execute(
            delete(SavedResource).where(
                SavedResource.id == resource_id,
                SavedResource.owner_id == owner_id,
            )
        )
        return bool(getattr(result, "rowcount", 0))
