"""Durable, bounded snapshots of the lesson media panel.

The retriever and the live-media service return richer dictionaries than the
panel renders.  Persisting those dictionaries wholesale would turn provider
responses into an unbounded database contract.  This module keeps only the
fields the UI and resource-bookmarking flow use, with the same per-section
limits as the templates.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from aix.webui.lessons.models import LessonMediaSnapshot, LessonMessage

logger = logging.getLogger(__name__)

SNAPSHOT_SCHEMA_VERSION = 1

_CURATED_FIELDS: dict[str, tuple[set[str], int]] = {
    "videos": (
        {
            "title",
            "url",
            "suggested_url",
            "duration_hint",
            "search_query",
            "video_id",
            "channel",
            "thumbnail_url",
            "source",
        },
        6,
    ),
    "resources": (
        {"title", "url", "suggested_url", "source", "license", "description"},
        4,
    ),
    "citations": (
        {"title", "authors", "year", "journal", "doi", "url", "source"},
        5,
    ),
    "open_textbooks": (
        {"title", "chapter", "source", "license", "url", "suggested_url"},
        3,
    ),
    "web_links": (
        {"title", "url", "snippet", "source"},
        9,
    ),
}

_LIVE_FIELDS: dict[str, tuple[set[str], int]] = {
    "videos": (
        {
            "title",
            "url",
            "channel",
            "duration_hint",
            "video_id",
            "thumbnail_url",
            "source",
        },
        6,
    ),
    "citations": (
        {"title", "authors", "year", "journal", "doi", "url", "source"},
        5,
    ),
    "wikipedia": (
        {"title", "summary", "url", "thumbnail_url", "source"},
        4,
    ),
}


def _safe_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:4000]
    if isinstance(value, (list, tuple)):
        return [_safe_scalar(item) for item in list(value)[:20]]
    return str(value)[:4000]


def _sanitize(
    media: Any,
    contract: dict[str, tuple[set[str], int]],
) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(media, dict):
        return {}

    snapshot: dict[str, list[dict[str, Any]]] = {}
    for bucket, (allowed, limit) in contract.items():
        raw_items = media.get(bucket)
        if not isinstance(raw_items, list):
            continue
        items: list[dict[str, Any]] = []
        for raw in raw_items[:limit]:
            if not isinstance(raw, dict):
                continue
            clean = {key: _safe_scalar(raw[key]) for key in allowed if key in raw}
            if clean:
                items.append(clean)
        if items:
            snapshot[bucket] = items
    return snapshot


def sanitize_curated_media(media: Any) -> dict[str, list[dict[str, Any]]]:
    return _sanitize(media, _CURATED_FIELDS)


def sanitize_live_media(media: Any) -> dict[str, list[dict[str, Any]]]:
    return _sanitize(media, _LIVE_FIELDS)


async def latest_user_turn(session: AsyncSession, lesson_id: uuid.UUID) -> int:
    return int(
        (
            await session.scalar(
                select(func.max(LessonMessage.turn_index)).where(
                    LessonMessage.lesson_id == lesson_id,
                    LessonMessage.role == "user",
                )
            )
        )
        or 1
    )


async def latest_display_snapshot(
    session: AsyncSession,
    *,
    lesson_id: uuid.UUID,
    owner_id: uuid.UUID,
) -> LessonMediaSnapshot | None:
    result = await session.execute(
        select(LessonMediaSnapshot)
        .where(
            LessonMediaSnapshot.lesson_id == lesson_id,
            LessonMediaSnapshot.owner_id == owner_id,
            LessonMediaSnapshot.curated_json.is_not(None),
        )
        .order_by(LessonMediaSnapshot.turn_index.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def latest_display_snapshot_safe(
    session: AsyncSession,
    *,
    lesson_id: uuid.UUID,
    owner_id: uuid.UUID,
) -> LessonMediaSnapshot | None:
    """Treat an unavailable additive table like a legacy lesson with no snapshot."""
    try:
        async with session.begin_nested():
            return await latest_display_snapshot(
                session,
                lesson_id=lesson_id,
                owner_id=owner_id,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[media-snapshot] replay unavailable for lesson=%s: %s", lesson_id, exc)
        return None


async def snapshot_for_turn(
    session: AsyncSession,
    *,
    lesson_id: uuid.UUID,
    owner_id: uuid.UUID,
    turn_index: int,
) -> LessonMediaSnapshot | None:
    result = await session.execute(
        select(LessonMediaSnapshot).where(
            LessonMediaSnapshot.lesson_id == lesson_id,
            LessonMediaSnapshot.owner_id == owner_id,
            LessonMediaSnapshot.turn_index == turn_index,
        )
    )
    return result.scalar_one_or_none()


async def snapshot_for_turn_safe(
    session: AsyncSession,
    *,
    lesson_id: uuid.UUID,
    owner_id: uuid.UUID,
    turn_index: int,
) -> LessonMediaSnapshot | None:
    try:
        async with session.begin_nested():
            return await snapshot_for_turn(
                session,
                lesson_id=lesson_id,
                owner_id=owner_id,
                turn_index=turn_index,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[media-snapshot] turn replay unavailable for lesson=%s: %s", lesson_id, exc)
        return None


async def save_curated_snapshot(
    session: AsyncSession,
    *,
    lesson_id: uuid.UUID,
    owner_id: uuid.UUID,
    turn_index: int,
    media: Any,
) -> None:
    await _upsert_snapshot(
        session,
        lesson_id=lesson_id,
        owner_id=owner_id,
        turn_index=turn_index,
        values={"curated_json": sanitize_curated_media(media)},
    )


async def save_live_snapshot(
    session: AsyncSession,
    *,
    lesson_id: uuid.UUID,
    owner_id: uuid.UUID,
    turn_index: int,
    media: Any,
) -> None:
    await _upsert_snapshot(
        session,
        lesson_id=lesson_id,
        owner_id=owner_id,
        turn_index=turn_index,
        values={
            "live_json": sanitize_live_media(media),
            "live_fetched": True,
        },
    )


async def _upsert_snapshot(
    session: AsyncSession,
    *,
    lesson_id: uuid.UUID,
    owner_id: uuid.UUID,
    turn_index: int,
    values: dict[str, Any],
) -> None:
    """Update one side of a snapshot without overwriting the other side."""

    insert_values = {
        "lesson_id": lesson_id,
        "owner_id": owner_id,
        "turn_index": turn_index,
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        **values,
    }
    update_values = {
        "owner_id": owner_id,
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "updated_at": func.now(),
        **values,
    }

    dialect = session.get_bind().dialect.name
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        pg_statement = pg_insert(LessonMediaSnapshot).values(**insert_values)
        pg_statement = pg_statement.on_conflict_do_update(
            index_elements=["lesson_id", "turn_index"],
            set_=update_values,
        )
        await session.execute(pg_statement)
        return

    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        sqlite_statement = sqlite_insert(LessonMediaSnapshot).values(**insert_values)
        sqlite_statement = sqlite_statement.on_conflict_do_update(
            index_elements=["lesson_id", "turn_index"],
            set_=update_values,
        )
        await session.execute(sqlite_statement)
        return

    snapshot = await snapshot_for_turn(
        session,
        lesson_id=lesson_id,
        owner_id=owner_id,
        turn_index=turn_index,
    )
    if snapshot is None:
        session.add(LessonMediaSnapshot(**insert_values))
        return
    for key, value in update_values.items():
        if key != "updated_at":
            setattr(snapshot, key, value)
