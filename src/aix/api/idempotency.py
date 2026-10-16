"""A retried request gets the first answer, not a second run (2026-10-16).

AIxLearning sends its model calls from a Dramatiq worker that retries a failed
job up to three times. Without a key, a turn that timed out on the client side
but finished on ours runs again when retried: a second lesson, a second bill,
and sometimes a ``409 already_running`` instead of the lesson that is about to
be ready. ``docs/product/Analisi_Integrazione_AIxLearning.md`` §3 E.

With an ``Idempotency-Key`` header, the first request with a key is recorded:

* while it runs, the same key answers ``409 idempotency_in_progress`` with a
  ``Retry-After``. The client waits and retries, or reads the session;
* once it has succeeded, the same key answers with the **stored response** and
  ``Idempotent-Replayed: true``, without running anything;
* if it failed, the record is dropped and the same key may try again: a failure
  is not an answer worth replaying;
* the same key with a different body is ``422 idempotency_key_reused``. That is
  a client bug, and replaying the other request's answer would hide it.

Keys belong to the owner (an end user, through the service account) and to the
operation, and last :data:`TTL`. A record still ``running`` after
:data:`STALE_AFTER` is from a run that crashed, and the key is free again.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import HTTPException
from fastapi_users_db_sqlalchemy.generics import GUID
from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    delete,
    func,
    select,
    update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from aix.webui.db import Base

#: The request header, as in the IETF draft and the payment APIs that use it.
HEADER = "Idempotency-Key"
#: Set on a response that was replayed rather than produced.
REPLAYED_HEADER = "Idempotent-Replayed"

KEY_MAX = 200
TTL = timedelta(hours=24)
#: A lesson takes a minute or two; a record this old and still running is orphaned.
STALE_AFTER = timedelta(minutes=15)
RETRY_AFTER_SECONDS = 5


class ApiIdempotencyKey(Base):
    __tablename__ = "api_idempotency_key"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("user.id", ondelete="CASCADE"), nullable=False
    )
    #: The operation and its target, e.g. ``turn:<session id>``.
    scope: Mapped[str] = mapped_column(String(120), nullable=False)
    key: Mapped[str] = mapped_column(String(KEY_MAX), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    #: ``running`` or ``done``.
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    status_code: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    response_json: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("owner_id", "scope", "key", name="uq_api_idempotency_owner_scope_key"),
        Index("ix_api_idempotency_created", "created_at"),
    )


@dataclass(frozen=True)
class Claim:
    """This request holds the key: :func:`finish` or :func:`abandon` it."""

    id: uuid.UUID


@dataclass(frozen=True)
class Replay:
    """The stored answer to a request already served."""

    status_code: int
    body: Any


def request_hash(*parts: Any) -> str:
    """A fingerprint of what was asked, to tell a retry from a different request."""
    canonical = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _error(status_code: int, code: str, message: str, retry_after: Optional[int] = None):
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
        headers={"Retry-After": str(retry_after)} if retry_after else None,
    )


def check_key(key: Optional[str]) -> Optional[str]:
    if key is None or not key.strip():
        return None
    key = key.strip()
    if len(key) > KEY_MAX or not key.isprintable():
        raise _error(
            422,
            "invalid_idempotency_key",
            f"Idempotency-Key: testo stampabile, al massimo {KEY_MAX} caratteri.",
        )
    return key


def _aware(moment: datetime) -> datetime:
    # SQLite gives naive datetimes back; Postgres gives aware ones.
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


async def begin(
    session: AsyncSession,
    *,
    owner_id: uuid.UUID,
    scope: str,
    key: Optional[str],
    fingerprint: str,
) -> Optional[Replay | Claim]:
    """Claim ``key`` for this request.

    ``None``: no key was sent, go ahead without one. A :class:`Replay`: answer
    with it. A :class:`Claim`: go ahead, then :func:`finish` or :func:`abandon`
    it. Raises the 409 and 422 described in the module docstring.
    """
    key = check_key(key)
    if key is None:
        return None
    now = datetime.now(timezone.utc)

    # Expired keys of this owner go first: the table stays as small as its TTL.
    await session.execute(
        delete(ApiIdempotencyKey).where(
            ApiIdempotencyKey.owner_id == owner_id, ApiIdempotencyKey.created_at < now - TTL
        )
    )
    record = ApiIdempotencyKey(
        id=uuid.uuid4(),
        owner_id=owner_id,
        scope=scope,
        key=key,
        request_hash=fingerprint,
        state="running",
        created_at=now,
    )
    claim = Claim(record.id)
    try:
        # A savepoint, not the session: rolling the whole session back would
        # expire every object the route already holds (its user, its lesson),
        # and an async session cannot reload them behind the route's back.
        async with session.begin_nested():
            session.add(record)
        await session.commit()
        return claim
    except IntegrityError:
        pass  # the key exists: only the savepoint was rolled back

    existing = await session.scalar(
        select(ApiIdempotencyKey).where(
            ApiIdempotencyKey.owner_id == owner_id,
            ApiIdempotencyKey.scope == scope,
            ApiIdempotencyKey.key == key,
        )
    )
    if existing is None:  # dropped between the insert and the read: claim it again
        return await begin(
            session, owner_id=owner_id, scope=scope, key=key, fingerprint=fingerprint
        )
    if existing.request_hash != fingerprint:
        raise _error(
            422,
            "idempotency_key_reused",
            "Questa Idempotency-Key è già stata usata per una richiesta diversa.",
        )
    if existing.state == "done":
        return Replay(existing.status_code or 200, existing.response_json)
    if now - _aware(existing.created_at) < STALE_AFTER:
        raise _error(
            409,
            "idempotency_in_progress",
            "La stessa richiesta è ancora in corso: riprova tra poco con la stessa chiave.",
            RETRY_AFTER_SECONDS,
        )
    # Orphaned by a crash: this request takes the key over.
    claim = Claim(existing.id)
    existing.created_at = now
    await session.commit()
    return claim


async def finish(
    session: AsyncSession, claim: Optional[Claim], status_code: int, body: Any
) -> None:
    """Remember the answer, so the next request with the key gets it."""
    if not isinstance(claim, Claim):
        return
    await session.execute(
        update(ApiIdempotencyKey)
        .where(ApiIdempotencyKey.id == claim.id)
        .values(state="done", status_code=status_code, response_json=body)
    )
    await session.commit()


async def abandon(session: AsyncSession, claim: Optional[Claim]) -> None:
    """The request failed: free the key for a retry. Best effort."""
    if not isinstance(claim, Claim):
        return
    try:
        await session.rollback()
        await session.execute(delete(ApiIdempotencyKey).where(ApiIdempotencyKey.id == claim.id))
        await session.commit()
    except Exception:  # noqa: BLE001 — the key then frees itself after STALE_AFTER
        await session.rollback()
