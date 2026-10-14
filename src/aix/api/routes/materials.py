"""Documents through the API: upload once, use by id (2026-10-14).

``POST /api/v1/materials`` stores a document in the caller's library and queues
its reading (parse, OCR where needed, chunks, embeddings) -- the same pipeline
and the same checks as the WebUI's paperclip. The answer comes at once with
``status: "processing"``; ``GET /api/v1/materials/{id}`` says when it is
``ready``. A session binds documents by id, and its turns search them.

Every route acts as the user ``aix.api.principals.current_owner`` resolves: a
document is visible only to the end user who uploaded it.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from aix.api.principals import current_owner
from aix.webui.auth.models import User
from aix.webui.db import get_async_session
from aix.webui.materials import library
from aix.webui.materials.parsing.intake import MAX_BYTES, IntakeRejected, too_large_message

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/materials", tags=["materials"])


class MaterialOut(BaseModel):
    """One document of the caller's library."""

    material_id: str
    name: str
    status: str = Field(
        ...,
        description=(
            "processing: being read and indexed · ready: searchable by the sessions it is "
            "bound to · failed: will never be searchable (see error_code)"
        ),
    )
    passages: int = Field(0, description="Searchable passages (chunks) once ready.")
    quality: Optional[dict[str, Any]] = Field(
        None,
        description="What the reader could and could not do: pages read, OCR used, unread pages.",
    )
    error_code: Optional[str] = None
    created_at: Optional[datetime] = None
    sessions: list[str] = Field(default_factory=list, description="Sessions that use it.")


def _out(info: library.MaterialInfo) -> MaterialOut:
    return MaterialOut(
        material_id=info.material_id,
        name=info.name,
        status=info.status,
        passages=info.passages,
        quality=info.quality,
        error_code=info.error_code,
        created_at=info.created_at,
        sessions=info.sessions,
    )


def _error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})


def _material_id(raw: str) -> uuid.UUID:
    try:
        return uuid.UUID(raw)
    except ValueError:
        raise _error(404, "unknown_material", "Documento non trovato.") from None


@router.post(
    "",
    response_model=MaterialOut,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document to the caller's library",
    description=(
        "Multipart upload (field ``file``). Same formats and limits as the WebUI: PDF, DOCX, "
        "PPTX, TXT, Markdown and images, magic bytes checked, macros and encrypted files "
        "refused. Returns at once with ``status: processing``; poll "
        "``GET /api/v1/materials/{id}`` until ``ready``."
    ),
)
async def upload_material(
    file: UploadFile = File(...),
    owner: User = Depends(current_owner),
) -> MaterialOut:
    filename = file.filename or "upload"
    # Measured before it is read: a 400 MB file must not reach RAM to be refused.
    if isinstance(file.size, int) and file.size > MAX_BYTES:
        raise _error(413, "too_large", too_large_message())
    content = await file.read()
    try:
        info = await library.add_material(owner_id=owner.id, filename=filename, content=content)
    except IntakeRejected as exc:
        logger.info("[api.materials] refused %s for owner=%s: %s", filename, owner.id, exc.code)
        raise _error(422, exc.code, str(exc)) from exc
    except library.LibraryUnavailable as exc:
        raise _error(503, exc.code, str(exc)) from exc
    logger.info("[api.materials] owner=%s material=%s queued", owner.id, info.material_id)
    return _out(info)


@router.get("", response_model=list[MaterialOut], summary="The caller's documents")
async def list_materials(
    owner: User = Depends(current_owner),
    session: AsyncSession = Depends(get_async_session),
) -> list[MaterialOut]:
    return [_out(info) for info in await library.list_materials(session, owner_id=owner.id)]


@router.get(
    "/{material_id}", response_model=MaterialOut, summary="One document and its reading state"
)
async def get_material(
    material_id: str,
    owner: User = Depends(current_owner),
    session: AsyncSession = Depends(get_async_session),
) -> MaterialOut:
    try:
        info = await library.get_material(
            session, owner_id=owner.id, material_id=_material_id(material_id)
        )
    except library.UnknownMaterial as exc:
        raise _error(404, exc.code, "Documento non trovato.") from exc
    return _out(info)


@router.delete(
    "/{material_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove a document from retrieval and from the library",
)
async def delete_material(
    material_id: str,
    owner: User = Depends(current_owner),
    session: AsyncSession = Depends(get_async_session),
) -> None:
    try:
        await library.remove_material(
            session, owner_id=owner.id, material_id=_material_id(material_id)
        )
    except library.UnknownMaterial as exc:
        raise _error(404, exc.code, "Documento non trovato.") from exc
