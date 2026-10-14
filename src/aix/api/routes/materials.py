"""Authenticated JSON API for the private material library.

This surface accepts only a real teacher JWT/cookie.  The service-account Basic
Auth used by the stateless agent endpoint cannot identify a material owner and
is therefore intentionally insufficient here.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from aix.api.schemas.materials import (
    MaterialAttachRequest,
    MaterialListResponse,
    MaterialResponse,
    MaterialVersionResponse,
    SavedResourceCreate,
    SavedResourceResponse,
)
from aix.webui.auth.dependencies import current_active_user
from aix.webui.auth.models import User
from aix.webui.db import get_async_session
from aix.webui.features import material_library_enabled
from aix.webui.lessons.models import Lesson
from aix.webui.lessons.uploads import MAX_FILE_BYTES, MAX_FILES_PER_LESSON, too_large_message
from aix.webui.materials.handlers import PARSER_PROFILE
from aix.webui.materials.models import (
    MATERIAL_PURPOSES,
    IngestionJob,
    Material,
    MaterialVersion,
)
from aix.webui.materials.parsing.intake import IntakeRejected, check_upload
from aix.webui.materials.repository import (
    READY_STATUSES,
    AssetBlobRepository,
    BindingRepository,
    MaterialRepository,
    MaterialService,
    MaterialVersionRepository,
)
from aix.webui.materials.saved_resources import SavedResourceRepository
from aix.webui.materials.shadow import retire_material
from aix.webui.materials.storage import StorageError, get_storage_backend

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/materials", tags=["materials"])


async def _accepted_upload(file: UploadFile) -> tuple[bytes, str, str]:
    filename = (file.filename or "materiale")[:240]
    content = await file.read(MAX_FILE_BYTES + 1)
    await file.close()
    if not content:
        raise HTTPException(status_code=422, detail="Empty file")
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail=too_large_message())
    try:
        accepted = check_upload(filename, content)
    except IntakeRejected as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return content, filename, accepted.media_type


async def _add_lesson_manifest_entry(
    session: AsyncSession,
    *,
    lesson: Lesson,
    material: Material,
    version: MaterialVersion,
    owner_id: uuid.UUID,
) -> None:
    """Keep API-created bindings visible in the existing WebUI attachment panel."""
    manifest = list(lesson.uploaded_files_json or [])
    if any(
        isinstance(entry, dict) and str(entry.get("material_id")) == str(material.id)
        for entry in manifest
    ):
        return
    if len(manifest) >= MAX_FILES_PER_LESSON:
        raise HTTPException(
            status_code=409,
            detail=f"Too many lesson attachments (maximum {MAX_FILES_PER_LESSON})",
        )
    blob = await AssetBlobRepository(session).get(version.blob_id, owner_id=owner_id)
    manifest.append(
        {
            "id": str(uuid.uuid4()),
            "filename": material.display_name,
            "mime": version.media_type,
            "size": blob.size_bytes if blob is not None else 0,
            "text_excerpt": "",
            "material_id": str(material.id),
            "material_version_id": str(version.id),
            "processing_status": version.status,
            "library_attached": True,
        }
    )
    lesson.uploaded_files_json = manifest


async def _serialize_material(
    session: AsyncSession,
    material: Material,
    *,
    owner_id: uuid.UUID,
) -> MaterialResponse:
    versions = await MaterialVersionRepository(session).list_for_material(
        material.id,
        owner_id=owner_id,
    )
    bindings = await BindingRepository(session).list_for_material(
        material.id,
        owner_id=owner_id,
        enabled_only=True,
    )
    version_models = [MaterialVersionResponse.model_validate(version) for version in versions]
    return MaterialResponse(
        id=material.id,
        display_name=material.display_name,
        purpose=material.purpose,
        created_at=material.created_at,
        latest_version=version_models[0] if version_models else None,
        versions=version_models,
        attached_lesson_ids=[binding.lesson_id for binding in bindings],
    )


@router.get("", response_model=MaterialListResponse)
async def list_materials(
    limit: int = 100,
    offset: int = 0,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> MaterialListResponse:
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    repository = MaterialRepository(session)
    materials = await repository.list_for_owner(
        user.id,
        limit=limit,
        offset=offset,
    )
    items = [
        await _serialize_material(session, material, owner_id=user.id) for material in materials
    ]
    return MaterialListResponse(
        items=items,
        total=await repository.count_for_owner(user.id),
    )


@router.post(
    "",
    response_model=MaterialResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_material(
    file: UploadFile = File(...),
    display_name: Optional[str] = Form(None),
    purpose: str = Form("reference"),
    lesson_id: Optional[uuid.UUID] = Form(None),
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> MaterialResponse:
    if not material_library_enabled():
        raise HTTPException(status_code=404)
    purpose = purpose.strip().lower()
    if purpose not in MATERIAL_PURPOSES:
        raise HTTPException(status_code=422, detail="Invalid material purpose")

    lesson = None
    if lesson_id is not None:
        lesson = await session.scalar(
            select(Lesson).where(Lesson.id == lesson_id, Lesson.owner_id == user.id)
        )
        if lesson is None:
            raise HTTPException(status_code=404, detail="Lesson not found")
        if lesson.status == "running":
            raise HTTPException(status_code=409, detail="Lesson generation is running")

    content, filename, media_type = await _accepted_upload(file)
    material, version = await MaterialService(session, get_storage_backend()).store_new_material(
        content,
        owner_id=user.id,
        display_name=(display_name or filename).strip()[:240] or filename,
        media_type=media_type,
        purpose=purpose,
    )
    if lesson is not None:
        await BindingRepository(session).bind(
            lesson_id=lesson.id,
            material_id=material.id,
            owner_id=user.id,
            selected_version_id=version.id,
            created_by=user.id,
            mode=purpose,
        )
        await _add_lesson_manifest_entry(
            session,
            lesson=lesson,
            material=material,
            version=version,
            owner_id=user.id,
        )
    session.add(
        IngestionJob(
            owner_id=user.id,
            material_version_id=version.id,
            parser_profile_version=PARSER_PROFILE,
        )
    )
    await session.commit()
    return await _serialize_material(session, material, owner_id=user.id)


@router.get("/saved/resources", response_model=list[SavedResourceResponse])
async def list_saved_resources(
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> list[SavedResourceResponse]:
    rows = await SavedResourceRepository(session).list_for_owner(user.id)
    return [SavedResourceResponse.model_validate(row) for row in rows]


@router.post(
    "/saved/resources",
    response_model=SavedResourceResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_saved_resource(
    body: SavedResourceCreate,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> SavedResourceResponse:
    try:
        saved = await SavedResourceRepository(session).save(
            owner_id=user.id,
            kind=body.kind,
            source=body.source,
            title=body.title,
            url=body.url,
            metadata=body.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    await session.commit()
    return SavedResourceResponse.model_validate(saved)


@router.delete(
    "/saved/resources/{resource_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def remove_saved_resource(
    resource_id: uuid.UUID,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> Response:
    removed = await SavedResourceRepository(session).delete(
        resource_id,
        owner_id=user.id,
    )
    if not removed:
        raise HTTPException(status_code=404, detail="Resource not found")
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{material_id}", response_model=MaterialResponse)
async def get_material(
    material_id: uuid.UUID,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> MaterialResponse:
    material = await MaterialRepository(session).get(material_id, owner_id=user.id)
    if material is None:
        raise HTTPException(status_code=404, detail="Material not found")
    return await _serialize_material(session, material, owner_id=user.id)


@router.post(
    "/{material_id}/versions",
    response_model=MaterialResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_material_version(
    material_id: uuid.UUID,
    file: UploadFile = File(...),
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> MaterialResponse:
    if not material_library_enabled():
        raise HTTPException(status_code=404)
    material = await MaterialRepository(session).get(material_id, owner_id=user.id)
    if material is None:
        raise HTTPException(status_code=404, detail="Material not found")
    content, _filename, media_type = await _accepted_upload(file)
    version = await MaterialService(session, get_storage_backend()).add_version(
        content,
        material_id=material.id,
        owner_id=user.id,
        media_type=media_type,
    )
    if version is None:
        raise HTTPException(status_code=404, detail="Material not found")
    session.add(
        IngestionJob(
            owner_id=user.id,
            material_version_id=version.id,
            parser_profile_version=PARSER_PROFILE,
        )
    )
    await session.commit()
    return await _serialize_material(session, material, owner_id=user.id)


@router.get("/{material_id}/versions/{version_id}/source")
async def download_material_version(
    material_id: uuid.UUID,
    version_id: uuid.UUID,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> Response:
    material = await MaterialRepository(session).get(material_id, owner_id=user.id)
    version = await MaterialVersionRepository(session).get(version_id, owner_id=user.id)
    if material is None or version is None or version.material_id != material.id:
        raise HTTPException(status_code=404, detail="Material not found")
    try:
        content = await MaterialService(session, get_storage_backend()).read(
            version,
            owner_id=user.id,
        )
    except StorageError as exc:
        logger.warning(
            "[materials-api] source unavailable material=%s version=%s: %s",
            material_id,
            version_id,
            exc,
        )
        raise HTTPException(status_code=410, detail="Source file unavailable") from exc
    if content is None:
        raise HTTPException(status_code=404, detail="Material not found")

    disposition = f"attachment; filename*=UTF-8''{quote(material.display_name, safe='')}"
    return Response(
        content=content,
        media_type=version.media_type or "application/octet-stream",
        headers={
            "Content-Disposition": disposition,
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post("/{material_id}/attach", response_model=MaterialResponse)
async def attach_material(
    material_id: uuid.UUID,
    body: MaterialAttachRequest,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> MaterialResponse:
    if not material_library_enabled():
        raise HTTPException(status_code=404)

    material = await MaterialRepository(session).get(material_id, owner_id=user.id)
    lesson = await session.scalar(
        select(Lesson).where(Lesson.id == body.lesson_id, Lesson.owner_id == user.id)
    )
    if material is None or lesson is None:
        raise HTTPException(status_code=404, detail="Lesson or material not found")
    if lesson.status == "running":
        raise HTTPException(status_code=409, detail="Lesson generation is running")

    versions = await MaterialVersionRepository(session).list_for_material(
        material.id,
        owner_id=user.id,
    )
    version: Optional[MaterialVersion]
    if body.version_id is not None:
        version = next((item for item in versions if item.id == body.version_id), None)
    else:
        version = next((item for item in versions if item.status in READY_STATUSES), None)
    if version is None or version.status not in READY_STATUSES:
        raise HTTPException(status_code=409, detail="Material version is not ready")

    binding = await BindingRepository(session).bind(
        lesson_id=lesson.id,
        material_id=material.id,
        owner_id=user.id,
        selected_version_id=version.id,
        created_by=user.id,
        mode=material.purpose,
    )
    if binding is None:
        raise HTTPException(status_code=404, detail="Lesson or material not found")
    await _add_lesson_manifest_entry(
        session,
        lesson=lesson,
        material=material,
        version=version,
        owner_id=user.id,
    )
    await session.commit()
    return await _serialize_material(session, material, owner_id=user.id)


@router.delete(
    "/{material_id}/lessons/{lesson_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def detach_material(
    material_id: uuid.UUID,
    lesson_id: uuid.UUID,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> Response:
    if not material_library_enabled():
        raise HTTPException(status_code=404)

    material = await MaterialRepository(session).get(material_id, owner_id=user.id)
    lesson = await session.scalar(
        select(Lesson).where(Lesson.id == lesson_id, Lesson.owner_id == user.id)
    )
    if material is None or lesson is None:
        raise HTTPException(status_code=404, detail="Lesson or material not found")
    if lesson.status == "running":
        raise HTTPException(status_code=409, detail="Lesson generation is running")

    detached = await BindingRepository(session).set_enabled(
        lesson_id=lesson.id,
        material_id=material.id,
        owner_id=user.id,
        enabled=False,
    )
    if not detached:
        raise HTTPException(status_code=404, detail="Material is not attached to lesson")
    lesson.uploaded_files_json = [
        entry
        for entry in list(lesson.uploaded_files_json or [])
        if not (isinstance(entry, dict) and str(entry.get("material_id")) == str(material.id))
    ]
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/{material_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_material(
    material_id: uuid.UUID,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> Response:
    if not material_library_enabled():
        raise HTTPException(status_code=404)
    material = await MaterialRepository(session).get(material_id, owner_id=user.id)
    if material is None:
        raise HTTPException(status_code=404, detail="Material not found")
    await session.commit()
    await retire_material(material_id=str(material.id), owner_id=user.id, force=True)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
