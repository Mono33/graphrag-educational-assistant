"""Authenticated JSON API for the private material library.

Every route acts as a user (``aix.api.principals.current_owner``): a teacher with
their own JWT/cookie is themselves; the service account of a chatbot platform
names the person it acts for with ``X-End-User-Id``, which is its own owner. So
the owner filter that keeps one teacher's documents from another's keeps one
end user's from another's too (2026-10-14: the service account used to be
refused here, because it could not identify an owner).

Errors carry ``{"code", "message"}``: the code for a client, the sentence for a
person. Uploads, versions, attach and detach need ``AIX_MATERIAL_LIBRARY_ENABLED``.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Optional
from urllib.parse import quote

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from aix.api import idempotency
from aix.api.principals import current_owner
from aix.api.schemas.materials import (
    MaterialAttachRequest,
    MaterialListResponse,
    MaterialResponse,
    MaterialVersionResponse,
    SavedResourceCreate,
    SavedResourceResponse,
)
from aix.webui.auth.models import User
from aix.webui.db import get_async_session
from aix.webui.features import material_library_enabled
from aix.webui.lessons.models import Lesson
from aix.webui.lessons.uploads import MAX_FILE_BYTES, MAX_FILES_PER_LESSON, too_large_message
from aix.webui.materials import library
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


def _error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})


LIBRARY_OFF = "La libreria dei documenti non è attiva su questo server."


async def _accepted_upload(file: UploadFile) -> tuple[bytes, str, str]:
    filename = (file.filename or "materiale")[:240]
    content = await file.read(MAX_FILE_BYTES + 1)
    await file.close()
    if not content:
        raise _error(422, "empty", "File vuoto.")
    if len(content) > MAX_FILE_BYTES:
        raise _error(413, "too_large", too_large_message())
    try:
        accepted = check_upload(filename, content)
    except IntakeRejected as exc:
        raise _error(422, exc.code, str(exc)) from exc
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
        raise _error(
            409,
            "too_many_attachments",
            f"Troppi documenti nella lezione (massimo {MAX_FILES_PER_LESSON}).",
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
    # What a client polls for: processing → ready | failed, with the passages a
    # session can search and, for a failure, why (``materials.library``).
    state = await library.list_materials(session, owner_id=owner_id, material_ids=[material.id])
    return MaterialResponse(
        status=state[0].status if state else library.public_status(None),
        passages=state[0].passages if state else 0,
        error_code=state[0].error_code if state else None,
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
    user: User = Depends(current_owner),
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
    user: User = Depends(current_owner),
    session: AsyncSession = Depends(get_async_session),
    idempotency_key: Optional[str] = Header(default=None, alias=idempotency.HEADER),
) -> Response:
    if not material_library_enabled():
        raise _error(404, "library_disabled", LIBRARY_OFF)
    purpose = purpose.strip().lower()
    if purpose not in MATERIAL_PURPOSES:
        raise _error(422, "invalid_purpose", "Scopo del documento non valido.")

    lesson = None
    if lesson_id is not None:
        lesson = await session.scalar(
            select(Lesson).where(Lesson.id == lesson_id, Lesson.owner_id == user.id)
        )
        if lesson is None:
            raise _error(404, "unknown_lesson", "Lezione non trovata.")
        if lesson.status == "running":
            raise _error(
                409, "already_running", "Una generazione è in corso: riprova appena finisce."
            )

    content, filename, media_type = await _accepted_upload(file)
    claim = await idempotency.begin(
        session,
        owner_id=user.id,
        scope="material",
        key=idempotency_key,
        fingerprint=idempotency.request_hash(
            hashlib.sha256(content).hexdigest(), display_name, purpose, lesson_id
        ),
    )
    if isinstance(claim, idempotency.Replay):
        # The document as it is now: "processing" then may be "ready" today.
        stored = await MaterialRepository(session).get(
            uuid.UUID(str(claim.body["id"])), owner_id=user.id
        )
        body = (
            (await _serialize_material(session, stored, owner_id=user.id)).model_dump(mode="json")
            if stored is not None
            else claim.body
        )
        return JSONResponse(
            body, status_code=claim.status_code, headers={idempotency.REPLAYED_HEADER: "true"}
        )
    try:
        out = await _store_upload(
            session, user, lesson, content, filename, media_type, display_name, purpose
        )
    except Exception:
        await idempotency.abandon(session, claim)
        raise
    body = out.model_dump(mode="json")
    await idempotency.finish(session, claim, 201, body)
    return JSONResponse(body, status_code=201)


async def _store_upload(
    session: AsyncSession,
    user: User,
    lesson: Optional[Lesson],
    content: bytes,
    filename: str,
    media_type: str,
    display_name: Optional[str],
    purpose: str,
) -> MaterialResponse:
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
    user: User = Depends(current_owner),
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
    user: User = Depends(current_owner),
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
        raise _error(422, "invalid_resource", str(exc)) from exc
    await session.commit()
    return SavedResourceResponse.model_validate(saved)


@router.delete(
    "/saved/resources/{resource_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def remove_saved_resource(
    resource_id: uuid.UUID,
    user: User = Depends(current_owner),
    session: AsyncSession = Depends(get_async_session),
) -> Response:
    removed = await SavedResourceRepository(session).delete(
        resource_id,
        owner_id=user.id,
    )
    if not removed:
        raise _error(404, "unknown_resource", "Risorsa non trovata.")
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{material_id}", response_model=MaterialResponse)
async def get_material(
    material_id: uuid.UUID,
    user: User = Depends(current_owner),
    session: AsyncSession = Depends(get_async_session),
) -> MaterialResponse:
    material = await MaterialRepository(session).get(material_id, owner_id=user.id)
    if material is None:
        raise _error(404, "unknown_material", "Documento non trovato.")
    return await _serialize_material(session, material, owner_id=user.id)


@router.post(
    "/{material_id}/versions",
    response_model=MaterialResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_material_version(
    material_id: uuid.UUID,
    file: UploadFile = File(...),
    user: User = Depends(current_owner),
    session: AsyncSession = Depends(get_async_session),
) -> MaterialResponse:
    if not material_library_enabled():
        raise _error(404, "library_disabled", LIBRARY_OFF)
    material = await MaterialRepository(session).get(material_id, owner_id=user.id)
    if material is None:
        raise _error(404, "unknown_material", "Documento non trovato.")
    content, _filename, media_type = await _accepted_upload(file)
    version = await MaterialService(session, get_storage_backend()).add_version(
        content,
        material_id=material.id,
        owner_id=user.id,
        media_type=media_type,
    )
    if version is None:
        raise _error(404, "unknown_material", "Documento non trovato.")
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
    user: User = Depends(current_owner),
    session: AsyncSession = Depends(get_async_session),
) -> Response:
    material = await MaterialRepository(session).get(material_id, owner_id=user.id)
    version = await MaterialVersionRepository(session).get(version_id, owner_id=user.id)
    if material is None or version is None or version.material_id != material.id:
        raise _error(404, "unknown_material", "Documento non trovato.")
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
        raise _error(410, "source_unavailable", "Il file originale non è più disponibile.") from exc
    if content is None:
        raise _error(404, "unknown_material", "Documento non trovato.")

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
    user: User = Depends(current_owner),
    session: AsyncSession = Depends(get_async_session),
) -> MaterialResponse:
    if not material_library_enabled():
        raise _error(404, "library_disabled", LIBRARY_OFF)

    material = await MaterialRepository(session).get(material_id, owner_id=user.id)
    lesson = await session.scalar(
        select(Lesson).where(Lesson.id == body.lesson_id, Lesson.owner_id == user.id)
    )
    if material is None or lesson is None:
        raise _error(404, "unknown_lesson_or_material", "Lezione o documento non trovati.")
    if lesson.status == "running":
        raise _error(409, "already_running", "Una generazione è in corso: riprova appena finisce.")

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
        raise _error(409, "version_not_ready", "Il documento non è ancora pronto.")

    binding = await BindingRepository(session).bind(
        lesson_id=lesson.id,
        material_id=material.id,
        owner_id=user.id,
        selected_version_id=version.id,
        created_by=user.id,
        mode=material.purpose,
    )
    if binding is None:
        raise _error(404, "unknown_lesson_or_material", "Lezione o documento non trovati.")
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
    user: User = Depends(current_owner),
    session: AsyncSession = Depends(get_async_session),
) -> Response:
    if not material_library_enabled():
        raise _error(404, "library_disabled", LIBRARY_OFF)

    material = await MaterialRepository(session).get(material_id, owner_id=user.id)
    lesson = await session.scalar(
        select(Lesson).where(Lesson.id == lesson_id, Lesson.owner_id == user.id)
    )
    if material is None or lesson is None:
        raise _error(404, "unknown_lesson_or_material", "Lezione o documento non trovati.")
    if lesson.status == "running":
        raise _error(409, "already_running", "Una generazione è in corso: riprova appena finisce.")

    detached = await BindingRepository(session).set_enabled(
        lesson_id=lesson.id,
        material_id=material.id,
        owner_id=user.id,
        enabled=False,
    )
    if not detached:
        raise _error(404, "not_attached", "Il documento non è collegato a questa lezione.")
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
    user: User = Depends(current_owner),
    session: AsyncSession = Depends(get_async_session),
) -> Response:
    if not material_library_enabled():
        raise _error(404, "library_disabled", LIBRARY_OFF)
    material = await MaterialRepository(session).get(material_id, owner_id=user.id)
    if material is None:
        raise _error(404, "unknown_material", "Documento non trovato.")
    await session.commit()
    await retire_material(material_id=str(material.id), owner_id=user.id, force=True)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
