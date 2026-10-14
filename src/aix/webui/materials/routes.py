"""Owner-scoped material library for the standalone WebUI."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from aix.webui.auth.csrf import token_for as _csrf_token_for
from aix.webui.auth.dependencies import optional_current_user
from aix.webui.auth.models import User
from aix.webui.db import get_async_session
from aix.webui.features import material_library_enabled
from aix.webui.features import register as register_feature_flags
from aix.webui.lessons.models import Lesson
from aix.webui.lessons.uploads import MAX_FILES_PER_LESSON
from aix.webui.materials.models import Material
from aix.webui.materials.repository import (
    READY_STATUSES,
    AssetBlobRepository,
    BindingRepository,
    MaterialRepository,
    MaterialService,
    MaterialVersionRepository,
)
from aix.webui.materials.saved_resources import (
    SavedResourceRepository,
    saved_keys_for_owner,
)
from aix.webui.materials.shadow import retire_material
from aix.webui.materials.storage import StorageError, get_storage_backend

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webui", tags=["webui-materials"])

_TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
templates.env.globals["csrf_token"] = _csrf_token_for
register_feature_flags(templates)


def _login_redirect(path: str) -> RedirectResponse:
    return RedirectResponse(f"/auth/login?next={path}", status_code=303)


async def _material_row(
    session: AsyncSession,
    material: Material,
    *,
    owner_id: uuid.UUID,
) -> dict[str, Any]:
    versions = await MaterialVersionRepository(session).list_for_material(
        material.id,
        owner_id=owner_id,
    )
    latest = versions[0] if versions else None
    blob = (
        await AssetBlobRepository(session).get(latest.blob_id, owner_id=owner_id)
        if latest is not None
        else None
    )
    bindings = await BindingRepository(session).list_for_material(
        material.id,
        owner_id=owner_id,
    )
    active = [binding for binding in bindings if binding.enabled]
    lesson_titles: dict[uuid.UUID, str] = {}
    if bindings:
        rows = await session.execute(
            select(Lesson.id, Lesson.title).where(
                Lesson.owner_id == owner_id,
                Lesson.id.in_([binding.lesson_id for binding in bindings]),
            )
        )
        lesson_titles = {lesson_id: title or "Lezione senza titolo" for lesson_id, title in rows}

    return {
        "material": material,
        "versions": versions,
        "latest": latest,
        "size_bytes": blob.size_bytes if blob is not None else None,
        "digest": blob.sha256 if blob is not None else None,
        "bindings": bindings,
        "active_bindings": active,
        "lesson_titles": lesson_titles,
    }


def _same_file_once(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One card per file: the same bytes uploaded to several lessons are one document.

    Each upload is its own material (it has its own versions and bindings), so
    the library showed «guerra-fredda-def.pdf» twice. The newest copy stands for
    the file; the others add their lessons and their count (2026-10-14).
    """
    by_digest: dict[str, dict[str, Any]] = {}
    out: list[dict[str, Any]] = []
    for row in rows:  # newest first
        digest = row.get("digest")
        first = by_digest.get(digest) if digest else None
        if first is None:
            row = {**row, "copies": 1, "all_lessons": dict(row["lesson_titles"])}
            row["active_lesson_ids"] = [b.lesson_id for b in row["active_bindings"]]
            if digest:
                by_digest[digest] = row
            out.append(row)
            continue
        first["copies"] += 1
        first["all_lessons"].update(row["lesson_titles"])
        first["active_lesson_ids"] += [b.lesson_id for b in row["active_bindings"]]
    return out


@router.get("/resources", response_class=HTMLResponse, name="webui_resources")
async def resource_library(
    request: Request,
    user: Optional[User] = Depends(optional_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> Response:
    if user is None:
        return _login_redirect("/webui/resources")

    materials = await MaterialRepository(session).list_for_owner(user.id)
    rows = _same_file_once(
        [await _material_row(session, material, owner_id=user.id) for material in materials]
    )
    saved_resources = await SavedResourceRepository(session).list_for_owner(user.id)
    # The lesson each resource was saved from (kept in its metadata), by title.
    saved_lesson_ids = {
        uuid.UUID(lesson_id)
        for resource in saved_resources
        if (lesson_id := (resource.metadata_json or {}).get("lesson_id"))
    }
    saved_lessons: dict[str, str] = {}
    if saved_lesson_ids:
        found = await session.execute(
            select(Lesson.id, Lesson.title).where(
                Lesson.owner_id == user.id, Lesson.id.in_(saved_lesson_ids)
            )
        )
        saved_lessons = {str(lid): title or "Lezione senza titolo" for lid, title in found}
    return templates.TemplateResponse(
        request,
        "pages/resources.html",
        {
            "title": "Risorse · AixLearning",
            "user": user,
            "active_nav": "resources",
            "rows": rows,
            "saved_resources": saved_resources,
            "saved_lessons": saved_lessons,
        },
    )


@router.get(
    "/materials/{material_id}",
    response_class=HTMLResponse,
    name="webui_material_detail",
)
async def material_detail(
    request: Request,
    material_id: uuid.UUID,
    user: Optional[User] = Depends(optional_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> Response:
    if user is None:
        return _login_redirect(f"/webui/materials/{material_id}")

    material = await MaterialRepository(session).get(material_id, owner_id=user.id)
    if material is None:
        raise HTTPException(status_code=404, detail="Materiale non trovato")

    lessons = list(
        (
            await session.execute(
                select(Lesson).where(Lesson.owner_id == user.id).order_by(Lesson.updated_at.desc())
            )
        ).scalars()
    )
    return templates.TemplateResponse(
        request,
        "pages/material_detail.html",
        {
            "title": f"{material.display_name} · Risorse",
            "user": user,
            "active_nav": "resources",
            "row": await _material_row(session, material, owner_id=user.id),
            "lessons": lessons,
        },
    )


@router.get(
    "/materials/{material_id}/versions/{version_id}/source",
    name="webui_material_download",
)
async def material_download(
    material_id: uuid.UUID,
    version_id: uuid.UUID,
    user: Optional[User] = Depends(optional_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> Response:
    if user is None:
        raise HTTPException(status_code=401)

    material = await MaterialRepository(session).get(material_id, owner_id=user.id)
    version = await MaterialVersionRepository(session).get(version_id, owner_id=user.id)
    if material is None or version is None or version.material_id != material.id:
        raise HTTPException(status_code=404, detail="Materiale non trovato")

    try:
        content = await MaterialService(
            session,
            get_storage_backend(),
        ).read(version, owner_id=user.id)
    except StorageError as exc:
        logger.warning(
            "[materials] source unavailable material=%s version=%s: %s",
            material_id,
            version_id,
            exc,
        )
        raise HTTPException(status_code=410, detail="File originale non disponibile") from exc
    if content is None:
        raise HTTPException(status_code=404, detail="Materiale non trovato")

    ascii_name = (
        "".join(
            character if character.isascii() and character not in '"\\\r\n' else "_"
            for character in material.display_name
        ).strip()
        or "materiale"
    )
    disposition = (
        f'attachment; filename="{ascii_name}"; '
        f"filename*=UTF-8''{quote(material.display_name, safe='')}"
    )
    return Response(
        content=content,
        media_type=version.media_type or "application/octet-stream",
        headers={
            "Content-Disposition": disposition,
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post(
    "/lesson/{lesson_id}/materials/{material_id}/attach",
    name="webui_material_attach",
)
async def attach_existing_material(
    lesson_id: uuid.UUID,
    material_id: uuid.UUID,
    user: Optional[User] = Depends(optional_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> Response:
    if user is None:
        raise HTTPException(status_code=401)
    if not material_library_enabled():
        raise HTTPException(status_code=404)

    lesson = (
        await session.execute(
            select(Lesson).where(Lesson.id == lesson_id, Lesson.owner_id == user.id)
        )
    ).scalar_one_or_none()
    material = await MaterialRepository(session).get(material_id, owner_id=user.id)
    if lesson is None or material is None:
        raise HTTPException(status_code=404, detail="Lezione o materiale non trovato")
    if lesson.status == "running":
        raise HTTPException(status_code=409, detail="La lezione è in generazione")

    versions = await MaterialVersionRepository(session).list_for_material(
        material.id,
        owner_id=user.id,
    )
    version = next((item for item in versions if item.status in READY_STATUSES), None)
    if version is None:
        raise HTTPException(status_code=409, detail="Il materiale non è ancora pronto")

    manifest = list(lesson.uploaded_files_json or [])
    existing = next(
        (
            entry
            for entry in manifest
            if isinstance(entry, dict) and str(entry.get("material_id")) == str(material.id)
        ),
        None,
    )
    if existing is None and len(manifest) >= MAX_FILES_PER_LESSON:
        raise HTTPException(
            status_code=409,
            detail=f"Troppi allegati (massimo {MAX_FILES_PER_LESSON})",
        )

    binding = await BindingRepository(session).bind(
        lesson_id=lesson.id,
        material_id=material.id,
        owner_id=user.id,
        selected_version_id=version.id,
        created_by=user.id,
        mode=material.purpose,
    )
    if binding is None:
        raise HTTPException(status_code=404, detail="Lezione o materiale non trovato")

    if existing is None:
        blob = await AssetBlobRepository(session).get(version.blob_id, owner_id=user.id)
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

    await session.commit()
    return RedirectResponse(f"/webui/lesson/{lesson.id}", status_code=303)


@router.post(
    "/materials/{material_id}/delete",
    name="webui_material_delete",
)
async def delete_material_from_library(
    material_id: uuid.UUID,
    user: Optional[User] = Depends(optional_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> Response:
    if user is None:
        raise HTTPException(status_code=401)
    if not material_library_enabled():
        raise HTTPException(status_code=404)

    material = await MaterialRepository(session).get(material_id, owner_id=user.id)
    if material is None:
        raise HTTPException(status_code=404, detail="Materiale non trovato")

    # End the route session's read transaction before the retirement service
    # opens its independent transaction (important for SQLite development).
    await session.commit()
    await retire_material(material_id=str(material.id), owner_id=user.id, force=True)
    return RedirectResponse("/webui/resources", status_code=303)


@router.post("/resources/saved", name="webui_saved_resource_create")
async def save_external_resource(
    request: Request,
    user: Optional[User] = Depends(optional_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> Response:
    if user is None:
        raise HTTPException(status_code=401)

    form = await request.form()
    metadata = {
        key: form.get(key)
        for key in (
            "doi",
            "video_id",
            "channel",
            "duration_hint",
            "year",
            "journal",
            "license",
            "provenance",
        )
        if form.get(key)
    }
    # The lesson it was saved from, kept only if it is this teacher's.
    lesson_id = None
    try:
        candidate = uuid.UUID(str(form.get("lesson_id") or ""))
    except ValueError:
        candidate = None
    if candidate is not None and await session.scalar(
        select(Lesson.id).where(Lesson.id == candidate, Lesson.owner_id == user.id)
    ):
        lesson_id = candidate
        metadata["lesson_id"] = str(candidate)

    try:
        await SavedResourceRepository(session).save(
            owner_id=user.id,
            kind=str(form.get("kind") or ""),
            source=str(form.get("source") or "unknown"),
            title=str(form.get("title") or ""),
            url=str(form.get("url") or "") or None,
            metadata=metadata,
        )
        await session.commit()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # From the panel (htmx): answer with the bookmark in its saved state, so the
    # page does not reload and the teacher sees it took (2026-10-14).
    if request.headers.get("HX-Request"):
        return templates.TemplateResponse(
            request,
            "partials/save_resource_form.html",
            {
                "resource_kind": str(form.get("kind") or ""),
                "resource_source": str(form.get("source") or "unknown"),
                "resource_title": str(form.get("title") or ""),
                "resource_url": str(form.get("url") or "") or None,
                "resource_item": {
                    "doi": form.get("doi") or "",
                    "video_id": form.get("video_id") or "",
                },
                "lesson_id": lesson_id,
                "saved_keys": await saved_keys_for_owner(session, user.id),
            },
        )

    next_path = str(form.get("next") or "")
    if not (next_path == "/webui/resources" or next_path.startswith("/webui/lesson/")):
        next_path = "/webui/resources"
    return RedirectResponse(next_path, status_code=303)


@router.post(
    "/resources/saved/{resource_id}/delete",
    name="webui_saved_resource_delete",
)
async def delete_saved_resource(
    resource_id: uuid.UUID,
    user: Optional[User] = Depends(optional_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> Response:
    if user is None:
        raise HTTPException(status_code=401)
    removed = await SavedResourceRepository(session).delete(
        resource_id,
        owner_id=user.id,
    )
    if not removed:
        raise HTTPException(status_code=404, detail="Risorsa non trovata")
    await session.commit()
    return RedirectResponse("/webui/resources", status_code=303)
