"""JSON contracts for a teacher's private material library."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class MaterialVersionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    version_number: int
    media_type: str
    status: str
    parse_quality: Optional[dict[str, Any]] = None
    created_at: datetime


class MaterialResponse(BaseModel):
    id: uuid.UUID
    display_name: str
    purpose: str
    created_at: datetime
    status: str = Field(
        "processing",
        description="processing: being read and indexed · ready: searchable · failed: never "
        "searchable (see error_code)",
    )
    passages: int = Field(0, description="Searchable passages once ready.")
    error_code: Optional[str] = None
    latest_version: Optional[MaterialVersionResponse] = None
    versions: list[MaterialVersionResponse] = Field(default_factory=list)
    attached_lesson_ids: list[uuid.UUID] = Field(default_factory=list)


class MaterialListResponse(BaseModel):
    items: list[MaterialResponse]
    total: int


class MaterialAttachRequest(BaseModel):
    lesson_id: uuid.UUID
    version_id: Optional[uuid.UUID] = None


class SavedResourceCreate(BaseModel):
    kind: str
    source: str = "unknown"
    title: str
    url: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class SavedResourceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    kind: str
    source: str
    title: str
    url: Optional[str] = None
    metadata_json: Optional[dict[str, Any]] = None
    created_at: datetime
