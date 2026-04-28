"""
Pydantic schemas for the User model — used by FastAPI-Users for input
validation on register/update and for serialization on /users/me responses.

Three distinct shapes per FastAPI-Users convention:
    UserRead    — what the API returns (no password)
    UserCreate  — what the register endpoint accepts (email + password)
    UserUpdate  — partial PATCH for /users/me
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi_users import schemas
from pydantic import Field


class UserRead(schemas.BaseUser[uuid.UUID]):
    """Public user profile returned by /users/me and similar."""

    display_name: Optional[str] = Field(
        default=None,
        description="Human-readable display name shown in the navbar (e.g. 'Prof.ssa Rossi').",
    )


class UserCreate(schemas.BaseUserCreate):
    """Payload for /auth/register and the HTML register form."""

    display_name: Optional[str] = Field(
        default=None,
        max_length=120,
        description="Optional display name; if omitted, the email is shown.",
    )


class UserUpdate(schemas.BaseUserUpdate):
    """PATCH payload for /users/me — every field is optional."""

    display_name: Optional[str] = Field(default=None, max_length=120)
