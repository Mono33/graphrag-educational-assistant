"""
SQLAlchemy ``User`` model for the webui (CORE 2 #6.6 P1).

Inherits ``SQLAlchemyBaseUserTableUUID`` from fastapi-users-db-sqlalchemy,
which provides the standard columns required by FastAPI-Users:
    id (UUID), email, hashed_password, is_active, is_superuser, is_verified

We add domain-relevant fields below. Keep this list lean — anything that's
truly per-lesson (e.g. EducationalProfile) belongs in the ``Lesson`` model,
not on the user.

Migration philosophy:
    - For dev/SQLite: ``Base.metadata.create_all`` in ``init_db`` is enough.
    - For production: Alembic migrations land in P5 / CORE 6 once the schema
      starts evolving; new fields added before then are recreated on a
      fresh DB.
"""

from __future__ import annotations

from fastapi_users_db_sqlalchemy import SQLAlchemyBaseUserTableUUID
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from aix.webui.db import Base


class User(SQLAlchemyBaseUserTableUUID, Base):
    """
    Webui user account.

    Inherited columns (from ``SQLAlchemyBaseUserTableUUID``):
        id              UUID, primary key
        email           str, unique, indexed
        hashed_password str
        is_active       bool, default True
        is_superuser    bool, default False
        is_verified     bool, default False
    """

    # Optional human-readable display name shown in the navbar / lesson author.
    # Italian "Nome visualizzato" — e.g. "Prof.ssa Rossi". Optional so the
    # register form can be filled with email only and updated later.
    display_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
