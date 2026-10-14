"""add saved external resources

Revision ID: ac54b8d1e902
Revises: 7e91c4a2b605
Create Date: 2026-10-13 14:30:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

import fastapi_users_db_sqlalchemy


revision: str = "ac54b8d1e902"
down_revision: Union[str, Sequence[str], None] = "7e91c4a2b605"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "saved_resource",
        sa.Column(
            "id",
            fastapi_users_db_sqlalchemy.generics.GUID(),
            nullable=False,
        ),
        sa.Column(
            "owner_id",
            fastapi_users_db_sqlalchemy.generics.GUID(),
            nullable=False,
        ),
        sa.Column("kind", sa.String(length=24), nullable=False),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("stable_key", sa.String(length=320), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["owner_id"], ["user.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "owner_id",
            "stable_key",
            name="uq_saved_resource_owner_key",
        ),
    )
    op.create_index(
        "ix_saved_resource_owner_created",
        "saved_resource",
        ["owner_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_saved_resource_owner_created", table_name="saved_resource")
    op.drop_table("saved_resource")
