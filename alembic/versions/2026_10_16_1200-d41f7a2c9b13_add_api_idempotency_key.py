"""add api idempotency keys

Revision ID: d41f7a2c9b13
Revises: ac54b8d1e902
Create Date: 2026-10-16 12:00:00.000000

A retried API request gets the first answer instead of a second run
(``aix.api.idempotency``). Additive: one new table, nothing existing changes.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

import fastapi_users_db_sqlalchemy


revision: str = "d41f7a2c9b13"
down_revision: Union[str, Sequence[str], None] = "ac54b8d1e902"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "api_idempotency_key",
        sa.Column("id", fastapi_users_db_sqlalchemy.generics.GUID(), nullable=False),
        sa.Column("owner_id", fastapi_users_db_sqlalchemy.generics.GUID(), nullable=False),
        sa.Column("scope", sa.String(length=120), nullable=False),
        sa.Column("key", sa.String(length=200), nullable=False),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("status_code", sa.Integer(), nullable=True),
        sa.Column("response_json", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["owner_id"], ["user.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("owner_id", "scope", "key", name="uq_api_idempotency_owner_scope_key"),
    )
    op.create_index(
        "ix_api_idempotency_created", "api_idempotency_key", ["created_at"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_api_idempotency_created", table_name="api_idempotency_key")
    op.drop_table("api_idempotency_key")
