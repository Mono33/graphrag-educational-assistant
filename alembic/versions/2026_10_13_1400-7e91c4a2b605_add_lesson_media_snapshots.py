"""persist lesson media snapshots

Revision ID: 7e91c4a2b605
Revises: 3b8e6d1f4a27
Create Date: 2026-10-13 14:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

import fastapi_users_db_sqlalchemy


revision: str = "7e91c4a2b605"
down_revision: Union[str, Sequence[str], None] = "3b8e6d1f4a27"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "lesson_media_snapshot",
        sa.Column(
            "lesson_id",
            fastapi_users_db_sqlalchemy.generics.GUID(),
            nullable=False,
        ),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column(
            "owner_id",
            fastapi_users_db_sqlalchemy.generics.GUID(),
            nullable=False,
        ),
        sa.Column("schema_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("curated_json", sa.JSON(), nullable=True),
        sa.Column("live_json", sa.JSON(), nullable=True),
        sa.Column("live_fetched", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["lesson_id"], ["lesson.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["owner_id"], ["user.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("lesson_id", "turn_index"),
    )
    op.create_index(
        "ix_lesson_media_snapshot_owner_lesson_turn",
        "lesson_media_snapshot",
        ["owner_id", "lesson_id", "turn_index"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_lesson_media_snapshot_owner_lesson_turn",
        table_name="lesson_media_snapshot",
    )
    op.drop_table("lesson_media_snapshot")
