"""add document_map, the overview of each indexed document

Revision ID: 3b8e6d1f4a27
Revises: 9a4f2b7c61de
Create Date: 2026-09-29 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# Primary keys and owner columns use fastapi-users' GUID type, which
# autogenerate emits fully qualified without adding the import.
import fastapi_users_db_sqlalchemy


# revision identifiers, used by Alembic.
revision: str = "3b8e6d1f4a27"
down_revision: Union[str, Sequence[str], None] = "9a4f2b7c61de"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "document_map",
        sa.Column("id", fastapi_users_db_sqlalchemy.generics.GUID(), nullable=False),
        sa.Column("owner_id", fastapi_users_db_sqlalchemy.generics.GUID(), nullable=False),
        sa.Column(
            "material_version_id", fastapi_users_db_sqlalchemy.generics.GUID(), nullable=False
        ),
        sa.Column("map_version", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("map_json", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["material_version_id"], ["material_version.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["owner_id"], ["user.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("material_version_id", "map_version", name="uq_document_map_version"),
    )
    with op.batch_alter_table("document_map", schema=None) as batch_op:
        batch_op.create_index("ix_document_map_owner", ["owner_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("document_map", schema=None) as batch_op:
        batch_op.drop_index("ix_document_map_owner")

    op.drop_table("document_map")
