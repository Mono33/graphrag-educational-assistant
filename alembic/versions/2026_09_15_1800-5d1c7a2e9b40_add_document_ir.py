"""add document_ir, the parser's structured result

Revision ID: 5d1c7a2e9b40
Revises: 48f042517314
Create Date: 2026-09-15 18:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# Primary keys and owner columns use fastapi-users' GUID type, which
# autogenerate emits fully qualified without adding the import.
import fastapi_users_db_sqlalchemy


# revision identifiers, used by Alembic.
revision: str = "5d1c7a2e9b40"
down_revision: Union[str, Sequence[str], None] = "48f042517314"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "document_ir",
        sa.Column("id", fastapi_users_db_sqlalchemy.generics.GUID(), nullable=False),
        sa.Column("owner_id", fastapi_users_db_sqlalchemy.generics.GUID(), nullable=False),
        sa.Column(
            "material_version_id", fastapi_users_db_sqlalchemy.generics.GUID(), nullable=False
        ),
        sa.Column("parser_profile_version", sa.String(length=64), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("ir_json", sa.JSON(), nullable=False),
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
        sa.UniqueConstraint(
            "material_version_id", "parser_profile_version", name="uq_document_ir_version_profile"
        ),
    )
    with op.batch_alter_table("document_ir", schema=None) as batch_op:
        batch_op.create_index("ix_document_ir_owner", ["owner_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("document_ir", schema=None) as batch_op:
        batch_op.drop_index("ix_document_ir_owner")

    op.drop_table("document_ir")
