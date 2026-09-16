"""add material_chunk, the private RAG index

Revision ID: 9a4f2b7c61de
Revises: 5d1c7a2e9b40
Create Date: 2026-09-16 15:00:00.000000

Dialect-aware on purpose (A.2, §8.5). The table is created portably, and on
PostgreSQL the embedding column is then converted to a real ``vector(1024)`` and
the text gains an Italian full-text index — so similarity search and keyword
search both run in the database. SQLite, the developer database, keeps the
embedding as JSON and the retriever computes the cosine in Python over the
owner's own chunks: correct, and fast enough for a handful of documents.

The extension is created before the table rather than assumed: the pilot VM
switches to ``pgvector/pgvector:pg16`` in the same deploy, and a migration that
discovered a missing extension halfway would leave the table behind.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# Primary keys and owner columns use fastapi-users' GUID type, which
# autogenerate emits fully qualified without adding the import.
import fastapi_users_db_sqlalchemy


# revision identifiers, used by Alembic.
revision: str = "9a4f2b7c61de"
down_revision: Union[str, Sequence[str], None] = "5d1c7a2e9b40"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

#: Must match ``aix.webui.materials.models.EMBEDDING_DIMENSION``: baai/bge-m3.
EMBEDDING_DIMENSION = 1024


def upgrade() -> None:
    """Upgrade schema."""
    is_postgres = op.get_bind().dialect.name == "postgresql"

    if is_postgres:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "material_chunk",
        sa.Column("id", fastapi_users_db_sqlalchemy.generics.GUID(), nullable=False),
        sa.Column("owner_id", fastapi_users_db_sqlalchemy.generics.GUID(), nullable=False),
        sa.Column(
            "material_version_id", fastapi_users_db_sqlalchemy.generics.GUID(), nullable=False
        ),
        sa.Column("parent_chunk_id", fastapi_users_db_sqlalchemy.generics.GUID(), nullable=True),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("text_hash", sa.String(length=64), nullable=False),
        sa.Column("locator_json", sa.JSON(), nullable=False),
        sa.Column("embedding", sa.JSON(), nullable=False),
        sa.Column("embedding_model", sa.String(length=128), nullable=False),
        sa.Column("chunker_version", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), server_default="active", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.CheckConstraint("ordinal >= 0", name="ck_material_chunk_ordinal_non_negative"),
        sa.CheckConstraint("status IN ('active', 'deleted')", name="ck_material_chunk_status"),
        sa.ForeignKeyConstraint(["owner_id"], ["user.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["material_version_id"], ["material_version.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["parent_chunk_id"], ["material_chunk.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "material_version_id", "chunker_version", "ordinal", name="uq_material_chunk_ordinal"
        ),
    )
    with op.batch_alter_table("material_chunk", schema=None) as batch_op:
        batch_op.create_index(
            "ix_material_chunk_owner_version", ["owner_id", "material_version_id"], unique=False
        )

    if is_postgres:
        op.execute(
            f"ALTER TABLE material_chunk "
            f"ALTER COLUMN embedding TYPE vector({EMBEDDING_DIMENSION}) "
            f"USING embedding::text::vector({EMBEDDING_DIMENSION})"
        )
        op.execute(
            "ALTER TABLE material_chunk ADD COLUMN fts tsvector "
            "GENERATED ALWAYS AS (to_tsvector('italian', coalesce(text, ''))) STORED"
        )
        op.execute("CREATE INDEX ix_material_chunk_fts ON material_chunk USING GIN (fts)")


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("material_chunk", schema=None) as batch_op:
        batch_op.drop_index("ix_material_chunk_owner_version")

    op.drop_table("material_chunk")
