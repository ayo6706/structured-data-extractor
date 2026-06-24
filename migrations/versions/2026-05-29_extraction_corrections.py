"""add extraction corrections

Revision ID: 20260529_corrections
Revises: 216044123518
Create Date: 2026-05-29 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260529_corrections"
down_revision: str | Sequence[str] | None = "216044123518"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "extraction_corrections",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("extraction_id", sa.Uuid(), nullable=False),
        sa.Column("original_data", postgresql.JSONB(), nullable=True),
        sa.Column("corrected_data", postgresql.JSONB(), nullable=False),
        sa.Column("correction_diff", postgresql.JSONB(), nullable=False),
        sa.Column("warnings", postgresql.JSONB(), nullable=False),
        sa.Column("corrected_by", sa.String(), nullable=True),
        sa.Column("note", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["extraction_id"], ["extractions.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_extraction_corrections_extraction_id"),
        "extraction_corrections",
        ["extraction_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_extraction_corrections_created_at"),
        "extraction_corrections",
        ["created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_extraction_corrections_created_at"),
        table_name="extraction_corrections",
    )
    op.drop_index(
        op.f("ix_extraction_corrections_extraction_id"),
        table_name="extraction_corrections",
    )
    op.drop_table("extraction_corrections")
