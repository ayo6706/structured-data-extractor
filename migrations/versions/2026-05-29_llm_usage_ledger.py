"""add llm usage ledger

Revision ID: 20260529_llm_usage
Revises: 20260529_correction_retention
Create Date: 2026-05-29 00:00:00.000000

"""
from collections.abc import Sequence
from uuid import uuid4

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260529_llm_usage"
down_revision: str | Sequence[str] | None = "20260529_correction_retention"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "llm_usages",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("extraction_id", sa.Uuid(), nullable=True),
        sa.Column("purpose", sa.String(), nullable=False),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "input_tokens >= 0",
            name="ck_llm_usages_input_tokens_non_negative",
        ),
        sa.CheckConstraint(
            "output_tokens >= 0",
            name="ck_llm_usages_output_tokens_non_negative",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"], ["documents.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["extraction_id"], ["extractions.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_llm_usages_document_id"),
        "llm_usages",
        ["document_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_llm_usages_extraction_id"),
        "llm_usages",
        ["extraction_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_llm_usages_purpose"),
        "llm_usages",
        ["purpose"],
        unique=False,
    )
    op.create_index(
        op.f("ix_llm_usages_model"),
        "llm_usages",
        ["model"],
        unique=False,
    )
    op.create_index(
        op.f("ix_llm_usages_created_at"),
        "llm_usages",
        ["created_at"],
        unique=False,
    )
    _backfill_extraction_usage()


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_llm_usages_created_at"), table_name="llm_usages")
    op.drop_index(op.f("ix_llm_usages_model"), table_name="llm_usages")
    op.drop_index(op.f("ix_llm_usages_purpose"), table_name="llm_usages")
    op.drop_index(op.f("ix_llm_usages_extraction_id"), table_name="llm_usages")
    op.drop_index(op.f("ix_llm_usages_document_id"), table_name="llm_usages")
    op.drop_table("llm_usages")


def _backfill_extraction_usage() -> None:
    connection = op.get_bind()
    rows = list(connection.execute(
        sa.text(
            """
            SELECT id, document_id, model_used, input_tokens, output_tokens,
                   created_at
            FROM extractions
            """
        )
    ))
    if not rows:
        return

    connection.execute(
        sa.table(
            "llm_usages",
            sa.column("id", sa.Uuid()),
            sa.column("document_id", sa.Uuid()),
            sa.column("extraction_id", sa.Uuid()),
            sa.column("purpose", sa.String()),
            sa.column("model", sa.String()),
            sa.column("input_tokens", sa.Integer()),
            sa.column("output_tokens", sa.Integer()),
            sa.column("created_at", sa.DateTime(timezone=True)),
        ).insert(),
        [
            {
                "id": uuid4(),
                "document_id": row.document_id,
                "extraction_id": row.id,
                "purpose": "extraction",
                "model": row.model_used,
                "input_tokens": row.input_tokens,
                "output_tokens": row.output_tokens,
                "created_at": row.created_at,
            }
            for row in rows
        ],
    )
