"""align correction retention and warnings storage

Revision ID: 20260529_correction_retention
Revises: 20260529_corrections
Create Date: 2026-05-29 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260529_correction_retention"
down_revision: str | Sequence[str] | None = "20260529_corrections"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

FK_NAME = "extraction_corrections_extraction_id_fkey"


def upgrade() -> None:
    op.drop_constraint(FK_NAME, "extraction_corrections", type_="foreignkey")
    op.create_foreign_key(
        FK_NAME,
        "extraction_corrections",
        "extractions",
        ["extraction_id"],
        ["id"],
        ondelete="RESTRICT",
    )

    op.add_column(
        "extraction_corrections",
        sa.Column("warnings_array", postgresql.ARRAY(sa.String())),
    )
    op.execute(
        """
        UPDATE extraction_corrections
        SET warnings_array = ARRAY(
            SELECT jsonb_array_elements_text(warnings)
        )
        """
    )
    op.drop_column("extraction_corrections", "warnings")
    op.alter_column(
        "extraction_corrections",
        "warnings_array",
        new_column_name="warnings",
        existing_type=postgresql.ARRAY(sa.String()),
        nullable=False,
    )


def downgrade() -> None:
    op.drop_constraint(FK_NAME, "extraction_corrections", type_="foreignkey")
    op.create_foreign_key(
        FK_NAME,
        "extraction_corrections",
        "extractions",
        ["extraction_id"],
        ["id"],
        ondelete="CASCADE",
    )

    op.add_column(
        "extraction_corrections",
        sa.Column("warnings_jsonb", postgresql.JSONB()),
    )
    op.execute(
        """
        UPDATE extraction_corrections
        SET warnings_jsonb = to_jsonb(warnings)
        """
    )
    op.drop_column("extraction_corrections", "warnings")
    op.alter_column(
        "extraction_corrections",
        "warnings_jsonb",
        new_column_name="warnings",
        existing_type=postgresql.JSONB(),
        nullable=False,
    )
