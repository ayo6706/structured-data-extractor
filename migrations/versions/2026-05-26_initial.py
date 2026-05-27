"""initial

Revision ID: 216044123518
Revises: 
Create Date: 2026-05-26 14:53:33.210887

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '216044123518'
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'documents',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('filename', sa.String(), nullable=False),
        sa.Column('file_path', sa.String(), nullable=False),
        sa.Column('file_size_bytes', sa.Integer(), nullable=False),
        sa.Column('page_count', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('uploaded_at', sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            'file_size_bytes > 0',
            name='ck_documents_file_size_positive',
        ),
        sa.CheckConstraint(
            'page_count > 0',
            name='ck_documents_page_count_positive',
        ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(
        op.f('ix_documents_status'), 'documents', ['status'], unique=False
    )
    op.create_index(
        op.f('ix_documents_uploaded_at'),
        'documents',
        ['uploaded_at'],
        unique=False,
    )

    op.create_table(
        'extractions',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('document_id', sa.Uuid(), nullable=False),
        sa.Column('doc_type', sa.String(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('extracted_data', postgresql.JSONB(), nullable=True),
        sa.Column('confidence_map', postgresql.JSONB(), nullable=True),
        sa.Column('warnings', sa.ARRAY(sa.String()), nullable=True),
        sa.Column('raw_tool_output', postgresql.JSONB(), nullable=True),
        sa.Column('strategy', sa.String(), nullable=False),
        sa.Column('source_pages', sa.ARRAY(sa.Integer()), nullable=True),
        sa.Column('model_used', sa.String(), nullable=False),
        sa.Column('input_tokens', sa.Integer(), nullable=False),
        sa.Column('output_tokens', sa.Integer(), nullable=False),
        sa.Column('extraction_duration_ms', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            'extraction_duration_ms >= 0',
            name='ck_extractions_duration_non_negative',
        ),
        sa.CheckConstraint(
            'input_tokens >= 0',
            name='ck_extractions_input_tokens_non_negative',
        ),
        sa.CheckConstraint(
            'output_tokens >= 0',
            name='ck_extractions_output_tokens_non_negative',
        ),
        sa.ForeignKeyConstraint(
            ['document_id'], ['documents.id'], ondelete='CASCADE'
        ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(
        op.f('ix_extractions_document_id'),
        'extractions',
        ['document_id'],
        unique=False,
    )
    op.create_index(
        op.f('ix_extractions_status'),
        'extractions',
        ['status'],
        unique=False,
    )
    op.create_index(
        op.f('ix_extractions_created_at'),
        'extractions',
        ['created_at'],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_extractions_created_at'), table_name='extractions')
    op.drop_index(op.f('ix_extractions_status'), table_name='extractions')
    op.drop_index(op.f('ix_extractions_document_id'), table_name='extractions')
    op.drop_table('extractions')
    op.drop_index(op.f('ix_documents_uploaded_at'), table_name='documents')
    op.drop_index(op.f('ix_documents_status'), table_name='documents')
    op.drop_table('documents')
