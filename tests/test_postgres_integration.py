import os
from collections.abc import AsyncGenerator
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlmodel import SQLModel

from app.models.document import Document, DocumentStatus
from app.models.extraction import ExtractionStatus
from app.repositories.extractions import ExtractionRepository

POSTGRES_TEST_DATABASE_URL = os.getenv("POSTGRES_TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not POSTGRES_TEST_DATABASE_URL,
    reason="POSTGRES_TEST_DATABASE_URL is not configured",
)


def _asyncpg_url(url: str) -> str:
    parsed = make_url(url)
    if parsed.drivername == "postgresql":
        parsed = parsed.set(drivername="postgresql+asyncpg")
    return parsed.render_as_string(hide_password=False)


@pytest_asyncio.fixture
async def postgres_session() -> AsyncGenerator[AsyncSession, None]:
    assert POSTGRES_TEST_DATABASE_URL is not None

    schema_name = f"test_{uuid4().hex}"
    url = _asyncpg_url(POSTGRES_TEST_DATABASE_URL)
    admin_engine = create_async_engine(url)

    async with admin_engine.begin() as conn:
        await conn.execute(text(f'CREATE SCHEMA "{schema_name}"'))

    engine = create_async_engine(
        url,
        connect_args={"server_settings": {"search_path": schema_name}},
    )

    try:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        async with AsyncSession(engine) as session:
            yield session
    finally:
        await engine.dispose()
        async with admin_engine.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA "{schema_name}" CASCADE'))
        await admin_engine.dispose()


@pytest.mark.asyncio
async def test_postgres_persists_extraction_jsonb_and_arrays(
    postgres_session: AsyncSession,
) -> None:
    document = Document(
        filename="invoice.pdf",
        file_path="uploads/invoice.pdf",
        file_size_bytes=100,
        page_count=2,
        status=DocumentStatus.COMPLETED,
    )
    postgres_session.add(document)
    await postgres_session.flush()

    extractions = ExtractionRepository(postgres_session)
    extraction = await extractions.create_from_fields(
        document_id=document.id,
        doc_type="invoice",
        status=ExtractionStatus.COMPLETED,
        extracted_data={"vendor_name": "Acme", "total_amount": "10.00"},
        confidence_map={"vendor_name": 0.85},
        warnings=["total_amount: quality warning"],
        raw_tool_output={"tool": {"vendor_name": "Acme"}},
        model_used="test-model",
        input_tokens=100,
        output_tokens=50,
        extraction_duration_ms=123,
        source_pages=[0, 1],
        strategy="full",
    )
    await postgres_session.commit()

    stored = await extractions.get_by_id(extraction.id)

    assert stored is not None
    assert stored.extracted_data == {
        "vendor_name": "Acme",
        "total_amount": "10.00",
    }
    assert stored.confidence_map == {"vendor_name": 0.85}
    assert stored.warnings == ["total_amount: quality warning"]
    assert stored.raw_tool_output == {"tool": {"vendor_name": "Acme"}}
    assert stored.source_pages == [0, 1]


@pytest.mark.asyncio
async def test_postgres_rolls_back_uncommitted_extraction(
    postgres_session: AsyncSession,
) -> None:
    document = Document(
        filename="invoice.pdf",
        file_path="uploads/invoice.pdf",
        file_size_bytes=100,
        page_count=1,
        status=DocumentStatus.UPLOADED,
    )
    postgres_session.add(document)
    await postgres_session.flush()

    extractions = ExtractionRepository(postgres_session)
    extraction = await extractions.create_from_fields(
        document_id=document.id,
        doc_type="invoice",
        status=ExtractionStatus.FAILED,
        extracted_data=None,
        confidence_map={},
        warnings=["failed"],
        raw_tool_output=None,
        model_used="test-model",
        input_tokens=1,
        output_tokens=1,
        extraction_duration_ms=1,
        source_pages=None,
        strategy="full",
    )
    extraction_id = extraction.id
    await postgres_session.rollback()

    stored = await extractions.get_by_id(extraction_id)

    assert stored is None
