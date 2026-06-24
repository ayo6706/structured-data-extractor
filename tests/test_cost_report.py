from datetime import datetime
from decimal import Decimal
from typing import NamedTuple
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import status
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_db
from app.main import app
from app.repositories.extractions import ExtractionRepository
from app.repositories.llm_usage import LLMUsageRepository

EXTRACTION_MODEL = "gemini/gemini-2.5-flash"
CLASSIFIER_MODEL = "openai/gpt-4.1-mini"


class CostRow(NamedTuple):
    doc_type: str
    count: int
    avg_duration_ms: float
    total_input_tokens: int
    total_output_tokens: int


class StrategyCostRow(NamedTuple):
    strategy: str
    count: int
    avg_duration_ms: float
    total_input_tokens: int
    total_output_tokens: int


class UsageCostRow(NamedTuple):
    group_key: str
    model: str
    total_input_tokens: int
    total_output_tokens: int


@pytest.fixture
def mock_db() -> AsyncMock:
    db = AsyncMock()
    app.dependency_overrides[get_db] = lambda: db
    try:
        yield db
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest.mark.asyncio
@patch.object(LLMUsageRepository, "get_purpose_usage_report_data")
@patch.object(LLMUsageRepository, "get_model_usage_report_data")
@patch.object(LLMUsageRepository, "get_strategy_usage_report_data")
@patch.object(LLMUsageRepository, "get_doc_type_usage_report_data")
@patch.object(ExtractionRepository, "get_strategy_cost_report_data")
@patch.object(ExtractionRepository, "get_cost_report_data")
async def test_cost_report_success(
    mock_get_data: AsyncMock,
    mock_get_strategy_data: AsyncMock,
    mock_get_doc_type_usage: AsyncMock,
    mock_get_strategy_usage: AsyncMock,
    mock_get_model_usage: AsyncMock,
    mock_get_purpose_usage: AsyncMock,
    mock_db: AsyncMock,
) -> None:
    mock_get_data.return_value = [
        CostRow(
            doc_type="contract",
            count=2,
            avg_duration_ms=1200.0,
            total_input_tokens=100,
            total_output_tokens=50,
        ),
        CostRow(
            doc_type="invoice",
            count=1,
            avg_duration_ms=600.0,
            total_input_tokens=40,
            total_output_tokens=20,
        ),
    ]
    mock_get_strategy_data.return_value = [
        StrategyCostRow(
            strategy="full",
            count=2,
            avg_duration_ms=1000.0,
            total_input_tokens=120,
            total_output_tokens=60,
        ),
        StrategyCostRow(
            strategy="smart",
            count=1,
            avg_duration_ms=400.0,
            total_input_tokens=20,
            total_output_tokens=10,
        ),
    ]
    mock_get_doc_type_usage.return_value = [
        UsageCostRow("contract", EXTRACTION_MODEL, 100, 50),
        UsageCostRow("contract", CLASSIFIER_MODEL, 10, 4),
        UsageCostRow("invoice", EXTRACTION_MODEL, 40, 20),
        UsageCostRow("invoice", CLASSIFIER_MODEL, 5, 2),
    ]
    mock_get_strategy_usage.return_value = [
        UsageCostRow("full", EXTRACTION_MODEL, 120, 60),
        UsageCostRow("full", CLASSIFIER_MODEL, 10, 4),
        UsageCostRow("smart", EXTRACTION_MODEL, 20, 10),
        UsageCostRow("smart", CLASSIFIER_MODEL, 5, 2),
    ]
    mock_get_model_usage.return_value = [
        UsageCostRow(EXTRACTION_MODEL, EXTRACTION_MODEL, 140, 70),
        UsageCostRow(CLASSIFIER_MODEL, CLASSIFIER_MODEL, 15, 6),
    ]
    mock_get_purpose_usage.return_value = [
        UsageCostRow("extraction", EXTRACTION_MODEL, 140, 70),
        UsageCostRow("classifier", CLASSIFIER_MODEL, 15, 6),
    ]

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/documents/extractions/cost-report")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["total_extractions"] == 3
    assert data["total_input_tokens"] == 155
    assert data["total_output_tokens"] == 76

    assert Decimal(data["total_estimated_cost_usd"]) == Decimal("0.0002326")

    by_type = data["by_doc_type"]
    assert len(by_type) == 2

    assert by_type["contract"]["count"] == 2
    assert by_type["contract"]["avg_extraction_ms"] == 1200.0
    contract_cost = by_type["contract"]["estimated_cost_usd"]
    assert Decimal(contract_cost) == Decimal("0.0001654")
    contract_avg_cost = by_type["contract"]["avg_cost_per_document_usd"]
    assert Decimal(contract_avg_cost) == Decimal("0.0000827")

    assert by_type["invoice"]["count"] == 1
    assert by_type["invoice"]["avg_extraction_ms"] == 600.0
    invoice_cost = by_type["invoice"]["estimated_cost_usd"]
    assert Decimal(invoice_cost) == Decimal("0.0000672")
    invoice_avg_cost = by_type["invoice"]["avg_cost_per_document_usd"]
    assert Decimal(invoice_avg_cost) == Decimal("0.0000672")

    by_strategy = data["by_strategy"]
    assert len(by_strategy) == 2
    assert by_strategy["full"]["count"] == 2
    assert Decimal(by_strategy["full"]["estimated_cost_usd"]) == Decimal(
        "0.0001964"
    )
    assert by_strategy["smart"]["count"] == 1
    assert Decimal(by_strategy["smart"]["estimated_cost_usd"]) == Decimal(
        "0.0000362"
    )
    assert Decimal(
        data["by_model"][CLASSIFIER_MODEL]["estimated_cost_usd"]
    ) == Decimal("0.0000156")
    assert Decimal(data["by_purpose"]["classifier"]["estimated_cost_usd"]) == (
        Decimal("0.0000156")
    )

    mock_get_data.assert_called_once_with(None, None)
    mock_get_strategy_data.assert_called_once_with(None, None)
    mock_get_doc_type_usage.assert_called_once_with(None, None)
    mock_get_strategy_usage.assert_called_once_with(None, None)
    mock_get_model_usage.assert_called_once_with(None, None)
    mock_get_purpose_usage.assert_called_once_with(None, None)


@pytest.mark.asyncio
@patch.object(LLMUsageRepository, "get_purpose_usage_report_data")
@patch.object(LLMUsageRepository, "get_model_usage_report_data")
@patch.object(LLMUsageRepository, "get_strategy_usage_report_data")
@patch.object(LLMUsageRepository, "get_doc_type_usage_report_data")
@patch.object(ExtractionRepository, "get_strategy_cost_report_data")
@patch.object(ExtractionRepository, "get_cost_report_data")
async def test_cost_report_empty(
    mock_get_data: AsyncMock,
    mock_get_strategy_data: AsyncMock,
    mock_get_doc_type_usage: AsyncMock,
    mock_get_strategy_usage: AsyncMock,
    mock_get_model_usage: AsyncMock,
    mock_get_purpose_usage: AsyncMock,
    mock_db: AsyncMock,
) -> None:
    mock_get_data.return_value = []
    mock_get_strategy_data.return_value = []
    mock_get_doc_type_usage.return_value = []
    mock_get_strategy_usage.return_value = []
    mock_get_model_usage.return_value = []
    mock_get_purpose_usage.return_value = []

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/documents/extractions/cost-report")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["total_extractions"] == 0
    assert data["total_input_tokens"] == 0
    assert data["total_output_tokens"] == 0
    assert Decimal(data["total_estimated_cost_usd"]) == Decimal("0.0")
    assert data["by_doc_type"] == {}
    assert data["by_strategy"] == {}
    assert data["by_model"] == {}
    assert data["by_purpose"] == {}


@pytest.mark.asyncio
@patch.object(LLMUsageRepository, "get_purpose_usage_report_data")
@patch.object(LLMUsageRepository, "get_model_usage_report_data")
@patch.object(LLMUsageRepository, "get_strategy_usage_report_data")
@patch.object(LLMUsageRepository, "get_doc_type_usage_report_data")
@patch.object(ExtractionRepository, "get_strategy_cost_report_data")
@patch.object(ExtractionRepository, "get_cost_report_data")
async def test_cost_report_filters_passed(
    mock_get_data: AsyncMock,
    mock_get_strategy_data: AsyncMock,
    mock_get_doc_type_usage: AsyncMock,
    mock_get_strategy_usage: AsyncMock,
    mock_get_model_usage: AsyncMock,
    mock_get_purpose_usage: AsyncMock,
    mock_db: AsyncMock,
) -> None:
    mock_get_data.return_value = []
    mock_get_strategy_data.return_value = []
    mock_get_doc_type_usage.return_value = []
    mock_get_strategy_usage.return_value = []
    mock_get_model_usage.return_value = []
    mock_get_purpose_usage.return_value = []

    from_val = "2026-05-28T00:00:00"
    to_val = "2026-05-28T23:59:59"

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.get(
            "/api/v1/documents/extractions/cost-report",
            params={"from": from_val, "to": to_val},
        )

    assert response.status_code == status.HTTP_200_OK

    mock_get_data.assert_called_once()
    args = mock_get_data.call_args[0]
    assert isinstance(args[0], datetime)
    assert isinstance(args[1], datetime)
    assert args[0].isoformat() == "2026-05-28T00:00:00"
    assert args[1].isoformat() == "2026-05-28T23:59:59"

    mock_get_strategy_data.assert_called_once()
    strategy_args = mock_get_strategy_data.call_args[0]
    assert isinstance(strategy_args[0], datetime)
    assert isinstance(strategy_args[1], datetime)
    assert strategy_args[0].isoformat() == "2026-05-28T00:00:00"
    assert strategy_args[1].isoformat() == "2026-05-28T23:59:59"


def test_cost_report_query_includes_failed_extractions() -> None:
    query_names = ExtractionRepository.get_cost_report_data.__code__.co_names
    assert "ExtractionStatus" not in query_names


def test_strategy_cost_report_query_includes_failed_extractions() -> None:
    query_names = (
        ExtractionRepository.get_strategy_cost_report_data.__code__.co_names
    )
    assert "ExtractionStatus" not in query_names
