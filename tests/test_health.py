import pytest
from httpx import ASGITransport, AsyncClient

from app.core.database import get_db
from app.main import app


class HealthyDb:
    async def execute(self, query):
        return None


class FailingDb:
    async def execute(self, query):
        raise RuntimeError("database unavailable")


async def healthy_db():
    yield HealthyDb()


async def failing_db():
    yield FailingDb()


@pytest.mark.asyncio
async def test_health_check_returns_200():
    """Verify that the health check endpoint returns 200 OK."""
    app.dependency_overrides[get_db] = healthy_db
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/health")

            assert response.status_code == 200
            assert response.json() == {"status": "ok", "database": "ok"}
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_health_check_returns_503_when_db_fails():
    """Verify that database failures make health checks unavailable."""
    app.dependency_overrides[get_db] = failing_db
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/health")

            assert response.status_code == 503
            assert response.json() == {
                "detail": {"status": "error", "database": "error"}
            }
    finally:
        app.dependency_overrides.clear()
