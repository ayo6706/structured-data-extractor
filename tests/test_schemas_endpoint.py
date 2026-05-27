import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_list_schemas_does_not_redirect():
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        follow_redirects=False,
    ) as client:
        response = await client.get("/api/v1/schemas")

    assert response.status_code == 200
    assert {schema["name"] for schema in response.json()["schemas"]} == {
        "invoice",
        "contract",
        "payslip",
        "receipt",
    }
