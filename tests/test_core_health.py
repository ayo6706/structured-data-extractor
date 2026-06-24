import pytest

from app.core.health import (
    check_database_connection,
    healthy_status,
    unhealthy_status,
)


class RecordingConnection:
    def __init__(self) -> None:
        self.executed = False

    async def execute(self, _query) -> None:
        self.executed = True


@pytest.mark.asyncio
async def test_check_database_connection_executes_probe():
    connection = RecordingConnection()

    await check_database_connection(connection)

    assert connection.executed is True


def test_health_status_payloads_are_stable():
    assert healthy_status() == {"status": "ok", "database": "ok"}
    assert unhealthy_status() == {"status": "error", "database": "error"}
