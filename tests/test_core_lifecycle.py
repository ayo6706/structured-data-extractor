import pytest

from app.core import lifecycle


class DisposableEngine:
    def __init__(self) -> None:
        self.disposed = False

    async def dispose(self) -> None:
        self.disposed = True


@pytest.mark.asyncio
async def test_startup_checks_database(monkeypatch):
    checked = False

    async def check_database(engine):
        nonlocal checked
        checked = True

    monkeypatch.setattr(lifecycle, "_validate_llm_api_keys", lambda: None)
    monkeypatch.setattr(lifecycle, "check_database_engine", check_database)

    await lifecycle.startup()

    assert checked is True


@pytest.mark.asyncio
async def test_shutdown_disposes_engine(monkeypatch):
    engine = DisposableEngine()
    monkeypatch.setattr(lifecycle, "get_engine", lambda: engine)

    await lifecycle.shutdown()

    assert engine.disposed is True
