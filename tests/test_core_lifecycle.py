from types import SimpleNamespace

import pytest

from app.core import lifecycle
from app.core.config import ArqConfig


class DisposableEngine:
    def __init__(self) -> None:
        self.disposed = False

    async def dispose(self) -> None:
        self.disposed = True


def test_get_arq_redis_settings_uses_zero_for_root_path(monkeypatch):
    redis_url = ArqConfig(REDIS_URL="redis://localhost:6379/").REDIS_URL
    monkeypatch.setattr(lifecycle.arq_settings, "REDIS_URL", redis_url)

    settings = lifecycle.get_arq_redis_settings()

    assert settings.database == 0


@pytest.mark.asyncio
async def test_startup_checks_database(monkeypatch):
    checked = False

    async def check_database(engine):
        nonlocal checked
        checked = True

    monkeypatch.setattr(lifecycle, "validate_llm_api_keys", lambda: None)
    monkeypatch.setattr(lifecycle, "check_database_engine", check_database)

    await lifecycle.startup()

    assert checked is True


@pytest.mark.asyncio
async def test_shutdown_disposes_engine(monkeypatch):
    engine = DisposableEngine()
    monkeypatch.setattr(lifecycle, "get_engine", lambda: engine)

    await lifecycle.shutdown()

    assert engine.disposed is True


@pytest.mark.asyncio
async def test_startup_stores_arq_pool_on_app_state(monkeypatch):
    pool = object()
    app = SimpleNamespace(state=SimpleNamespace())

    async def check_database(engine):
        return None

    async def create_arq_pool():
        return pool

    monkeypatch.setattr(lifecycle, "validate_llm_api_keys", lambda: None)
    monkeypatch.setattr(lifecycle, "check_database_engine", check_database)
    monkeypatch.setattr(lifecycle, "_create_arq_pool", create_arq_pool)

    await lifecycle.startup(app)

    assert app.state.arq_pool is pool


@pytest.mark.asyncio
async def test_shutdown_closes_app_arq_pool(monkeypatch):
    engine = DisposableEngine()
    pool = SimpleNamespace(closed=False)
    app = SimpleNamespace(state=SimpleNamespace(arq_pool=pool))

    async def close_arq_pool(pool_arg):
        pool_arg.closed = True

    monkeypatch.setattr(lifecycle, "get_engine", lambda: engine)
    monkeypatch.setattr(lifecycle, "_close_arq_pool", close_arq_pool)

    await lifecycle.shutdown(app)

    assert pool.closed is True
    assert app.state.arq_pool is None
    assert engine.disposed is True


@pytest.mark.asyncio
async def test_create_arq_pool_logs_and_returns_none_on_failure(
    monkeypatch,
    caplog,
):
    async def create_pool(settings):
        raise OSError("redis down")

    monkeypatch.setattr(lifecycle, "create_pool", create_pool)

    with caplog.at_level("WARNING"):
        pool = await lifecycle._create_arq_pool()

    assert pool is None
    assert "Redis unavailable for Arq" in caplog.text
