from types import SimpleNamespace

from app.api.dependencies import get_arq_pool, get_storage


def test_get_storage_is_cached() -> None:
    get_storage.cache_clear()

    try:
        assert get_storage() is get_storage()
    finally:
        get_storage.cache_clear()


def test_get_arq_pool_returns_app_state_pool() -> None:
    pool = object()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    request.app.state.arq_pool = pool

    assert get_arq_pool(request) is pool


def test_get_arq_pool_returns_none_when_pool_missing() -> None:
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))

    assert get_arq_pool(request) is None
