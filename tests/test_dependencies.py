from app.api.dependencies import get_storage


def test_get_storage_is_cached() -> None:
    get_storage.cache_clear()

    try:
        assert get_storage() is get_storage()
    finally:
        get_storage.cache_clear()
