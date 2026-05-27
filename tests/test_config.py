import pytest

from app.core.config import AppConfig, get_database_url


def test_get_database_url_requires_database_url():
    settings = AppConfig(DATABASE_URL=None)

    with pytest.raises(RuntimeError, match="DATABASE_URL must be configured"):
        get_database_url(settings)
