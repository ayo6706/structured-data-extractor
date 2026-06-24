from decimal import Decimal

import pytest

from app.core.config import (
    AppConfig,
    CostConfig,
    DEFAULT_MODEL_TOKEN_PRICES_USD,
    ModelTokenPrice,
    get_database_url,
)


def test_get_database_url_requires_database_url():
    settings = AppConfig(DATABASE_URL=None)

    with pytest.raises(RuntimeError, match="DATABASE_URL must be configured"):
        get_database_url(settings)


def test_cost_config_uses_model_specific_price_when_available():
    settings = CostConfig(
        MODEL_TOKEN_PRICES_USD={
            "expensive-model": ModelTokenPrice(
                input_token_price_usd="0.1",
                output_token_price_usd="0.2",
            )
        }
    )

    price = settings.price_for_model("expensive-model")

    assert price.input_token_price_usd == Decimal("0.1")
    assert price.output_token_price_usd == Decimal("0.2")


def test_cost_config_has_default_provider_prices():
    settings = CostConfig()

    assert settings.price_for_model(
        "gemini/gemini-2.5-flash"
    ) == DEFAULT_MODEL_TOKEN_PRICES_USD["gemini/gemini-2.5-flash"]
    assert settings.price_for_model(
        "openai/gpt-4.1-mini"
    ) == DEFAULT_MODEL_TOKEN_PRICES_USD["openai/gpt-4.1-mini"]
    assert settings.price_for_model(
        "anthropic/claude-haiku-4.5"
    ) == DEFAULT_MODEL_TOKEN_PRICES_USD["anthropic/claude-haiku-4.5"]
    assert settings.price_for_model(
        "anthropic/claude-haiku-4-5"
    ) == DEFAULT_MODEL_TOKEN_PRICES_USD["anthropic/claude-haiku-4-5"]


def test_cost_config_requires_price_for_unknown_model():
    settings = CostConfig()

    with pytest.raises(ValueError, match="No token pricing configured"):
        settings.price_for_model("unconfigured-model")
