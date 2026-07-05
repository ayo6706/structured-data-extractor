from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
    ENVIRONMENT: str = "local"
    PROJECT_NAME: str = "Structured Data Extractor"
    VERSION: str = "0.1.0"
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = False
    DATABASE_URL: PostgresDsn | None = None
    STORAGE_BACKEND: Literal["local"] = "local"
    STORAGE_LOCAL_DIR: Path = Path("./uploads")
    DB_POOL_PRE_PING: bool = True
    DB_POOL_SIZE: int = Field(default=5, gt=0)
    DB_MAX_OVERFLOW: int = Field(default=10, ge=0)
    DB_POOL_RECYCLE: int = Field(default=1800, ge=-1)
    DB_ECHO: bool = False
    MAX_UPLOAD_SIZE_BYTES: int = Field(default=25 * 1024 * 1024, gt=0)

    @field_validator("STORAGE_LOCAL_DIR")
    @classmethod
    def resolve_storage_local_dir(cls, value: Path) -> Path:
        return value.expanduser().resolve()


@lru_cache
def get_app_settings() -> AppConfig:
    return AppConfig()


def get_database_url(settings: AppConfig | None = None) -> PostgresDsn:
    app_settings = settings or get_app_settings()
    if app_settings.DATABASE_URL is None:
        raise RuntimeError("DATABASE_URL must be configured.")

    return app_settings.DATABASE_URL


class LLMConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    EXTRACTION_MODEL: str = "gemini/gemini-2.5-flash"
    CLASSIFIER_MODEL: str = "gemini/gemini-2.5-flash"
    MAX_RETRIES: int = Field(default=2, ge=0, le=10)


class LLMAPIKeyConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    GEMINI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None


class ArqConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ARQ_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    REDIS_URL: RedisDsn = "redis://localhost:6379"


class ModelTokenPrice(BaseModel):
    input_token_price_usd: Decimal
    output_token_price_usd: Decimal


DEFAULT_MODEL_TOKEN_PRICES_USD = {
    "gemini/gemini-2.5-flash": ModelTokenPrice(
        input_token_price_usd=Decimal("0.0000003"),
        output_token_price_usd=Decimal("0.0000025"),
    ),
    "openai/gpt-4.1-mini": ModelTokenPrice(
        input_token_price_usd=Decimal("0.0000004"),
        output_token_price_usd=Decimal("0.0000016"),
    ),
    "anthropic/claude-haiku-4.5": ModelTokenPrice(
        input_token_price_usd=Decimal("0.000001"),
        output_token_price_usd=Decimal("0.000005"),
    ),
    "anthropic/claude-haiku-4-5": ModelTokenPrice(
        input_token_price_usd=Decimal("0.000001"),
        output_token_price_usd=Decimal("0.000005"),
    ),
}


class CostConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="COST_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    MODEL_TOKEN_PRICES_USD: dict[str, ModelTokenPrice] = Field(
        default_factory=lambda: {
            model: price.model_copy()
            for model, price in DEFAULT_MODEL_TOKEN_PRICES_USD.items()
        }
    )

    def price_for_model(self, model: str) -> ModelTokenPrice:
        try:
            return self.MODEL_TOKEN_PRICES_USD[model]
        except KeyError as exc:
            raise ValueError(
                "No token pricing configured for model "
                f"'{model}'. Add it to COST_MODEL_TOKEN_PRICES_USD."
            ) from exc


llm_settings = LLMConfig()
arq_settings = ArqConfig()
cost_settings = CostConfig()
