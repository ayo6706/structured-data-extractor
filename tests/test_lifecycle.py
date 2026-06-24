import pytest

from app.core import lifecycle
from app.core.config import LLMAPIKeyConfig


def _set_models(
    monkeypatch: pytest.MonkeyPatch,
    *,
    extraction_model: str,
    classifier_model: str = "local-test-model",
) -> None:
    monkeypatch.setattr(
        lifecycle.llm_settings, "EXTRACTION_MODEL", extraction_model
    )
    monkeypatch.setattr(
        lifecycle.llm_settings, "CLASSIFIER_MODEL", classifier_model
    )


def test_validate_llm_api_keys_gemini_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_models(monkeypatch, extraction_model="gemini/gemini-2.5-flash")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        lifecycle.validate_llm_api_keys(LLMAPIKeyConfig(_env_file=None))


def test_validate_llm_api_keys_gemini_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_models(monkeypatch, extraction_model="gemini/gemini-2.5-flash")

    lifecycle.validate_llm_api_keys(
        LLMAPIKeyConfig(_env_file=None, GEMINI_API_KEY="x")
    )


def test_validate_llm_api_keys_gemini_present_in_env_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=x\n")
    _set_models(monkeypatch, extraction_model="gemini/gemini-2.5-flash")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        lifecycle,
        "LLMAPIKeyConfig",
        lambda: LLMAPIKeyConfig(_env_file=env_file),
    )

    lifecycle.validate_llm_api_keys()


def test_validate_llm_api_keys_anthropic_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_models(monkeypatch, extraction_model="anthropic/claude-sonnet-4")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        lifecycle.validate_llm_api_keys(LLMAPIKeyConfig(_env_file=None))


def test_validate_llm_api_keys_openai_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_models(monkeypatch, extraction_model="openai/gpt-4.1-mini")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        lifecycle.validate_llm_api_keys(LLMAPIKeyConfig(_env_file=None))
