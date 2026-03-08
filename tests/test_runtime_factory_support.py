"""runtime_factory_support 모듈 테스트."""

from __future__ import annotations

from contextlib import AsyncExitStack
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config import (
    AppSettings,
    BotConfig,
    LemonadeConfig,
    MemoryConfig,
    OllamaConfig,
    OpenAIConfig,
    SecurityConfig,
    TelegramConfig,
)
from core.runtime_factory_support import (
    StartupError,
    _create_llm_client,
    _create_retrieval_client,
    _release_runtime_lock,
    handle_optional_component_failure,
    log_degraded_startup_summary,
    model_for_provider,
    validate_required_settings,
)


def _make_config(**overrides) -> AppSettings:
    defaults = dict(
        bot=BotConfig(llm_provider="lemonade"),
        lemonade=LemonadeConfig(
            host="http://localhost:8000",
            default_model="lemonade-model",
        ),
        security=SecurityConfig(allowed_users=[111]),
        memory=MemoryConfig(),
        telegram=TelegramConfig(bot_token="valid_token", allowed_users="111"),
    )
    defaults.update(overrides)
    return AppSettings(**defaults)


class TestModelForProvider:
    def test_ollama_provider(self) -> None:
        config = _make_config(bot=BotConfig(llm_provider="ollama"))
        result = model_for_provider(config)
        assert result == config.ollama.embedding_model

    def test_openai_provider(self) -> None:
        config = _make_config(bot=BotConfig(llm_provider="openai"))
        result = model_for_provider(config)
        assert result == config.openai.default_model

    def test_lemonade_provider(self) -> None:
        config = _make_config(bot=BotConfig(llm_provider="lemonade"))
        result = model_for_provider(config)
        assert result == "lemonade-model"


class TestCreateLlmClient:
    def test_ollama_provider(self) -> None:
        config = _make_config(bot=BotConfig(llm_provider="ollama"))
        client = _create_llm_client(config)
        from core.ollama_client import OllamaClient
        assert isinstance(client, OllamaClient)

    def test_openai_provider(self) -> None:
        config = _make_config(bot=BotConfig(llm_provider="openai"))
        client = _create_llm_client(config)
        from core.lemonade_client import LemonadeClient
        assert isinstance(client, LemonadeClient)

    def test_lemonade_provider(self) -> None:
        config = _make_config(bot=BotConfig(llm_provider="lemonade"))
        client = _create_llm_client(config)
        from core.lemonade_client import LemonadeClient
        assert isinstance(client, LemonadeClient)


class TestCreateRetrievalClient:
    def test_creates_ollama_client(self) -> None:
        config = _make_config()
        client = _create_retrieval_client(config)
        from core.ollama_client import OllamaClient
        assert isinstance(client, OllamaClient)


class TestValidateRequiredSettings:
    def test_missing_bot_token_raises(self) -> None:
        config = _make_config(
            telegram=TelegramConfig(bot_token="your_telegram_bot_token_here", allowed_users="111"),
        )
        with pytest.raises(StartupError, match="bot_token"):
            validate_required_settings(config, MagicMock())

    def test_empty_bot_token_raises(self) -> None:
        config = _make_config(
            telegram=TelegramConfig(bot_token="", allowed_users="111"),
        )
        with pytest.raises(StartupError, match="bot_token"):
            validate_required_settings(config, MagicMock())

    def test_empty_allowed_users_raises(self) -> None:
        config = _make_config(
            security=SecurityConfig(allowed_users=[]),
            telegram=TelegramConfig(bot_token="real_token", allowed_users=""),
        )
        with pytest.raises(StartupError, match="allowed_users"):
            validate_required_settings(config, MagicMock())

    def test_valid_config_passes(self) -> None:
        config = _make_config()
        validate_required_settings(config, MagicMock())  # No exception


class TestHandleOptionalComponentFailure:
    def test_non_strict_logs_degraded(self) -> None:
        config = _make_config()
        config = AppSettings(**{**config.__dict__, "strict_startup": False})
        logger = MagicMock()
        degraded = []
        handle_optional_component_failure(
            config, logger, degraded,
            component="semantic_cache", error=RuntimeError("init failed"),
        )
        assert len(degraded) == 1
        assert degraded[0]["component"] == "semantic_cache"

    def test_strict_raises(self) -> None:
        config = _make_config()
        config = AppSettings(**{**config.__dict__, "strict_startup": True})
        logger = MagicMock()
        degraded = []
        with pytest.raises(StartupError, match="strict_startup"):
            handle_optional_component_failure(
                config, logger, degraded,
                component="rag", error=RuntimeError("fail"),
            )


class TestLogDegradedStartupSummary:
    def test_empty_list_noop(self) -> None:
        logger = MagicMock()
        log_degraded_startup_summary(logger, [])
        logger.warning.assert_not_called()

    def test_logs_summary(self) -> None:
        logger = MagicMock()
        degraded = [{"component": "cache", "error": "init failed"}]
        log_degraded_startup_summary(logger, degraded)
        logger.warning.assert_called_once()


class TestReleaseRuntimeLock:
    def test_release_closes_file(self) -> None:
        lock_file = MagicMock()
        lock_file.fileno.return_value = 999
        _release_runtime_lock(lock_file)
        lock_file.close.assert_called_once()
