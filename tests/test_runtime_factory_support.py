"""runtime_factory_support 모듈 테스트."""

from __future__ import annotations

from contextlib import AsyncExitStack
from unittest.mock import MagicMock

import pytest

from core.config import (
    AppSettings,
    BotConfig,
    MemoryConfig,
    RetrievalProviderConfig,
    SecurityConfig,
    TelegramConfig,
)
from core.runtime_factory_support import (
    StartupError,
    _create_retrieval_client,
    _release_runtime_lock,
    handle_optional_component_failure,
    log_degraded_startup_summary,
    validate_required_settings,
)


def _make_config(**overrides) -> AppSettings:
    defaults = dict(
        bot=BotConfig(),
        ollama=RetrievalProviderConfig(chat_model="chat-model"),
        security=SecurityConfig(allowed_users=[111]),
        memory=MemoryConfig(),
        telegram=TelegramConfig(bot_token="valid_token", allowed_users="111"),
    )
    defaults.update(overrides)
    return AppSettings(**defaults)


class TestCreateRetrievalClient:
    def test_creates_ollama_client(self) -> None:
        config = _make_config()
        client = _create_retrieval_client(config)
        from core.ollama_client import OllamaClient
        assert isinstance(client, OllamaClient)


class TestValidateRequiredSettings:
    def test_missing_ollama_chat_model_raises(self) -> None:
        config = _make_config(
            ollama=RetrievalProviderConfig(chat_model=""),
        )
        with pytest.raises(StartupError, match="ollama\\.chat_model"):
            validate_required_settings(config, MagicMock())

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
