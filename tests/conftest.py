"""테스트 공유 픽스처."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from core.config import (
    AppSettings,
    BotConfig,
    LemonadeConfig,
    MemoryConfig,
    SecurityConfig,
    TelegramConfig,
)
from core.llm_types import ChatResponse
from core.memory import MemoryManager
from core.security import SecurityManager


@pytest.fixture
def security_config() -> SecurityConfig:
    return SecurityConfig(
        allowed_users=[111, 222],
        rate_limit=10,
        max_file_size=10_485_760,
        blocked_paths=["/etc/*", "/proc/*", "/sys/*"],
    )


@pytest.fixture
def app_settings(security_config: SecurityConfig) -> AppSettings:
    return AppSettings(
        log_level="DEBUG",
        data_dir="/tmp/test_data",
        bot=BotConfig(max_conversation_length=10),
        lemonade=LemonadeConfig(
            host="http://localhost:8000",
            default_model="test-model",
            system_prompt="You are a test assistant.",
        ),
        telegram=TelegramConfig(
            bot_token="test_token_123",
            allowed_users="111,222",
        ),
        security=security_config,
        memory=MemoryConfig(),
    )


@pytest.fixture
def security_manager(security_config: SecurityConfig) -> SecurityManager:
    return SecurityManager(security_config)


@pytest_asyncio.fixture
async def memory_manager(tmp_path: Path) -> MemoryManager:
    config = MemoryConfig()
    mm = MemoryManager(config=config, data_dir=str(tmp_path), max_conversation_length=10)
    await mm.initialize()
    yield mm
    await mm.close()


@pytest.fixture
def mock_ollama_client() -> AsyncMock:
    client = AsyncMock()
    client.default_model = "test-model"
    client.system_prompt = "You are a test assistant."
    client.chat = AsyncMock(return_value=ChatResponse(content="Test response"))
    client.health_check = AsyncMock(
        return_value={"status": "ok", "models_count": 1}
    )
    client.list_models = AsyncMock(
        return_value=[{"name": "test-model", "size": 1024}]
    )
    return client
