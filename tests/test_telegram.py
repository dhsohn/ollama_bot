"""텔레그램 핸들러 테스트."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from core.config import AppSettings, BotConfig, OllamaConfig, SecurityConfig, MemoryConfig, TelegramConfig
from core.security import SecurityManager, AuthenticationError, RateLimitError
from core.telegram_handler import TelegramHandler


@pytest.fixture
def app_config() -> AppSettings:
    return AppSettings(
        telegram_bot_token="test_token",
        data_dir="/tmp/test",
        bot=BotConfig(),
        ollama=OllamaConfig(),
        security=SecurityConfig(allowed_users=[111, 222]),
        memory=MemoryConfig(),
        telegram=TelegramConfig(),
    )


@pytest.fixture
def security() -> SecurityManager:
    return SecurityManager(SecurityConfig(allowed_users=[111, 222], rate_limit=30))


@pytest.fixture
def mock_engine() -> AsyncMock:
    engine = AsyncMock()
    engine.process_message = AsyncMock(return_value="Bot response")
    engine.process_message_stream = AsyncMock()
    engine.get_status = AsyncMock(return_value={
        "uptime_seconds": 100,
        "uptime_human": "1분 40초",
        "ollama": {"status": "ok"},
        "skills_loaded": 3,
        "current_model": "test-model",
    })
    engine._skills = MagicMock()
    engine._skills.list_skills.return_value = [
        {"name": "summarize", "description": "요약", "triggers": ["/summarize"], "security_level": "safe"}
    ]
    engine._ollama = MagicMock()
    engine._ollama.default_model = "test-model"
    engine._memory = AsyncMock()
    engine._memory.get_memory_stats = AsyncMock(return_value={
        "chat_id": 111,
        "conversation_count": 10,
        "memory_count": 5,
        "oldest_conversation": "2026-01-01",
    })
    return engine


@pytest.fixture
def telegram_handler(app_config, mock_engine, security) -> TelegramHandler:
    return TelegramHandler(config=app_config, engine=mock_engine, security=security)


class TestMessageSplitting:
    def test_short_message_not_split(self, telegram_handler: TelegramHandler) -> None:
        parts = telegram_handler._split_message("Hello!")
        assert len(parts) == 1
        assert parts[0] == "Hello!"

    def test_long_message_split_at_paragraph(self, telegram_handler: TelegramHandler) -> None:
        text = ("A" * 4000) + "\n\n" + ("B" * 100)
        parts = telegram_handler._split_message(text, max_length=4096)
        assert len(parts) == 2

    def test_very_long_message_split(self, telegram_handler: TelegramHandler) -> None:
        text = "A" * 10000
        parts = telegram_handler._split_message(text, max_length=4096)
        assert len(parts) >= 3
        for part in parts:
            assert len(part) <= 4096


class TestAuthDecorator:
    def test_unauthorized_user_blocked(self, security: SecurityManager) -> None:
        with pytest.raises(AuthenticationError):
            security.authenticate(999)

    def test_authorized_user_passes(self, security: SecurityManager) -> None:
        assert security.authenticate(111) is True

    def test_rate_limited_user(self, security: SecurityManager) -> None:
        # rate_limit=30이므로 30번 소진
        for _ in range(30):
            security.check_rate_limit(111)
        with pytest.raises(RateLimitError):
            security.check_rate_limit(111)


class TestPrivateChatOnly:
    @pytest.mark.asyncio
    async def test_group_chat_blocked(self, telegram_handler: TelegramHandler) -> None:
        chat = MagicMock()
        chat.id = -100123
        chat.type = "group"

        message = MagicMock()
        message.reply_text = AsyncMock()

        update = MagicMock()
        update.effective_chat = chat
        update.effective_message = message

        context = MagicMock()
        await telegram_handler._cmd_help(update, context)

        message.reply_text.assert_awaited_once_with(
            "이 봇은 private chat에서만 동작합니다."
        )

    @pytest.mark.asyncio
    async def test_private_chat_allowed(self, telegram_handler: TelegramHandler) -> None:
        chat = MagicMock()
        chat.id = 111
        chat.type = "private"

        message = MagicMock()
        message.reply_text = AsyncMock()

        update = MagicMock()
        update.effective_chat = chat
        update.effective_message = message

        context = MagicMock()
        await telegram_handler._cmd_help(update, context)

        assert message.reply_text.await_count == 1


class TestModelCommand:
    @pytest.mark.asyncio
    async def test_model_list_error_returns_message(self, telegram_handler: TelegramHandler) -> None:
        telegram_handler._engine._ollama.list_models = AsyncMock(side_effect=RuntimeError("down"))

        chat = MagicMock()
        chat.id = 111
        chat.type = "private"

        message = MagicMock()
        message.reply_text = AsyncMock()

        update = MagicMock()
        update.effective_chat = chat
        update.effective_message = message

        context = MagicMock()
        context.args = ["list"]

        await telegram_handler._cmd_model(update, context)

        message.reply_text.assert_awaited_once_with(
            "모델 목록을 가져오지 못했습니다. Ollama 상태를 확인해주세요."
        )

    @pytest.mark.asyncio
    async def test_model_change_error_returns_message(self, telegram_handler: TelegramHandler) -> None:
        telegram_handler._engine.change_model = AsyncMock(side_effect=RuntimeError("failed"))

        chat = MagicMock()
        chat.id = 111
        chat.type = "private"

        message = MagicMock()
        message.reply_text = AsyncMock()

        update = MagicMock()
        update.effective_chat = chat
        update.effective_message = message

        context = MagicMock()
        context.args = ["new-model"]

        await telegram_handler._cmd_model(update, context)

        message.reply_text.assert_awaited_once_with(
            "모델 변경 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        )


class TestReloadCommands:
    @pytest.mark.asyncio
    async def test_skills_reload_command(
        self, telegram_handler: TelegramHandler
    ) -> None:
        telegram_handler._engine._skills.reload_skills = AsyncMock(return_value=5)

        chat = MagicMock()
        chat.id = 111
        chat.type = "private"

        message = MagicMock()
        message.reply_text = AsyncMock()

        update = MagicMock()
        update.effective_chat = chat
        update.effective_message = message

        context = MagicMock()
        context.args = ["reload"]

        await telegram_handler._cmd_skills(update, context)

        telegram_handler._engine._skills.reload_skills.assert_awaited_once()
        message.reply_text.assert_awaited_once_with("스킬을 다시 로드했습니다: 5개")

    @pytest.mark.asyncio
    async def test_auto_reload_command(
        self, telegram_handler: TelegramHandler
    ) -> None:
        scheduler = MagicMock()
        scheduler.reload_automations = AsyncMock(return_value=4)
        telegram_handler.set_scheduler(scheduler)

        chat = MagicMock()
        chat.id = 111
        chat.type = "private"

        message = MagicMock()
        message.reply_text = AsyncMock()

        update = MagicMock()
        update.effective_chat = chat
        update.effective_message = message

        context = MagicMock()
        context.args = ["reload"]

        await telegram_handler._cmd_auto(update, context)

        scheduler.reload_automations.assert_awaited_once()
        message.reply_text.assert_awaited_once_with("자동화를 다시 로드했습니다: 4개")


class TestHandleMessage:
    @pytest.mark.asyncio
    async def test_handle_message_uses_index_for_split_parts(self, telegram_handler: TelegramHandler) -> None:
        async def _stream():
            yield "result"

        telegram_handler._engine.process_message_stream = MagicMock(return_value=_stream())
        telegram_handler._split_message = MagicMock(return_value=["SAME", "SAME"])

        chat = MagicMock()
        chat.id = 111
        chat.type = "private"
        chat.send_action = AsyncMock()

        sent_message = MagicMock()
        sent_message.edit_text = AsyncMock()

        message = MagicMock()
        message.text = "hello"
        message.reply_text = AsyncMock(return_value=sent_message)

        update = MagicMock()
        update.effective_chat = chat
        update.effective_message = message

        context = MagicMock()
        await telegram_handler._handle_message(update, context)

        # 첫 파트는 edit_text, 두 번째 파트는 reply_text로 전송되어야 한다.
        sent_message.edit_text.assert_awaited_once_with("SAME")
        message.reply_text.assert_has_awaits([call("..."), call("SAME")])
