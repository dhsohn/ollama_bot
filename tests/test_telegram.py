"""텔레그램 핸들러 테스트."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from core.config import AppSettings, BotConfig, FeedbackConfig, LemonadeConfig, SecurityConfig, MemoryConfig, TelegramConfig
from core.security import SecurityManager, AuthenticationError, RateLimitError
from core.telegram_handler import TelegramHandler


@pytest.fixture
def app_config() -> AppSettings:
    return AppSettings(
        telegram_bot_token="test_token",
        data_dir="/tmp/test",
        bot=BotConfig(),
        lemonade=LemonadeConfig(),
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
    engine.classify_intent = MagicMock(return_value=None)
    engine.consume_last_stream_meta = MagicMock(return_value=None)
    engine.get_status = AsyncMock(return_value={
        "uptime_seconds": 100,
        "uptime_human": "1분 40초",
        "llm": {"status": "ok"},
        "skills_loaded": 3,
        "current_model": "test-model",
    })
    engine.list_skills = MagicMock(return_value=[
        {"name": "summarize", "description": "요약", "triggers": ["/summarize"], "security_level": "safe"}
    ])
    engine.reload_skills = AsyncMock(return_value=5)
    engine.list_models = AsyncMock(return_value=[{"name": "test-model", "size": 1024}])
    engine.get_current_model = MagicMock(return_value="test-model")
    engine.get_memory_stats = AsyncMock(return_value={
        "chat_id": 111,
        "conversation_count": 10,
        "memory_count": 5,
        "oldest_conversation": "2026-01-01",
    })
    engine.analyze_all_corpus = AsyncMock(return_value={
        "answer": "전체 분석 결과",
        "stats": {
            "total_chunks": 10,
            "total_segments": 2,
            "mapped_segments": 2,
            "evidence_lines": 4,
            "duration_ms": 1234.5,
        },
    })
    engine.clear_conversation = AsyncMock(return_value=0)
    engine.export_conversation_markdown = AsyncMock()
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

    def test_default_split_uses_configured_max_message_length(
        self,
        app_config: AppSettings,
        mock_engine: AsyncMock,
        security: SecurityManager,
    ) -> None:
        app_config.telegram.max_message_length = 100
        handler = TelegramHandler(config=app_config, engine=mock_engine, security=security)

        parts = handler._split_message("A" * 250)
        assert len(parts) == 3
        assert all(len(part) <= 100 for part in parts)


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


class TestCommandRegistration:
    def test_analyze_all_command_is_not_registered(self, telegram_handler: TelegramHandler) -> None:
        handler_names = {cmd for handler in telegram_handler._build_command_handlers() for cmd in handler.commands}
        bot_command_names = [cmd.command for cmd in telegram_handler._build_bot_commands()]

        assert "analyze_all" not in handler_names
        assert "analyze_all" not in bot_command_names


class TestReloadCommands:
    @pytest.mark.asyncio
    async def test_skills_reload_command(
        self, telegram_handler: TelegramHandler
    ) -> None:
        telegram_handler._engine.reload_skills = AsyncMock(return_value=5)

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

        telegram_handler._engine.reload_skills.assert_awaited_once_with(strict=True)
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

        scheduler.reload_automations.assert_awaited_once_with(strict=True)
        message.reply_text.assert_awaited_once_with("자동화를 다시 로드했습니다: 4개")

    @pytest.mark.asyncio
    async def test_skills_reload_command_shows_partial_load_warnings(
        self, telegram_handler: TelegramHandler
    ) -> None:
        telegram_handler._engine.reload_skills = AsyncMock(return_value=2)
        telegram_handler._engine.get_last_skill_load_errors = MagicMock(
            return_value=["bad_skill.yaml: security violation"]
        )

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

        reply = message.reply_text.await_args[0][0]
        assert "스킬을 다시 로드했습니다: 2개" in reply
        assert "일부 항목 로드 실패" in reply
        assert "bad_skill.yaml" in reply

    @pytest.mark.asyncio
    async def test_auto_reload_command_shows_partial_load_warnings(
        self, telegram_handler: TelegramHandler
    ) -> None:
        scheduler = MagicMock()
        scheduler.reload_automations = AsyncMock(return_value=1)
        scheduler.get_last_load_errors = MagicMock(return_value=["bad.yaml: Invalid cron"])
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

        reply = message.reply_text.await_args[0][0]
        assert "자동화를 다시 로드했습니다: 1개" in reply
        assert "일부 항목 로드 실패" in reply
        assert "bad.yaml" in reply


class TestHandleMessage:
    @pytest.mark.asyncio
    async def test_handle_message_auto_routes_to_analyze_all_when_analysis_keyword_exists(
        self,
        app_config: AppSettings,
        mock_engine: AsyncMock,
        security: SecurityManager,
    ) -> None:
        handler = TelegramHandler(config=app_config, engine=mock_engine, security=security)
        handler._engine.analyze_all_corpus = AsyncMock(return_value={
            "answer": "자동 전환 결과",
            "stats": {
                "total_chunks": 10,
                "total_segments": 3,
                "mapped_segments": 2,
                "evidence_lines": 4,
                "duration_ms": 100.0,
            },
        })
        handler._engine.process_message_stream = MagicMock()

        chat = MagicMock()
        chat.id = 111
        chat.type = "private"
        chat.send_action = AsyncMock()

        sent_message = MagicMock()
        sent_message.edit_text = AsyncMock()
        sent_message.message_id = 42
        sent_message.edit_reply_markup = AsyncMock()

        message = MagicMock()
        message.text = "이 요청 좀 분석해줘"
        message.reply_text = AsyncMock(return_value=sent_message)

        update = MagicMock()
        update.effective_chat = chat
        update.effective_message = message

        await handler._handle_message(update, MagicMock())

        handler._engine.analyze_all_corpus.assert_awaited_once()
        handler._engine.process_message_stream.assert_not_called()
        edited_texts = [call.args[0] for call in sent_message.edit_text.await_args_list]
        assert any("자동 전환" in text for text in edited_texts)

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
        sent_message.message_id = 42
        sent_message.edit_reply_markup = AsyncMock()

        reply_msg = MagicMock()
        reply_msg.message_id = 43
        reply_msg.edit_reply_markup = AsyncMock()

        # reply_text는 처음에 진행 안내 메시지, 두 번째 파트에서 reply_msg를 반환
        message = MagicMock()
        message.text = "hello"
        message.reply_text = AsyncMock(side_effect=[sent_message, reply_msg])

        update = MagicMock()
        update.effective_chat = chat
        update.effective_message = message

        context = MagicMock()
        await telegram_handler._handle_message(update, context)

        # 첫 파트는 edit_text, 두 번째 파트는 reply_text로 전송되어야 한다.
        sent_message.edit_text.assert_awaited_once_with("SAME")
        expected_placeholder = (
            f"{telegram_handler._config.bot.name}이 답변을 위해 생각 중입니다..."
        )
        message.reply_text.assert_has_awaits([call(expected_placeholder), call("SAME")])

    @pytest.mark.asyncio
    async def test_handle_message_continue_without_pending_shows_notice(
        self,
        telegram_handler: TelegramHandler,
    ) -> None:
        chat = MagicMock()
        chat.id = 111
        chat.type = "private"
        chat.send_action = AsyncMock()

        message = MagicMock()
        message.text = "계속"
        message.reply_text = AsyncMock()

        update = MagicMock()
        update.effective_chat = chat
        update.effective_message = message

        await telegram_handler._handle_message(update, MagicMock())

        message.reply_text.assert_awaited_once_with("이어볼 답변이 없습니다. 먼저 질문을 해주세요.")
        telegram_handler._engine.process_message_stream.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_message_auto_continues_then_fallback_when_still_truncated(
        self,
        app_config: AppSettings,
        mock_engine: AsyncMock,
        security: SecurityManager,
    ) -> None:
        handler = TelegramHandler(config=app_config, engine=mock_engine, security=security)

        async def _stream():
            yield "긴"

        handler._engine.process_message_stream = MagicMock(return_value=_stream())

        chat = MagicMock()
        chat.id = 111
        chat.type = "private"
        chat.send_action = AsyncMock()

        sent_message = MagicMock()
        sent_message.edit_text = AsyncMock()
        sent_message.message_id = 42
        sent_message.edit_reply_markup = AsyncMock()

        message = MagicMock()
        message.text = "매우 긴 설명 부탁해"
        message.reply_text = AsyncMock(return_value=sent_message)

        update = MagicMock()
        update.effective_chat = chat
        update.effective_message = message

        calls = 0

        async def fake_stream_and_render(**kwargs):
            nonlocal calls
            calls += 1
            return SimpleNamespace(
                full_response="핵심 요점 1\n핵심 요점 2\n핵심 요점 3",
                last_message=sent_message,
                stop_reason="max_total_chars",
                tier="full",
                intent=None,
                cache_id=None,
                usage=None,
            )

        with patch("core.telegram_handler.stream_and_render", new=fake_stream_and_render):
            await handler._handle_message(update, MagicMock())

        assert calls == 4
        assert 111 in handler._pending_continuation
        assert handler._pending_continuation[111]["root_query"] == "매우 긴 설명 부탁해"
        reply_texts = [item.args[0] for item in message.reply_text.await_args_list if item.args]
        assert any("자동으로 이어서" in text for text in reply_texts)
        followup_text = reply_texts[-1]
        assert "지금까지 요약" in followup_text
        assert "/continue" in followup_text

    @pytest.mark.asyncio
    async def test_handle_message_auto_continues_without_manual_prompt_when_next_turn_finishes(
        self,
        app_config: AppSettings,
        mock_engine: AsyncMock,
        security: SecurityManager,
    ) -> None:
        handler = TelegramHandler(config=app_config, engine=mock_engine, security=security)

        async def _stream():
            yield "긴"

        handler._engine.process_message_stream = MagicMock(return_value=_stream())

        chat = MagicMock()
        chat.id = 111
        chat.type = "private"
        chat.send_action = AsyncMock()

        sent_message = MagicMock()
        sent_message.edit_text = AsyncMock()
        sent_message.message_id = 42
        sent_message.edit_reply_markup = AsyncMock()

        message = MagicMock()
        message.text = "정리해서 길게 설명해줘"
        message.reply_text = AsyncMock(return_value=sent_message)

        update = MagicMock()
        update.effective_chat = chat
        update.effective_message = message

        calls = 0

        async def fake_stream_and_render(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return SimpleNamespace(
                    full_response="1차 응답",
                    last_message=sent_message,
                    stop_reason="max_total_chars",
                    tier="full",
                    intent=None,
                    cache_id=None,
                    usage=None,
                )
            return SimpleNamespace(
                full_response="2차 응답",
                last_message=sent_message,
                stop_reason=None,
                tier="full",
                intent=None,
                cache_id=None,
                usage=None,
            )

        with patch("core.telegram_handler.stream_and_render", new=fake_stream_and_render):
            await handler._handle_message(update, MagicMock())

        assert calls == 2
        assert 111 not in handler._pending_continuation
        reply_texts = [item.args[0] for item in message.reply_text.await_args_list if item.args]
        assert any("자동으로 이어서" in text for text in reply_texts)
        assert not any("/continue" in text for text in reply_texts)

    @pytest.mark.asyncio
    async def test_handle_message_passes_configured_max_edit_length(
        self,
        app_config: AppSettings,
        mock_engine: AsyncMock,
        security: SecurityManager,
    ) -> None:
        app_config.telegram.max_message_length = 123
        handler = TelegramHandler(config=app_config, engine=mock_engine, security=security)
        async def _stream():
            if False:
                yield ""
        handler._engine.process_message_stream = MagicMock(return_value=_stream())

        chat = MagicMock()
        chat.id = 111
        chat.type = "private"
        chat.send_action = AsyncMock()

        sent_message = MagicMock()
        sent_message.edit_text = AsyncMock()
        sent_message.message_id = 42
        sent_message.edit_reply_markup = AsyncMock()

        message = MagicMock()
        message.text = "hello"
        message.reply_text = AsyncMock(return_value=sent_message)

        update = MagicMock()
        update.effective_chat = chat
        update.effective_message = message

        captured: dict[str, float] = {}

        async def fake_stream_and_render(**kwargs):
            captured["max_edit_length"] = kwargs["max_edit_length"]
            captured["first_chunk_timeout_seconds"] = kwargs["first_chunk_timeout_seconds"]
            captured["chunk_timeout_seconds"] = kwargs["chunk_timeout_seconds"]
            return SimpleNamespace(full_response="", last_message=None)

        with patch("core.telegram_handler.stream_and_render", new=fake_stream_and_render):
            await handler._handle_message(update, MagicMock())

        assert captured["max_edit_length"] == 123
        assert captured["first_chunk_timeout_seconds"] > captured["chunk_timeout_seconds"]

    @pytest.mark.asyncio
    async def test_handle_message_uses_long_timeouts_for_complex_intent(
        self,
        app_config: AppSettings,
        mock_engine: AsyncMock,
        security: SecurityManager,
    ) -> None:
        app_config.bot.response_timeout = 300
        handler = TelegramHandler(config=app_config, engine=mock_engine, security=security)

        async def _stream():
            if False:
                yield ""

        handler._engine.process_message_stream = MagicMock(return_value=_stream())
        handler._engine.classify_intent = MagicMock(return_value="complex")

        chat = MagicMock()
        chat.id = 111
        chat.type = "private"
        chat.send_action = AsyncMock()

        sent_message = MagicMock()
        sent_message.edit_text = AsyncMock()
        sent_message.message_id = 42
        sent_message.edit_reply_markup = AsyncMock()

        message = MagicMock()
        message.text = "심층 검토해줘"
        message.photo = []
        message.reply_text = AsyncMock(return_value=sent_message)

        update = MagicMock()
        update.effective_chat = chat
        update.effective_message = message

        captured: dict[str, float] = {}

        async def fake_stream_and_render(**kwargs):
            captured["first_chunk_timeout_seconds"] = kwargs["first_chunk_timeout_seconds"]
            captured["chunk_timeout_seconds"] = kwargs["chunk_timeout_seconds"]
            captured["max_stream_seconds"] = kwargs["max_stream_seconds"]
            return SimpleNamespace(full_response="", last_message=None)

        with patch("core.telegram_handler.stream_and_render", new=fake_stream_and_render):
            await handler._handle_message(update, MagicMock())

        assert captured["first_chunk_timeout_seconds"] == 600.0
        assert captured["chunk_timeout_seconds"] == 60.0
        assert captured["max_stream_seconds"] == 3600.0

    @pytest.mark.asyncio
    async def test_handle_message_uses_long_timeouts_for_image_input(
        self,
        app_config: AppSettings,
        mock_engine: AsyncMock,
        security: SecurityManager,
    ) -> None:
        app_config.bot.response_timeout = 300
        handler = TelegramHandler(config=app_config, engine=mock_engine, security=security)

        async def _stream():
            if False:
                yield ""

        handler._engine.process_message_stream = MagicMock(return_value=_stream())
        handler._engine.classify_intent = MagicMock(return_value=None)

        chat = MagicMock()
        chat.id = 111
        chat.type = "private"
        chat.send_action = AsyncMock()

        sent_message = MagicMock()
        sent_message.edit_text = AsyncMock()
        sent_message.message_id = 42
        sent_message.edit_reply_markup = AsyncMock()

        photo_file = MagicMock()
        photo_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"img-bytes"))
        photo = MagicMock()
        photo.get_file = AsyncMock(return_value=photo_file)

        message = MagicMock()
        message.text = None
        message.caption = None
        message.photo = [photo]
        message.reply_text = AsyncMock(return_value=sent_message)

        update = MagicMock()
        update.effective_chat = chat
        update.effective_message = message

        captured: dict[str, float] = {}

        async def fake_stream_and_render(**kwargs):
            captured["first_chunk_timeout_seconds"] = kwargs["first_chunk_timeout_seconds"]
            captured["chunk_timeout_seconds"] = kwargs["chunk_timeout_seconds"]
            captured["max_stream_seconds"] = kwargs["max_stream_seconds"]
            return SimpleNamespace(full_response="", last_message=None)

        with patch("core.telegram_handler.stream_and_render", new=fake_stream_and_render):
            await handler._handle_message(update, MagicMock())

        assert captured["first_chunk_timeout_seconds"] == 600.0
        assert captured["chunk_timeout_seconds"] == 60.0
        assert captured["max_stream_seconds"] == 3600.0

    @pytest.mark.asyncio
    async def test_handle_message_recovers_with_non_stream_call_when_stream_stops_early(
        self,
        app_config: AppSettings,
        mock_engine: AsyncMock,
        security: SecurityManager,
    ) -> None:
        handler = TelegramHandler(config=app_config, engine=mock_engine, security=security)

        async def _stream():
            yield "partial"

        handler._engine.process_message_stream = MagicMock(return_value=_stream())
        handler._engine.consume_last_stream_meta = MagicMock(return_value=None)
        handler._engine.process_message = AsyncMock(return_value="복구된 최종 답변")

        chat = MagicMock()
        chat.id = 111
        chat.type = "private"
        chat.send_action = AsyncMock()

        sent_message = MagicMock()
        sent_message.edit_text = AsyncMock()
        sent_message.message_id = 42
        sent_message.edit_reply_markup = AsyncMock()

        message = MagicMock()
        message.text = "hello"
        message.reply_text = AsyncMock(return_value=sent_message)

        update = MagicMock()
        update.effective_chat = chat
        update.effective_message = message

        async def fake_stream_and_render(**kwargs):
            return SimpleNamespace(
                full_response="partial",
                last_message=sent_message,
                stop_reason="repeated_chunks",
                tier="full",
                intent=None,
                cache_id=None,
                usage=None,
            )

        with patch("core.telegram_handler.stream_and_render", new=fake_stream_and_render):
            await handler._handle_message(update, MagicMock())

        handler._engine.process_message.assert_awaited_once_with(
            111,
            "hello",
            images=None,
        )
        edited_texts = [call.args[0] for call in sent_message.edit_text.await_args_list]
        assert "복구된 최종 답변" in edited_texts

    @pytest.mark.asyncio
    async def test_handle_image_only_message_routes_with_empty_text(
        self, telegram_handler: TelegramHandler,
    ) -> None:
        captured: dict[str, object] = {}

        def _stream_call(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

            async def _stream():
                yield "vision result"

            return _stream()

        telegram_handler._engine.process_message_stream = MagicMock(side_effect=_stream_call)

        chat = MagicMock()
        chat.id = 111
        chat.type = "private"
        chat.send_action = AsyncMock()

        sent_message = MagicMock()
        sent_message.edit_text = AsyncMock()
        sent_message.message_id = 44
        sent_message.edit_reply_markup = AsyncMock()

        photo_file = MagicMock()
        photo_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"img-bytes"))
        photo = MagicMock()
        photo.get_file = AsyncMock(return_value=photo_file)

        message = MagicMock()
        message.text = None
        message.caption = None
        message.photo = [photo]
        message.reply_text = AsyncMock(return_value=sent_message)

        update = MagicMock()
        update.effective_chat = chat
        update.effective_message = message

        await telegram_handler._handle_message(update, MagicMock())

        assert captured["args"] == (111, "")
        kwargs = captured["kwargs"]
        assert isinstance(kwargs, dict)
        assert kwargs["images"] == [b"img-bytes"]

    @pytest.mark.asyncio
    async def test_handle_image_only_download_failure_notifies_user(
        self, telegram_handler: TelegramHandler,
    ) -> None:
        telegram_handler._engine.process_message_stream = MagicMock()

        chat = MagicMock()
        chat.id = 111
        chat.type = "private"
        chat.send_action = AsyncMock()

        photo = MagicMock()
        photo.get_file = AsyncMock(side_effect=RuntimeError("download fail"))

        message = MagicMock()
        message.text = None
        message.caption = None
        message.photo = [photo]
        message.reply_text = AsyncMock()

        update = MagicMock()
        update.effective_chat = chat
        update.effective_message = message

        await telegram_handler._handle_message(update, MagicMock())

        message.reply_text.assert_awaited_once_with(
            "이미지 다운로드에 실패했어요. 잠시 후 다시 시도해주세요."
        )
        telegram_handler._engine.process_message_stream.assert_not_called()


class TestFeedbackButtons:
    @pytest.fixture
    def mock_feedback(self) -> AsyncMock:
        fb = AsyncMock()
        fb.store_feedback = AsyncMock(return_value=False)
        fb.get_user_stats = AsyncMock(return_value={
            "total": 10, "positive": 7, "negative": 3, "satisfaction_rate": 0.7,
        })
        fb.get_global_stats = AsyncMock(return_value={
            "total": 20, "positive": 15, "negative": 5, "satisfaction_rate": 0.75,
        })
        return fb

    @pytest.fixture
    def feedback_handler(self, app_config, mock_engine, security, mock_feedback) -> TelegramHandler:
        return TelegramHandler(
            config=app_config, engine=mock_engine, security=security, feedback=mock_feedback,
        )

    @pytest.fixture
    def no_feedback_handler(self, mock_engine, security) -> TelegramHandler:
        """feedback=None인 핸들러."""
        config = AppSettings(
            telegram_bot_token="test_token",
            data_dir="/tmp/test",
            bot=BotConfig(),
            lemonade=LemonadeConfig(),
            security=SecurityConfig(allowed_users=[111, 222]),
            memory=MemoryConfig(),
            telegram=TelegramConfig(),
            feedback=FeedbackConfig(enabled=False),
        )
        return TelegramHandler(config=config, engine=mock_engine, security=security, feedback=None)

    @pytest.mark.asyncio
    async def test_feedback_buttons_attached_after_response(self, feedback_handler: TelegramHandler) -> None:
        """피드백 활성화 시 응답 후 버튼이 부착된다."""
        async def _stream():
            yield "response"

        feedback_handler._engine.process_message_stream = MagicMock(return_value=_stream())

        chat = MagicMock()
        chat.id = 111
        chat.type = "private"
        chat.send_action = AsyncMock()

        sent_message = MagicMock()
        sent_message.edit_text = AsyncMock()
        sent_message.message_id = 42
        sent_message.edit_reply_markup = AsyncMock()

        message = MagicMock()
        message.text = "hello"
        message.reply_text = AsyncMock(return_value=sent_message)

        update = MagicMock()
        update.effective_chat = chat
        update.effective_message = message

        context = MagicMock()
        await feedback_handler._handle_message(update, context)

        sent_message.edit_reply_markup.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_buttons_when_feedback_disabled(self, no_feedback_handler: TelegramHandler) -> None:
        """피드백 비활성화 시 버튼이 부착되지 않는다."""
        async def _stream():
            yield "response"

        no_feedback_handler._engine.process_message_stream = MagicMock(return_value=_stream())

        chat = MagicMock()
        chat.id = 111
        chat.type = "private"
        chat.send_action = AsyncMock()

        sent_message = MagicMock()
        sent_message.edit_text = AsyncMock()
        sent_message.message_id = 42
        sent_message.edit_reply_markup = AsyncMock()

        message = MagicMock()
        message.text = "hello"
        message.reply_text = AsyncMock(return_value=sent_message)

        update = MagicMock()
        update.effective_chat = chat
        update.effective_message = message

        context = MagicMock()
        await no_feedback_handler._handle_message(update, context)

        sent_message.edit_reply_markup.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_buttons_when_response_empty(self, feedback_handler: TelegramHandler) -> None:
        """응답 본문이 비어 있으면 피드백 버튼을 붙이지 않는다."""
        async def _stream():
            if False:
                yield ""

        feedback_handler._engine.process_message_stream = MagicMock(return_value=_stream())

        chat = MagicMock()
        chat.id = 111
        chat.type = "private"
        chat.send_action = AsyncMock()

        sent_message = MagicMock()
        sent_message.edit_text = AsyncMock()
        sent_message.message_id = 42
        sent_message.edit_reply_markup = AsyncMock()

        message = MagicMock()
        message.text = "hello"
        message.reply_text = AsyncMock(return_value=sent_message)

        update = MagicMock()
        update.effective_chat = chat
        update.effective_message = message

        context = MagicMock()
        await feedback_handler._handle_message(update, context)

        sent_message.edit_reply_markup.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cache_feedback_link_saved_when_stream_meta_has_cache_id(
        self,
        app_config: AppSettings,
        mock_engine: AsyncMock,
        security: SecurityManager,
        mock_feedback: AsyncMock,
    ) -> None:
        semantic_cache = AsyncMock()
        handler = TelegramHandler(
            config=app_config,
            engine=mock_engine,
            security=security,
            feedback=mock_feedback,
            semantic_cache=semantic_cache,
        )

        async def _stream():
            yield "response"

        handler._engine.process_message_stream = MagicMock(return_value=_stream())
        handler._engine.consume_last_stream_meta = MagicMock(return_value={"cache_id": 77})

        chat = MagicMock()
        chat.id = 111
        chat.type = "private"
        chat.send_action = AsyncMock()

        sent_message = MagicMock()
        sent_message.edit_text = AsyncMock()
        sent_message.message_id = 42
        sent_message.edit_reply_markup = AsyncMock()

        message = MagicMock()
        message.text = "hello"
        message.reply_text = AsyncMock(return_value=sent_message)

        update = MagicMock()
        update.effective_chat = chat
        update.effective_message = message

        await handler._handle_message(update, MagicMock())

        semantic_cache.link_feedback_target.assert_awaited_once_with(111, 42, 77)

    @pytest.mark.asyncio
    async def test_callback_new_feedback(self, feedback_handler: TelegramHandler, mock_feedback) -> None:
        """콜백으로 새 피드백이 저장된다."""
        # 프리뷰 캐시에 항목 추가
        feedback_handler._preview_cache[(111, 42)] = {
            "user": "q",
            "bot": "a",
            "ts": time.monotonic(),
        }

        query = MagicMock()
        query.data = "fb:1:42"
        query.answer = AsyncMock()

        chat = MagicMock()
        chat.id = 111
        chat.type = "private"

        update = MagicMock()
        update.callback_query = query
        update.effective_chat = chat

        context = MagicMock()
        await feedback_handler._handle_feedback_callback(update, context)

        mock_feedback.store_feedback.assert_awaited_once_with(
            chat_id=111, bot_message_id=42, rating=1, user_preview="q", bot_preview="a",
        )
        query.answer.assert_awaited_once_with("피드백 감사합니다!", show_alert=False)

    @pytest.mark.asyncio
    async def test_callback_prunes_expired_preview_cache(
        self,
        feedback_handler: TelegramHandler,
        mock_feedback,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        feedback_handler._config.feedback.preview_cache_ttl_hours = 1
        feedback_handler._preview_cache[(111, 42)] = {"user": "q", "bot": "a", "ts": 0.0}
        feedback_handler._pending_reason[111] = {
            "bot_message_id": 41,
            "expires": 1.0,
        }
        monkeypatch.setattr("core.telegram_handler.time.monotonic", lambda: 7201.0)

        query = MagicMock()
        query.data = "fb:1:42"
        query.answer = AsyncMock()

        chat = MagicMock()
        chat.id = 111
        chat.type = "private"

        update = MagicMock()
        update.callback_query = query
        update.effective_chat = chat

        context = MagicMock()
        await feedback_handler._handle_feedback_callback(update, context)

        mock_feedback.store_feedback.assert_awaited_once_with(
            chat_id=111, bot_message_id=42, rating=1, user_preview=None, bot_preview=None,
        )
        assert (111, 42) not in feedback_handler._preview_cache
        assert 111 not in feedback_handler._pending_reason

    @pytest.mark.asyncio
    async def test_callback_negative_feedback_replaces_existing_pending_with_notice(
        self,
        feedback_handler: TelegramHandler,
        mock_feedback,
    ) -> None:
        feedback_handler._pending_reason[111] = {
            "bot_message_id": 40,
            "expires": time.monotonic() + 60,
        }
        mock_feedback.store_feedback = AsyncMock(return_value=False)

        query = MagicMock()
        query.data = "fb:-1:42"
        query.answer = AsyncMock()
        query.message = MagicMock()
        query.message.reply_text = AsyncMock()

        chat = MagicMock()
        chat.id = 111
        chat.type = "private"

        update = MagicMock()
        update.callback_query = query
        update.effective_chat = chat

        context = MagicMock()
        await feedback_handler._handle_feedback_callback(update, context)

        assert feedback_handler._pending_reason[111]["bot_message_id"] == 42
        assert query.message.reply_text.await_count == 2
        first_notice = query.message.reply_text.await_args_list[0].args[0]
        second_prompt = query.message.reply_text.await_args_list[1].args[0]
        assert "자동 만료" in first_notice
        assert "사유를 입력" in second_prompt

    @pytest.mark.asyncio
    async def test_callback_update_feedback(self, feedback_handler: TelegramHandler, mock_feedback) -> None:
        """재평가 시 업데이트 메시지가 표시된다."""
        mock_feedback.store_feedback = AsyncMock(return_value=True)

        query = MagicMock()
        query.data = "fb:-1:42"
        query.answer = AsyncMock()

        chat = MagicMock()
        chat.id = 111
        chat.type = "private"

        update = MagicMock()
        update.callback_query = query
        update.effective_chat = chat

        context = MagicMock()
        await feedback_handler._handle_feedback_callback(update, context)

        query.answer.assert_awaited_once_with("피드백을 업데이트했어요.", show_alert=False)

    @pytest.mark.asyncio
    async def test_callback_update_clears_pending_reason(
        self,
        feedback_handler: TelegramHandler,
        mock_feedback,
    ) -> None:
        """같은 메시지를 재평가하면 사유 대기 상태를 정리한다."""
        mock_feedback.store_feedback = AsyncMock(return_value=True)
        feedback_handler._pending_reason[111] = {
            "bot_message_id": 42,
            "expires": time.monotonic() + 120,
        }

        query = MagicMock()
        query.data = "fb:1:42"
        query.answer = AsyncMock()

        chat = MagicMock()
        chat.id = 111
        chat.type = "private"

        update = MagicMock()
        update.callback_query = query
        update.effective_chat = chat

        context = MagicMock()
        await feedback_handler._handle_feedback_callback(update, context)

        assert 111 not in feedback_handler._pending_reason

    @pytest.mark.asyncio
    async def test_callback_invalid_data(self, feedback_handler: TelegramHandler) -> None:
        """잘못된 콜백 데이터는 에러 메시지를 표시한다."""
        query = MagicMock()
        query.data = "fb:invalid"
        query.answer = AsyncMock()

        chat = MagicMock()
        chat.id = 111
        chat.type = "private"

        update = MagicMock()
        update.callback_query = query
        update.effective_chat = chat

        context = MagicMock()
        await feedback_handler._handle_feedback_callback(update, context)

        query.answer.assert_awaited_once_with("잘못된 피드백 요청입니다.", show_alert=True)

    @pytest.mark.asyncio
    async def test_callback_invalid_rating(self, feedback_handler: TelegramHandler) -> None:
        """지원하지 않는 rating 값은 에러 메시지를 표시한다."""
        query = MagicMock()
        query.data = "fb:5:42"
        query.answer = AsyncMock()

        chat = MagicMock()
        chat.id = 111
        chat.type = "private"

        update = MagicMock()
        update.callback_query = query
        update.effective_chat = chat

        context = MagicMock()
        await feedback_handler._handle_feedback_callback(update, context)

        query.answer.assert_awaited_once_with("지원하지 않는 피드백 값입니다.", show_alert=True)

    @pytest.mark.asyncio
    async def test_callback_non_private_answers(self, feedback_handler: TelegramHandler) -> None:
        """private chat이 아니면 콜백을 종료하고 스피너를 해제한다."""
        query = MagicMock()
        query.data = "fb:1:42"
        query.answer = AsyncMock()

        chat = MagicMock()
        chat.id = -100
        chat.type = "group"

        update = MagicMock()
        update.callback_query = query
        update.effective_chat = chat

        context = MagicMock()
        await feedback_handler._handle_feedback_callback(update, context)

        query.answer.assert_awaited_once_with("private chat에서만 사용할 수 있습니다.", show_alert=False)

    @pytest.mark.asyncio
    async def test_callback_auth_failure_answers(self, feedback_handler: TelegramHandler) -> None:
        """인증 실패 시에도 콜백 스피너가 남지 않도록 answer를 호출한다."""
        query = MagicMock()
        query.data = "fb:1:42"
        query.answer = AsyncMock()

        chat = MagicMock()
        chat.id = 999  # allowed_users에 없음
        chat.type = "private"

        update = MagicMock()
        update.callback_query = query
        update.effective_chat = chat

        context = MagicMock()
        await feedback_handler._handle_feedback_callback(update, context)

        query.answer.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_cmd_feedback(self, feedback_handler: TelegramHandler) -> None:
        """/feedback 명령이 통계를 표시한다."""
        chat = MagicMock()
        chat.id = 111
        chat.type = "private"

        message = MagicMock()
        message.reply_text = AsyncMock()

        update = MagicMock()
        update.effective_chat = chat
        update.effective_message = message

        context = MagicMock()
        await feedback_handler._cmd_feedback(update, context)

        assert message.reply_text.await_count == 1
        call_text = message.reply_text.await_args[0][0]
        assert "피드백 통계" in call_text
        assert "10건" in call_text

    @pytest.mark.asyncio
    async def test_status_includes_feedback(self, feedback_handler: TelegramHandler) -> None:
        """/status에 피드백 현황이 포함된다."""
        chat = MagicMock()
        chat.id = 111
        chat.type = "private"

        message = MagicMock()
        message.reply_text = AsyncMock()

        update = MagicMock()
        update.effective_chat = chat
        update.effective_message = message

        context = MagicMock()
        await feedback_handler._cmd_status(update, context)

        call_text = message.reply_text.await_args[0][0]
        assert "피드백" in call_text
        assert "20건" in call_text

    @pytest.mark.asyncio
    async def test_status_includes_degraded_components(self, feedback_handler: TelegramHandler) -> None:
        """degraded 컴포넌트가 있으면 /status에 노출된다."""
        feedback_handler._engine.get_status = AsyncMock(return_value={
            "uptime_seconds": 100,
            "uptime_human": "1분 40초",
            "llm": {"status": "ok"},
            "skills_loaded": 3,
            "current_model": "test-model",
            "degraded_components": {
                "semantic_cache": {
                    "reason": "encoder_unavailable",
                    "degraded_for_seconds": 45,
                }
            },
        })

        chat = MagicMock()
        chat.id = 111
        chat.type = "private"

        message = MagicMock()
        message.reply_text = AsyncMock()

        update = MagicMock()
        update.effective_chat = chat
        update.effective_message = message

        await feedback_handler._cmd_status(update, MagicMock())

        call_text = message.reply_text.await_args[0][0]
        assert "degraded 상태" in call_text
        assert "semantic_cache" in call_text
        assert "encoder_unavailable" in call_text

    @pytest.mark.asyncio
    async def test_reason_or_message_path_does_not_double_rate_limit(
        self,
        mock_engine: AsyncMock,
        mock_feedback: AsyncMock,
    ) -> None:
        """collect_reason 경로에서도 인증/레이트리밋이 1회만 적용된다."""
        config = AppSettings(
            telegram_bot_token="test_token",
            data_dir="/tmp/test",
            bot=BotConfig(),
            lemonade=LemonadeConfig(),
            security=SecurityConfig(allowed_users=[111], rate_limit=1),
            memory=MemoryConfig(),
            telegram=TelegramConfig(),
            feedback=FeedbackConfig(enabled=True, collect_reason=True),
        )
        handler = TelegramHandler(
            config=config,
            engine=mock_engine,
            security=SecurityManager(config.security),
            feedback=mock_feedback,
        )

        async def _stream():
            yield "response"

        handler._engine.process_message_stream = MagicMock(return_value=_stream())

        chat = MagicMock()
        chat.id = 111
        chat.type = "private"
        chat.send_action = AsyncMock()

        sent_message = MagicMock()
        sent_message.edit_text = AsyncMock()
        sent_message.message_id = 42
        sent_message.edit_reply_markup = AsyncMock()

        message = MagicMock()
        message.text = "hello"
        message.reply_text = AsyncMock(return_value=sent_message)

        update = MagicMock()
        update.effective_chat = chat
        update.effective_message = message

        await handler._handle_reason_or_message(update, MagicMock())

        handler._engine.process_message_stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_expired_reason_is_consumed(self, feedback_handler: TelegramHandler) -> None:
        """만료된 사유 입력은 일반 메시지 처리로 전달되지 않는다."""
        feedback_handler._pending_reason[111] = {
            "bot_message_id": 42,
            "expires": time.monotonic() - 1,
        }
        feedback_handler._engine.process_message_stream = MagicMock()

        chat = MagicMock()
        chat.id = 111
        chat.type = "private"
        chat.send_action = AsyncMock()

        message = MagicMock()
        message.text = "늦은 사유"
        message.reply_text = AsyncMock()

        update = MagicMock()
        update.effective_chat = chat
        update.effective_message = message

        await feedback_handler._handle_reason_or_message(update, MagicMock())

        feedback_handler._engine.process_message_stream.assert_not_called()
        message.reply_text.assert_awaited_once_with("사유 입력 시간이 만료되었습니다.")


class TestApplicationGuards:
    def test_application_before_initialize_raises(self, telegram_handler: TelegramHandler) -> None:
        with pytest.raises(RuntimeError, match="초기화"):
            _ = telegram_handler.application

    @pytest.mark.asyncio
    async def test_send_message_before_initialize_raises(self, telegram_handler: TelegramHandler) -> None:
        with pytest.raises(RuntimeError, match="초기화"):
            await telegram_handler.send_message(111, "hello")
