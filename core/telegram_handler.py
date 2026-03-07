"""텔레그램 봇 핸들러 — 메시지 수신/발신, 명령어 처리.

사용자 인터페이스 계층. 인증과 레이트리밋을 적용하고,
엔진에 메시지를 전달하여 응답을 텔레그램으로 전송한다.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from telegram import BotCommand, Update
from telegram.constants import ChatType
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from core import telegram_commands, telegram_feedback, telegram_messages, telegram_sim
from core.config import AppSettings
from core.engine import Engine
from core.feedback_manager import FeedbackManager
from core.logging_setup import get_logger
from core.security import (
    AuthenticationError,
    GlobalConcurrencyError,
    RateLimitError,
    SecurityManager,
)
from core.telegram_message_renderer import escape_html, split_message, stream_and_render
from core.text_utils import detect_output_anomalies

if TYPE_CHECKING:
    from core.semantic_cache import SemanticCache
    from core.sim_scheduler import SimJobScheduler


def _auth_required(func: Callable) -> Callable:
    """private chat + 인증 + 레이트리밋을 적용하는 데코레이터."""

    async def wrapper(self: TelegramHandler, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_chat or not update.effective_message:
            return

        chat = update.effective_chat
        message = update.effective_message

        if chat.type != ChatType.PRIVATE:
            self._logger.warning(
                "non_private_chat_blocked",
                chat_id=chat.id,
                chat_type=chat.type,
            )
            with suppress(Exception):
                await message.reply_text("이 봇은 private chat에서만 동작합니다.")
            return

        chat_id = chat.id

        try:
            self._authorize_chat_id(chat_id)
        except AuthenticationError:
            self._logger.warning("unauthorized_access", chat_id=chat_id)
            return
        except RateLimitError:
            await update.effective_message.reply_text(
                "요청이 너무 많습니다. 잠시 후 다시 시도해주세요."
            )
            return

        return await func(self, update, context)

    return wrapper


def _global_slot_required(func: Callable) -> Callable:
    """전역 동시성 슬롯을 적용하는 데코레이터."""

    async def wrapper(self: TelegramHandler, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_chat:
            return

        chat_id = update.effective_chat.id
        try:
            async with self._security.global_slot(chat_id):
                return await func(self, update, context)
        except GlobalConcurrencyError:
            query = update.callback_query
            if query is not None:
                await query.answer("현재 요청이 많습니다. 잠시 후 다시 시도해주세요.", show_alert=True)
                return
            if update.effective_message is not None:
                await update.effective_message.reply_text(
                    "현재 요청이 많습니다. 잠시 후 다시 시도해주세요."
                )
            return

    return wrapper


class TelegramHandler:
    """텔레그램 봇 메시지 핸들러."""

    def __init__(
        self,
        config: AppSettings,
        engine: Engine,
        security: SecurityManager,
        feedback: FeedbackManager | None = None,
        semantic_cache: SemanticCache | None = None,
    ) -> None:
        self._config = config
        self._engine = engine
        self._security = security
        self._feedback = feedback
        self._semantic_cache = semantic_cache
        self._app: Application | None = None
        self._logger = get_logger("telegram")
        self._max_message_length = config.telegram.max_message_length
        self._scheduler = None
        self._sim_scheduler: SimJobScheduler | None = None
        self._preview_cache: dict[tuple[int, int], dict] = {}
        self._pending_reason: dict[int, dict] = {}
        self._pending_continuation: dict[int, dict[str, Any]] = {}

    def set_scheduler(self, scheduler) -> None:
        self._scheduler = scheduler

    def has_scheduler(self) -> bool:
        return self._scheduler is not None

    def set_sim_scheduler(self, sim_scheduler: SimJobScheduler) -> None:
        self._sim_scheduler = sim_scheduler

    @property
    def _feedback_enabled(self) -> bool:
        return self._feedback is not None and self._config.feedback.enabled

    def _authorize_chat_id(self, chat_id: int) -> None:
        self._security.authenticate(chat_id)
        self._security.check_rate_limit(chat_id)

    def _build_command_handlers(self) -> list:
        handlers = [
            CommandHandler("start", self._cmd_start),
            CommandHandler("help", self._cmd_help),
            CommandHandler("skills", self._cmd_skills),
            CommandHandler("auto", self._cmd_auto),
            CommandHandler("memory", self._cmd_memory),
            CommandHandler("status", self._cmd_status),
            CommandHandler("continue", self._cmd_continue),
        ]
        if self._feedback_enabled:
            handlers.append(CommandHandler("feedback", self._cmd_feedback))
        if self._sim_scheduler is not None:
            handlers.append(CommandHandler("sim", self._cmd_sim))
        return handlers

    def _build_bot_commands(self) -> list[BotCommand]:
        commands = [
            BotCommand("start", "봇 시작"),
            BotCommand("help", "도움말"),
            BotCommand("skills", "스킬 목록"),
            BotCommand("auto", "자동화 관리"),
            BotCommand("memory", "메모리 관리"),
            BotCommand("status", "시스템 상태"),
            BotCommand("continue", "긴 답변 이어보기"),
        ]
        if self._feedback_enabled:
            commands.append(BotCommand("feedback", "피드백 통계"))
        if self._sim_scheduler is not None:
            commands.append(BotCommand("sim", "시뮬레이션 큐 관리"))
        return commands

    async def initialize(self) -> Application:
        """텔레그램 Application을 생성하고 핸들러를 등록한다.

        Returns:
            초기화된 telegram.ext.Application 인스턴스.
        """
        self._app = ApplicationBuilder().token(self._config.telegram_bot_token).build()

        for handler in self._build_command_handlers():
            self._app.add_handler(handler)

        if self._feedback_enabled:
            self._app.add_handler(
                CallbackQueryHandler(self._handle_feedback_callback, pattern=r"^fb:")
            )

        if self._feedback_enabled and self._config.feedback.collect_reason:
            self._app.add_handler(CommandHandler("skip", self._handle_reason_skip))
            self._app.add_handler(
                MessageHandler(
                    filters.TEXT & ~filters.COMMAND,
                    self._handle_reason_or_message,
                )
            )
        else:
            self._app.add_handler(
                MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
            )

        self._app.add_handler(MessageHandler(filters.PHOTO, self._handle_message))
        self._app.add_error_handler(self._error_handler)
        await self._app.bot.set_my_commands(self._build_bot_commands())

        self._logger.info("telegram_handler_initialized")
        return self._app

    @_auth_required
    @_global_slot_required
    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await telegram_commands.cmd_start(self, update, context)

    @_auth_required
    @_global_slot_required
    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await telegram_commands.cmd_help(self, update, context)

    @_auth_required
    @_global_slot_required
    async def _cmd_skills(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await telegram_commands.cmd_skills(self, update, context)

    @_auth_required
    @_global_slot_required
    async def _cmd_continue(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await telegram_commands.cmd_continue(self, update, context)

    @_auth_required
    @_global_slot_required
    async def _cmd_auto(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await telegram_commands.cmd_auto(self, update, context)

    async def _handle_auto_list(self, update: Update) -> None:
        await telegram_commands.handle_auto_list(self, update)

    async def _handle_auto_disable(self, update: Update, name: str) -> None:
        await telegram_commands.handle_auto_disable(self, update, name)

    async def _handle_auto_reload(self, update: Update) -> None:
        await telegram_commands.handle_auto_reload(self, update)

    async def _handle_auto_run(self, update: Update, name: str) -> None:
        await telegram_commands.handle_auto_run(self, update, name)

    @_auth_required
    @_global_slot_required
    async def _cmd_memory(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await telegram_commands.cmd_memory(self, update, context)

    @_auth_required
    @_global_slot_required
    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await telegram_commands.cmd_status(self, update, context)

    @staticmethod
    def _should_auto_trigger_analyze_all(text: str) -> bool:
        return telegram_messages.should_auto_trigger_analyze_all(text)

    async def _run_analyze_all_flow(
        self,
        *,
        chat: Any,
        message: Any,
        query: str,
        auto_triggered: bool,
    ) -> None:
        await telegram_messages.run_analyze_all_flow(
            self,
            chat=chat,
            message=message,
            query=query,
            auto_triggered=auto_triggered,
        )

    @_auth_required
    @_global_slot_required
    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """자유 텍스트/이미지 메시지를 수신하여 엔진에 전달하고 스트리밍 응답을 반환한다."""
        await telegram_messages.handle_message(self, update, context)

    async def _handle_message_impl(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        *,
        text_override: str | None = None,
        force_continuation: bool = False,
        auto_continuation_turn: int = 0,
    ) -> None:
        """메시지 처리 핵심 구현. 스트리밍·폴백·자동 이어보기를 처리한다."""
        await telegram_messages.handle_message_impl(
            self,
            update,
            context,
            text_override=text_override,
            force_continuation=force_continuation,
            auto_continuation_turn=auto_continuation_turn,
            stream_and_render_fn=stream_and_render,
            detect_output_anomalies_fn=detect_output_anomalies,
        )

    @staticmethod
    def _is_continue_request(text: str) -> bool:
        return telegram_messages.is_continue_request(text)

    def _cleanup_pending_continuations(self) -> None:
        telegram_messages.cleanup_pending_continuations(
            self,
            monotonic_fn=time.monotonic,
        )

    def _take_pending_continuation(self, chat_id: int) -> dict[str, Any] | None:
        return telegram_messages.take_pending_continuation(
            self,
            chat_id,
            monotonic_fn=time.monotonic,
        )

    def _set_pending_continuation(
        self,
        chat_id: int,
        *,
        root_query: str,
        turn: int,
    ) -> None:
        telegram_messages.set_pending_continuation(
            self,
            chat_id,
            root_query=root_query,
            turn=turn,
            monotonic_fn=time.monotonic,
        )

    @staticmethod
    def _build_continuation_prompt(pending: dict[str, Any]) -> str:
        return telegram_messages.build_continuation_prompt(pending)

    @staticmethod
    def _truncate_summary_line(text: str, *, max_chars: int) -> str:
        return telegram_messages.truncate_summary_line(text, max_chars=max_chars)

    @classmethod
    def _extract_summary_points(
        cls,
        text: str,
        *,
        max_points: int = 3,
    ) -> list[str]:
        return telegram_messages.extract_summary_points(
            cls,
            text,
            max_points=max_points,
        )

    @classmethod
    def _build_long_response_followup_message(cls, response_text: str) -> str:
        return telegram_messages.build_long_response_followup_message(cls, response_text)

    @staticmethod
    async def _keep_typing(chat: Any, stop_event) -> None:
        await telegram_messages.keep_typing(chat, stop_event)

    async def _link_feedback_target(self, chat_id: int, result: Any) -> None:
        await telegram_feedback.link_feedback_target(self, chat_id, result)

    async def _attach_feedback_controls(
        self,
        chat_id: int,
        user_text: str,
        result: Any,
    ) -> None:
        await telegram_feedback.attach_feedback_controls(
            self,
            chat_id,
            user_text,
            result,
        )

    def _cleanup_preview_cache(self) -> None:
        telegram_feedback.cleanup_preview_cache(self)

    def _cleanup_pending_reasons(self) -> None:
        telegram_feedback.cleanup_pending_reasons(self)

    def _cache_preview(
        self,
        chat_id: int,
        bot_message_id: int,
        user_text: str,
        bot_text: str,
    ) -> None:
        telegram_feedback.cache_preview(
            self,
            chat_id,
            bot_message_id,
            user_text,
            bot_text,
        )

    @staticmethod
    def _parse_feedback_callback_data(data: str | None) -> tuple[int, int] | None:
        return telegram_feedback.parse_feedback_callback_data(data)

    async def _authorize_feedback_callback(self, chat_id: int, query) -> bool:
        return await telegram_feedback.authorize_feedback_callback(self, chat_id, query)

    @_global_slot_required
    async def _handle_feedback_callback(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        await telegram_feedback.handle_feedback_callback(self, update, context)

    @_auth_required
    @_global_slot_required
    async def _cmd_feedback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await telegram_feedback.cmd_feedback(self, update, context)

    async def _handle_reason_input(self, chat_id: int, text: str, update: Update) -> bool:
        return await telegram_feedback.handle_reason_input(self, chat_id, text, update)

    @_auth_required
    @_global_slot_required
    async def _handle_reason_skip(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await telegram_feedback.handle_reason_skip(self, update, context)

    @_auth_required
    @_global_slot_required
    async def _handle_reason_or_message(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        await telegram_feedback.handle_reason_or_message(self, update, context)

    @_auth_required
    @_global_slot_required
    async def _cmd_sim(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await telegram_sim.cmd_sim(self, update, context)

    async def _get_sim_scheduler(self, update: Update) -> SimJobScheduler | None:
        return await telegram_sim.get_sim_scheduler(self, update)

    async def _sim_submit(self, update: Update, args: list[str]) -> None:
        await telegram_sim.sim_submit(self, update, args)

    @staticmethod
    def _sim_elapsed_text(job: dict[str, Any]) -> str:
        return telegram_sim.sim_elapsed_text(job)

    async def _sim_list(self, update: Update, args: list[str]) -> None:
        await telegram_sim.sim_list(self, update, args)

    async def _sim_clear(self, update: Update, args: list[str]) -> None:
        await telegram_sim.sim_clear(self, update, args)

    async def _sim_status(self, update: Update, args: list[str]) -> None:
        await telegram_sim.sim_status(self, update, args)

    async def _sim_info(self, update: Update, args: list[str]) -> None:
        await telegram_sim.sim_info(self, update, args)

    async def _sim_cancel(self, update: Update, args: list[str]) -> None:
        await telegram_sim.sim_cancel(self, update, args)

    async def _sim_priority(self, update: Update, args: list[str]) -> None:
        await telegram_sim.sim_priority(self, update, args)

    async def _sim_retry(self, update: Update, args: list[str]) -> None:
        await telegram_sim.sim_retry(self, update, args)

    async def _sim_tools(self, update: Update, args: list[str]) -> None:
        await telegram_sim.sim_tools(self, update, args)

    async def _error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        self._logger.error(
            "telegram_error",
            error=str(context.error),
            update=str(update),
        )

    async def send_message(
        self,
        chat_id: int,
        text: str,
        parse_mode: str | None = None,
    ) -> None:
        """지정 채팅에 메시지를 전송한다. 긴 메시지는 자동 분할된다."""
        if self._app is None:
            raise RuntimeError("TelegramHandler가 아직 초기화되지 않았습니다.")
        for part in self._split_message(text):
            await self._app.bot.send_message(
                chat_id=chat_id,
                text=part,
                parse_mode=parse_mode,
            )

    @staticmethod
    def _escape_html(value: object) -> str:
        return escape_html(value)

    @staticmethod
    def _format_memory_gb(value_mb: object) -> str:
        mb = 0.0
        if isinstance(value_mb, bool):
            mb = 0.0
        elif isinstance(value_mb, int | float):
            mb = max(0.0, float(value_mb))
        elif isinstance(value_mb, str):
            try:
                mb = max(0.0, float(value_mb.strip()))
            except ValueError:
                mb = 0.0

        gb = mb / 1024.0
        rounded = round(gb)
        if abs(gb - rounded) < 1e-9:
            return f"{int(rounded)}GB"
        if gb >= 10:
            return f"{gb:.1f}GB"
        return f"{gb:.2f}GB"

    @staticmethod
    def _coerce_error_list(value: object) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        return []

    def _get_skill_reload_errors(self) -> list[str]:
        getter = getattr(self._engine, "get_last_skill_load_errors", None)
        if not callable(getter):
            return []
        try:
            errors = getter()
        except Exception:
            return []
        if inspect.isawaitable(errors):
            closer = getattr(errors, "close", None)
            if callable(closer):
                closer()
            return []
        return self._coerce_error_list(errors)

    def _get_auto_reload_errors(self) -> list[str]:
        if self._scheduler is None:
            return []
        getter = getattr(self._scheduler, "get_last_load_errors", None)
        if not callable(getter):
            return []
        try:
            errors = getter()
        except Exception:
            return []
        if inspect.isawaitable(errors):
            closer = getattr(errors, "close", None)
            if callable(closer):
                closer()
            return []
        return self._coerce_error_list(errors)

    @staticmethod
    def _format_reload_warnings(errors: list[str], max_items: int = 3) -> str:
        preview = errors[:max_items]
        lines = [f"\n\n⚠️ 일부 항목 로드 실패({len(errors)}건)"]
        lines.extend(f"- {item}" for item in preview)
        if len(errors) > max_items:
            lines.append(f"- ... 외 {len(errors) - max_items}건")
        return "\n".join(lines)

    def _split_message(self, text: str, max_length: int | None = None) -> list[str]:
        max_length = max_length or self._max_message_length
        return split_message(text, max_length=max_length)

    @property
    def application(self) -> Application:
        if self._app is None:
            raise RuntimeError("TelegramHandler가 아직 초기화되지 않았습니다.")
        return self._app
