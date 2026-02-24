"""텔레그램 봇 핸들러 — 메시지 수신/발신, 명령어 처리.

사용자 인터페이스 계층. 인증과 레이트리밋을 적용하고,
엔진에 메시지를 전달하여 응답을 텔레그램으로 전송한다.
"""

from __future__ import annotations

import functools
import inspect
import time
from collections.abc import Callable
from pathlib import Path

from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ChatType, ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from core.config import AppSettings
from core.engine import Engine
from core.feedback_manager import FeedbackManager
from core.logging_setup import get_logger
from core.security import AuthenticationError, RateLimitError, SecurityManager
from core.telegram_message_renderer import escape_html, split_message, stream_and_render

# 메시지 편집 최소 간격 (초) — 텔레그램 API 제한 대응
_EDIT_INTERVAL = 1.0
_EDIT_CHAR_THRESHOLD = 100


def _auth_required(func: Callable) -> Callable:
    """인증, 레이트리밋, 입력 검증을 적용하는 데코레이터."""

    @functools.wraps(func)
    async def wrapper(self: TelegramHandler, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_chat or not update.effective_message:
            return

        chat = update.effective_chat
        message = update.effective_message

        # private chat만 지원
        if chat.type != ChatType.PRIVATE:
            self._logger.warning(
                "non_private_chat_blocked",
                chat_id=chat.id,
                chat_type=chat.type,
            )
            try:
                await message.reply_text("이 봇은 private chat에서만 동작합니다.")
            except Exception:
                pass
            return

        chat_id = chat.id

        try:
            self._security.authenticate(chat_id)
            self._security.check_rate_limit(chat_id)
        except AuthenticationError:
            self._logger.warning("unauthorized_access", chat_id=chat_id)
            return  # 미인증 사용자에게는 아무 응답도 하지 않음
        except RateLimitError:
            await update.effective_message.reply_text(
                "요청이 너무 많습니다. 잠시 후 다시 시도해주세요."
            )
            return

        return await func(self, update, context)

    return wrapper


class TelegramHandler:
    """텔레그램 봇 메시지 핸들러."""

    def __init__(
        self,
        config: AppSettings,
        engine: Engine,
        security: SecurityManager,
        feedback: FeedbackManager | None = None,
    ) -> None:
        self._config = config
        self._engine = engine
        self._security = security
        self._feedback = feedback
        self._app: Application | None = None
        self._logger = get_logger("telegram")
        self._max_message_length = config.telegram.max_message_length
        # auto_scheduler 참조 (main.py에서 주입)
        self._scheduler = None
        # 프리뷰 캐시: {(chat_id, bot_message_id): {"user": str, "bot": str, "ts": float}}
        self._preview_cache: dict[tuple[int, int], dict] = {}

    def set_scheduler(self, scheduler) -> None:
        """auto_scheduler 참조를 설정한다 (순환 의존 방지)."""
        self._scheduler = scheduler

    @property
    def _feedback_enabled(self) -> bool:
        return self._feedback is not None and self._config.feedback.enabled

    def _build_command_handlers(self) -> list:
        handlers = [
            CommandHandler("start", self._cmd_start),
            CommandHandler("help", self._cmd_help),
            CommandHandler("skills", self._cmd_skills),
            CommandHandler("auto", self._cmd_auto),
            CommandHandler("model", self._cmd_model),
            CommandHandler("memory", self._cmd_memory),
            CommandHandler("status", self._cmd_status),
        ]
        if self._feedback_enabled:
            handlers.append(CommandHandler("feedback", self._cmd_feedback))
        return handlers

    def _build_bot_commands(self) -> list[BotCommand]:
        commands = [
            BotCommand("start", "봇 시작"),
            BotCommand("help", "도움말"),
            BotCommand("skills", "스킬 목록"),
            BotCommand("auto", "자동화 관리"),
            BotCommand("model", "모델 관리"),
            BotCommand("memory", "메모리 관리"),
            BotCommand("status", "시스템 상태"),
        ]
        if self._feedback_enabled:
            commands.append(BotCommand("feedback", "피드백 통계"))
        return commands

    async def initialize(self) -> Application:
        """텔레그램 Application을 생성하고 핸들러를 등록한다."""
        self._app = (
            ApplicationBuilder()
            .token(self._config.telegram_bot_token)
            .build()
        )

        # 명령어 핸들러 등록
        handlers = self._build_command_handlers()
        for handler in handlers:
            self._app.add_handler(handler)

        # 피드백 콜백 핸들러
        if self._feedback_enabled:
            self._app.add_handler(
                CallbackQueryHandler(self._handle_feedback_callback, pattern=r"^fb:")
            )

        # 일반 텍스트 메시지 핸들러 (명령어 제외)
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )

        # 에러 핸들러
        self._app.add_error_handler(self._error_handler)

        # 봇 명령어 목록 설정
        commands = self._build_bot_commands()
        await self._app.bot.set_my_commands(commands)

        self._logger.info("telegram_handler_initialized")
        return self._app

    # ── 명령어 핸들러 ──

    @_auth_required
    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        welcome = (
            f"안녕하세요! {self._config.bot.name} 입니다.\n\n"
            "로컬 LLM(Ollama) 기반 AI 어시스턴트입니다.\n"
            "자유롭게 대화하거나, /help 명령으로 도움말을 확인하세요."
        )
        await update.effective_message.reply_text(welcome)  # type: ignore[union-attr]

    @_auth_required
    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        command_lines = [
            "/start — 봇 시작",
            "/help — 이 도움말 표시",
            "/skills — 스킬 목록/리로드",
            "/auto — 자동화 관리/리로드",
            "/model — 모델 확인/변경",
            "/memory — 메모리 관리",
            "/status — 시스템 상태",
        ]
        if self._feedback_enabled:
            command_lines.insert(6, "/feedback — 피드백 통계")

        help_text = (
            "📋 *사용 가능한 명령어*\n\n"
            + "\n".join(command_lines)
            + "\n\n"
            "💬 *대화 모드*\n"
            "명령어 없이 자유롭게 대화하세요.\n\n"
            "🔧 *스킬 모드*\n"
            "스킬 트리거 키워드를 사용하면 전문 기능이 활성화됩니다.\n"
            "/skills 명령으로 스킬 목록을 확인하고, /skills reload로 다시 로드할 수 있습니다."
        )
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            help_text, parse_mode=ParseMode.MARKDOWN
        )

    @_auth_required
    async def _cmd_skills(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        args = context.args or []
        if args and args[0] == "reload":
            try:
                count = await self._engine.reload_skills(strict=True)
                errors = self._get_skill_reload_errors()
                message = f"스킬을 다시 로드했습니다: {count}개"
                if errors:
                    message += self._format_reload_warnings(errors)
                await update.effective_message.reply_text(  # type: ignore[union-attr]
                    message
                )
            except Exception as exc:
                self._logger.error("skills_reload_failed", error=str(exc))
                await update.effective_message.reply_text(  # type: ignore[union-attr]
                    f"스킬 로드 실패: {exc}"
                )
            return

        skills = self._engine.list_skills()
        if not skills:
            await update.effective_message.reply_text("등록된 스킬이 없습니다.")  # type: ignore[union-attr]
            return

        lines = ["🔧 <b>사용 가능한 스킬</b>\n"]
        for s in skills:
            triggers = ", ".join(f"<code>{self._escape_html(t)}</code>" for t in s["triggers"])
            lines.append(
                f"• <b>{self._escape_html(s['name'])}</b> — "
                f"{self._escape_html(s['description'])}\n"
                f"  트리거: {triggers}"
            )

        await update.effective_message.reply_text(  # type: ignore[union-attr]
            "\n".join(lines), parse_mode=ParseMode.HTML
        )

    @_auth_required
    async def _cmd_auto(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._scheduler:
            await update.effective_message.reply_text("자동화 스케줄러가 초기화되지 않았습니다.")  # type: ignore[union-attr]
            return

        args = context.args or []
        if not args or args[0] == "list":
            await self._handle_auto_list(update)
            return

        if len(args) >= 2 and args[0] == "enable":
            await self._handle_auto_toggle(update, name=args[1], enable=True)
            return

        if len(args) >= 2 and args[0] == "disable":
            await self._handle_auto_toggle(update, name=args[1], enable=False)
            return

        if args[0] == "reload":
            await self._handle_auto_reload(update)
            return

        await update.effective_message.reply_text(
            "사용법: /auto [list|enable <이름>|disable <이름>|reload]"
        )

    async def _handle_auto_list(self, update: Update) -> None:
        if self._scheduler is None:
            await update.effective_message.reply_text("자동화 스케줄러가 초기화되지 않았습니다.")  # type: ignore[union-attr]
            return
        automations = self._scheduler.list_automations()
        if not automations:
            await update.effective_message.reply_text("등록된 자동화가 없습니다.")
            return

        lines = ["⏰ <b>자동화 목록</b>\n"]
        for auto in automations:
            status = "✅" if auto["enabled"] else "❌"
            lines.append(
                f"{status} <b>{self._escape_html(auto['name'])}</b> — "
                f"{self._escape_html(auto['description'])}\n"
                f"  스케줄: <code>{self._escape_html(auto['schedule'])}</code>"
            )
        await update.effective_message.reply_text(
            "\n".join(lines), parse_mode=ParseMode.HTML
        )

    async def _handle_auto_toggle(self, update: Update, name: str, *, enable: bool) -> None:
        if self._scheduler is None:
            await update.effective_message.reply_text("자동화 스케줄러가 초기화되지 않았습니다.")  # type: ignore[union-attr]
            return
        if enable:
            result = await self._scheduler.enable_automation(name)
            message = f"'{name}' 자동화가 활성화되었습니다." if result else f"'{name}' 자동화를 찾을 수 없습니다."
        else:
            result = await self._scheduler.disable_automation(name)
            message = f"'{name}' 자동화가 비활성화되었습니다." if result else f"'{name}' 자동화를 찾을 수 없습니다."
        await update.effective_message.reply_text(message)

    async def _handle_auto_reload(self, update: Update) -> None:
        if self._scheduler is None:
            await update.effective_message.reply_text("자동화 스케줄러가 초기화되지 않았습니다.")  # type: ignore[union-attr]
            return
        try:
            count = await self._scheduler.reload_automations(strict=True)
            errors = self._get_auto_reload_errors()
            message = f"자동화를 다시 로드했습니다: {count}개"
            if errors:
                message += self._format_reload_warnings(errors)
            await update.effective_message.reply_text(
                message
            )
        except Exception as exc:
            self._logger.error("auto_reload_failed", error=str(exc))
            await update.effective_message.reply_text(
                f"자동화 로드 실패: {exc}"
            )

    @_auth_required
    async def _cmd_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        args = context.args or []
        if not args:
            await self._show_current_model(update)
            return

        if args[0] == "list":
            await self._show_available_models(update, chat_id)
            return

        await self._change_model(update, chat_id, args[0])

    async def _show_current_model(self, update: Update) -> None:
        current = self._engine.get_current_model()
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            f"현재 모델: <code>{self._escape_html(current)}</code>\n\n"
            "모델 변경: <code>/model &lt;모델명&gt;</code>\n"
            "모델 목록: <code>/model list</code>",
            parse_mode=ParseMode.HTML,
        )

    async def _show_available_models(self, update: Update, chat_id: int) -> None:
        try:
            models = await self._engine.list_models()
        except Exception as exc:
            self._logger.error(
                "model_list_failed",
                chat_id=chat_id,
                error=str(exc),
            )
            await update.effective_message.reply_text(  # type: ignore[union-attr]
                "모델 목록을 가져오지 못했습니다. Ollama 상태를 확인해주세요."
            )
            return

        if not models:
            await update.effective_message.reply_text("설치된 모델이 없습니다.")  # type: ignore[union-attr]
            return
        lines = ["📦 <b>설치된 모델</b>\n"]
        for model in models:
            size_mb = model["size"] / (1024 * 1024) if model["size"] else 0
            lines.append(
                f"• <code>{self._escape_html(model['name'])}</code> ({size_mb:.0f}MB)"
            )
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            "\n".join(lines), parse_mode=ParseMode.HTML
        )

    async def _change_model(self, update: Update, chat_id: int, requested_model: str) -> None:
        try:
            result = await self._engine.change_model(requested_model)
        except Exception as exc:
            self._logger.error(
                "model_change_failed",
                chat_id=chat_id,
                requested_model=requested_model,
                error=str(exc),
            )
            await update.effective_message.reply_text(  # type: ignore[union-attr]
                "모델 변경 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
            )
            return

        if result["success"]:
            await update.effective_message.reply_text(  # type: ignore[union-attr]
                "모델 변경: "
                f"<code>{self._escape_html(result['old_model'])}</code> → "
                f"<code>{self._escape_html(result['new_model'])}</code>",
                parse_mode=ParseMode.HTML,
            )
            return

        await update.effective_message.reply_text(  # type: ignore[union-attr]
            f"모델 변경 실패: {result['error']}"
        )

    @_auth_required
    async def _cmd_memory(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        args = context.args or []

        if not args:
            stats = await self._engine.get_memory_stats(chat_id)
            await update.effective_message.reply_text(  # type: ignore[union-attr]
                f"🧠 <b>메모리 상태</b>\n\n"
                f"대화 기록: {stats['conversation_count']}건\n"
                f"장기 메모리: {stats['memory_count']}건\n"
                f"가장 오래된 대화: "
                f"{self._escape_html(stats['oldest_conversation'] or '없음')}",
                parse_mode=ParseMode.HTML,
            )
        elif args[0] == "clear":
            deleted = await self._engine.clear_conversation(chat_id)
            await update.effective_message.reply_text(  # type: ignore[union-attr]
                f"대화 기록 {deleted}건이 삭제되었습니다."
            )
        elif args[0] == "export":
            output_dir = Path(self._config.data_dir) / "conversations"
            filepath = await self._engine.export_conversation_markdown(
                chat_id, output_dir
            )
            await update.effective_message.reply_text(  # type: ignore[union-attr]
                "대화 기록이 내보내기되었습니다: "
                f"<code>{self._escape_html(filepath.name)}</code>",
                parse_mode=ParseMode.HTML,
            )
        else:
            await update.effective_message.reply_text(  # type: ignore[union-attr]
                "사용법: /memory [clear|export]"
            )

    @_auth_required
    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        status = await self._engine.get_status()
        ollama = status["ollama"]
        ollama_status = "🟢 정상" if ollama.get("status") == "ok" else "🔴 오류"

        text = (
            f"📊 <b>시스템 상태</b>\n\n"
            f"가동 시간: {status['uptime_human']}\n"
            f"Ollama: {ollama_status}\n"
            f"모델: <code>{self._escape_html(status['current_model'])}</code>\n"
            f"로드된 스킬: {status['skills_loaded']}개"
        )

        if self._scheduler:
            autos = self._scheduler.list_automations()
            enabled = sum(1 for a in autos if a["enabled"])
            text += f"\n자동화: {enabled}/{len(autos)}개 활성"

        if self._feedback:
            global_stats = await self._feedback.get_global_stats()
            text += (
                f"\n📊 피드백: {global_stats['total']}건 "
                f"(만족도 {global_stats['satisfaction_rate']:.0%})"
            )

        await update.effective_message.reply_text(  # type: ignore[union-attr]
            text, parse_mode=ParseMode.HTML
        )

    # ── 일반 메시지 핸들러 (스트리밍) ──

    @_auth_required
    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """자유 텍스트 메시지를 처리한다. 스트리밍 UX를 제공한다."""
        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        text = update.effective_message.text  # type: ignore[union-attr]
        if text is None:
            return

        # 입력 정제
        text = self._security.sanitize_input(text)
        if not text.strip():
            return

        # 타이핑 표시
        await update.effective_chat.send_action(ChatAction.TYPING)  # type: ignore[union-attr]

        try:
            # 초기 placeholder 메시지 전송
            sent_message = await update.effective_message.reply_text("...")  # type: ignore[union-attr]

            result = await stream_and_render(
                stream=self._engine.process_message_stream(chat_id, text),
                sent_message=sent_message,
                reply_text=update.effective_message.reply_text,  # type: ignore[union-attr]
                split_message_fn=self._split_message,
                edit_interval=_EDIT_INTERVAL,
                edit_char_threshold=_EDIT_CHAR_THRESHOLD,
                max_edit_length=self._max_message_length,
            )

            # 피드백 버튼 부착
            if (
                self._feedback_enabled
                and self._config.feedback.show_buttons
                and result.full_response.strip()
                and result.last_message
            ):
                target_msg = result.last_message
                self._cache_preview(chat_id, target_msg.message_id, text, result.full_response)
                keyboard = InlineKeyboardMarkup([
                    [
                        InlineKeyboardButton("\U0001f44d", callback_data=f"fb:1:{target_msg.message_id}"),
                        InlineKeyboardButton("\U0001f44e", callback_data=f"fb:-1:{target_msg.message_id}"),
                    ]
                ])
                try:
                    await target_msg.edit_reply_markup(reply_markup=keyboard)
                except Exception:
                    pass  # 편집 실패 시 버튼 없이 진행

        except Exception as exc:
            self._logger.error(
                "message_processing_error",
                chat_id=chat_id,
                error=str(exc),
            )
            await update.effective_message.reply_text(  # type: ignore[union-attr]
                "죄송합니다. 메시지 처리 중 오류가 발생했습니다."
            )

    # ── 피드백 ──

    def _cache_preview(self, chat_id: int, bot_message_id: int, user_text: str, bot_text: str) -> None:
        """프리뷰를 캐시에 저장한다. TTL 초과/크기 초과 시 정리."""
        max_chars = self._config.feedback.preview_max_chars
        max_size = self._config.feedback.preview_cache_max_size
        ttl_hours = self._config.feedback.preview_cache_ttl_hours
        if max_chars <= 0 or max_size <= 0 or ttl_hours <= 0:
            return
        now = time.monotonic()

        # TTL 만료 항목 정리
        ttl_seconds = ttl_hours * 3600
        expired = [k for k, v in self._preview_cache.items() if now - v["ts"] > ttl_seconds]
        for k in expired:
            del self._preview_cache[k]

        # 최대 크기 초과 시 가장 오래된 항목 제거
        while len(self._preview_cache) >= max_size:
            oldest_key = min(self._preview_cache, key=lambda k: self._preview_cache[k]["ts"])
            del self._preview_cache[oldest_key]

        self._preview_cache[(chat_id, bot_message_id)] = {
            "user": user_text[:max_chars],
            "bot": bot_text[:max_chars],
            "ts": now,
        }

    @staticmethod
    def _parse_feedback_callback_data(data: str | None) -> tuple[int, int] | None:
        if not data:
            return None
        try:
            _, rating_str, msg_id_str = data.split(":")
            return int(rating_str), int(msg_id_str)
        except (ValueError, AttributeError):
            return None

    async def _authorize_feedback_callback(self, chat_id: int, query) -> bool:
        try:
            self._security.authenticate(chat_id)
            self._security.check_rate_limit(chat_id)
        except AuthenticationError:
            await query.answer()
            return False
        except RateLimitError:
            await query.answer("요청이 너무 많습니다. 잠시 후 다시 시도해주세요.", show_alert=True)
            return False
        return True

    async def _handle_feedback_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """인라인 피드백 버튼 콜백을 처리한다."""
        query = update.callback_query
        if not query or not update.effective_chat:
            return
        if self._feedback is None:
            await query.answer()
            return
        if update.effective_chat.type != ChatType.PRIVATE:
            await query.answer("private chat에서만 사용할 수 있습니다.", show_alert=False)
            return

        parsed = self._parse_feedback_callback_data(query.data)
        if parsed is None:
            await query.answer("잘못된 피드백 요청입니다.", show_alert=True)
            return
        rating, bot_message_id = parsed

        chat_id = update.effective_chat.id
        if not await self._authorize_feedback_callback(chat_id, query):
            return

        if rating not in (-1, 1):
            await query.answer("지원하지 않는 피드백 값입니다.", show_alert=True)
            return

        preview = self._preview_cache.get((chat_id, bot_message_id), {})
        is_update = await self._feedback.store_feedback(
            chat_id=chat_id,
            bot_message_id=bot_message_id,
            rating=rating,
            user_preview=preview.get("user"),
            bot_preview=preview.get("bot"),
        )

        if is_update:
            await query.answer("피드백을 업데이트했어요.", show_alert=False)
        else:
            await query.answer("피드백 감사합니다!", show_alert=False)

    @_auth_required
    async def _cmd_feedback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """피드백 통계를 표시한다."""
        if self._feedback is None:
            await update.effective_message.reply_text("피드백 기능이 비활성화되어 있습니다.")  # type: ignore[union-attr]
            return
        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        stats = await self._feedback.get_user_stats(chat_id)
        text = (
            "📊 <b>피드백 통계</b>\n\n"
            f"전체: {stats['total']}건\n"
            f"👍 긍정: {stats['positive']}건\n"
            f"👎 부정: {stats['negative']}건\n"
            f"만족도: {stats['satisfaction_rate']:.0%}"
        )
        await update.effective_message.reply_text(text, parse_mode=ParseMode.HTML)  # type: ignore[union-attr]

    # ── 에러 핸들러 ──

    async def _error_handler(
        self, update: object, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        self._logger.error(
            "telegram_error",
            error=str(context.error),
            update=str(update),
        )

    # ── 유틸리티 ──

    async def send_message(self, chat_id: int, text: str) -> None:
        """능동적으로 메시지를 전송한다 (auto_scheduler용)."""
        if self._app is None:
            raise RuntimeError("TelegramHandler가 아직 초기화되지 않았습니다.")
        for part in self._split_message(text):
            await self._app.bot.send_message(chat_id=chat_id, text=part)

    @staticmethod
    def _escape_html(value: object) -> str:
        """HTML parse mode용 최소 이스케이프."""
        return escape_html(value)

    def _get_skill_reload_errors(self) -> list[str]:
        if (
            "get_last_skill_load_errors" not in getattr(self._engine, "__dict__", {})
            and not hasattr(type(self._engine), "get_last_skill_load_errors")
        ):
            return []
        getter = getattr(self._engine, "get_last_skill_load_errors", None)
        if not callable(getter) or inspect.iscoroutinefunction(getter):
            return []
        try:
            errors = getter()
        except Exception:
            return []
        if inspect.isawaitable(errors):
            return []
        if isinstance(errors, list):
            return [str(item) for item in errors]
        return []

    def _get_auto_reload_errors(self) -> list[str]:
        if self._scheduler is None:
            return []
        if (
            "get_last_load_errors" not in getattr(self._scheduler, "__dict__", {})
            and not hasattr(type(self._scheduler), "get_last_load_errors")
        ):
            return []
        getter = getattr(self._scheduler, "get_last_load_errors", None)
        if not callable(getter) or inspect.iscoroutinefunction(getter):
            return []
        try:
            errors = getter()
        except Exception:
            return []
        if inspect.isawaitable(errors):
            return []
        if isinstance(errors, list):
            return [str(item) for item in errors]
        return []

    @staticmethod
    def _format_reload_warnings(errors: list[str], max_items: int = 3) -> str:
        preview = errors[:max_items]
        lines = [f"\n\n⚠️ 일부 항목 로드 실패({len(errors)}건)"]
        lines.extend(f"- {item}" for item in preview)
        if len(errors) > max_items:
            lines.append(f"- ... 외 {len(errors) - max_items}건")
        return "\n".join(lines)

    def _split_message(
        self,
        text: str,
        max_length: int | None = None,
    ) -> list[str]:
        """긴 메시지를 단락 기준으로 분할한다."""
        max_length = max_length or self._max_message_length
        return split_message(text, max_length=max_length)

    @property
    def application(self) -> Application:
        if self._app is None:
            raise RuntimeError("TelegramHandler가 아직 초기화되지 않았습니다.")
        return self._app
