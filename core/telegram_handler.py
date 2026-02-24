"""텔레그램 봇 핸들러 — 메시지 수신/발신, 명령어 처리.

사용자 인터페이스 계층. 인증과 레이트리밋을 적용하고,
엔진에 메시지를 전달하여 응답을 텔레그램으로 전송한다.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path

from telegram import BotCommand, Update
from telegram.constants import ChatAction, ChatType, ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from core.config import AppSettings
from core.engine import Engine
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
    ) -> None:
        self._config = config
        self._engine = engine
        self._security = security
        self._app: Application | None = None
        self._logger = get_logger("telegram")
        # auto_scheduler 참조 (main.py에서 주입)
        self._scheduler = None

    def set_scheduler(self, scheduler) -> None:
        """auto_scheduler 참조를 설정한다 (순환 의존 방지)."""
        self._scheduler = scheduler

    async def initialize(self) -> Application:
        """텔레그램 Application을 생성하고 핸들러를 등록한다."""
        self._app = (
            ApplicationBuilder()
            .token(self._config.telegram_bot_token)
            .build()
        )

        # 명령어 핸들러 등록
        handlers = [
            CommandHandler("start", self._cmd_start),
            CommandHandler("help", self._cmd_help),
            CommandHandler("skills", self._cmd_skills),
            CommandHandler("auto", self._cmd_auto),
            CommandHandler("model", self._cmd_model),
            CommandHandler("memory", self._cmd_memory),
            CommandHandler("status", self._cmd_status),
        ]
        for handler in handlers:
            self._app.add_handler(handler)

        # 일반 텍스트 메시지 핸들러 (명령어 제외)
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )

        # 에러 핸들러
        self._app.add_error_handler(self._error_handler)

        # 봇 명령어 목록 설정
        await self._app.bot.set_my_commands([
            BotCommand("start", "봇 시작"),
            BotCommand("help", "도움말"),
            BotCommand("skills", "스킬 목록"),
            BotCommand("auto", "자동화 관리"),
            BotCommand("model", "모델 관리"),
            BotCommand("memory", "메모리 관리"),
            BotCommand("status", "시스템 상태"),
        ])

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
        help_text = (
            "📋 *사용 가능한 명령어*\n\n"
            "/start — 봇 시작\n"
            "/help — 이 도움말 표시\n"
            "/skills — 스킬 목록/리로드\n"
            "/auto — 자동화 관리/리로드\n"
            "/model — 모델 확인/변경\n"
            "/memory — 메모리 관리\n"
            "/status — 시스템 상태\n\n"
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
                count = await self._engine.reload_skills()
                await update.effective_message.reply_text(  # type: ignore[union-attr]
                    f"스킬을 다시 로드했습니다: {count}개"
                )
            except Exception as exc:
                self._logger.error("skills_reload_failed", error=str(exc))
                await update.effective_message.reply_text(  # type: ignore[union-attr]
                    "스킬 로드 중 오류가 발생했습니다. YAML 중복/형식을 확인하세요."
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
            automations = self._scheduler.list_automations()
            if not automations:
                await update.effective_message.reply_text("등록된 자동화가 없습니다.")  # type: ignore[union-attr]
                return

            lines = ["⏰ <b>자동화 목록</b>\n"]
            for a in automations:
                status = "✅" if a["enabled"] else "❌"
                lines.append(
                    f"{status} <b>{self._escape_html(a['name'])}</b> — "
                    f"{self._escape_html(a['description'])}\n"
                    f"  스케줄: <code>{self._escape_html(a['schedule'])}</code>"
                )
            await update.effective_message.reply_text(  # type: ignore[union-attr]
                "\n".join(lines), parse_mode=ParseMode.HTML
            )

        elif len(args) >= 2 and args[0] == "enable":
            result = await self._scheduler.enable_automation(args[1])
            msg = f"'{args[1]}' 자동화가 활성화되었습니다." if result else f"'{args[1]}' 자동화를 찾을 수 없습니다."
            await update.effective_message.reply_text(msg)  # type: ignore[union-attr]

        elif len(args) >= 2 and args[0] == "disable":
            result = await self._scheduler.disable_automation(args[1])
            msg = f"'{args[1]}' 자동화가 비활성화되었습니다." if result else f"'{args[1]}' 자동화를 찾을 수 없습니다."
            await update.effective_message.reply_text(msg)  # type: ignore[union-attr]

        elif args[0] == "reload":
            try:
                count = await self._scheduler.reload_automations()
                await update.effective_message.reply_text(  # type: ignore[union-attr]
                    f"자동화를 다시 로드했습니다: {count}개"
                )
            except Exception as exc:
                self._logger.error("auto_reload_failed", error=str(exc))
                await update.effective_message.reply_text(  # type: ignore[union-attr]
                    "자동화 로드 중 오류가 발생했습니다. YAML 중복/형식을 확인하세요."
                )

        else:
            await update.effective_message.reply_text(  # type: ignore[union-attr]
                "사용법: /auto [list|enable <이름>|disable <이름>|reload]"
            )

    @_auth_required
    async def _cmd_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        args = context.args or []
        if not args:
            current = self._engine.get_current_model()
            await update.effective_message.reply_text(  # type: ignore[union-attr]
                f"현재 모델: <code>{self._escape_html(current)}</code>\n\n"
                "모델 변경: <code>/model &lt;모델명&gt;</code>\n"
                "모델 목록: <code>/model list</code>",
                parse_mode=ParseMode.HTML,
            )
        elif args[0] == "list":
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
            for m in models:
                size_mb = m["size"] / (1024 * 1024) if m["size"] else 0
                lines.append(
                    f"• <code>{self._escape_html(m['name'])}</code> ({size_mb:.0f}MB)"
                )
            await update.effective_message.reply_text(  # type: ignore[union-attr]
                "\n".join(lines), parse_mode=ParseMode.HTML
            )
        else:
            try:
                result = await self._engine.change_model(args[0])
            except Exception as exc:
                self._logger.error(
                    "model_change_failed",
                    chat_id=chat_id,
                    requested_model=args[0],
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
            else:
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

        await update.effective_message.reply_text(  # type: ignore[union-attr]
            text, parse_mode=ParseMode.HTML
        )

    # ── 일반 메시지 핸들러 (스트리밍) ──

    @_auth_required
    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """자유 텍스트 메시지를 처리한다. 스트리밍 UX를 제공한다."""
        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        text = update.effective_message.text  # type: ignore[union-attr]

        # 입력 정제
        text = self._security.sanitize_input(text)
        if not text.strip():
            return

        # 타이핑 표시
        await update.effective_chat.send_action(ChatAction.TYPING)  # type: ignore[union-attr]

        try:
            # 초기 placeholder 메시지 전송
            sent_message = await update.effective_message.reply_text("...")  # type: ignore[union-attr]

            await stream_and_render(
                stream=self._engine.process_message_stream(chat_id, text),
                sent_message=sent_message,
                reply_text=update.effective_message.reply_text,  # type: ignore[union-attr]
                split_message_fn=self._split_message,
                edit_interval=_EDIT_INTERVAL,
                edit_char_threshold=_EDIT_CHAR_THRESHOLD,
            )

        except Exception as exc:
            self._logger.error(
                "message_processing_error",
                chat_id=chat_id,
                error=str(exc),
            )
            await update.effective_message.reply_text(  # type: ignore[union-attr]
                "죄송합니다. 메시지 처리 중 오류가 발생했습니다."
            )

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
        assert self._app is not None
        for part in self._split_message(text):
            await self._app.bot.send_message(chat_id=chat_id, text=part)

    @staticmethod
    def _escape_html(value: object) -> str:
        """HTML parse mode용 최소 이스케이프."""
        return escape_html(value)

    def _split_message(
        self, text: str, max_length: int = 4096
    ) -> list[str]:
        """긴 메시지를 단락 기준으로 분할한다."""
        return split_message(text, max_length=max_length)

    @property
    def application(self) -> Application:
        assert self._app is not None
        return self._app
