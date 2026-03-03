"""텔레그램 봇 핸들러 — 메시지 수신/발신, 명령어 처리.

사용자 인터페이스 계층. 인증과 레이트리밋을 적용하고,
엔진에 메시지를 전달하여 응답을 텔레그램으로 전송한다.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

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

# 메시지 편집 최소 간격 (초) — 텔레그램 API 제한 대응
_EDIT_INTERVAL = 1.0
_EDIT_CHAR_THRESHOLD = 100

# 연속 typing 인디케이터 간격 (초)
_TYPING_INTERVAL = 4.0

# 스트리밍 안전 가드
# 기본 첫 청크 대기 시간(120초)
_STREAM_DEFAULT_FIRST_CHUNK_TIMEOUT_SECONDS = 120.0
_STREAM_DEFAULT_CHUNK_TIMEOUT_SECONDS = 20.0
_STREAM_DEFAULT_MAX_SECONDS_CAP = 300.0
# 추론/코딩/비전 첫 청크 대기 시간(10분)
_STREAM_REASONING_FIRST_CHUNK_TIMEOUT_SECONDS = 600.0
_STREAM_REASONING_CHUNK_TIMEOUT_SECONDS = 60.0
_STREAM_REASONING_MAX_SECONDS_CAP = 3600.0
_STREAM_LONG_TIMEOUT_INTENTS = {"complex", "code"}
_STREAM_MAX_TOTAL_CHARS = 8_192
_STREAM_MAX_REPEATED_CHUNKS = 30
_STREAM_RENDER_WAIT_GRACE_SECONDS = 5.0
_CONTINUATION_TTL_SECONDS = 30 * 60
_CONTINUE_REQUEST_RE = re.compile(
    r"^\s*(continue|more|계속|이어서|이어줘|더\s*보여줘)\s*$",
    re.IGNORECASE,
)
_LONG_RESPONSE_STOP_NOTICE_PREFIX = "⚠️ 응답이 길어서 여기서 끊었습니다."

# 전체 문서 분석 자동 우회 트리거(일반 채팅용)
_FULL_SCAN_AUTO_TRIGGER_RE = re.compile(
    r"(전체\s*문서|전수\s*분석|전체\s*읽고|모든\s*문서)",
    re.IGNORECASE,
)

# 사용자 체감용 즉시 안내 메시지
_THINKING_PLACEHOLDER_TEMPLATE = "{bot_name}이 답변을 위해 생각 중입니다..."


class _AutoEvaluatorLike(Protocol):
    def schedule_evaluation(
        self,
        chat_id: int,
        bot_message_id: int,
        user_input: str,
        bot_response: str,
    ) -> None: ...


def _auth_required(func: Callable) -> Callable:
    """private chat + 인증 + 레이트리밋을 적용하는 데코레이터."""

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
            self._authorize_chat_id(chat_id)
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


def _global_slot_required(func: Callable) -> Callable:
    """전역 동시성 슬롯을 적용하는 데코레이터."""

    @functools.wraps(func)
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
        auto_evaluator: _AutoEvaluatorLike | None = None,
        semantic_cache: SemanticCache | None = None,
    ) -> None:
        self._config = config
        self._engine = engine
        self._security = security
        self._feedback = feedback
        self._auto_evaluator = auto_evaluator
        self._semantic_cache = semantic_cache
        self._app: Application | None = None
        self._logger = get_logger("telegram")
        self._max_message_length = config.telegram.max_message_length
        # auto_scheduler 참조 (main.py에서 주입)
        self._scheduler = None
        # 프리뷰 캐시: {(chat_id, bot_message_id): {"user": str, "bot": str, "ts": float}}
        self._preview_cache: dict[tuple[int, int], dict] = {}
        # 사유 수집 대기: {chat_id: {"bot_message_id": int, "expires": float}}
        self._pending_reason: dict[int, dict] = {}
        # 긴 응답 이어보기 대기: {chat_id: {"root_query": str, "turn": int, "expires": float}}
        self._pending_continuation: dict[int, dict[str, Any]] = {}

    def set_scheduler(self, scheduler) -> None:
        """auto_scheduler 참조를 설정한다 (순환 의존 방지)."""
        self._scheduler = scheduler

    def has_scheduler(self) -> bool:
        """auto_scheduler 참조 주입 여부를 반환한다."""
        return self._scheduler is not None

    @property
    def _feedback_enabled(self) -> bool:
        return self._feedback is not None and self._config.feedback.enabled

    def _authorize_chat_id(self, chat_id: int) -> None:
        """chat_id 인증 + 레이트리밋 검사를 수행한다."""
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
            CommandHandler("analyze_all", self._cmd_analyze_all),
            CommandHandler("continue", self._cmd_continue),
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
            BotCommand("memory", "메모리 관리"),
            BotCommand("status", "시스템 상태"),
            BotCommand("analyze_all", "전체 문서 분석"),
            BotCommand("continue", "긴 답변 이어보기"),
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

        # 사진 메시지 핸들러 (caption 포함 이미지)
        self._app.add_handler(
            MessageHandler(filters.PHOTO, self._handle_message)
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
    @_global_slot_required
    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        welcome = (
            f"안녕하세요! {self._config.bot.name} 입니다.\n\n"
            "Dual-Provider(Lemonade + Ollama retrieval) 기반 AI 어시스턴트입니다.\n"
            "자유롭게 대화하거나, /help 명령으로 도움말을 확인하세요."
        )
        await update.effective_message.reply_text(welcome)  # type: ignore[union-attr]

    @_auth_required
    @_global_slot_required
    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        command_lines = [
            "/start — 봇 시작",
            "/help — 이 도움말 표시",
            "/skills — 스킬 목록/리로드",
            "/auto — 자동화 관리/리로드",
            "/memory — 메모리 관리",
            "/status — 시스템 상태",
            "/analyze_all — RAG 전체 문서 분석",
            "/continue — 긴 답변 이어보기",
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
    @_global_slot_required
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
    @_global_slot_required
    async def _cmd_continue(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._handle_message_impl(
            update,
            context,
            text_override="",
            force_continuation=True,
        )

    @_auth_required
    @_global_slot_required
    async def _cmd_auto(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._scheduler:
            await update.effective_message.reply_text("자동화 스케줄러가 초기화되지 않았습니다.")  # type: ignore[union-attr]
            return

        args = context.args or []
        if not args or args[0] == "list":
            await self._handle_auto_list(update)
            return

        if len(args) >= 2 and args[0] == "disable":
            await self._handle_auto_disable(update, name=args[1])
            return

        if len(args) >= 2 and args[0] == "run":
            await self._handle_auto_run(update, name=args[1])
            return

        if args[0] == "reload":
            await self._handle_auto_reload(update)
            return

        await update.effective_message.reply_text(
            "사용법: /auto [list|disable <이름>|run <이름>|reload]"
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

    async def _handle_auto_disable(self, update: Update, name: str) -> None:
        if self._scheduler is None:
            await update.effective_message.reply_text("자동화 스케줄러가 초기화되지 않았습니다.")  # type: ignore[union-attr]
            return
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

    async def _handle_auto_run(self, update: Update, name: str) -> None:
        if self._scheduler is None:
            await update.effective_message.reply_text("자동화 스케줄러가 초기화되지 않았습니다.")  # type: ignore[union-attr]
            return

        autos = {item["name"]: item for item in self._scheduler.list_automations()}
        target = autos.get(name)
        if target is None:
            await update.effective_message.reply_text(f"'{name}' 자동화를 찾을 수 없습니다.")
            return
        if not bool(target.get("enabled", False)):
            await update.effective_message.reply_text(f"'{name}' 자동화는 비활성화 상태입니다.")
            return

        ok = await self._scheduler.run_automation_once(name)
        if ok:
            await update.effective_message.reply_text(
                f"'{name}' 자동화를 수동 실행했습니다."
            )
            return
        await update.effective_message.reply_text(
            f"'{name}' 자동화 실행에 실패했습니다. 로그를 확인하세요."
        )

    @_auth_required
    @_global_slot_required
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
    @_global_slot_required
    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        status = await self._engine.get_status()
        llm = status["llm"]
        llm_status = "🟢 정상" if llm.get("status") == "ok" else "🔴 오류"

        text = (
            f"📊 <b>시스템 상태</b>\n\n"
            f"가동 시간: {status['uptime_human']}\n"
            f"LLM 백엔드: {llm_status}\n"
            f"모델: <code>{self._escape_html(status['current_model'])}</code>\n"
            f"로드된 스킬: {status['skills_loaded']}개"
        )

        degraded_components = status.get("degraded_components", {})
        if degraded_components:
            degraded_lines = []
            for name, detail in degraded_components.items():
                reason = detail.get("reason") or "unknown"
                duration = detail.get("degraded_for_seconds")
                if isinstance(duration, int):
                    degraded_lines.append(
                        f"- {name}: {reason} ({duration}초)"
                    )
                else:
                    degraded_lines.append(f"- {name}: {reason}")
            text += "\n⚠️ degraded 상태:\n" + "\n".join(degraded_lines)
        else:
            text += "\n✅ degraded 상태 없음"

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

    @_auth_required
    @_global_slot_required
    async def _cmd_analyze_all(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """RAG 인덱스 전체를 읽어 map-reduce 분석을 수행한다."""
        chat = update.effective_chat
        message = update.effective_message
        if chat is None or message is None:
            return

        args = context.args or []
        query = " ".join(args).strip()
        if not query:
            await message.reply_text("사용법: /analyze_all [질문]")
            return

        await self._run_analyze_all_flow(
            chat=chat,
            message=message,
            query=query,
            auto_triggered=False,
        )

    @staticmethod
    def _should_auto_trigger_analyze_all(text: str) -> bool:
        """일반 채팅에서 full-scan 분석으로 우회할 문구인지 검사한다."""
        text_norm = text.strip()
        if not text_norm:
            return False
        if text_norm.startswith("/"):
            return False
        return bool(_FULL_SCAN_AUTO_TRIGGER_RE.search(text_norm))

    async def _run_analyze_all_flow(
        self,
        *,
        chat: Any,
        message: Any,
        query: str,
        auto_triggered: bool,
    ) -> None:
        """전체 문서 분석 실행 + 진행률 렌더링 공통 처리."""
        query_text = query.strip()
        if not query_text:
            await message.reply_text("사용법: /analyze_all [질문]")
            return

        await chat.send_action(ChatAction.TYPING)
        progress_message = await message.reply_text(
            "전체 문서 분석을 시작합니다.\n"
            "- 단계: 준비\n"
            "- 진행: 0%"
        )

        typing_stop = asyncio.Event()
        typing_task = asyncio.create_task(
            self._keep_typing(chat, typing_stop),
            name=f"analyze_all_typing_{chat.id}",
        )
        last_progress_update = 0.0

        async def _on_progress(payload: dict[str, Any]) -> None:
            nonlocal last_progress_update
            now = time.monotonic()
            phase = str(payload.get("phase", "")).strip().lower()
            force_update = phase in {"final", "map_start", "collect"}
            if not force_update and (now - last_progress_update) < 1.5:
                return
            last_progress_update = now

            if phase == "collect":
                text = "전체 문서 분석을 시작합니다.\n- 단계: 인덱스 수집\n- 진행: 준비 중"
            elif phase == "map_start":
                total_chunks = int(payload.get("total_chunks", 0))
                total_segments = int(payload.get("total_segments", 0))
                text = (
                    "전체 문서 분석을 시작합니다.\n"
                    "- 단계: 맵 분석 시작\n"
                    f"- 청크: {total_chunks}개\n"
                    f"- 세그먼트: {total_segments}개"
                )
            elif phase == "map":
                processed = int(payload.get("processed_segments", 0))
                total = max(1, int(payload.get("total_segments", 1)))
                mapped = int(payload.get("mapped_segments", 0))
                evidence = int(payload.get("evidence_lines", 0))
                percent = int((processed / total) * 100)
                text = (
                    "전체 문서 분석 진행 중\n"
                    "- 단계: 맵 분석\n"
                    f"- 진행: {processed}/{total} ({percent}%)\n"
                    f"- 근거 세그먼트: {mapped}개\n"
                    f"- 근거 라인: {evidence}개"
                )
            elif phase == "reduce":
                reduce_pass = int(payload.get("reduce_pass", 0))
                groups = int(payload.get("groups", 0))
                text = (
                    "전체 문서 분석 진행 중\n"
                    "- 단계: 리듀스(통합)\n"
                    f"- 패스: {reduce_pass}\n"
                    f"- 그룹: {groups}개"
                )
            elif phase == "final":
                text = "전체 문서 분석 진행 중\n- 단계: 최종 답변 생성"
            else:
                return

            try:
                await progress_message.edit_text(text)
            except Exception:
                pass

        try:
            result = await self._engine.analyze_all_corpus(
                query_text,
                progress_callback=_on_progress,
            )
            answer = str(result.get("answer", "")).strip()
            stats = result.get("stats", {}) if isinstance(result, dict) else {}
            if not answer:
                answer = "분석 결과를 생성하지 못했습니다."

            stats_lines = []
            if isinstance(stats, dict):
                total_chunks = stats.get("total_chunks")
                total_segments = stats.get("total_segments")
                mapped_segments = stats.get("mapped_segments")
                evidence_lines = stats.get("evidence_lines")
                duration_ms = stats.get("duration_ms")
                if total_chunks is not None:
                    stats_lines.append(f"- 총 청크: {total_chunks}")
                if total_segments is not None:
                    stats_lines.append(f"- 총 세그먼트: {total_segments}")
                if mapped_segments is not None:
                    stats_lines.append(f"- 근거 세그먼트: {mapped_segments}")
                if evidence_lines is not None:
                    stats_lines.append(f"- 근거 라인: {evidence_lines}")
                if duration_ms is not None:
                    stats_lines.append(f"- 소요 시간: {duration_ms}ms")

            header = "📚 전체 문서 분석 결과"
            if auto_triggered:
                header += " (자동 전환)"
            final_text = f"{header}\n\n{answer}"
            if stats_lines:
                final_text += "\n\n[분석 통계]\n" + "\n".join(stats_lines)

            parts = self._split_message(final_text)
            if parts:
                try:
                    await progress_message.edit_text(parts[0])
                except Exception:
                    await message.reply_text(parts[0])
                for part in parts[1:]:
                    await message.reply_text(part)
        except Exception as exc:
            self._logger.error(
                "analyze_all_failed",
                chat_id=chat.id,
                error=str(exc),
            )
            await message.reply_text(
                "전체 문서 분석 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
            )
        finally:
            typing_stop.set()
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

    # ── 일반 메시지 핸들러 (스트리밍) ──

    @_auth_required
    @_global_slot_required
    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """자유 텍스트 메시지를 처리한다. 스트리밍 UX를 제공한다."""
        await self._handle_message_impl(update, context)

    async def _handle_message_impl(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        *,
        text_override: str | None = None,
        force_continuation: bool = False,
    ) -> None:
        """자유 텍스트 메시지를 처리한다. 스트리밍 UX를 제공한다."""
        chat = update.effective_chat
        message = update.effective_message
        if chat is None or message is None:
            return
        chat_id = chat.id
        raw_text = text_override if text_override is not None else (message.text or message.caption or "")

        # 이미지 처리
        images: list[bytes] | None = None
        image_download_failed = False
        if message.photo:
            try:
                photo = message.photo[-1]  # 최고 해상도
                file = await photo.get_file()
                image_bytes = await file.download_as_bytearray()
                images = [bytes(image_bytes)]
            except Exception as exc:
                image_download_failed = True
                self._logger.warning(
                    "image_download_failed",
                    chat_id=chat_id,
                    error=str(exc),
                )

        if not raw_text and not images and not force_continuation:
            if image_download_failed:
                await message.reply_text(
                    "이미지 다운로드에 실패했어요. 잠시 후 다시 시도해주세요."
                )
            return

        # 입력 정제
        text = self._security.sanitize_input(raw_text) if raw_text else ""
        self._cleanup_pending_continuations()
        continuation_state: dict[str, Any] | None = None
        continuation_root_query: str | None = None
        if force_continuation or (not images and self._is_continue_request(text)):
            continuation_state = self._take_pending_continuation(chat_id)
            if continuation_state is None:
                await message.reply_text("이어볼 답변이 없습니다. 먼저 질문을 해주세요.")
                return
            continuation_root_query = str(continuation_state.get("root_query", "")).strip()
            text = self._build_continuation_prompt(continuation_state)
        else:
            # 새 질문이 들어오면 이전 이어보기 상태는 정리한다.
            self._pending_continuation.pop(chat_id, None)

        if not text.strip() and not images:
            return

        # 전체 문서 분석 의도가 명확하면 일반 RAG(top-k) 대신 full-scan 경로로 우회한다.
        if continuation_state is None and not images and self._should_auto_trigger_analyze_all(text):
            self._logger.info("analyze_all_auto_triggered", chat_id=chat_id)
            await self._run_analyze_all_flow(
                chat=chat,
                message=message,
                query=text,
                auto_triggered=True,
            )
            return

        # 타이핑 표시
        await chat.send_action(ChatAction.TYPING)

        # 연속 typing 인디케이터
        typing_stop = asyncio.Event()
        typing_task = asyncio.create_task(
            self._keep_typing(chat, typing_stop),
            name=f"typing_{chat_id}",
        )
        render_timeout: float | None = None

        try:
            # 사용자가 즉시 진행 상태를 인지할 수 있도록 먼저 안내한다.
            sent_message = await message.reply_text(
                _THINKING_PLACEHOLDER_TEMPLATE.format(
                    bot_name=self._config.bot.name,
                )
            )

            # 인텐트 분류 결과는 timeout 정책 결정에만 사용한다.
            raw_intent = self._engine.classify_intent(text)
            if inspect.isawaitable(raw_intent):
                intent = await raw_intent
            else:
                intent = raw_intent

            intent_key = str(intent).strip().lower() if intent is not None else None
            if images or intent_key in _STREAM_LONG_TIMEOUT_INTENTS:
                first_chunk_timeout_seconds = _STREAM_REASONING_FIRST_CHUNK_TIMEOUT_SECONDS
                chunk_timeout_seconds = _STREAM_REASONING_CHUNK_TIMEOUT_SECONDS
                effective_stream_seconds = max(
                    float(self._config.bot.response_timeout),
                    _STREAM_REASONING_MAX_SECONDS_CAP,
                )
            else:
                first_chunk_timeout_seconds = _STREAM_DEFAULT_FIRST_CHUNK_TIMEOUT_SECONDS
                chunk_timeout_seconds = _STREAM_DEFAULT_CHUNK_TIMEOUT_SECONDS
                effective_stream_seconds = min(
                    float(self._config.bot.response_timeout),
                    _STREAM_DEFAULT_MAX_SECONDS_CAP,
                )
            render_timeout = effective_stream_seconds + _STREAM_RENDER_WAIT_GRACE_SECONDS
            result = await asyncio.wait_for(
                stream_and_render(
                    stream=self._engine.process_message_stream(chat_id, text, images=images),
                    sent_message=sent_message,
                    reply_text=message.reply_text,
                    split_message_fn=self._split_message,
                    edit_interval=_EDIT_INTERVAL,
                    edit_char_threshold=_EDIT_CHAR_THRESHOLD,
                    max_edit_length=self._max_message_length,
                    first_chunk_timeout_seconds=first_chunk_timeout_seconds,
                    chunk_timeout_seconds=chunk_timeout_seconds,
                    max_stream_seconds=effective_stream_seconds,
                    max_total_chars=_STREAM_MAX_TOTAL_CHARS,
                    max_repeated_chunks=_STREAM_MAX_REPEATED_CHUNKS,
                ),
                timeout=render_timeout,
            )
            stream_meta_found = False
            consume_meta = getattr(self._engine, "consume_last_stream_meta", None)
            if callable(consume_meta):
                stream_meta = consume_meta(chat_id)
                if inspect.isawaitable(stream_meta):
                    stream_meta = await stream_meta
                if isinstance(stream_meta, dict):
                    stream_meta_found = True
                    result.tier = stream_meta.get("tier", result.tier)
                    result.intent = stream_meta.get("intent")
                    result.cache_id = stream_meta.get("cache_id")
                    result.usage = stream_meta.get("usage")
            stop_reason = getattr(result, "stop_reason", None)
            recovery_reason: str | None = None
            anomaly_reasons: list[str] = []
            if stop_reason in {"chunk_timeout", "repeated_chunks"} and not stream_meta_found:
                recovery_reason = stop_reason
            elif stop_reason is None:
                anomaly_reasons = detect_output_anomalies(
                    result.full_response,
                    result.full_response,
                )
                actionable_reasons = [
                    reason for reason in anomaly_reasons if reason != "empty_after_sanitize"
                ]
                if actionable_reasons:
                    anomaly_reasons = actionable_reasons
                    recovery_reason = "response_anomaly"

            if recovery_reason is not None:
                self._logger.warning(
                    "stream_recovery_triggered",
                    chat_id=chat_id,
                    reason=recovery_reason,
                    anomaly_reasons=anomaly_reasons or None,
                )
                # 스트리밍에서 이미 저장된 비정상 턴을 삭제하여
                # recovery LLM 호출이 오염된 히스토리를 보지 않도록 한다.
                try:
                    rollback_fn = getattr(self._engine, "rollback_last_turn", None)
                    if callable(rollback_fn):
                        deleted = await rollback_fn(chat_id)
                        self._logger.info(
                            "stream_recovery_turn_rolled_back",
                            chat_id=chat_id,
                            deleted=deleted,
                        )
                except Exception as rb_exc:
                    self._logger.warning(
                        "stream_recovery_rollback_failed",
                        chat_id=chat_id,
                        error=str(rb_exc),
                    )
                try:
                    recovered_response = await self._engine.process_message(
                        chat_id,
                        text,
                        images=images,
                    )
                except Exception as exc:
                    self._logger.warning(
                        "stream_recovery_failed",
                        chat_id=chat_id,
                        reason=recovery_reason,
                        error=str(exc),
                    )
                else:
                    recovered_text = str(recovered_response).strip()
                    if recovered_text:
                        recovered_parts = self._split_message(recovered_text)
                        if recovered_parts:
                            last_recovered = None
                            try:
                                await sent_message.edit_text(recovered_parts[0])
                                last_recovered = sent_message
                            except Exception:
                                last_recovered = await message.reply_text(recovered_parts[0])

                            for part in recovered_parts[1:]:
                                last_recovered = await message.reply_text(part)

                            if last_recovered is not None:
                                result.last_message = last_recovered
                                result.full_response = recovered_text

            # 캐시 피드백 링크 저장
            if (
                self._semantic_cache is not None
                and result.cache_id is not None
                and result.last_message
            ):
                try:
                    await self._semantic_cache.link_feedback_target(
                        chat_id, result.last_message.message_id, result.cache_id,
                    )
                except Exception:
                    pass

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

            # LLM-as-Judge 자동 평가 트리거
            if (
                self._auto_evaluator is not None
                and result.full_response.strip()
                and result.last_message
            ):
                self._auto_evaluator.schedule_evaluation(
                    chat_id,
                    result.last_message.message_id,
                    text,
                    result.full_response,
                )

            stop_reason = getattr(result, "stop_reason", None)
            if stop_reason == "max_total_chars":
                next_turn = 1
                if continuation_state is not None:
                    next_turn = max(1, int(continuation_state.get("turn", 0)) + 1)
                root_query = (continuation_root_query or "").strip() or text
                self._set_pending_continuation(
                    chat_id,
                    root_query=root_query,
                    turn=next_turn,
                )
                await message.reply_text(
                    self._build_long_response_followup_message(result.full_response)
                )
            else:
                self._pending_continuation.pop(chat_id, None)

        except asyncio.TimeoutError:
            self._logger.error(
                "stream_render_timeout",
                chat_id=chat_id,
                timeout_seconds=render_timeout,
            )
            await message.reply_text(
                "⚠️ 응답 시간이 길어져 중단했습니다. 질문을 더 짧게 나눠 다시 시도해주세요."
            )
        except Exception as exc:
            self._logger.error(
                "message_processing_error",
                chat_id=chat_id,
                error=str(exc),
            )
            await message.reply_text(
                "죄송합니다. 메시지 처리 중 오류가 발생했습니다."
            )
        finally:
            typing_stop.set()
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

    @staticmethod
    def _is_continue_request(text: str) -> bool:
        return bool(_CONTINUE_REQUEST_RE.match(text.strip()))

    def _cleanup_pending_continuations(self) -> None:
        if not self._pending_continuation:
            return
        now = time.monotonic()
        expired_chat_ids = [
            chat_id
            for chat_id, pending in self._pending_continuation.items()
            if now > float(pending.get("expires", 0.0))
        ]
        for chat_id in expired_chat_ids:
            del self._pending_continuation[chat_id]

    def _take_pending_continuation(self, chat_id: int) -> dict[str, Any] | None:
        self._cleanup_pending_continuations()
        pending = self._pending_continuation.get(chat_id)
        if pending is None:
            return None
        del self._pending_continuation[chat_id]
        return pending

    def _set_pending_continuation(
        self,
        chat_id: int,
        *,
        root_query: str,
        turn: int,
    ) -> None:
        self._cleanup_pending_continuations()
        self._pending_continuation[chat_id] = {
            "root_query": root_query,
            "turn": max(1, turn),
            "expires": time.monotonic() + _CONTINUATION_TTL_SECONDS,
        }

    @staticmethod
    def _build_continuation_prompt(pending: dict[str, Any]) -> str:
        root_query = str(pending.get("root_query", "")).strip()
        turn = max(1, int(pending.get("turn", 1)))
        return (
            "직전 답변을 이어서 작성해줘.\n"
            "- 이미 설명한 내용은 반복하지 말고 중단 지점부터 이어서 설명해줘.\n"
            "- 먼저 3줄 이내로 지금까지 핵심을 요약해줘.\n"
            "- 답변이 다시 길어지면 마지막 줄에 '계속하려면 계속이라고 입력해주세요.'를 적어줘.\n"
            f"- 이어보기 턴: {turn}\n"
            f"[원 질문]\n{root_query}"
        ).strip()

    @staticmethod
    def _truncate_summary_line(text: str, *, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    @classmethod
    def _extract_summary_points(cls, text: str, *, max_points: int = 3) -> list[str]:
        content = text.strip()
        marker = f"\n\n{_LONG_RESPONSE_STOP_NOTICE_PREFIX}"
        if marker in content:
            content = content.split(marker, 1)[0].strip()
        elif content.startswith(_LONG_RESPONSE_STOP_NOTICE_PREFIX):
            content = ""

        points: list[str] = []
        seen: set[str] = set()
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("```"):
                continue
            line = re.sub(r"^(?:[-*•]|\d+[.)])\s*", "", line).strip()
            if len(line) < 8:
                continue
            key = line.casefold()
            if key in seen:
                continue
            seen.add(key)
            points.append(cls._truncate_summary_line(line, max_chars=140))
            if len(points) >= max_points:
                return points

        collapsed = " ".join(part.strip() for part in content.splitlines() if part.strip())
        if collapsed:
            points.append(cls._truncate_summary_line(collapsed, max_chars=180))
        return points

    @classmethod
    def _build_long_response_followup_message(cls, response_text: str) -> str:
        points = cls._extract_summary_points(response_text, max_points=3)
        if points:
            summary = "\n".join(f"- {point}" for point in points)
            return (
                "📌 지금까지 요약\n"
                f"{summary}\n\n"
                "계속 보려면 /continue 또는 '계속'이라고 입력하세요."
            )
        return "응답이 길어서 여기서 끊었습니다. /continue 또는 '계속'이라고 입력하면 이어서 보여드릴게요."

    @staticmethod
    async def _keep_typing(chat: Any, stop_event: asyncio.Event) -> None:
        """typing 인디케이터를 주기적으로 전송한다."""
        while not stop_event.is_set():
            try:
                await chat.send_action(ChatAction.TYPING)
            except Exception:
                pass
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=_TYPING_INTERVAL)
                return
            except asyncio.TimeoutError:
                pass

    # ── 피드백 ──

    def _cleanup_preview_cache(self) -> None:
        """프리뷰 캐시의 TTL 만료 항목을 정리하고 크기 제한을 유지한다."""
        self._cleanup_pending_reasons()
        self._cleanup_pending_continuations()
        max_size = self._config.feedback.preview_cache_max_size
        ttl_hours = self._config.feedback.preview_cache_ttl_hours
        if max_size <= 0 or ttl_hours <= 0:
            self._preview_cache.clear()
            return

        now = time.monotonic()
        ttl_seconds = ttl_hours * 3600
        expired = [
            key for key, value in self._preview_cache.items()
            if now - value["ts"] > ttl_seconds
        ]
        for key in expired:
            del self._preview_cache[key]

        while len(self._preview_cache) > max_size:
            oldest_key = min(
                self._preview_cache,
                key=lambda key: self._preview_cache[key]["ts"],
            )
            del self._preview_cache[oldest_key]

    def _cleanup_pending_reasons(self) -> None:
        """사유 입력 대기 상태의 만료 항목을 주기적으로 정리한다."""
        if not self._pending_reason:
            return
        now = time.monotonic()
        expired_chat_ids = [
            chat_id
            for chat_id, pending in self._pending_reason.items()
            if now > float(pending.get("expires", 0.0))
        ]
        for chat_id in expired_chat_ids:
            del self._pending_reason[chat_id]

    def _cache_preview(self, chat_id: int, bot_message_id: int, user_text: str, bot_text: str) -> None:
        """프리뷰를 캐시에 저장한다. TTL 초과/크기 초과 시 정리."""
        max_chars = self._config.feedback.preview_max_chars
        max_size = self._config.feedback.preview_cache_max_size
        ttl_hours = self._config.feedback.preview_cache_ttl_hours
        if max_chars <= 0 or max_size <= 0 or ttl_hours <= 0:
            return
        self._cleanup_preview_cache()

        # 최대 크기 초과 시 가장 오래된 항목 제거
        while len(self._preview_cache) >= max_size:
            oldest_key = min(self._preview_cache, key=lambda k: self._preview_cache[k]["ts"])
            del self._preview_cache[oldest_key]

        now = time.monotonic()
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
            self._authorize_chat_id(chat_id)
        except AuthenticationError:
            await query.answer()
            return False
        except RateLimitError:
            await query.answer("요청이 너무 많습니다. 잠시 후 다시 시도해주세요.", show_alert=True)
            return False
        return True

    @_global_slot_required
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

        self._cleanup_preview_cache()
        preview = self._preview_cache.get((chat_id, bot_message_id), {})
        is_update = await self._feedback.store_feedback(
            chat_id=chat_id,
            bot_message_id=bot_message_id,
            rating=rating,
            user_preview=preview.get("user"),
            bot_preview=preview.get("bot"),
        )

        # 기존 요청(동일 메시지)에 대한 pending reason은 재평가 시 정리한다.
        pending = self._pending_reason.get(chat_id)
        if (
            pending is not None
            and pending.get("bot_message_id") == bot_message_id
            and (is_update or rating == 1)
        ):
            del self._pending_reason[chat_id]

        # 👎 피드백 시 시맨틱 캐시 무효화
        if (
            rating == -1
            and self._semantic_cache is not None
            and self._config.semantic_cache.invalidate_on_negative_feedback
        ):
            try:
                linked_cache_id = await self._semantic_cache.get_feedback_cache_id(
                    chat_id, bot_message_id,
                )
                if linked_cache_id is not None:
                    await self._semantic_cache.invalidate_by_id(linked_cache_id)
                    self._logger.info(
                        "cache_invalidated_by_feedback",
                        chat_id=chat_id,
                        cache_id=linked_cache_id,
                    )
            except Exception as exc:
                self._logger.debug("cache_feedback_invalidation_failed", error=str(exc))

        # 👎이면서 사유 수집이 활성화되어 있으면 사유 입력 요청
        if (
            rating == -1
            and not is_update
            and self._config.feedback.collect_reason
        ):
            existing_pending = self._pending_reason.get(chat_id)
            replaced_pending = False
            if existing_pending is not None:
                previous_expires = float(existing_pending.get("expires", 0.0))
                previous_bot_message_id = existing_pending.get("bot_message_id")
                del self._pending_reason[chat_id]
                replaced_pending = (
                    time.monotonic() <= previous_expires
                    and previous_bot_message_id != bot_message_id
                )

            timeout = self._config.feedback.reason_timeout_seconds
            self._pending_reason[chat_id] = {
                "bot_message_id": bot_message_id,
                "expires": time.monotonic() + timeout,
            }
            await query.answer("피드백 감사합니다!", show_alert=False)
            if query.message is not None and hasattr(query.message, "reply_text"):
                if replaced_pending:
                    await query.message.reply_text(
                        "이전 사유 입력 요청은 자동 만료되어 최신 요청으로 교체되었어요."
                    )
                await query.message.reply_text(
                    "어떤 점이 아쉬웠나요? 사유를 입력해주세요.\n"
                    "건너뛰려면 /skip 을 입력하세요."
                )
            return

        if is_update:
            await query.answer("피드백을 업데이트했어요.", show_alert=False)
        else:
            await query.answer("피드백 감사합니다!", show_alert=False)

    @_auth_required
    @_global_slot_required
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

    # ── 사유 수집 ──

    async def _handle_reason_input(self, chat_id: int, text: str, update: Update) -> bool:
        """대기 중인 사유가 있으면 저장하고 True를 반환한다."""
        pending = self._pending_reason.get(chat_id)
        if pending is None:
            return False

        # 만료 확인
        if time.monotonic() > pending["expires"]:
            del self._pending_reason[chat_id]
            await update.effective_message.reply_text(  # type: ignore[union-attr]
                "사유 입력 시간이 만료되었습니다."
            )
            return True

        # 길이 검증
        min_chars = self._config.feedback.reason_min_chars
        max_chars = self._config.feedback.reason_max_chars
        reason = self._security.sanitize_input(text).strip()

        if len(reason) < min_chars:
            await update.effective_message.reply_text(  # type: ignore[union-attr]
                f"사유는 최소 {min_chars}자 이상 입력해주세요. 건너뛰려면 /skip"
            )
            return True

        reason = reason[:max_chars]

        updated = False
        if self._feedback is not None:
            updated = await self._feedback.update_reason(
                chat_id=chat_id,
                bot_message_id=pending["bot_message_id"],
                reason=reason,
            )

        del self._pending_reason[chat_id]
        if updated:
            await update.effective_message.reply_text(  # type: ignore[union-attr]
                "사유가 기록되었습니다. 감사합니다!"
            )
        else:
            await update.effective_message.reply_text(  # type: ignore[union-attr]
                "사유를 저장할 대상 피드백을 찾지 못했습니다."
            )
        return True

    @_auth_required
    @_global_slot_required
    async def _handle_reason_skip(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """사유 입력을 건너뛴다."""
        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        if chat_id in self._pending_reason:
            del self._pending_reason[chat_id]
            await update.effective_message.reply_text("사유 입력을 건너뛰었습니다.")  # type: ignore[union-attr]
        else:
            await update.effective_message.reply_text("건너뛸 사유 요청이 없습니다.")  # type: ignore[union-attr]

    @_auth_required
    @_global_slot_required
    async def _handle_reason_or_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """사유 대기 상태이면 사유를 처리하고, 아니면 일반 메시지를 처리한다."""
        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        text = update.effective_message.text  # type: ignore[union-attr]
        if text is None:
            return

        if await self._handle_reason_input(chat_id, text, update):
            return

        # 사유 대기가 아니면 일반 메시지 처리로 위임
        await self._handle_message_impl(update, context)

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
