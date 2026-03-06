"""TelegramHandler 명령어 조립 레이어."""

from __future__ import annotations

from typing import TYPE_CHECKING

from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from core import (
    telegram_commands_automation,
    telegram_commands_memory,
    telegram_commands_status,
)

if TYPE_CHECKING:
    from telegram import Update

    from core.telegram_handler import TelegramHandler


async def cmd_start(
    self: TelegramHandler,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    _ = context
    welcome = (
        f"안녕하세요! {self._config.bot.name} 입니다.\n\n"
        "Dual-Provider(Lemonade + Ollama retrieval) 기반 AI 어시스턴트입니다.\n"
        "자유롭게 대화하거나, /help 명령으로 도움말을 확인하세요."
    )
    await update.effective_message.reply_text(welcome)  # type: ignore[union-attr]


async def cmd_help(
    self: TelegramHandler,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    _ = context
    commands: list[tuple[str, str]] = [
        ("/start", "봇 시작"),
        ("/help", "이 도움말 표시"),
        ("/skills", "스킬 목록/리로드"),
        ("/auto", "자동화 관리/리로드"),
        ("/memory", "메모리 관리"),
        ("/status", "시스템 상태"),
        ("/continue", "긴 답변 이어보기"),
    ]
    if self._feedback_enabled:
        commands.insert(6, ("/feedback", "피드백 통계"))
    if self._sim_scheduler is not None:
        commands.append(("/sim", "시뮬레이션 큐 관리"))

    table_lines = ["명령어       설명", "─" * 25]
    for cmd, desc in commands:
        table_lines.append(f"{cmd:<12s} {desc}")

    help_text = (
        "📋 <b>사용 가능한 명령어</b>\n\n"
        f"<pre>{chr(10).join(table_lines)}</pre>\n\n"
        "💬 <b>대화 모드</b>\n"
        "명령어 없이 자유롭게 대화하세요.\n\n"
        "🔧 <b>스킬 모드</b>\n"
        "스킬 트리거 키워드를 사용하면 전문 기능이 활성화됩니다.\n"
        "/skills 명령으로 스킬 목록을 확인하세요."
    )
    await update.effective_message.reply_text(  # type: ignore[union-attr]
        help_text,
        parse_mode=ParseMode.HTML,
    )


async def cmd_skills(
    self: TelegramHandler,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    args = context.args or []
    if args and args[0] == "reload":
        try:
            count = await self._engine.reload_skills(strict=True)
            errors = self._get_skill_reload_errors()
            message = f"스킬을 다시 로드했습니다: {count}개"
            if errors:
                message += self._format_reload_warnings(errors)
            await update.effective_message.reply_text(message)  # type: ignore[union-attr]
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

    name_w = max(4, max(len(skill["name"]) for skill in skills))
    desc_w = max(4, max(len(skill["description"]) for skill in skills))
    header = f"{'스킬':<{name_w}s}  {'설명':<{desc_w}s}  트리거"
    sep = "─" * min(len(header) + 4, 50)
    table_lines = [header, sep]
    for skill in skills:
        triggers = ", ".join(skill["triggers"])
        table_lines.append(
            f"{self._escape_html(skill['name']):<{name_w}s}  "
            f"{self._escape_html(skill['description']):<{desc_w}s}  "
            f"{self._escape_html(triggers)}"
        )

    text = f"🔧 <b>사용 가능한 스킬</b>\n\n<pre>{chr(10).join(table_lines)}</pre>"
    await update.effective_message.reply_text(  # type: ignore[union-attr]
        text,
        parse_mode=ParseMode.HTML,
    )


async def cmd_continue(
    self: TelegramHandler,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    await self._handle_message_impl(
        update,
        context,
        text_override="",
        force_continuation=True,
    )


async def cmd_auto(
    self: TelegramHandler,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    await telegram_commands_automation.cmd_auto(self, update, context)


async def handle_auto_list(self: TelegramHandler, update: Update) -> None:
    await telegram_commands_automation.handle_auto_list(self, update)


async def handle_auto_disable(self: TelegramHandler, update: Update, name: str) -> None:
    await telegram_commands_automation.handle_auto_disable(self, update, name)


async def handle_auto_reload(self: TelegramHandler, update: Update) -> None:
    await telegram_commands_automation.handle_auto_reload(self, update)


async def handle_auto_run(self: TelegramHandler, update: Update, name: str) -> None:
    await telegram_commands_automation.handle_auto_run(self, update, name)


async def cmd_memory(
    self: TelegramHandler,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    await telegram_commands_memory.cmd_memory(self, update, context)


async def cmd_status(
    self: TelegramHandler,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    await telegram_commands_status.cmd_status(self, update, context)
