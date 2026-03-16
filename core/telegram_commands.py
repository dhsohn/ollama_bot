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
from core.i18n import t
from core.telegram_menus import get_user_language, handle_start_with_onboarding

if TYPE_CHECKING:
    from telegram import Update

    from core.telegram_handler import TelegramHandler


async def cmd_start(
    self: TelegramHandler,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    await handle_start_with_onboarding(self, update, context)


async def cmd_help(
    self: TelegramHandler,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    _ = context
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    lang = await get_user_language(self, chat_id)

    commands: list[tuple[str, str]] = [
        ("/start", t("cmd_start", lang)),
        ("/help", t("cmd_help", lang)),
        ("/skills", t("cmd_skills", lang)),
        ("/auto", t("cmd_auto", lang)),
        ("/memory", t("cmd_memory", lang)),
        ("/status", t("cmd_status", lang)),
    ]
    if self._feedback_enabled:
        commands.append(("/feedback", t("cmd_feedback", lang)))

    header_cmd = t("help_header_cmd", lang)
    header_desc = t("help_header_desc", lang)
    table_lines = [f"{header_cmd:<12s} {header_desc}", "\u2500" * 25]
    for cmd, desc in commands:
        table_lines.append(f"{cmd:<12s} {desc}")

    help_text = (
        f"\U0001f4cb <b>{t('help_title', lang)}</b>\n\n"
        f"<pre>{chr(10).join(table_lines)}</pre>\n\n"
        f"\U0001f4ac <b>{t('help_chat_mode', lang)}</b>\n"
        f"{t('help_chat_desc', lang)}\n\n"
        f"\U0001f527 <b>{t('help_skill_mode', lang)}</b>\n"
        f"{t('help_skill_desc', lang)}"
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
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    lang = await get_user_language(self, chat_id)
    args = context.args or []

    if args and args[0] == "reload":
        try:
            count = await self._engine.reload_skills(strict=True)
            errors = self._get_skill_reload_errors()
            message = t("skills_reloaded", lang, count=count)
            if errors:
                message += self._format_reload_warnings(errors, lang=lang)
            await update.effective_message.reply_text(message)  # type: ignore[union-attr]
        except Exception as exc:
            self._logger.error("skills_reload_failed", error=str(exc))
            await update.effective_message.reply_text(  # type: ignore[union-attr]
                t("skills_reload_failed", lang, error=str(exc))
            )
        return

    skills = self._engine.list_skills(lang=lang)
    if not skills:
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            t("skills_empty", lang)
        )
        return

    name_w = max(4, max(len(skill["name"]) for skill in skills))
    desc_w = max(4, max(len(skill["description"]) for skill in skills))
    h_name = t("skills_header_name", lang)
    h_desc = t("skills_header_desc", lang)
    h_trigger = t("skills_header_trigger", lang)
    header = f"{h_name:<{name_w}s}  {h_desc:<{desc_w}s}  {h_trigger}"
    sep = "\u2500" * min(len(header) + 4, 50)
    table_lines = [header, sep]
    for skill in skills:
        triggers = ", ".join(skill["triggers"])
        table_lines.append(
            f"{self._escape_html(skill['name']):<{name_w}s}  "
            f"{self._escape_html(skill['description']):<{desc_w}s}  "
            f"{self._escape_html(triggers)}"
        )

    text = (
        f"\U0001f527 <b>{t('skills_title', lang)}</b>"
        f"\n\n<pre>{chr(10).join(table_lines)}</pre>"
    )
    await update.effective_message.reply_text(  # type: ignore[union-attr]
        text,
        parse_mode=ParseMode.HTML,
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
