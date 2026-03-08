from __future__ import annotations

from typing import TYPE_CHECKING

from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from core.i18n import t
from core.telegram_menus import get_user_language

if TYPE_CHECKING:
    from telegram import Update

    from core.telegram_handler import TelegramHandler


async def cmd_auto(
    self: TelegramHandler,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    lang = await get_user_language(self, chat_id)

    if not self._scheduler:
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            t("auto_no_scheduler", lang)
        )
        return

    args = context.args or []
    if not args or args[0] == "list":
        await handle_auto_list(self, update)
        return

    if len(args) >= 2 and args[0] == "disable":
        await handle_auto_disable(self, update, name=args[1])
        return

    if len(args) >= 2 and args[0] == "run":
        await handle_auto_run(self, update, name=args[1])
        return

    if args[0] == "reload":
        await handle_auto_reload(self, update)
        return

    await update.effective_message.reply_text(t("auto_usage", lang))


async def handle_auto_list(self: TelegramHandler, update: Update) -> None:
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    lang = await get_user_language(self, chat_id)

    if self._scheduler is None:
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            t("auto_no_scheduler", lang)
        )
        return
    automations = self._scheduler.list_automations()
    if not automations:
        await update.effective_message.reply_text(t("auto_empty", lang))
        return

    name_w = max(4, max(len(auto["name"]) for auto in automations))
    sched_w = max(6, max(len(auto["schedule"]) for auto in automations))
    h_name = t("auto_header_name", lang)
    h_sched = t("auto_header_schedule", lang)
    h_desc = t("auto_header_desc", lang)
    header = f"     {h_name:<{name_w}s}  {h_sched:<{sched_w}s}  {h_desc}"
    sep = "\u2500" * min(len(header) + 4, 50)
    table_lines = [header, sep]
    for auto in automations:
        icon = " \u2705 " if auto["enabled"] else " \u274c "
        table_lines.append(
            f"{icon} {self._escape_html(auto['name']):<{name_w}s}  "
            f"{self._escape_html(auto['schedule']):<{sched_w}s}  "
            f"{self._escape_html(auto['description'])}"
        )

    text = (
        f"\u23f0 <b>{t('auto_title', lang)}</b>"
        f"\n\n<pre>{chr(10).join(table_lines)}</pre>"
    )
    await update.effective_message.reply_text(text, parse_mode=ParseMode.HTML)


async def handle_auto_disable(self: TelegramHandler, update: Update, name: str) -> None:
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    lang = await get_user_language(self, chat_id)

    if self._scheduler is None:
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            t("auto_no_scheduler", lang)
        )
        return
    result = await self._scheduler.disable_automation(name)
    message = (
        t("auto_disabled", lang, name=name)
        if result
        else t("auto_not_found", lang, name=name)
    )
    await update.effective_message.reply_text(message)


async def handle_auto_reload(self: TelegramHandler, update: Update) -> None:
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    lang = await get_user_language(self, chat_id)

    if self._scheduler is None:
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            t("auto_no_scheduler", lang)
        )
        return
    try:
        count = await self._scheduler.reload_automations(strict=True)
        errors = self._get_auto_reload_errors()
        message = t("auto_reloaded", lang, count=count)
        if errors:
            message += self._format_reload_warnings(errors, lang=lang)
        await update.effective_message.reply_text(message)
    except Exception as exc:
        self._logger.error("auto_reload_failed", error=str(exc))
        await update.effective_message.reply_text(
            t("auto_reload_failed", lang, error=str(exc))
        )


async def handle_auto_run(self: TelegramHandler, update: Update, name: str) -> None:
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    lang = await get_user_language(self, chat_id)

    if self._scheduler is None:
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            t("auto_no_scheduler", lang)
        )
        return

    autos = {item["name"]: item for item in self._scheduler.list_automations()}
    target = autos.get(name)
    if target is None:
        await update.effective_message.reply_text(t("auto_not_found", lang, name=name))
        return
    if not bool(target.get("enabled", False)):
        await update.effective_message.reply_text(t("auto_is_disabled", lang, name=name))
        return

    ok = await self._scheduler.run_automation_once(name)
    if ok:
        await update.effective_message.reply_text(t("auto_run_success", lang, name=name))
        return
    await update.effective_message.reply_text(t("auto_run_failed", lang, name=name))
