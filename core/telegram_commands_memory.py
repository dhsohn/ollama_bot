from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from core.i18n import t
from core.telegram_menus import get_user_language

if TYPE_CHECKING:
    from telegram import Update

    from core.telegram_handler import TelegramHandler


async def cmd_memory(
    self: TelegramHandler,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    lang = await get_user_language(self, chat_id)
    args = context.args or []

    if not args:
        stats = await self._engine.get_memory_stats(chat_id)
        oldest = self._escape_html(stats["oldest_conversation"] or t("memory_none", lang))
        h_item = t("status_header_item", lang)
        h_value = t("status_header_value", lang)
        table = (
            f"{h_item:<16s}  {h_value}\n"
            "\u2500" * 25 + "\n"
            f"{t('memory_conversations', lang):<16s}  "
            f"{t('memory_count', lang, count=stats['conversation_count'])}\n"
            f"{t('memory_long_term', lang):<16s}  "
            f"{t('memory_count', lang, count=stats['memory_count'])}\n"
            f"{t('memory_oldest', lang):<16s}  {oldest}"
        )
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            f"\U0001f9e0 <b>{t('memory_title', lang)}</b>\n\n<pre>{table}</pre>",
            parse_mode=ParseMode.HTML,
        )
        return

    if args[0] == "clear":
        deleted = await self._engine.clear_conversation(chat_id)
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            t("memory_cleared", lang, count=deleted)
        )
        return

    if args[0] == "export":
        output_dir = Path(self._config.data_dir) / "conversations"
        filepath = await self._engine.export_conversation_markdown(chat_id, output_dir)
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            t("memory_exported", lang)
            + f"<code>{self._escape_html(filepath.name)}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    await update.effective_message.reply_text(  # type: ignore[union-attr]
        t("memory_usage", lang)
    )
