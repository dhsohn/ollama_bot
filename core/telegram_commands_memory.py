from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from telegram.constants import ParseMode
from telegram.ext import ContextTypes

if TYPE_CHECKING:
    from telegram import Update

    from core.telegram_handler import TelegramHandler


async def cmd_memory(
    self: TelegramHandler,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    args = context.args or []

    if not args:
        stats = await self._engine.get_memory_stats(chat_id)
        oldest = self._escape_html(stats["oldest_conversation"] or "없음")
        table = (
            "항목              값\n"
            "─" * 25 + "\n"
            f"대화 기록         {stats['conversation_count']}건\n"
            f"장기 메모리       {stats['memory_count']}건\n"
            f"가장 오래된 대화  {oldest}"
        )
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            f"🧠 <b>메모리 상태</b>\n\n<pre>{table}</pre>",
            parse_mode=ParseMode.HTML,
        )
        return

    if args[0] == "clear":
        deleted = await self._engine.clear_conversation(chat_id)
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            f"대화 기록 {deleted}건이 삭제되었습니다."
        )
        return

    if args[0] == "export":
        output_dir = Path(self._config.data_dir) / "conversations"
        filepath = await self._engine.export_conversation_markdown(chat_id, output_dir)
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            "대화 기록이 내보내기되었습니다: "
            f"<code>{self._escape_html(filepath.name)}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    await update.effective_message.reply_text(  # type: ignore[union-attr]
        "사용법: /memory [clear|export]"
    )
