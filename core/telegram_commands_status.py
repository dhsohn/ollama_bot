from __future__ import annotations

from typing import TYPE_CHECKING

from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from core.i18n import t
from core.telegram_menus import get_user_language

if TYPE_CHECKING:
    from telegram import Update

    from core.telegram_handler import TelegramHandler


async def cmd_status(
    self: TelegramHandler,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    _ = context
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    lang = await get_user_language(self, chat_id)

    status = await self._engine.get_status()
    llm = status["llm"]
    llm_ok = llm.get("status") == "ok"
    llm_status = (
        f"\U0001f7e2 {t('status_llm_ok', lang)}"
        if llm_ok
        else f"\U0001f534 {t('status_llm_error', lang)}"
    )
    model = self._escape_html(status["current_model"])

    rows: list[tuple[str, str]] = [
        (t("status_uptime", lang), status["uptime_human"]),
        (t("status_llm_backend", lang), llm_status),
        (t("status_model", lang), model),
        (t("status_skills", lang), t("status_count_suffix", lang, count=status["skills_loaded"])),
    ]

    degraded_components = status.get("degraded_components", {})
    if degraded_components:
        parts = []
        for name, detail in degraded_components.items():
            reason = detail.get("reason") or "unknown"
            duration = detail.get("degraded_for_seconds")
            if isinstance(duration, int):
                parts.append(
                    f"{name}: {reason} ({t('status_seconds_suffix', lang, seconds=duration)})"
                )
            else:
                parts.append(f"{name}: {reason}")
        rows.append((t("status_degraded", lang), "\u26a0\ufe0f " + ", ".join(parts)))
    else:
        rows.append((t("status_degraded", lang), f"\u2705 {t('status_degraded_none', lang)}"))

    if self._scheduler:
        autos = self._scheduler.list_automations()
        enabled = sum(1 for auto in autos if auto["enabled"])
        rows.append((
            t("status_automations", lang),
            t("status_automations_active", lang, enabled=enabled, total=len(autos)),
        ))

    if self._feedback:
        global_stats = await self._feedback.get_global_stats()
        cnt = t("memory_count", lang, count=global_stats["total"])
        rows.append(
            (
                t("status_feedback", lang),
                f"{cnt} ({global_stats['satisfaction_rate']:.0%})",
            )
        )

    label_w = max(len(row[0]) for row in rows)
    table_lines = [
        f"{t('status_header_item', lang):<{label_w}s}  {t('status_header_value', lang)}",
        "\u2500" * (label_w + 20),
    ]
    for label, value in rows:
        table_lines.append(f"{label:<{label_w}s}  {value}")

    text = (
        f"\U0001f4ca <b>{t('status_title', lang)}</b>"
        f"\n\n<pre>{chr(10).join(table_lines)}</pre>"
    )
    await update.effective_message.reply_text(  # type: ignore[union-attr]
        text,
        parse_mode=ParseMode.HTML,
    )
