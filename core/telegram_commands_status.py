from __future__ import annotations

from typing import TYPE_CHECKING

from telegram.constants import ParseMode
from telegram.ext import ContextTypes

if TYPE_CHECKING:
    from telegram import Update

    from core.telegram_handler import TelegramHandler


async def cmd_status(
    self: TelegramHandler,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    _ = context
    status = await self._engine.get_status()
    llm = status["llm"]
    llm_status = "🟢 정상" if llm.get("status") == "ok" else "🔴 오류"
    model = self._escape_html(status["current_model"])

    rows: list[tuple[str, str]] = [
        ("가동 시간", status["uptime_human"]),
        ("LLM 백엔드", llm_status),
        ("모델", model),
        ("로드된 스킬", f"{status['skills_loaded']}개"),
    ]

    degraded_components = status.get("degraded_components", {})
    if degraded_components:
        parts = []
        for name, detail in degraded_components.items():
            reason = detail.get("reason") or "unknown"
            duration = detail.get("degraded_for_seconds")
            if isinstance(duration, int):
                parts.append(f"{name}: {reason} ({duration}초)")
            else:
                parts.append(f"{name}: {reason}")
        rows.append(("Degraded", "⚠️ " + ", ".join(parts)))
    else:
        rows.append(("Degraded", "✅ 없음"))

    if self._scheduler:
        autos = self._scheduler.list_automations()
        enabled = sum(1 for auto in autos if auto["enabled"])
        rows.append(("자동화", f"{enabled}/{len(autos)}개 활성"))

    if self._feedback:
        global_stats = await self._feedback.get_global_stats()
        rows.append(
            (
                "피드백",
                f"{global_stats['total']}건 ({global_stats['satisfaction_rate']:.0%})",
            )
        )

    label_w = max(len(row[0]) for row in rows)
    table_lines = [
        f"{'항목':<{label_w}s}  값",
        "─" * (label_w + 20),
    ]
    for label, value in rows:
        table_lines.append(f"{label:<{label_w}s}  {value}")

    text = f"📊 <b>시스템 상태</b>\n\n<pre>{chr(10).join(table_lines)}</pre>"
    await update.effective_message.reply_text(  # type: ignore[union-attr]
        text,
        parse_mode=ParseMode.HTML,
    )
