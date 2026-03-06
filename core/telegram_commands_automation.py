from __future__ import annotations

from typing import TYPE_CHECKING

from telegram.constants import ParseMode
from telegram.ext import ContextTypes

if TYPE_CHECKING:
    from telegram import Update

    from core.telegram_handler import TelegramHandler


async def cmd_auto(
    self: TelegramHandler,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    if not self._scheduler:
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            "자동화 스케줄러가 초기화되지 않았습니다."
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

    await update.effective_message.reply_text(
        "사용법: /auto [list|disable <이름>|run <이름>|reload]"
    )


async def handle_auto_list(self: TelegramHandler, update: Update) -> None:
    if self._scheduler is None:
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            "자동화 스케줄러가 초기화되지 않았습니다."
        )
        return
    automations = self._scheduler.list_automations()
    if not automations:
        await update.effective_message.reply_text("등록된 자동화가 없습니다.")
        return

    name_w = max(4, max(len(auto["name"]) for auto in automations))
    sched_w = max(6, max(len(auto["schedule"]) for auto in automations))
    header = f"     {'이름':<{name_w}s}  {'스케줄':<{sched_w}s}  설명"
    sep = "─" * min(len(header) + 4, 50)
    table_lines = [header, sep]
    for auto in automations:
        icon = " ✅ " if auto["enabled"] else " ❌ "
        table_lines.append(
            f"{icon} {self._escape_html(auto['name']):<{name_w}s}  "
            f"{self._escape_html(auto['schedule']):<{sched_w}s}  "
            f"{self._escape_html(auto['description'])}"
        )

    text = f"⏰ <b>자동화 목록</b>\n\n<pre>{chr(10).join(table_lines)}</pre>"
    await update.effective_message.reply_text(text, parse_mode=ParseMode.HTML)


async def handle_auto_disable(self: TelegramHandler, update: Update, name: str) -> None:
    if self._scheduler is None:
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            "자동화 스케줄러가 초기화되지 않았습니다."
        )
        return
    result = await self._scheduler.disable_automation(name)
    message = (
        f"'{name}' 자동화가 비활성화되었습니다."
        if result
        else f"'{name}' 자동화를 찾을 수 없습니다."
    )
    await update.effective_message.reply_text(message)


async def handle_auto_reload(self: TelegramHandler, update: Update) -> None:
    if self._scheduler is None:
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            "자동화 스케줄러가 초기화되지 않았습니다."
        )
        return
    try:
        count = await self._scheduler.reload_automations(strict=True)
        errors = self._get_auto_reload_errors()
        message = f"자동화를 다시 로드했습니다: {count}개"
        if errors:
            message += self._format_reload_warnings(errors)
        await update.effective_message.reply_text(message)
    except Exception as exc:
        self._logger.error("auto_reload_failed", error=str(exc))
        await update.effective_message.reply_text(f"자동화 로드 실패: {exc}")


async def handle_auto_run(self: TelegramHandler, update: Update, name: str) -> None:
    if self._scheduler is None:
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            "자동화 스케줄러가 초기화되지 않았습니다."
        )
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
