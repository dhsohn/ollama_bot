"""Direct tests for Telegram command helper modules."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram.constants import ParseMode

from core import (
    telegram_commands,
    telegram_commands_automation,
    telegram_commands_memory,
)


def _make_update(chat_id: int = 111) -> SimpleNamespace:
    return SimpleNamespace(
        effective_chat=SimpleNamespace(id=chat_id),
        effective_message=SimpleNamespace(reply_text=AsyncMock()),
    )


def _make_handler() -> SimpleNamespace:
    return SimpleNamespace(
        _feedback_enabled=False,
        _config=SimpleNamespace(data_dir="/tmp"),
        _engine=SimpleNamespace(
            list_skills=MagicMock(return_value=[]),
            get_memory_stats=AsyncMock(return_value={
                "conversation_count": 3,
                "memory_count": 2,
                "oldest_conversation": "2026-01-01",
            }),
            clear_conversation=AsyncMock(return_value=4),
            export_conversation_markdown=AsyncMock(return_value=Path("/tmp/export.md")),
        ),
        _scheduler=None,
        _feedback=None,
        _escape_html=lambda value: value,
        _get_skill_reload_errors=lambda: [],
        _format_reload_warnings=lambda errors, lang: f"\nWARN {len(errors)} {lang}",
        _get_auto_reload_errors=lambda: [],
        _logger=MagicMock(),
    )


@pytest.mark.asyncio
async def test_cmd_help_includes_feedback_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = _make_handler()
    handler._feedback_enabled = True
    update = _make_update()

    monkeypatch.setattr(telegram_commands, "get_user_language", AsyncMock(return_value="ko"))

    await telegram_commands.cmd_help(handler, update, MagicMock())

    update.effective_message.reply_text.assert_awaited_once()
    text = update.effective_message.reply_text.await_args.args[0]
    assert "/feedback" in text
    assert update.effective_message.reply_text.await_args.kwargs["parse_mode"] == ParseMode.HTML


@pytest.mark.asyncio
async def test_cmd_skills_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = _make_handler()
    update = _make_update()
    context = MagicMock(args=[])
    monkeypatch.setattr(telegram_commands, "get_user_language", AsyncMock(return_value="ko"))

    await telegram_commands.cmd_skills(handler, update, context)

    update.effective_message.reply_text.assert_awaited_once_with("등록된 스킬이 없습니다.")


@pytest.mark.asyncio
async def test_cmd_skills_lists_skills(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = _make_handler()
    handler._engine.list_skills = MagicMock(return_value=[
        {"name": "summarize", "description": "요약", "triggers": ["/sum"]},
        {"name": "review", "description": "리뷰", "triggers": ["/review", "/rv"]},
    ])
    update = _make_update()
    context = MagicMock(args=[])
    monkeypatch.setattr(telegram_commands, "get_user_language", AsyncMock(return_value="ko"))

    await telegram_commands.cmd_skills(handler, update, context)

    text = update.effective_message.reply_text.await_args.args[0]
    assert "사용 가능한 스킬" in text
    assert "/review, /rv" in text
    assert update.effective_message.reply_text.await_args.kwargs["parse_mode"] == ParseMode.HTML


@pytest.mark.asyncio
async def test_cmd_memory_handles_stats_clear_export_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = _make_handler()
    update = _make_update()
    monkeypatch.setattr(telegram_commands_memory, "get_user_language", AsyncMock(return_value="ko"))

    await telegram_commands_memory.cmd_memory(handler, update, MagicMock(args=[]))
    stats_text = update.effective_message.reply_text.await_args.args[0]
    assert "메모리 상태" in stats_text

    await telegram_commands_memory.cmd_memory(handler, update, MagicMock(args=["clear"]))
    assert update.effective_message.reply_text.await_args.args[0] == "대화 기록 4건이 삭제되었습니다."

    await telegram_commands_memory.cmd_memory(handler, update, MagicMock(args=["export"]))
    export_text = update.effective_message.reply_text.await_args.args[0]
    assert "export.md" in export_text
    assert update.effective_message.reply_text.await_args.kwargs["parse_mode"] == ParseMode.HTML

    await telegram_commands_memory.cmd_memory(handler, update, MagicMock(args=["unknown"]))
    assert update.effective_message.reply_text.await_args.args[0] == "사용법: /memory [clear|export]"


@pytest.mark.asyncio
async def test_cmd_auto_without_scheduler_and_invalid_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = _make_handler()
    update = _make_update()
    monkeypatch.setattr(telegram_commands_automation, "get_user_language", AsyncMock(return_value="ko"))

    await telegram_commands_automation.cmd_auto(handler, update, MagicMock(args=[]))
    assert update.effective_message.reply_text.await_args.args[0] == "자동화 스케줄러가 초기화되지 않았습니다."

    scheduler = MagicMock()
    handler._scheduler = scheduler
    await telegram_commands_automation.cmd_auto(handler, update, MagicMock(args=["invalid"]))
    assert update.effective_message.reply_text.await_args.args[0].startswith("사용법: /auto")


@pytest.mark.asyncio
async def test_handle_auto_list_disable_reload_and_run_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = _make_handler()
    update = _make_update()
    scheduler = MagicMock()
    scheduler.list_automations = MagicMock(return_value=[
        {"name": "daily", "schedule": "0 9 * * *", "description": "리포트", "enabled": True},
        {"name": "weekly", "schedule": "0 10 * * 1", "description": "주간", "enabled": False},
    ])
    scheduler.disable_automation = AsyncMock(return_value=True)
    scheduler.reload_automations = AsyncMock(return_value=2)
    scheduler.run_automation_once = AsyncMock(side_effect=[True, False])
    scheduler.get_last_load_errors = MagicMock(return_value=["bad.yaml"])
    handler._scheduler = scheduler
    handler._get_auto_reload_errors = lambda: ["bad.yaml"]
    monkeypatch.setattr(telegram_commands_automation, "get_user_language", AsyncMock(return_value="ko"))

    await telegram_commands_automation.handle_auto_list(handler, update)
    list_text = update.effective_message.reply_text.await_args.args[0]
    assert "자동화 목록" in list_text
    assert "daily" in list_text and "weekly" in list_text

    await telegram_commands_automation.handle_auto_disable(handler, update, "daily")
    assert update.effective_message.reply_text.await_args.args[0] == "'daily' 자동화가 비활성화되었습니다."

    scheduler.disable_automation = AsyncMock(return_value=False)
    await telegram_commands_automation.handle_auto_disable(handler, update, "missing")
    assert update.effective_message.reply_text.await_args.args[0] == "'missing' 자동화를 찾을 수 없습니다."

    await telegram_commands_automation.handle_auto_reload(handler, update)
    reload_text = update.effective_message.reply_text.await_args.args[0]
    assert "자동화를 다시 로드했습니다: 2개" in reload_text
    assert "WARN 1 ko" in reload_text

    await telegram_commands_automation.handle_auto_run(handler, update, "daily")
    assert update.effective_message.reply_text.await_args.args[0] == "'daily' 자동화를 수동 실행했습니다."

    await telegram_commands_automation.handle_auto_run(handler, update, "daily")
    assert "실행에 실패" in update.effective_message.reply_text.await_args.args[0]

    await telegram_commands_automation.handle_auto_run(handler, update, "weekly")
    assert "비활성화 상태" in update.effective_message.reply_text.await_args.args[0]

    await telegram_commands_automation.handle_auto_run(handler, update, "missing")
    assert "찾을 수 없습니다" in update.effective_message.reply_text.await_args.args[0]
