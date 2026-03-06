from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.engine import Engine


def trigger_background_summary(engine: Engine, chat_id: int) -> None:
    """백그라운드에서 요약 갱신을 트리거한다."""
    if engine._context_compressor is None:
        return
    if not engine._config.context_compressor.background_summarize:
        return
    if (
        engine._config.context_compressor.run_only_when_idle
        and engine._active_request_count > 1
    ):
        engine._logger.debug(
            "summary_refresh_skipped_busy",
            chat_id=chat_id,
            active_requests=engine._active_request_count,
        )
        return
    if len(engine._summary_tasks) >= engine._summary_task_limit:
        engine._logger.debug(
            "summary_refresh_skipped_task_limit",
            chat_id=chat_id,
            pending_summary_tasks=len(engine._summary_tasks),
            task_limit=engine._summary_task_limit,
        )
        return
    task = asyncio.create_task(
        engine._context_compressor.maybe_refresh_summary(chat_id),
        name=f"summary_refresh_{chat_id}",
    )
    engine._summary_tasks.add(task)
    task.add_done_callback(engine._handle_summary_task_done)


def handle_summary_task_done(engine: Engine, task: asyncio.Task[Any]) -> None:
    engine._summary_tasks.discard(task)
    engine._handle_background_task_error(task)


def handle_background_task_error(engine: Engine, task: asyncio.Task[Any]) -> None:
    """백그라운드 태스크 실패를 누락하지 않고 기록한다."""
    if task.cancelled():
        return
    try:
        exc = task.exception()
    except Exception as callback_exc:
        engine._logger.error("background_task_error_check_failed", error=str(callback_exc))
        return
    if exc is not None:
        engine._logger.error(
            "background_task_failed",
            task_name=task.get_name(),
            error=str(exc),
        )
