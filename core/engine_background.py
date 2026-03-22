from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.engine import Engine


class BackgroundSummaryController:
    """Manage background summary refresh task lifecycle."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def trigger_summary(self, chat_id: int) -> None:
        """Trigger a background summary refresh."""
        engine = self._engine
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

    def handle_summary_task_done(self, task: asyncio.Task[Any]) -> None:
        engine = self._engine
        engine._summary_tasks.discard(task)
        engine._handle_background_task_error(task)

    def handle_background_task_error(self, task: asyncio.Task[Any]) -> None:
        """Log background-task failures instead of silently dropping them."""
        if task.cancelled():
            return
        try:
            exc = task.exception()
        except Exception as callback_exc:
            self._engine._logger.error(
                "background_task_error_check_failed",
                error=str(callback_exc),
            )
            return
        if exc is not None:
            self._engine._logger.error(
                "background_task_failed",
                task_name=task.get_name(),
                error=str(exc),
            )


def _controller(engine: Engine) -> BackgroundSummaryController:
    existing = getattr(engine, "_background_summary_controller", None)
    if isinstance(existing, BackgroundSummaryController):
        return existing
    return BackgroundSummaryController(engine)


def trigger_background_summary(engine: Engine, chat_id: int) -> None:
    _controller(engine).trigger_summary(chat_id)


def handle_summary_task_done(engine: Engine, task: asyncio.Task[Any]) -> None:
    _controller(engine).handle_summary_task_done(task)


def handle_background_task_error(engine: Engine, task: asyncio.Task[Any]) -> None:
    _controller(engine).handle_background_task_error(task)
