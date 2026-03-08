"""engine_background 모듈 테스트."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.engine_background import (
    handle_background_task_error,
    handle_summary_task_done,
    trigger_background_summary,
)


def _make_engine(**overrides):
    engine = MagicMock()
    engine._context_compressor = MagicMock()
    engine._context_compressor.maybe_refresh_summary = AsyncMock()
    engine._config = MagicMock()
    engine._config.context_compressor.background_summarize = True
    engine._config.context_compressor.run_only_when_idle = False
    engine._active_request_count = 0
    engine._summary_tasks = set()
    engine._summary_task_limit = 5
    engine._handle_summary_task_done = MagicMock()
    engine._handle_background_task_error = MagicMock()
    engine._logger = MagicMock()
    for key, value in overrides.items():
        setattr(engine, key, value)
    return engine


class TestTriggerBackgroundSummary:
    def test_skips_when_no_compressor(self) -> None:
        engine = _make_engine(_context_compressor=None)
        trigger_background_summary(engine, 111)
        assert len(engine._summary_tasks) == 0

    def test_skips_when_background_disabled(self) -> None:
        engine = _make_engine()
        engine._config.context_compressor.background_summarize = False
        trigger_background_summary(engine, 111)
        assert len(engine._summary_tasks) == 0

    def test_skips_when_busy_and_idle_only(self) -> None:
        engine = _make_engine()
        engine._config.context_compressor.run_only_when_idle = True
        engine._active_request_count = 3
        trigger_background_summary(engine, 111)
        assert len(engine._summary_tasks) == 0
        engine._logger.debug.assert_called()

    def test_skips_when_task_limit_reached(self) -> None:
        engine = _make_engine()
        engine._summary_tasks = {MagicMock() for _ in range(5)}
        engine._summary_task_limit = 5
        trigger_background_summary(engine, 111)
        assert len(engine._summary_tasks) == 5
        engine._logger.debug.assert_called()

    @pytest.mark.asyncio
    async def test_creates_task_on_success(self) -> None:
        engine = _make_engine()
        trigger_background_summary(engine, 111)
        assert len(engine._summary_tasks) == 1
        # Clean up the task
        for task in engine._summary_tasks:
            task.cancel()


class TestHandleSummaryTaskDone:
    def test_discards_task_and_checks_error(self) -> None:
        engine = _make_engine()
        task = MagicMock()
        engine._summary_tasks = {task}
        handle_summary_task_done(engine, task)
        assert task not in engine._summary_tasks
        engine._handle_background_task_error.assert_called_once_with(task)


class TestHandleBackgroundTaskError:
    def test_cancelled_task_ignored(self) -> None:
        engine = _make_engine()
        task = MagicMock()
        task.cancelled.return_value = True
        handle_background_task_error(engine, task)
        engine._logger.error.assert_not_called()

    def test_task_with_exception_logged(self) -> None:
        engine = _make_engine()
        task = MagicMock()
        task.cancelled.return_value = False
        task.exception.return_value = RuntimeError("boom")
        task.get_name.return_value = "test_task"
        handle_background_task_error(engine, task)
        engine._logger.error.assert_called_once()

    def test_task_without_exception_ok(self) -> None:
        engine = _make_engine()
        task = MagicMock()
        task.cancelled.return_value = False
        task.exception.return_value = None
        handle_background_task_error(engine, task)
        engine._logger.error.assert_not_called()

    def test_exception_check_failure_logged(self) -> None:
        engine = _make_engine()
        task = MagicMock()
        task.cancelled.return_value = False
        task.exception.side_effect = RuntimeError("check failed")
        handle_background_task_error(engine, task)
        engine._logger.error.assert_called_once()
