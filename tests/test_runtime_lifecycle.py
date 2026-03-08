"""Runtime lifecycle tests."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.runtime_lifecycle import run_runtime, shutdown_runtime


def _make_runtime() -> SimpleNamespace:
    logger = MagicMock()
    updater = SimpleNamespace(
        start_polling=AsyncMock(),
        stop=AsyncMock(),
    )

    class FakeApp:
        def __init__(self) -> None:
            self.updater = updater
            self.start = AsyncMock()
            self.stop = AsyncMock()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    runtime = SimpleNamespace(
        logger=logger,
        app=FakeApp(),
        scheduler=SimpleNamespace(start=MagicMock(), stop=MagicMock()),
        memory=SimpleNamespace(),
        llm=SimpleNamespace(),
        config=SimpleNamespace(
            telegram=SimpleNamespace(polling_interval=1),
            runtime_maintenance=SimpleNamespace(
                memory_maintenance_interval_seconds=60,
                memory_maintenance_jitter_ratio=0.1,
                llm_recovery_interval_seconds=30,
            ),
            feedback=SimpleNamespace(retention_days=90),
        ),
        feedback=None,
        semantic_cache=None,
        llm_provider="lemonade",
        skill_count=3,
        auto_count=4,
        cleanup_stack=AsyncExitStack(),
        rag_startup_index_task=None,
        degraded_components=[],
    )
    return runtime


@pytest.mark.asyncio
async def test_shutdown_runtime_cleans_up_and_logs_errors() -> None:
    runtime = _make_runtime()
    runtime.scheduler.stop = MagicMock(side_effect=RuntimeError("scheduler boom"))
    runtime.app.updater.stop = AsyncMock(side_effect=RuntimeError("updater boom"))
    runtime.app.stop = AsyncMock(side_effect=RuntimeError("app boom"))
    runtime.cleanup_stack.aclose = AsyncMock(side_effect=RuntimeError("cleanup boom"))

    memory_task = asyncio.create_task(asyncio.sleep(30))
    recovery_task = asyncio.create_task(asyncio.sleep(30))
    rag_task = asyncio.create_task(asyncio.sleep(30))
    runtime.rag_startup_index_task = rag_task

    await shutdown_runtime(
        runtime=runtime,
        memory_maintenance_task=memory_task,
        llm_recovery_task=recovery_task,
        scheduler_started=True,
        app_started=True,
        updater_started=True,
    )

    assert memory_task.cancelled()
    assert recovery_task.cancelled()
    assert rag_task.cancelled()
    runtime.logger.error.assert_any_call("scheduler_stop_failed", error="scheduler boom")
    runtime.logger.error.assert_any_call("updater_stop_failed", error="updater boom")
    runtime.logger.error.assert_any_call("app_stop_failed", error="app boom")
    runtime.logger.error.assert_any_call("resource_cleanup_failed", error="cleanup boom")
    runtime.logger.info.assert_any_call("shutdown_complete")


@pytest.mark.asyncio
async def test_run_runtime_starts_background_loops_and_stops_on_signal(monkeypatch) -> None:
    runtime = _make_runtime()
    callbacks: list[callable] = []

    class FakeLoop:
        def add_signal_handler(self, _sig, callback) -> None:
            callbacks.append(callback)

    async def fake_memory_loop(*args, **kwargs) -> None:
        _ = (args, kwargs)
        await asyncio.Event().wait()

    async def fake_recovery_loop(*args, **kwargs) -> None:
        _ = (args, kwargs)
        await asyncio.Event().wait()

    async def start_polling(**kwargs) -> None:
        assert kwargs["poll_interval"] == 1
        callbacks[0]()

    runtime.app.updater.start_polling = AsyncMock(side_effect=start_polling)

    monkeypatch.setattr("core.runtime_lifecycle.asyncio.get_running_loop", lambda: FakeLoop())
    monkeypatch.setattr("core.runtime_lifecycle.memory_maintenance_loop", fake_memory_loop)
    monkeypatch.setattr("core.runtime_lifecycle.llm_recovery_loop", fake_recovery_loop)
    monkeypatch.setattr("core.runtime_lifecycle.model_for_provider", lambda _config: "test-model")

    await run_runtime(runtime)

    runtime.app.start.assert_awaited_once()
    runtime.scheduler.start.assert_called_once()
    runtime.app.updater.start_polling.assert_awaited_once()
    runtime.app.updater.stop.assert_awaited_once()
    runtime.app.stop.assert_awaited_once()
    runtime.logger.info.assert_any_call("shutdown_signal_received")
    runtime.logger.info.assert_any_call(
        "bot_running",
        provider="lemonade",
        model="test-model",
        skills=3,
        automations=4,
        degraded_count=0,
        degraded_components=None,
    )
