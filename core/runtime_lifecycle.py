"""Runtime start and shutdown lifecycle."""

from __future__ import annotations

import asyncio
import signal
from contextlib import suppress
from typing import Any

from core.runtime_factory import RuntimeState
from core.runtime_loop_state import RuntimeTaskHandles
from core.runtime_tasks import llm_recovery_loop, memory_maintenance_loop


async def _cancel_runtime_task(task: asyncio.Task[Any] | None) -> None:
    if task is None:
        return
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


def _register_shutdown_signals(logger: Any, stop_event: asyncio.Event) -> None:
    def _signal_handler() -> None:
        logger.info("shutdown_signal_received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, _signal_handler)


def _start_background_tasks(runtime: RuntimeState) -> RuntimeTaskHandles:
    handles = RuntimeTaskHandles()
    handles.memory_maintenance_task = asyncio.create_task(
        memory_maintenance_loop(
            runtime.memory,
            runtime.logger,
            interval_seconds=runtime.config.runtime_maintenance.memory_maintenance_interval_seconds,
            jitter_ratio=runtime.config.runtime_maintenance.memory_maintenance_jitter_ratio,
            feedback=runtime.feedback,
            feedback_retention_days=runtime.config.feedback.retention_days,
            semantic_cache=runtime.semantic_cache,
        ),
        name="memory_maintenance",
    )
    handles.llm_recovery_task = asyncio.create_task(
        llm_recovery_loop(
            runtime.llm,
            runtime.logger,
            interval_seconds=runtime.config.runtime_maintenance.llm_recovery_interval_seconds,
        ),
        name="llm_recovery",
    )
    return handles


async def shutdown_runtime(
    runtime: RuntimeState,
    memory_maintenance_task: asyncio.Task | None,
    llm_recovery_task: asyncio.Task | None,
    scheduler_started: bool,
    app_started: bool,
    updater_started: bool,
) -> None:
    """Release runtime resources in reverse order."""
    logger = runtime.logger
    app = runtime.app
    logger.info("shutting_down")

    if scheduler_started:
        try:
            runtime.scheduler.stop()
        except Exception as exc:
            logger.error("scheduler_stop_failed", error=str(exc))

    await _cancel_runtime_task(memory_maintenance_task)
    await _cancel_runtime_task(llm_recovery_task)
    await _cancel_runtime_task(runtime.rag_startup_index_task)

    if updater_started:
        try:
            await app.updater.stop()
        except Exception as exc:
            logger.error("updater_stop_failed", error=str(exc))

    if app_started:
        try:
            await app.stop()
        except Exception as exc:
            logger.error("app_stop_failed", error=str(exc))

    try:
        await runtime.cleanup_stack.aclose()
    except Exception as exc:
        logger.error("resource_cleanup_failed", error=str(exc))

    logger.info("shutdown_complete")


async def run_runtime(runtime: RuntimeState) -> None:
    """Run the initialized runtime and clean up on shutdown."""
    logger = runtime.logger
    app = runtime.app
    stop_event = asyncio.Event()
    _register_shutdown_signals(logger, stop_event)
    handles = RuntimeTaskHandles()

    async with app:
        try:
            await app.start()
            handles.app_started = True

            runtime.scheduler.start()
            handles.scheduler_started = True
            task_handles = _start_background_tasks(runtime)
            handles.memory_maintenance_task = task_handles.memory_maintenance_task
            handles.llm_recovery_task = task_handles.llm_recovery_task

            logger.info(
                "bot_running",
                model=runtime.llm.default_model,
                skills=runtime.skill_count,
                automations=runtime.auto_count,
                degraded_count=len(runtime.degraded_components),
                degraded_components=runtime.degraded_components or None,
            )

            await app.updater.start_polling(
                poll_interval=runtime.config.telegram.polling_interval,
                drop_pending_updates=True,
            )
            handles.updater_started = True
            await stop_event.wait()
        finally:
            await shutdown_runtime(
                runtime=runtime,
                memory_maintenance_task=handles.memory_maintenance_task,
                llm_recovery_task=handles.llm_recovery_task,
                scheduler_started=handles.scheduler_started,
                app_started=handles.app_started,
                updater_started=handles.updater_started,
            )
