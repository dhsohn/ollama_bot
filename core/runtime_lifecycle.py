"""Runtime start and shutdown lifecycle."""

from __future__ import annotations

import asyncio
import signal
from contextlib import suppress

from core.runtime_factory import RuntimeState
from core.runtime_tasks import llm_recovery_loop, memory_maintenance_loop


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

    if memory_maintenance_task is not None:
        memory_maintenance_task.cancel()
        await asyncio.gather(memory_maintenance_task, return_exceptions=True)

    if llm_recovery_task is not None:
        llm_recovery_task.cancel()
        await asyncio.gather(llm_recovery_task, return_exceptions=True)

    if runtime.rag_startup_index_task is not None:
        runtime.rag_startup_index_task.cancel()
        await asyncio.gather(runtime.rag_startup_index_task, return_exceptions=True)

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

    def _signal_handler() -> None:
        logger.info("shutdown_signal_received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, _signal_handler)

    memory_maintenance_task: asyncio.Task | None = None
    llm_recovery_task: asyncio.Task | None = None
    scheduler_started = False
    app_started = False
    updater_started = False

    async with app:
        try:
            await app.start()
            app_started = True

            runtime.scheduler.start()
            scheduler_started = True

            memory_maintenance_task = asyncio.create_task(
                memory_maintenance_loop(
                    runtime.memory,
                    logger,
                    interval_seconds=runtime.config.runtime_maintenance.memory_maintenance_interval_seconds,
                    jitter_ratio=runtime.config.runtime_maintenance.memory_maintenance_jitter_ratio,
                    feedback=runtime.feedback,
                    feedback_retention_days=runtime.config.feedback.retention_days,
                    semantic_cache=runtime.semantic_cache,
                ),
                name="memory_maintenance",
            )
            llm_recovery_task = asyncio.create_task(
                llm_recovery_loop(
                    runtime.llm,
                    logger,
                    interval_seconds=runtime.config.runtime_maintenance.llm_recovery_interval_seconds,
                ),
                name="llm_recovery",
            )

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
            updater_started = True
            await stop_event.wait()
        finally:
            await shutdown_runtime(
                runtime=runtime,
                memory_maintenance_task=memory_maintenance_task,
                llm_recovery_task=llm_recovery_task,
                scheduler_started=scheduler_started,
                app_started=app_started,
                updater_started=updater_started,
            )
