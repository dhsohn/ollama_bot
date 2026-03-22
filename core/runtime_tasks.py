"""Runtime maintenance tasks."""

from __future__ import annotations

import asyncio
import random
from typing import Any

from core.feedback_manager import FeedbackManager
from core.llm_protocol import LLMClientProtocol
from core.memory import MemoryManager


async def run_memory_maintenance_cycle(
    memory: MemoryManager,
    logger: Any,
    *,
    feedback: FeedbackManager | None = None,
    feedback_retention_days: int | None = None,
    semantic_cache: Any = None,
) -> None:
    """Run one retention-maintenance pass for all runtime stores."""
    try:
        deleted = await memory.prune_old_conversations()
        logger.debug("memory_retention_pruned", deleted=deleted)
    except Exception as exc:
        logger.error("memory_retention_prune_failed", error=str(exc))

    if feedback is not None and feedback_retention_days is not None:
        try:
            fb_deleted = await feedback.prune_old_feedback(feedback_retention_days)
            logger.debug("feedback_retention_pruned", deleted=fb_deleted)
        except Exception as exc:
            logger.error("feedback_retention_prune_failed", error=str(exc))

    if semantic_cache is not None:
        try:
            cache_pruned = await semantic_cache.prune_expired()
            if cache_pruned:
                logger.debug("semantic_cache_pruned", deleted=cache_pruned)
        except Exception as exc:
            logger.error("semantic_cache_prune_failed", error=str(exc))


def resolve_maintenance_sleep(interval_seconds: int, jitter_ratio: float) -> float:
    """Return the randomized sleep duration between maintenance passes."""
    base_interval = max(1, interval_seconds)
    jitter = random.uniform(0.0, base_interval * max(0.0, jitter_ratio))
    return base_interval + jitter


async def memory_maintenance_loop(
    memory: MemoryManager,
    logger: Any,
    interval_seconds: int,
    jitter_ratio: float,
    feedback: FeedbackManager | None = None,
    feedback_retention_days: int | None = None,
    semantic_cache: Any = None,
) -> None:
    """Periodically prune old conversation, feedback, and cache data."""
    while True:
        await run_memory_maintenance_cycle(
            memory,
            logger,
            feedback=feedback,
            feedback_retention_days=feedback_retention_days,
            semantic_cache=semantic_cache,
        )
        await asyncio.sleep(
            resolve_maintenance_sleep(interval_seconds, jitter_ratio)
        )


async def run_llm_recovery_cycle(
    llm: LLMClientProtocol,
    logger: Any,
) -> None:
    """Run one backend-health pass and attempt recovery if needed."""
    try:
        health = await llm.health_check(attempt_recovery=False)
        if health.get("status") != "ok":
            recovered = await llm.recover_connection(force=True)
            if recovered:
                logger.info("llm_recovered_by_loop")
            else:
                logger.warning(
                    "llm_still_unhealthy",
                    error=health.get("error"),
                )
    except Exception as exc:
        logger.error("llm_recovery_loop_failed", error=str(exc))


async def llm_recovery_loop(
    llm: LLMClientProtocol,
    logger: Any,
    interval_seconds: int,
) -> None:
    """Periodically check backend health and attempt recovery when needed."""
    while True:
        await run_llm_recovery_cycle(llm, logger)
        await asyncio.sleep(interval_seconds)
