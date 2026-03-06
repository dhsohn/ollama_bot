"""런타임 유지보수 태스크."""

from __future__ import annotations

import asyncio
import random
from typing import Any

from core.feedback_manager import FeedbackManager
from core.llm_protocol import LLMClientProtocol
from core.memory import MemoryManager


async def memory_maintenance_loop(
    memory: MemoryManager,
    logger: Any,
    interval_seconds: int,
    jitter_ratio: float,
    feedback: FeedbackManager | None = None,
    feedback_retention_days: int | None = None,
    semantic_cache: Any = None,
) -> None:
    """주기적으로 오래된 대화/피드백/캐시 데이터를 정리한다."""
    while True:
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

        base_interval = max(1, interval_seconds)
        jitter = random.uniform(0.0, base_interval * max(0.0, jitter_ratio))
        await asyncio.sleep(base_interval + jitter)


async def llm_recovery_loop(
    llm: LLMClientProtocol,
    logger: Any,
    interval_seconds: int,
) -> None:
    """LLM 백엔드 상태를 주기 점검하고 장애 시 재연결을 시도한다."""
    while True:
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

        await asyncio.sleep(interval_seconds)
