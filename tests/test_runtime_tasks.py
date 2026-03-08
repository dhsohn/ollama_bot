"""Runtime background loop tests."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from core import runtime_tasks


@pytest.mark.asyncio
async def test_memory_maintenance_loop_prunes_all_components(monkeypatch: pytest.MonkeyPatch) -> None:
    memory = MagicMock(prune_old_conversations=AsyncMock(return_value=3))
    feedback = MagicMock(prune_old_feedback=AsyncMock(return_value=2))
    semantic_cache = MagicMock(prune_expired=AsyncMock(return_value=1))
    logger = MagicMock()
    sleep_calls: list[float] = []

    monkeypatch.setattr(runtime_tasks.random, "uniform", lambda _a, _b: 0.5)

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)
        raise asyncio.CancelledError

    monkeypatch.setattr(runtime_tasks.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await runtime_tasks.memory_maintenance_loop(
            memory=memory,
            logger=logger,
            interval_seconds=10,
            jitter_ratio=0.1,
            feedback=feedback,
            feedback_retention_days=90,
            semantic_cache=semantic_cache,
        )

    memory.prune_old_conversations.assert_awaited_once()
    feedback.prune_old_feedback.assert_awaited_once_with(90)
    semantic_cache.prune_expired.assert_awaited_once()
    logger.debug.assert_any_call("memory_retention_pruned", deleted=3)
    logger.debug.assert_any_call("feedback_retention_pruned", deleted=2)
    logger.debug.assert_any_call("semantic_cache_pruned", deleted=1)
    assert sleep_calls == [10.5]


@pytest.mark.asyncio
async def test_memory_maintenance_loop_logs_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    memory = MagicMock(prune_old_conversations=AsyncMock(side_effect=RuntimeError("memory boom")))
    feedback = MagicMock(prune_old_feedback=AsyncMock(side_effect=RuntimeError("feedback boom")))
    semantic_cache = MagicMock(prune_expired=AsyncMock(side_effect=RuntimeError("cache boom")))
    logger = MagicMock()

    async def fake_sleep(_delay: float) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(runtime_tasks.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await runtime_tasks.memory_maintenance_loop(
            memory=memory,
            logger=logger,
            interval_seconds=1,
            jitter_ratio=0.0,
            feedback=feedback,
            feedback_retention_days=90,
            semantic_cache=semantic_cache,
        )

    logger.error.assert_any_call("memory_retention_prune_failed", error="memory boom")
    logger.error.assert_any_call("feedback_retention_prune_failed", error="feedback boom")
    logger.error.assert_any_call("semantic_cache_prune_failed", error="cache boom")


@pytest.mark.asyncio
async def test_llm_recovery_loop_recovers_when_unhealthy(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = MagicMock(
        health_check=AsyncMock(return_value={"status": "error", "error": "timeout"}),
        recover_connection=AsyncMock(return_value=True),
    )
    logger = MagicMock()

    async def fake_sleep(_delay: float) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(runtime_tasks.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await runtime_tasks.llm_recovery_loop(llm, logger, interval_seconds=5)

    llm.recover_connection.assert_awaited_once_with(force=True)
    logger.info.assert_any_call("llm_recovered_by_loop")


@pytest.mark.asyncio
async def test_llm_recovery_loop_warns_when_still_unhealthy(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = MagicMock(
        health_check=AsyncMock(return_value={"status": "error", "error": "down"}),
        recover_connection=AsyncMock(return_value=False),
    )
    logger = MagicMock()

    async def fake_sleep(_delay: float) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(runtime_tasks.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await runtime_tasks.llm_recovery_loop(llm, logger, interval_seconds=5)

    logger.warning.assert_any_call("llm_still_unhealthy", error="down")


@pytest.mark.asyncio
async def test_llm_recovery_loop_logs_health_check_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = MagicMock(
        health_check=AsyncMock(side_effect=RuntimeError("boom")),
        recover_connection=AsyncMock(),
    )
    logger = MagicMock()

    async def fake_sleep(_delay: float) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(runtime_tasks.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await runtime_tasks.llm_recovery_loop(llm, logger, interval_seconds=5)

    logger.error.assert_any_call("llm_recovery_loop_failed", error="boom")
