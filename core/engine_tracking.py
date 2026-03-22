from __future__ import annotations

import contextlib
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from core.engine import Engine
    from core.rag.types import RAGTrace
    from core.skill_manager import SkillDefinition


class EngineTrackingOperations:
    """Own request lifecycle bookkeeping and stream metadata operations."""

    def __init__(
        self,
        engine: Engine,
        *,
        meta_factory: Callable[..., Any] | None = None,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._engine = engine
        self._meta_factory = meta_factory
        self._monotonic_fn = monotonic_fn

    @contextlib.asynccontextmanager
    async def track_request(
        self,
        chat_id: int,
        *,
        stream: bool,
    ) -> AsyncGenerator[None, None]:
        """Manage request counters and log context consistently per request."""
        engine = self._engine
        request_id = uuid.uuid4().hex[:8]
        structlog.contextvars.bind_contextvars(request_id=request_id, chat_id=chat_id)
        engine._logger.info("request_started", stream=stream)
        engine._active_request_count += 1
        try:
            yield
        finally:
            engine._active_request_count -= 1
            if engine._active_request_count < 0:
                engine._logger.error(
                    "active_request_count_underflow",
                    active_requests=engine._active_request_count,
                )
                engine._active_request_count = 0
            structlog.contextvars.unbind_contextvars("request_id", "chat_id")

    def consume_last_stream_meta(self, chat_id: int) -> dict[str, Any] | None:
        """Return one-shot metadata captured during streaming."""
        self.cleanup_stream_meta(now=None)
        meta = self._engine._last_stream_meta.pop(chat_id, None)
        if meta is None:
            return None
        result: dict[str, Any] = {
            "tier": meta.tier,
            "intent": meta.intent,
            "cache_id": meta.cache_id,
            "usage": meta.usage,
        }
        if meta.stop_reason is not None:
            result["stop_reason"] = meta.stop_reason
        if meta.model_role is not None:
            result["model_role"] = meta.model_role
        if meta.rag_trace is not None:
            result["rag_trace"] = meta.rag_trace
        return result

    def set_stream_meta(
        self,
        chat_id: int,
        *,
        tier: str,
        intent: str | None = None,
        cache_id: int | None = None,
        stop_reason: str | None = None,
        usage: Any = None,
        model_role: str | None = None,
        rag_trace: dict | None = None,
    ) -> None:
        if self._meta_factory is None:
            raise RuntimeError("EngineTrackingOperations requires meta_factory for stream metadata.")
        now = self._monotonic_fn()
        self._engine._last_stream_meta[chat_id] = self._meta_factory(
            tier=tier,
            intent=intent,
            cache_id=cache_id,
            stop_reason=stop_reason,
            usage=usage,
            model_role=model_role,
            rag_trace=rag_trace,
            created_at=now,
        )
        self.cleanup_stream_meta(now=now)

    def cleanup_stream_meta(self, *, now: float | None) -> None:
        """Prune unconsumed stream metadata by TTL and max-entry limits."""
        engine = self._engine
        if not engine._last_stream_meta:
            return
        current = self._monotonic_fn() if now is None else now

        expired_chat_ids = [
            chat_id
            for chat_id, meta in engine._last_stream_meta.items()
            if current - float(getattr(meta, "created_at", 0.0)) >= engine._stream_meta_ttl_seconds
        ]
        for chat_id in expired_chat_ids:
            engine._last_stream_meta.pop(chat_id, None)

        overflow = len(engine._last_stream_meta) - engine._stream_meta_max_entries
        if overflow <= 0:
            return
        oldest_chat_ids = sorted(
            engine._last_stream_meta,
            key=lambda cid: float(getattr(engine._last_stream_meta[cid], "created_at", 0.0)),
        )[:overflow]
        for chat_id in oldest_chat_ids:
            engine._last_stream_meta.pop(chat_id, None)

    def log_request(
        self,
        t0: float,
        chat_id: int,
        tier: str,
        usage: Any = None,
        history_count: int = 0,
        *,
        intent: str | None = None,
        cache_hit: bool = False,
        rule: str | None = None,
        rag_trace: RAGTrace | None = None,
    ) -> None:
        elapsed_ms = (self._monotonic_fn() - t0) * 1000
        extra: dict[str, Any] = {}
        if rag_trace is not None:
            extra["rag_trace"] = {
                "rag_used": rag_trace.rag_used,
                "rerank_used": rag_trace.rerank_used,
                "context_tokens": rag_trace.context_tokens_estimate,
                "latency_ms": round(rag_trace.total_latency_ms, 1),
            }
        self._engine._logger.info(
            "request_completed",
            chat_id=chat_id,
            tier=tier,
            intent=intent,
            cache_hit=cache_hit,
            rule=rule,
            latency_ms=round(elapsed_ms, 1),
            history_count=history_count,
            tokens_input=usage.prompt_eval_count if usage else None,
            tokens_output=usage.eval_count if usage else None,
            **extra,
        )

    async def persist_turn(
        self,
        chat_id: int,
        user_text: str,
        assistant_text: str,
        skill: SkillDefinition | None = None,
    ) -> None:
        """Persist a user/assistant turn pair consistently in memory."""
        metadata = {"skill": skill.name} if skill else None
        await self._engine._memory.add_message(chat_id, "user", user_text, metadata=metadata)
        await self._engine._memory.add_message(chat_id, "assistant", assistant_text)

    async def persist_failed_turn(
        self,
        chat_id: int,
        user_text: str,
        *,
        error: Exception,
        tier: str | None,
        skill: SkillDefinition | None = None,
    ) -> None:
        """Persist only the user turn with failure metadata after stream failure."""
        if not user_text.strip():
            return
        metadata: dict[str, Any] = {
            "turn_status": "failed",
            "failure_path": "stream",
            "error_type": type(error).__name__,
        }
        if tier is not None:
            metadata["tier"] = tier
        if skill is not None:
            metadata["skill"] = skill.name
        error_text = str(error).strip()
        if error_text:
            metadata["error"] = error_text[:200]
        try:
            await self._engine._memory.add_message(
                chat_id,
                "user",
                user_text,
                metadata=metadata,
            )
        except Exception as persist_exc:
            self._engine._logger.warning(
                "failed_turn_persist_failed",
                chat_id=chat_id,
                error=str(persist_exc),
            )


def _build_ops(
    engine: Engine,
    *,
    meta_factory: Callable[..., Any] | None = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> EngineTrackingOperations:
    existing = getattr(engine, "_tracking_ops", None)
    if isinstance(existing, EngineTrackingOperations):
        return existing
    return EngineTrackingOperations(
        engine,
        meta_factory=meta_factory,
        monotonic_fn=monotonic_fn,
    )


@contextlib.asynccontextmanager
async def track_request(
    engine: Engine,
    chat_id: int,
    *,
    stream: bool,
) -> AsyncGenerator[None, None]:
    async with _build_ops(engine).track_request(chat_id, stream=stream):
        yield


def consume_last_stream_meta(
    engine: Engine,
    chat_id: int,
    *,
    monotonic_fn: Callable[[], float],
) -> dict[str, Any] | None:
    return _build_ops(engine, monotonic_fn=monotonic_fn).consume_last_stream_meta(chat_id)


def set_stream_meta(
    engine: Engine,
    chat_id: int,
    *,
    tier: str,
    intent: str | None,
    cache_id: int | None,
    stop_reason: str | None,
    usage: Any,
    model_role: str | None,
    rag_trace: dict | None,
    meta_factory: Callable[..., Any],
    monotonic_fn: Callable[[], float],
) -> None:
    _build_ops(
        engine,
        meta_factory=meta_factory,
        monotonic_fn=monotonic_fn,
    ).set_stream_meta(
        chat_id,
        tier=tier,
        intent=intent,
        cache_id=cache_id,
        stop_reason=stop_reason,
        usage=usage,
        model_role=model_role,
        rag_trace=rag_trace,
    )


def cleanup_stream_meta(
    engine: Engine,
    *,
    now: float | None,
    monotonic_fn: Callable[[], float],
) -> None:
    _build_ops(engine, monotonic_fn=monotonic_fn).cleanup_stream_meta(now=now)


def log_request(
    engine: Engine,
    t0: float,
    chat_id: int,
    tier: str,
    usage: Any = None,
    history_count: int = 0,
    *,
    intent: str | None = None,
    cache_hit: bool = False,
    rule: str | None = None,
    rag_trace: RAGTrace | None = None,
    monotonic_fn: Callable[[], float],
) -> None:
    _build_ops(engine, monotonic_fn=monotonic_fn).log_request(
        t0,
        chat_id,
        tier,
        usage,
        history_count,
        intent=intent,
        cache_hit=cache_hit,
        rule=rule,
        rag_trace=rag_trace,
    )


async def persist_turn(
    engine: Engine,
    chat_id: int,
    user_text: str,
    assistant_text: str,
    skill: SkillDefinition | None = None,
) -> None:
    await _build_ops(engine).persist_turn(
        chat_id,
        user_text,
        assistant_text,
        skill=skill,
    )


async def persist_failed_turn(
    engine: Engine,
    chat_id: int,
    user_text: str,
    *,
    error: Exception,
    tier: str | None,
    skill: SkillDefinition | None = None,
) -> None:
    await _build_ops(engine).persist_failed_turn(
        chat_id,
        user_text,
        error=error,
        tier=tier,
        skill=skill,
    )
