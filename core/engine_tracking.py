from __future__ import annotations

import contextlib
import uuid
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from core.engine import Engine
    from core.rag.types import RAGTrace
    from core.skill_manager import SkillDefinition


@contextlib.asynccontextmanager
async def track_request(
    engine: Engine,
    chat_id: int,
    *,
    stream: bool,
) -> AsyncGenerator[None, None]:
    """요청 수/로그 컨텍스트를 요청 단위로 일관되게 관리한다."""
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


def consume_last_stream_meta(
    engine: Engine,
    chat_id: int,
    *,
    monotonic_fn: Callable[[], float],
) -> dict[str, Any] | None:
    """스트리밍 처리 후 메타데이터를 1회성으로 반환한다."""
    cleanup_stream_meta(engine, now=None, monotonic_fn=monotonic_fn)
    meta = engine._last_stream_meta.pop(chat_id, None)
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
    now = monotonic_fn()
    engine._last_stream_meta[chat_id] = meta_factory(
        tier=tier,
        intent=intent,
        cache_id=cache_id,
        stop_reason=stop_reason,
        usage=usage,
        model_role=model_role,
        rag_trace=rag_trace,
        created_at=now,
    )
    cleanup_stream_meta(engine, now=now, monotonic_fn=monotonic_fn)


def cleanup_stream_meta(
    engine: Engine,
    *,
    now: float | None,
    monotonic_fn: Callable[[], float],
) -> None:
    """미소비 스트리밍 메타데이터를 TTL/최대 개수 기준으로 정리한다."""
    if not engine._last_stream_meta:
        return
    current = monotonic_fn() if now is None else now

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
        key=lambda chat_id: float(getattr(engine._last_stream_meta[chat_id], "created_at", 0.0)),
    )[:overflow]
    for chat_id in oldest_chat_ids:
        engine._last_stream_meta.pop(chat_id, None)


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
    elapsed_ms = (monotonic_fn() - t0) * 1000
    extra: dict[str, Any] = {}
    if rag_trace is not None:
        extra["rag_trace"] = {
            "rag_used": rag_trace.rag_used,
            "rerank_used": rag_trace.rerank_used,
            "context_tokens": rag_trace.context_tokens_estimate,
            "latency_ms": round(rag_trace.total_latency_ms, 1),
        }
    engine._logger.info(
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
    engine: Engine,
    chat_id: int,
    user_text: str,
    assistant_text: str,
    skill: SkillDefinition | None = None,
) -> None:
    """사용자/어시스턴트 턴을 메모리에 일관되게 저장한다."""
    metadata = {"skill": skill.name} if skill else None
    await engine._memory.add_message(chat_id, "user", user_text, metadata=metadata)
    await engine._memory.add_message(chat_id, "assistant", assistant_text)


async def persist_failed_turn(
    engine: Engine,
    chat_id: int,
    user_text: str,
    *,
    error: Exception,
    tier: str | None,
    skill: SkillDefinition | None = None,
) -> None:
    """스트리밍 실패 시 사용자 턴만 실패 메타데이터와 함께 저장한다."""
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
        await engine._memory.add_message(
            chat_id,
            "user",
            user_text,
            metadata=metadata,
        )
    except Exception as persist_exc:
        engine._logger.warning(
            "failed_turn_persist_failed",
            chat_id=chat_id,
            error=str(persist_exc),
        )
