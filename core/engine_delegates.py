"""Thin engine delegations extracted from ``core.engine``."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

from core import (
    engine_background,
    engine_context,
    engine_models,
    engine_rag,
    engine_reviewer,
    engine_routing,
    engine_summary,
    engine_tracking,
)
from core.engine_types import (
    _FullScanSegment,
    _PreparedFullRequest,
    _PreparedRequest,
    _RoutingDecision,
    _StreamMeta,
)
from core.skill_manager import SkillDefinition

if TYPE_CHECKING:
    from core.intent_router import ContextStrategy, RouteResult
    from core.rag.types import RAGTrace
    from core.semantic_cache import CacheContext


@contextlib.asynccontextmanager
async def _track_request(
    self: Any,
    chat_id: int,
    *,
    stream: bool,
) -> AsyncGenerator[None, None]:
    async with engine_tracking.track_request(self, chat_id, stream=stream):
        yield


async def _classify_route(self: Any, text: str) -> RouteResult | None:
    return await engine_routing.classify_route(self, text)


async def _decide_routing(
    self: Any,
    chat_id: int,
    text: str,
    model_override: str | None = None,
    *,
    images: list[bytes] | None = None,
) -> _RoutingDecision:
    return await engine_routing.decide_routing(
        self,
        chat_id,
        text,
        model_override=model_override,
        images=images,
        decision_factory=_RoutingDecision,
    )


def _set_stream_meta(
    self: Any,
    chat_id: int,
    *,
    tier,
    intent: str | None = None,
    cache_id: int | None = None,
    usage: Any = None,
    model_role: str | None = None,
    rag_trace: dict | None = None,
) -> None:
    engine_tracking.set_stream_meta(
        self,
        chat_id,
        tier=tier,
        intent=intent,
        cache_id=cache_id,
        usage=usage,
        model_role=model_role,
        rag_trace=rag_trace,
        meta_factory=_StreamMeta,
        monotonic_fn=time.monotonic,
    )


def _cleanup_stream_meta(self: Any, now: float | None = None) -> None:
    engine_tracking.cleanup_stream_meta(
        self,
        now=now,
        monotonic_fn=time.monotonic,
    )


def _build_cache_context(
    self: Any,
    model_override: str | None,
    intent: str | None,
    chat_id: int,
) -> CacheContext:
    return engine_routing.build_cache_context(self, model_override, intent, chat_id)


def _is_cache_response_acceptable(query: str, response: str) -> bool:
    return engine_routing.is_cache_response_acceptable(query, response)


def _log_request(
    self: Any,
    t0: float,
    chat_id: int,
    tier,
    usage=None,
    history_count: int = 0,
    *,
    intent: str | None = None,
    cache_hit: bool = False,
    rule: str | None = None,
    rag_trace: RAGTrace | None = None,
) -> None:
    engine_tracking.log_request(
        self,
        t0,
        chat_id,
        tier,
        usage,
        history_count,
        intent=intent,
        cache_hit=cache_hit,
        rule=rule,
        rag_trace=rag_trace,
        monotonic_fn=time.monotonic,
    )


def _inject_rag_context(
    messages: list[dict[str, str]],
    rag_result: Any,
) -> list[dict[str, str]]:
    return engine_rag.inject_rag_context(messages, rag_result)


def _inject_extra_context(
    messages: list[dict[str, str]],
    context: str,
) -> list[dict[str, str]]:
    return engine_rag.inject_extra_context(messages, context)


def _trigger_background_summary(self: Any, chat_id: int) -> None:
    engine_background.trigger_background_summary(self, chat_id)


def _handle_summary_task_done(self: Any, task: asyncio.Task[Any]) -> None:
    engine_background.handle_summary_task_done(self, task)


def _handle_background_task_error(self: Any, task: asyncio.Task[Any]) -> None:
    engine_background.handle_background_task_error(self, task)


async def _emit_full_scan_progress(
    callback: Callable[[dict[str, Any]], Awaitable[None] | None] | None,
    payload: dict[str, Any],
) -> None:
    await engine_rag.emit_full_scan_progress(callback, payload)


def _build_full_scan_segments(
    chunks: list[Any],
    *,
    max_chars: int,
) -> list[_FullScanSegment]:
    return engine_rag.build_full_scan_segments(
        chunks,
        max_chars=max_chars,
        segment_factory=_FullScanSegment,
    )


def _pack_blocks_for_reduction(
    blocks: list[str],
    *,
    max_chars: int,
) -> list[str]:
    return engine_rag.pack_blocks_for_reduction(blocks, max_chars=max_chars)


async def _prepare_request(
    self: Any,
    chat_id: int,
    text: str,
    *,
    stream: bool,
    strategy: ContextStrategy | None = None,
) -> _PreparedRequest:
    messages = await self._build_context(chat_id, text, strategy=strategy)
    timeout = self._config.bot.response_timeout
    max_tokens = strategy.max_tokens if strategy else None
    return _PreparedRequest(
        skill=None,
        messages=messages,
        timeout=timeout,
        max_tokens=max_tokens,
    )


def _resolve_inference_timeout(
    *,
    base_timeout: int,
    intent: str | None,
    model_role: str | None,
    has_images: bool = False,
) -> int:
    return engine_routing.resolve_inference_timeout(
        base_timeout=base_timeout,
        intent=intent,
        model_role=model_role,
        has_images=has_images,
    )


async def _prepare_full_request(
    self: Any,
    *,
    chat_id: int,
    text: str,
    model_override: str | None,
    images: list[bytes] | None,
    metadata: dict | None,
    intent: str | None,
    strategy: ContextStrategy | None,
    stream: bool,
) -> _PreparedFullRequest:
    prepared = await engine_rag.prepare_full_request(
        self,
        chat_id=chat_id,
        text=text,
        model_override=model_override,
        images=images,
        metadata=metadata,
        intent=intent,
        strategy=strategy,
        stream=stream,
    )
    return _PreparedFullRequest(**prepared)


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    return engine_rag.extract_json_payload(text)


async def _maybe_store_semantic_cache(
    self: Any,
    *,
    chat_id: int,
    text: str,
    response: str,
    images: list[bytes] | None,
    model_override: str | None,
    intent: str | None,
) -> int | None:
    if (
        self._semantic_cache is None
        or images
        or not self._semantic_cache.is_cacheable(text)
    ):
        return None
    if not self._is_cache_response_acceptable(text, response):
        self._logger.info("semantic_cache_put_skipped_low_quality", chat_id=chat_id)
        return None
    cache_ctx = self._build_cache_context(model_override, intent, chat_id)
    return await self._semantic_cache.put(text, response, context=cache_ctx)


async def _maybe_review_full_response(
    self: Any,
    *,
    chat_id: int,
    text: str,
    response: str,
    raw_response: str,
    intent: str | None,
    prepared_full: _PreparedFullRequest,
    images: list[bytes] | None,
    anomaly_reasons: list[str] | None = None,
) -> str:
    return await engine_reviewer.maybe_review_response(
        self,
        chat_id=chat_id,
        text=text,
        response=response,
        raw_response=raw_response,
        intent=intent,
        target_model=prepared_full.target_model,
        timeout=prepared_full.timeout,
        planner_applied=prepared_full.planner_applied,
        rag_used=bool(prepared_full.rag_result and prepared_full.rag_result.contexts),
        images=images,
        anomaly_reasons=anomaly_reasons,
    )


def _is_summarize_skill(skill: SkillDefinition) -> bool:
    return engine_summary.is_summarize_skill(skill)


def _extract_skill_user_input(messages: list[dict[str, str]]) -> str:
    return engine_summary.extract_skill_user_input(messages)


def _should_use_chunked_summary(
    self: Any,
    *,
    skill: SkillDefinition,
    input_text: str,
) -> bool:
    return engine_summary.should_use_chunked_summary(
        skill=skill,
        input_text=input_text,
    )


def _split_text_for_summary(text: str) -> list[str]:
    return engine_summary.split_text_for_summary(text)


async def _run_skill_chat(
    self: Any,
    *,
    skill: SkillDefinition,
    messages: list[dict[str, str]],
    model_override: str | None,
    model_role_override: str | None = None,
    max_tokens_override: int | None = None,
    temperature_override: float | None = None,
    timeout_override: int | None = None,
    chat_id: int | None = None,
) -> tuple[str, Any, str | None]:
    return await engine_summary.run_skill_chat(
        self,
        skill=skill,
        messages=messages,
        model_override=model_override,
        model_role_override=model_role_override,
        max_tokens_override=max_tokens_override,
        temperature_override=temperature_override,
        timeout_override=timeout_override,
        chat_id=chat_id,
    )


async def _run_chunked_summary_pipeline(
    self: Any,
    *,
    skill: SkillDefinition,
    messages: list[dict[str, str]],
    model_override: str | None,
    timeout_override: int | None = None,
    chat_id: int | None = None,
) -> tuple[str, Any, str | None]:
    return await engine_summary.run_chunked_summary_pipeline(
        self,
        skill=skill,
        messages=messages,
        model_override=model_override,
        timeout_override=timeout_override,
        chat_id=chat_id,
    )


async def _prepare_target_model(
    self: Any,
    *,
    model: str | None,
    role: str | None,
    timeout: int,
) -> tuple[str | None, str | None]:
    return await engine_models.prepare_target_model(
        self,
        model=model,
        role=role,
        timeout=timeout,
    )


def _resolve_model_for_role(self: Any, role: str | None) -> str | None:
    return engine_models.resolve_model_for_role(self, role)


async def _persist_turn(
    self: Any,
    chat_id: int,
    user_text: str,
    assistant_text: str,
    skill: SkillDefinition | None = None,
) -> None:
    await engine_tracking.persist_turn(
        self,
        chat_id,
        user_text,
        assistant_text,
        skill=skill,
    )


async def _persist_failed_turn(
    self: Any,
    chat_id: int,
    user_text: str,
    *,
    error: Exception,
    tier: str | None,
    skill: SkillDefinition | None = None,
) -> None:
    await engine_tracking.persist_failed_turn(
        self,
        chat_id,
        user_text,
        error=error,
        tier=tier,
        skill=skill,
    )


async def _build_context(
    self: Any,
    chat_id: int,
    text: str,
    skill: SkillDefinition | None = None,
    strategy: ContextStrategy | None = None,
) -> list[dict[str, str]]:
    return await engine_context.build_context(
        self,
        chat_id,
        text,
        skill=skill,
        strategy=strategy,
    )


async def _build_base_context(
    self: Any,
    chat_id: int,
    *,
    skill: SkillDefinition | None,
    strategy: ContextStrategy | None,
) -> tuple[str, list[dict[str, str]]]:
    return await engine_context.build_base_context(
        self,
        chat_id,
        skill=skill,
        strategy=strategy,
    )


def _sanitize_history_for_prompt(
    self: Any,
    history: list[dict[str, str]],
) -> list[dict[str, str]]:
    return engine_context.sanitize_history_for_prompt(self, history)


async def _inject_preferences(self: Any, system: str, chat_id: int) -> str:
    return await engine_context.inject_preferences(self, system, chat_id)


async def _inject_guidelines(self: Any, system: str, chat_id: int) -> str:
    return await engine_context.inject_guidelines(self, system, chat_id)


async def _inject_dicl_examples(
    self: Any,
    system: str,
    *,
    chat_id: int,
    text: str,
    include_dicl: bool,
    skill: SkillDefinition | None,
) -> str:
    return await engine_context.inject_dicl_examples(
        self,
        system,
        chat_id=chat_id,
        text=text,
        include_dicl=include_dicl,
        skill=skill,
    )


def _inject_intent_suffix(system: str, strategy: ContextStrategy | None) -> str:
    return engine_context.inject_intent_suffix(system, strategy)


def _normalize_language(value: str) -> str:
    return engine_context.normalize_language(value)


def _inject_language_policy(self: Any, system: str) -> str:
    return engine_context.inject_language_policy(self, system)


def _assemble_messages(
    system: str,
    history: list[dict[str, str]],
    text: str,
    skill: SkillDefinition | None,
) -> list[dict[str, str]]:
    return engine_context.assemble_messages(system, history, text, skill)
