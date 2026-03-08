"""Engine 스트리밍 실행 경로 오케스트레이터."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from core.enums import RoutingTier
from core.llm_types import ChatStreamState
from core.skill_manager import SkillDefinition
from core.text_utils import detect_output_anomalies, sanitize_model_output

if TYPE_CHECKING:
    from core.engine import Engine


class EngineStreamOrchestrator:
    """Engine.process_message_stream의 제어 흐름을 담당한다."""

    def __init__(self, engine: Engine, *, repeated_chunk_abort_threshold: int) -> None:
        self._engine = engine
        self._repeated_chunk_abort_threshold = repeated_chunk_abort_threshold

    async def process_message_stream(
        self,
        chat_id: int,
        text: str,
        model_override: str | None = None,
        *,
        images: list[bytes] | None = None,
        metadata: dict | None = None,
    ) -> AsyncGenerator[str, None]:
        engine = self._engine
        t0 = time.monotonic()
        async with engine._track_request(chat_id, stream=True):
            engine._cleanup_stream_meta()
            engine._last_stream_meta.pop(chat_id, None)
            turn_persisted = False
            routing_tier: RoutingTier | str | None = None
            active_skill: SkillDefinition | None = None
            try:
                routing = await engine._decide_routing(
                    chat_id, text, model_override, images=images,
                )
                routing_tier = routing.tier
                active_skill = getattr(routing, "skill", None)

                if routing.tier is RoutingTier.SKILL:
                    skill = routing.skill
                    if skill is None:
                        raise RuntimeError("routing_decision_invalid: missing skill")
                    engine._logger.info("skill_triggered_stream", chat_id=chat_id, skill=skill.name)
                    messages = await engine._build_context(chat_id, text, skill=skill)
                    full_response = ""
                    target_model: str | None = None
                    usage = None
                    skill_stream_stop_reason: str | None = None
                    if (
                        not skill.streaming
                        or engine._should_use_chunked_summary(
                            skill=skill,
                            input_text=engine._extract_skill_user_input(messages),
                        )
                    ):
                        full_response, usage, target_model = await engine._run_skill_chat(
                            skill=skill,
                            messages=messages,
                            model_override=model_override,
                            chat_id=chat_id,
                        )
                        if full_response:
                            yield full_response
                    else:
                        target_model, _ = await engine._prepare_target_model(
                            model=model_override,
                            role=skill.model_role,
                            timeout=skill.timeout,
                        )
                        stream_state = ChatStreamState()
                        skill_stream_error: Exception | None = None
                        skill_last_stream_chunk: str | None = None
                        skill_repeated_stream_chunk_count = 0
                        try:
                            async for chunk in engine._llm_client.chat_stream(
                                messages=messages,
                                model=target_model,
                                temperature=skill.temperature,
                                max_tokens=skill.max_tokens,
                                timeout=skill.timeout,
                                stream_state=stream_state,
                            ):
                                if not chunk:
                                    continue
                                if chunk == skill_last_stream_chunk:
                                    skill_repeated_stream_chunk_count += 1
                                    if (
                                        skill_repeated_stream_chunk_count
                                        >= self._repeated_chunk_abort_threshold
                                    ):
                                        engine._logger.warning(
                                            "stream_repeating_chunk_abort",
                                            tier=RoutingTier.SKILL,
                                            chat_id=chat_id,
                                            repeated_chunks=skill_repeated_stream_chunk_count,
                                        )
                                        skill_stream_stop_reason = "repeated_chunks"
                                        break
                                    continue
                                skill_last_stream_chunk = chunk
                                skill_repeated_stream_chunk_count = 0
                                full_response += chunk
                                yield chunk
                        except Exception as exc:
                            skill_stream_error = exc
                            if full_response.strip():
                                engine._logger.warning(
                                    "stream_interrupted_partial_response",
                                    tier=RoutingTier.SKILL,
                                    chat_id=chat_id,
                                    error=str(exc),
                                )
                            else:
                                engine._logger.warning(
                                    "stream_failed_fallback_to_chat",
                                    tier=RoutingTier.SKILL,
                                    chat_id=chat_id,
                                    error=str(exc),
                                )
                                chat_response = await engine._llm_client.chat(
                                    messages=messages,
                                    model=target_model,
                                    temperature=skill.temperature,
                                    max_tokens=skill.max_tokens,
                                    timeout=skill.timeout,
                                )
                                full_response = sanitize_model_output(chat_response.content)
                                usage = chat_response.usage
                                if full_response:
                                    yield full_response
                        if usage is None:
                            usage = stream_state.usage
                        if not full_response.strip() and skill_stream_error is None:
                            engine._logger.warning(
                                "stream_empty_fallback_to_chat",
                                tier=RoutingTier.SKILL,
                                chat_id=chat_id,
                            )
                            chat_response = await engine._llm_client.chat(
                                messages=messages,
                                model=target_model,
                                temperature=skill.temperature,
                                max_tokens=skill.max_tokens,
                                timeout=skill.timeout,
                            )
                            full_response = sanitize_model_output(chat_response.content)
                            usage = chat_response.usage or usage
                            if full_response:
                                yield full_response
                        if skill_stream_error is not None and not full_response.strip():
                            raise skill_stream_error
                    full_response = engine._finalize_stream_response(full_response)
                    await engine._persist_turn(chat_id, text, full_response, skill=skill)
                    turn_persisted = True
                    engine._set_stream_meta(
                        chat_id,
                        tier=RoutingTier.SKILL,
                        stop_reason=skill_stream_stop_reason,
                        usage=usage,
                    )
                    engine._log_request(t0, chat_id, "skill", usage, len(messages))
                    return

                if routing.tier is RoutingTier.INSTANT:
                    instant = routing.instant
                    if instant is None:
                        raise RuntimeError("routing_decision_invalid: missing instant")
                    await engine._persist_turn(chat_id, text, instant.response)
                    turn_persisted = True
                    engine._set_stream_meta(chat_id, tier=RoutingTier.INSTANT)
                    engine._log_request(t0, chat_id, "instant", None, 0, rule=instant.rule_name)
                    yield instant.response
                    return

                if routing.tier is RoutingTier.CACHE:
                    cached = routing.cached
                    if cached is None:
                        raise RuntimeError("routing_decision_invalid: missing cache")
                    await engine._persist_turn(chat_id, text, cached.response)
                    turn_persisted = True
                    engine._set_stream_meta(
                        chat_id,
                        tier=RoutingTier.CACHE,
                        intent=routing.intent,
                        cache_id=cached.cache_id,
                    )
                    engine._log_request(
                        t0,
                        chat_id,
                        "cache",
                        None,
                        0,
                        intent=routing.intent,
                        cache_hit=True,
                    )
                    yield cached.response
                    return

                prepared_full = await engine._prepare_full_request(
                    chat_id=chat_id,
                    text=text,
                    model_override=model_override,
                    images=images,
                    metadata=metadata,
                    intent=routing.intent,
                    strategy=routing.strategy,
                    stream=True,
                )

                full_response = ""
                raw_full_response = ""
                stream_state = ChatStreamState()
                usage = None
                stream_error: Exception | None = None
                stream_stop_reason: str | None = None
                should_stream_chunks = not prepared_full.stream_buffering
                full_last_stream_chunk: str | None = None
                full_repeated_stream_chunk_count = 0
                try:
                    async for chunk in engine._llm_client.chat_stream(
                        messages=prepared_full.messages,
                        model=prepared_full.target_model,
                        timeout=prepared_full.timeout,
                        max_tokens=prepared_full.max_tokens,
                        stream_state=stream_state,
                    ):
                        if not chunk:
                            continue
                        if chunk == full_last_stream_chunk:
                            full_repeated_stream_chunk_count += 1
                            if (
                                full_repeated_stream_chunk_count
                                >= self._repeated_chunk_abort_threshold
                            ):
                                engine._logger.warning(
                                    "stream_repeating_chunk_abort",
                                    tier=RoutingTier.FULL,
                                    chat_id=chat_id,
                                    repeated_chunks=full_repeated_stream_chunk_count,
                                )
                                stream_stop_reason = "repeated_chunks"
                                break
                            continue
                        full_last_stream_chunk = chunk
                        full_repeated_stream_chunk_count = 0
                        full_response += chunk
                        raw_full_response += chunk
                        if should_stream_chunks:
                            yield chunk
                except Exception as exc:
                    stream_error = exc
                    if full_response.strip():
                        engine._logger.warning(
                            "stream_interrupted_partial_response",
                            tier=RoutingTier.FULL,
                            chat_id=chat_id,
                            error=str(exc),
                        )
                    else:
                        engine._logger.warning(
                            "stream_failed_fallback_to_chat",
                            tier=RoutingTier.FULL,
                            chat_id=chat_id,
                            error=str(exc),
                        )
                        chat_response = await engine._llm_client.chat(
                            messages=prepared_full.messages,
                            model=prepared_full.target_model,
                            timeout=prepared_full.timeout,
                            max_tokens=prepared_full.max_tokens,
                        )
                        raw_full_response = chat_response.content
                        full_response = sanitize_model_output(chat_response.content)
                        usage = chat_response.usage
                        if full_response and should_stream_chunks:
                            yield full_response
                if usage is None:
                    usage = stream_state.usage
                if not full_response.strip() and stream_error is None:
                    engine._logger.warning(
                        "stream_empty_fallback_to_chat",
                        tier=RoutingTier.FULL,
                        chat_id=chat_id,
                    )
                    chat_response = await engine._llm_client.chat(
                        messages=prepared_full.messages,
                        model=prepared_full.target_model,
                        timeout=prepared_full.timeout,
                        max_tokens=prepared_full.max_tokens,
                    )
                    raw_full_response = chat_response.content
                    full_response = sanitize_model_output(chat_response.content)
                    usage = chat_response.usage or usage
                    if full_response and should_stream_chunks:
                        yield full_response
                if stream_error is not None and not full_response.strip():
                    raise stream_error
                if not raw_full_response:
                    raw_full_response = full_response
                full_response = engine._finalize_stream_response(full_response)
                anomaly_reasons = detect_output_anomalies(raw_full_response, full_response)
                if anomaly_reasons:
                    engine._logger.warning(
                        "response_anomaly_detected",
                        chat_id=chat_id,
                        model=prepared_full.target_model,
                        reasons=anomaly_reasons,
                    )
                full_response = await engine._maybe_review_full_response(
                    chat_id=chat_id,
                    text=text,
                    response=full_response,
                    raw_response=raw_full_response,
                    intent=routing.intent,
                    prepared_full=prepared_full,
                    images=images,
                    anomaly_reasons=anomaly_reasons,
                )
                if not should_stream_chunks and full_response:
                    yield full_response

                await engine._persist_turn(chat_id, text, full_response)
                turn_persisted = True

                cache_id = await engine._maybe_store_semantic_cache(
                    chat_id=chat_id,
                    text=text,
                    response=full_response,
                    images=images,
                    model_override=model_override,
                    intent=routing.intent,
                )

                engine._set_stream_meta(
                    chat_id,
                    tier=RoutingTier.FULL,
                    intent=routing.intent,
                    cache_id=cache_id,
                    stop_reason=stream_stop_reason,
                    usage=usage,
                    rag_trace=(
                        prepared_full.rag_result.trace.to_dict()
                        if prepared_full.rag_result
                        else None
                    ),
                )
                engine._log_request(
                    t0,
                    chat_id,
                    "full",
                    usage,
                    len(prepared_full.messages),
                    intent=routing.intent,
                    rag_trace=prepared_full.rag_result.trace if prepared_full.rag_result else None,
                )

                engine._trigger_background_summary(chat_id)
            except Exception as exc:
                if not turn_persisted:
                    await engine._persist_failed_turn(
                        chat_id=chat_id,
                        user_text=text,
                        error=exc,
                        tier=routing_tier,
                        skill=active_skill,
                    )
                engine._logger.error("request_failed", error=str(exc))
                raise
