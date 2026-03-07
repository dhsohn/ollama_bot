"""메인 엔진 — 대화 오케스트레이션, 컨텍스트 관리, 계층형 라우팅.

모든 사용자 메시지 처리의 중앙 허브.
텔레그램 핸들러로부터 입력을 받아 적절한 처리 후 응답을 반환한다.

계층형 처리 경로:
  [선행] 스킬 트리거 매칭
  [Tier 1] 규칙 기반 즉시 응답 (InstantResponder)
  [Tier 2] 인텐트 라우팅 (IntentRouter)
  [Tier 3] 시맨틱 캐시 (SemanticCache)
  [Tier 4] Full LLM (최적화된 컨텍스트)
"""

from __future__ import annotations

import asyncio
import contextlib
import re as _re
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core import (
    engine_background,
    engine_context,
    engine_models,
    engine_rag,
    engine_routing,
    engine_status,
    engine_summary,
    engine_tracking,
)
from core.config import AppSettings
from core.constants import (
    CONTEXT_HISTORY_MESSAGE_MAX_CHARS,
    FULL_SCAN_REDUCE_GROUP_MAX_CHARS,
    FULL_SCAN_REDUCE_MAX_PASSES,
    FULL_SCAN_SEGMENT_MAX_CHARS,
    REASONING_INTENTS,
    REASONING_MODEL_ROLES,
    REASONING_TIMEOUT_SECONDS,
    SUMMARY_CHUNK_MAX_CHARS,
    SUMMARY_CHUNK_OVERLAP_CHARS,
    SUMMARY_CHUNK_TRIGGER_CHARS,
    SUMMARY_MAP_MAX_TOKENS,
    SUMMARY_MAP_TIMEOUT_SECONDS,
    SUMMARY_REDUCE_MAX_TOKENS,
    SUMMARY_REDUCE_TIMEOUT_SECONDS,
)
from core.enums import RoutingTier
from core.llm_protocol import LLMClientProtocol
from core.logging_setup import get_logger
from core.memory import MemoryManager
from core.skill_manager import SkillDefinition, SkillManager
from core.stream_orchestrator import EngineStreamOrchestrator
from core.text_utils import detect_output_anomalies, sanitize_model_output

if TYPE_CHECKING:
    from core.context_compressor import ContextCompressor
    from core.feedback_manager import FeedbackManager
    from core.instant_responder import InstantResponder
    from core.intent_router import ContextStrategy, IntentRouter, RouteResult
    from core.rag.pipeline import RAGPipeline
    from core.rag.types import RAGTrace
    from core.semantic_cache import CacheContext, SemanticCache


class ContextProvider:
    """메시지 처리 시 추가 컨텍스트를 주입하는 범용 인터페이스.

    구현체는 사용자 텍스트를 받아 관련 컨텍스트를 반환한다.
    컨텍스트가 없으면 None을 반환한다.
    """

    async def get_context(self, text: str) -> str | None:
        """사용자 텍스트에 대한 추가 컨텍스트를 반환한다."""
        raise NotImplementedError


@dataclass
class _PreparedRequest:
    """LLM 호출에 필요한 사전 계산 결과."""

    skill: SkillDefinition | None
    messages: list[dict[str, str]]
    timeout: int
    max_tokens: int | None = None


@dataclass
class _PreparedFullRequest:
    """Tier 4(full LLM) 공통 준비 결과."""

    messages: list[dict[str, str]]
    timeout: int
    max_tokens: int | None
    target_model: str | None
    rag_result: Any = None  # RAGResult | None


@dataclass
class _StreamMeta:
    """최근 스트리밍 요청 메타데이터."""

    tier: RoutingTier = RoutingTier.FULL
    intent: str | None = None
    cache_id: int | None = None
    usage: Any = None  # ChatUsage | None
    model_role: str | None = None
    rag_trace: dict | None = None
    created_at: float = 0.0


@dataclass
class _RoutingDecision:
    """LLM 호출 전 라우팅 판단 결과."""

    tier: RoutingTier = RoutingTier.FULL
    skill: SkillDefinition | None = None
    instant: Any = None
    route: RouteResult | None = None
    cached: Any = None
    rag_result: Any = None  # RAGResult | None

    @property
    def intent(self) -> str | None:
        return self.route.intent if self.route else None

    @property
    def strategy(self) -> ContextStrategy | None:
        return self.route.context_strategy if self.route else None


@dataclass
class _FullScanSegment:
    """RAG full-scan 분석을 위한 문서 세그먼트."""

    source_path: str
    start_chunk_id: int
    end_chunk_id: int
    text: str


_INJECTION_RE = _re.compile(
    r"\[/?(?:system|user|assistant|INST)\]"
    r"|<\|(?:im_start|im_end|system|user|assistant)\|>"
    r"|(?:^|\n)\s*(?:system|user|assistant|human)\s*:",
    _re.IGNORECASE,
)
_CODE_BLOCK_RE = _re.compile(r"```.*?```", _re.DOTALL)
_STREAM_META_TTL_SECONDS = 600.0
_STREAM_META_MAX_ENTRIES = 2048
_STREAM_REPEATED_CHUNK_ABORT_THRESHOLD = 30


def _strip_prompt_injection(text: str) -> str:
    """프리뷰 텍스트에서 코드블록을 보존한 채 인젝션 패턴만 제거한다."""
    if not text:
        return ""

    parts: list[str] = []
    last = 0
    for match in _CODE_BLOCK_RE.finditer(text):
        outside = text[last:match.start()]
        outside = _INJECTION_RE.sub("", outside)
        outside = _re.sub(r"\n{3,}", "\n\n", outside)
        parts.append(outside)
        parts.append(match.group(0))
        last = match.end()

    tail = text[last:]
    tail = _INJECTION_RE.sub("", tail)
    tail = _re.sub(r"\n{3,}", "\n\n", tail)
    parts.append(tail)

    sanitized = "".join(parts)
    sanitized = _re.sub(r"\n{3,}", "\n\n", sanitized)
    return sanitized.strip()


class Engine:
    """대화 처리 엔진. 계층형 라우팅과 컨텍스트 관리를 오케스트레이션한다."""

    def __init__(
        self,
        config: AppSettings,
        llm_client: LLMClientProtocol,
        memory: MemoryManager,
        skills: SkillManager,
        feedback_manager: FeedbackManager | None = None,
        instant_responder: InstantResponder | None = None,
        semantic_cache: SemanticCache | None = None,
        intent_router: IntentRouter | None = None,
        context_compressor: ContextCompressor | None = None,
        rag_pipeline: RAGPipeline | None = None,
        context_providers: list[ContextProvider] | None = None,
    ) -> None:
        self._config = config
        self._llm_client = llm_client
        self._memory = memory
        self._skills = skills
        self._feedback_manager = feedback_manager
        self._instant_responder = instant_responder
        self._semantic_cache = semantic_cache
        self._intent_router = intent_router
        self._context_compressor = context_compressor
        self._rag_pipeline = rag_pipeline
        self._context_providers: list[ContextProvider] = context_providers or []
        self._system_prompt = getattr(llm_client, "system_prompt", config.lemonade.system_prompt)
        self._max_conversation_length = config.bot.max_conversation_length
        self._start_time = time.monotonic()
        self._logger = get_logger("engine")
        self._last_stream_meta: dict[int, _StreamMeta] = {}
        self._stream_meta_ttl_seconds = _STREAM_META_TTL_SECONDS
        self._stream_meta_max_entries = _STREAM_META_MAX_ENTRIES
        self._active_request_count = 0
        summary_concurrency = max(1, int(config.context_compressor.summarize_concurrency))
        # 요약 태스크가 무제한 생성되지 않도록 생성량 상한을 둔다.
        self._summary_task_limit = max(2, summary_concurrency * 3)
        self._summary_tasks: set[asyncio.Task[Any]] = set()
        self._degraded_since: dict[str, float] = {}
        self._rag_reindex_lock = asyncio.Lock()
        self._stream_orchestrator = EngineStreamOrchestrator(
            self,
            repeated_chunk_abort_threshold=_STREAM_REPEATED_CHUNK_ABORT_THRESHOLD,
        )

    @staticmethod
    def _finalize_stream_response(full_response: str) -> str:
        """스트리밍/폴백 응답의 공백·이상치를 정리하고 유효성을 보장한다."""
        if not full_response.strip():
            raise RuntimeError("empty_response_from_llm")
        normalized = sanitize_model_output(full_response).strip()
        if not normalized:
            raise RuntimeError("empty_response_from_llm")
        return normalized

    # ── 메시지 처리 (계층형 라우팅) ──

    async def process_message(
        self,
        chat_id: int,
        text: str,
        model_override: str | None = None,
        *,
        images: list[bytes] | None = None,
        metadata: dict | None = None,
    ) -> str:
        """사용자 메시지를 처리하고 응답을 반환한다.

        계층형 처리: 스킬 → Tier 1 (즉시) → Tier 2 (인텐트) → Tier 3 (캐시) → Tier 4 (LLM)
        """
        t0 = time.monotonic()
        async with self._track_request(chat_id, stream=False):
            try:
                routing = await self._decide_routing(
                    chat_id, text, model_override, images=images,
                )

                if routing.tier is RoutingTier.SKILL:
                    skill = routing.skill
                    if skill is None:
                        raise RuntimeError("routing_decision_invalid: missing skill")
                    self._logger.info("skill_triggered", chat_id=chat_id, skill=skill.name)
                    messages = await self._build_context(chat_id, text, skill=skill)
                    content, usage, target_model = await self._run_skill_chat(
                        skill=skill,
                        messages=messages,
                        model_override=model_override,
                        chat_id=chat_id,
                    )
                    content = sanitize_model_output(content).strip()
                    if not content:
                        raise RuntimeError("empty_response_from_llm")
                    await self._persist_turn(chat_id, text, content, skill=skill)
                    self._log_request(t0, chat_id, RoutingTier.SKILL, usage, len(messages))
                    return content

                if routing.tier is RoutingTier.INSTANT:
                    instant = routing.instant
                    if instant is None:
                        raise RuntimeError("routing_decision_invalid: missing instant")
                    await self._persist_turn(chat_id, text, instant.response)
                    self._log_request(t0, chat_id, RoutingTier.INSTANT, None, 0, rule=instant.rule_name)
                    return instant.response

                if routing.tier is RoutingTier.CACHE:
                    cached = routing.cached
                    if cached is None:
                        raise RuntimeError("routing_decision_invalid: missing cache")
                    await self._persist_turn(chat_id, text, cached.response)
                    self._log_request(
                        t0, chat_id, RoutingTier.CACHE, None, 0,
                        intent=routing.intent, cache_hit=True,
                    )
                    return cached.response

                prepared_full = await self._prepare_full_request(
                    chat_id=chat_id,
                    text=text,
                    model_override=model_override,
                    images=images,
                    metadata=metadata,
                    intent=routing.intent,
                    strategy=routing.strategy,
                    stream=False,
                )

                chat_response = await self._llm_client.chat(
                    messages=prepared_full.messages,
                    model=prepared_full.target_model,
                    timeout=prepared_full.timeout,
                    max_tokens=prepared_full.max_tokens,
                )
                content = sanitize_model_output(chat_response.content).strip()
                usage = chat_response.usage
                anomaly_reasons = detect_output_anomalies(chat_response.content, content)
                if anomaly_reasons:
                    self._logger.warning(
                        "response_anomaly_detected",
                        chat_id=chat_id,
                        model=prepared_full.target_model,
                        reasons=anomaly_reasons,
                    )
                if not content:
                    raise RuntimeError("empty_response_from_llm")
                await self._persist_turn(chat_id, text, content)

                await self._maybe_store_semantic_cache(
                    chat_id=chat_id,
                    text=text,
                    response=content,
                    images=images,
                    model_override=model_override,
                    intent=routing.intent,
                )

                self._log_request(
                    t0, chat_id, RoutingTier.FULL, usage, len(prepared_full.messages),
                    intent=routing.intent,
                    rag_trace=prepared_full.rag_result.trace if prepared_full.rag_result else None,
                )

                # 백그라운드 요약 갱신
                self._trigger_background_summary(chat_id)

                return content
            except Exception as exc:
                self._logger.error("request_failed", error=str(exc))
                raise

    async def process_message_stream(
        self,
        chat_id: int,
        text: str,
        model_override: str | None = None,
        *,
        images: list[bytes] | None = None,
        metadata: dict | None = None,
    ) -> AsyncGenerator[str, None]:
        """스트리밍 방식으로 메시지를 처리한다. 청크를 순차 반환한다."""
        async for chunk in self._stream_orchestrator.process_message_stream(
            chat_id=chat_id,
            text=text,
            model_override=model_override,
            images=images,
            metadata=metadata,
        ):
            yield chunk

    async def rollback_last_turn(self, chat_id: int) -> int:
        """최근 스트리밍 턴을 롤백한다 (recovery 전 호출용)."""
        return await self._memory.delete_last_turn(chat_id)

    # ── 인텐트 분류 (외부 접근용) ──

    async def classify_intent(self, text: str) -> str | None:
        """사용자 입력의 의도를 분류한다 (텔레그램 핸들러용)."""
        route = await self._classify_route(text)
        return route.intent if route else None

    async def route_request(
        self,
        text: str,
        images: list[bytes] | None = None,
        metadata: dict | None = None,
    ) -> dict[str, Any]:
        """PLAN 인터페이스: 단일 모델 고정 라우팅 결과를 반환한다."""
        _ = (text, images, metadata)
        return {
            "selected_model": self._llm_client.default_model,
            "selected_role": "default",
            "trigger": "single_model",
            "confidence": 1.0,
            "anchor_scores": {},
            "fallback_used": False,
            "original_role": None,
            "classifier_used": False,
            "latency_ms": 0.0,
            "degraded": False,
            "degradation_reasons": [],
        }

    async def retrieve(
        self,
        text: str,
        metadata: dict | None = None,
    ) -> dict[str, Any]:
        """PLAN 인터페이스: RAG 검색 결과를 반환한다."""
        if self._rag_pipeline is None:
            return {
                "candidates": [],
                "contexts": [],
                "rag_trace_partial": {
                    "rag_used": False,
                    "error": "rag_pipeline_disabled",
                },
            }

        rag_result = await self._rag_pipeline.execute(text, metadata)
        candidates = [
            {
                "chunk_text": item.chunk.text,
                "retrieval_score": item.retrieval_score,
                "rerank_score": item.rerank_score,
                "metadata": {
                    "doc_id": item.chunk.metadata.doc_id,
                    "source_path": item.chunk.metadata.source_path,
                    "chunk_id": item.chunk.metadata.chunk_id,
                    "section_title": item.chunk.metadata.section_title,
                    "tokens_estimate": item.chunk.metadata.tokens_estimate,
                },
            }
            for item in rag_result.candidates
        ]
        return {
            "candidates": candidates,
            "contexts": rag_result.contexts,
            "rag_trace_partial": rag_result.trace.to_dict(),
        }

    async def generate(
        self,
        text: str,
        images: list[bytes] | None = None,
        metadata: dict | None = None,
    ) -> dict[str, Any]:
        """PLAN 인터페이스: answer + routing_decision + rag_trace를 반환한다."""
        routing_decision = await self.route_request(text, images=images, metadata=metadata)
        target_model = routing_decision["selected_model"]

        user_text = text.strip()
        if not user_text and images:
            user_text = "이미지를 분석해줘."

        system_prompt = self._inject_language_policy(self._system_prompt)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

        rag_trace: dict[str, Any] = {"rag_used": False}
        if self._rag_pipeline is not None and self._rag_pipeline.should_trigger_rag(text, metadata):
            rag_result = await self._rag_pipeline.execute(text, metadata)
            if rag_result.contexts:
                messages = self._inject_rag_context(messages, rag_result)
            rag_trace = rag_result.trace.to_dict()

        chat_response = await self._llm_client.chat(
            messages=messages,
            model=target_model,
            timeout=self._config.bot.response_timeout,
        )
        answer = sanitize_model_output(chat_response.content)

        return {
            "answer": answer,
            "routing_decision": routing_decision,
            "rag_trace": rag_trace,
        }

    async def analyze_all_corpus(
        self,
        query: str,
        *,
        progress_callback: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
    ) -> dict[str, Any]:
        return await engine_rag.analyze_all_corpus(
            self,
            query,
            progress_callback=progress_callback,
        )

    async def reindex_rag_corpus(self, kb_dirs: list[str] | None = None) -> dict[str, Any]:
        return await engine_rag.reindex_rag_corpus(self, kb_dirs)

    def consume_last_stream_meta(self, chat_id: int) -> dict[str, Any] | None:
        return engine_tracking.consume_last_stream_meta(
            self,
            chat_id,
            monotonic_fn=time.monotonic,
        )

    # ── 내부 메서드 ──

    @contextlib.asynccontextmanager
    async def _track_request(
        self,
        chat_id: int,
        *,
        stream: bool,
    ) -> AsyncGenerator[None, None]:
        async with engine_tracking.track_request(self, chat_id, stream=stream):
            yield

    async def _classify_route(self, text: str) -> RouteResult | None:
        return await engine_routing.classify_route(self, text)

    async def _decide_routing(
        self,
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
        self,
        chat_id: int,
        *,
        tier: RoutingTier,
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

    def _cleanup_stream_meta(self, now: float | None = None) -> None:
        engine_tracking.cleanup_stream_meta(
            self,
            now=now,
            monotonic_fn=time.monotonic,
        )

    def _build_cache_context(
        self, model_override: str | None, intent: str | None, chat_id: int,
    ) -> CacheContext:
        return engine_routing.build_cache_context(self, model_override, intent, chat_id)

    @staticmethod
    def _is_cache_response_acceptable(query: str, response: str) -> bool:
        return engine_routing.is_cache_response_acceptable(query, response)

    def _log_request(
        self,
        t0: float,
        chat_id: int,
        tier: RoutingTier | str,
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

    @staticmethod
    def _inject_rag_context(
        messages: list[dict[str, str]],
        rag_result: Any,
    ) -> list[dict[str, str]]:
        return engine_rag.inject_rag_context(messages, rag_result)

    @staticmethod
    def _inject_extra_context(
        messages: list[dict[str, str]],
        context: str,
    ) -> list[dict[str, str]]:
        return engine_rag.inject_extra_context(messages, context)

    def _trigger_background_summary(self, chat_id: int) -> None:
        engine_background.trigger_background_summary(self, chat_id)

    def _handle_summary_task_done(self, task: asyncio.Task[Any]) -> None:
        engine_background.handle_summary_task_done(self, task)

    def _handle_background_task_error(self, task: asyncio.Task[Any]) -> None:
        engine_background.handle_background_task_error(self, task)

    @staticmethod
    async def _emit_full_scan_progress(
        callback: Callable[[dict[str, Any]], Awaitable[None] | None] | None,
        payload: dict[str, Any],
    ) -> None:
        await engine_rag.emit_full_scan_progress(callback, payload)

    @staticmethod
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

    @staticmethod
    def _pack_blocks_for_reduction(
        blocks: list[str],
        *,
        max_chars: int,
    ) -> list[str]:
        return engine_rag.pack_blocks_for_reduction(blocks, max_chars=max_chars)

    async def _prepare_request(
        self,
        chat_id: int,
        text: str,
        *,
        stream: bool,
        strategy: ContextStrategy | None = None,
    ) -> _PreparedRequest:
        """컨텍스트 빌드를 처리한다 (스킬 제외 — 일반 메시지용)."""
        messages = await self._build_context(chat_id, text, strategy=strategy)
        timeout = self._config.bot.response_timeout
        max_tokens = strategy.max_tokens if strategy else None
        return _PreparedRequest(
            skill=None, messages=messages, timeout=timeout, max_tokens=max_tokens,
        )

    @staticmethod
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
        self,
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

    @staticmethod
    def _extract_json_payload(text: str) -> dict[str, Any] | None:
        return engine_rag.extract_json_payload(text)

    async def _maybe_store_semantic_cache(
        self,
        *,
        chat_id: int,
        text: str,
        response: str,
        images: list[bytes] | None,
        model_override: str | None,
        intent: str | None,
    ) -> int | None:
        """응답을 시맨틱 캐시에 저장하고 cache_id를 반환한다."""
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

    @staticmethod
    def _is_summarize_skill(skill: SkillDefinition) -> bool:
        return engine_summary.is_summarize_skill(skill)

    @staticmethod
    def _extract_skill_user_input(messages: list[dict[str, str]]) -> str:
        return engine_summary.extract_skill_user_input(messages)

    def _should_use_chunked_summary(
        self,
        *,
        skill: SkillDefinition,
        input_text: str,
    ) -> bool:
        return engine_summary.should_use_chunked_summary(
            skill=skill,
            input_text=input_text,
        )

    @staticmethod
    def _split_text_for_summary(text: str) -> list[str]:
        return engine_summary.split_text_for_summary(text)

    async def _run_skill_chat(
        self,
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
        self,
        *,
        skill: SkillDefinition,
        messages: list[dict[str, str]],
        model_override: str | None,
        timeout_override: int | None = None,
        chat_id: int | None,
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
        self,
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

    def _resolve_model_for_role(self, role: str | None) -> str | None:
        return engine_models.resolve_model_for_role(self, role)

    async def _persist_turn(
        self,
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
        self,
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
        self,
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
        self,
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
        self,
        history: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        return engine_context.sanitize_history_for_prompt(self, history)

    async def _inject_preferences(self, system: str, chat_id: int) -> str:
        return await engine_context.inject_preferences(self, system, chat_id)

    async def _inject_guidelines(self, system: str, chat_id: int) -> str:
        return await engine_context.inject_guidelines(self, system, chat_id)

    async def _inject_dicl_examples(
        self,
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

    @staticmethod
    def _inject_intent_suffix(system: str, strategy: ContextStrategy | None) -> str:
        return engine_context.inject_intent_suffix(system, strategy)

    @staticmethod
    def _normalize_language(value: str) -> str:
        return engine_context.normalize_language(value)

    def _inject_language_policy(self, system: str) -> str:
        return engine_context.inject_language_policy(self, system)

    @staticmethod
    def _assemble_messages(
        system: str,
        history: list[dict[str, str]],
        text: str,
        skill: SkillDefinition | None,
    ) -> list[dict[str, str]]:
        return engine_context.assemble_messages(system, history, text, skill)

    # ── 스킬/모델/메모리 관리 ──

    async def execute_skill(
        self,
        skill_name: str,
        parameters: dict,
        chat_id: int | None = None,
        model_override: str | None = None,
        model_role_override: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: int | None = None,
    ) -> str:
        """프로그래밍 방식으로 스킬을 실행한다 (auto_scheduler용)."""
        skill = self._skills.get_skill(skill_name)
        if not skill:
            return f"스킬 '{skill_name}'을(를) 찾을 수 없습니다."

        input_text = parameters.get("input_text", parameters.get("query", ""))
        skill_system = self._inject_language_policy(skill.system_prompt)
        messages = [
            {"role": "system", "content": skill_system},
            {"role": "user", "content": input_text},
        ]

        content, _, _ = await self._run_skill_chat(
            skill=skill,
            messages=messages,
            model_override=model_override,
            model_role_override=model_role_override,
            max_tokens_override=max_tokens,
            temperature_override=temperature,
            timeout_override=timeout,
            chat_id=chat_id,
        )

        if chat_id:
            await self._memory.add_message(
                chat_id, "assistant", content,
                metadata={"skill": skill_name, "auto": True},
            )

        return content

    async def process_prompt(
        self,
        prompt: str,
        chat_id: int | None = None,
        response_format: str | dict | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        model_override: str | None = None,
        model_role: str | None = None,
        timeout: int | None = None,
        system_prompt_override: str | None = None,
    ) -> str:
        """단순 프롬프트를 LLM에 전달한다 (auto_scheduler의 prompt 타입용)."""
        base = system_prompt_override if system_prompt_override is not None else self._system_prompt
        system_prompt = self._inject_language_policy(base)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        base_timeout = int(timeout or self._config.bot.response_timeout)
        effective_timeout = self._resolve_inference_timeout(
            base_timeout=base_timeout,
            intent=None,
            model_role=model_role,
            has_images=False,
        )
        target_model, _ = await self._prepare_target_model(
            model=model_override,
            role=model_role,
            timeout=effective_timeout,
        )
        chat_response = await self._llm_client.chat(
            messages=messages,
            model=target_model,
            timeout=effective_timeout,
            response_format=response_format,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return sanitize_model_output(chat_response.content)

    async def change_model(self, model: str) -> dict:
        """런타임 기본 모델을 변경한다."""
        models = await self._llm_client.list_models()
        available_names = [m["name"] for m in models]

        if model not in available_names:
            return {
                "success": False,
                "error": f"모델 '{model}'을(를) 찾을 수 없습니다.",
                "available": available_names,
            }

        old_model = self._llm_client.default_model
        self._llm_client.default_model = model

        # 모델 변경 시 시맨틱 캐시 무효화
        if self._semantic_cache is not None:
            await self._semantic_cache.invalidate()

        self._logger.info(
            "model_changed", old_model=old_model, new_model=model
        )
        return {"success": True, "old_model": old_model, "new_model": model}

    async def list_models(self) -> list[dict]:
        """설치된 모델 목록을 반환한다."""
        return await self._llm_client.list_models()

    def get_current_model(self) -> str:
        """현재 기본 모델 이름을 반환한다."""
        return self._llm_client.default_model

    async def reload_skills(self, *, strict: bool = False) -> int:
        """스킬 정의를 다시 로드한다."""
        return await self._skills.reload_skills(strict=strict)

    def list_skills(self) -> list[dict]:
        """로드된 스킬 목록을 반환한다."""
        return self._skills.list_skills()

    def get_last_skill_load_errors(self) -> list[str]:
        """최근 스킬 로드 중 발생한 오류 목록을 반환한다."""
        return self._skills.get_last_load_errors()

    async def get_memory_stats(self, chat_id: int) -> dict:
        """채팅 메모리 통계를 조회한다."""
        return await self._memory.get_memory_stats(chat_id)

    async def clear_conversation(self, chat_id: int) -> int:
        """채팅 대화 기록을 삭제한다."""
        deleted = await self._memory.clear_conversation(chat_id)
        if self._semantic_cache is not None:
            await self._semantic_cache.invalidate(chat_id=chat_id)
        return deleted

    async def export_conversation_markdown(
        self,
        chat_id: int,
        output_dir: Path,
    ) -> Path:
        """채팅 대화 기록을 마크다운으로 내보낸다."""
        return await self._memory.export_conversation_markdown(chat_id, output_dir)

    async def get_status(self) -> dict:
        return await engine_status.get_status(self)

    def _build_optimization_tier_details(self) -> dict[str, dict[str, Any]]:
        return engine_status.build_optimization_tier_details(self)

    def _build_rag_tier_detail(self) -> dict[str, Any]:
        return engine_status.build_rag_tier_detail(self)

    def _manual_tier_detail(
        self,
        *,
        name: str,
        configured: bool,
        enabled: bool,
        degraded: bool,
        reason: str | None = None,
    ) -> dict[str, Any]:
        return engine_status.manual_tier_detail(
            self,
            name=name,
            configured=configured,
            enabled=enabled,
            degraded=degraded,
            reason=reason,
        )

    def _make_tier_detail(
        self,
        *,
        name: str,
        configured: bool,
        instance: Any,
        unavailable_reason: str,
        enabled_attr: str | None = None,
        disabled_reason: str = "disabled",
    ) -> dict[str, Any]:
        return engine_status.make_tier_detail(
            self,
            name=name,
            configured=configured,
            instance=instance,
            unavailable_reason=unavailable_reason,
            enabled_attr=enabled_attr,
            disabled_reason=disabled_reason,
        )

    @staticmethod
    def _format_uptime(seconds: float) -> str:
        return engine_status.format_uptime(seconds)
