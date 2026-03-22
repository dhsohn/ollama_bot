"""Main engine for conversation orchestration, context management, and routing.

This is the central hub for user-message handling. It receives input from the
Telegram handler and returns a response after the appropriate processing path.

Routing order:
  [pre-step] skill trigger matching
  [Tier 1] rule-based instant response (`InstantResponder`)
  [Tier 2] intent routing (`IntentRouter`)
  [Tier 3] semantic cache (`SemanticCache`)
  [Tier 4] full LLM with optimized context
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from core import engine_background, engine_management, engine_status, engine_tracking
from core.config import AppSettings, get_system_prompt
from core.engine_mixins import EngineDelegateMixin, EngineManagementMixin
from core.engine_types import (
    ContextProvider,
    _RoutingDecision,
    _StreamMeta,
)
from core.enums import RoutingTier
from core.llm_protocol import LLMClientProtocol
from core.logging_setup import get_logger
from core.memory import MemoryManager
from core.skill_manager import SkillManager
from core.stream_orchestrator import EngineStreamOrchestrator
from core.text_utils import detect_output_anomalies, sanitize_model_output

if TYPE_CHECKING:
    from core.context_compressor import ContextCompressor
    from core.feedback_manager import FeedbackManager
    from core.instant_responder import InstantResponder
    from core.intent_router import IntentRouter
    from core.rag.pipeline import RAGPipeline
    from core.semantic_cache import SemanticCache
_STREAM_META_TTL_SECONDS = 600.0
_STREAM_META_MAX_ENTRIES = 2048
_STREAM_REPEATED_CHUNK_ABORT_THRESHOLD = 30


class Engine(EngineManagementMixin, EngineDelegateMixin):
    """Conversation engine that orchestrates routing and context management."""

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
        self._system_prompt = getattr(llm_client, "system_prompt", "") or self._resolve_system_prompt(config)
        self._max_conversation_length = config.bot.max_conversation_length
        self._start_time = time.monotonic()
        self._logger = get_logger("engine")
        self._last_stream_meta: dict[int, _StreamMeta] = {}
        self._stream_meta_ttl_seconds = _STREAM_META_TTL_SECONDS
        self._stream_meta_max_entries = _STREAM_META_MAX_ENTRIES
        self._active_request_count = 0
        summary_concurrency = max(1, int(config.context_compressor.summarize_concurrency))
        # Cap summary-task creation to avoid unbounded background growth.
        self._summary_task_limit = max(2, summary_concurrency * 3)
        self._summary_tasks: set[asyncio.Task[Any]] = set()
        self._degraded_since: dict[str, float] = {}
        self._rag_reindex_lock = asyncio.Lock()
        self._tracking_ops = engine_tracking.EngineTrackingOperations(
            self,
            meta_factory=_StreamMeta,
            monotonic_fn=lambda: time.monotonic(),
        )
        self._background_summary_controller = engine_background.BackgroundSummaryController(self)
        self._status_service = engine_status.EngineStatusService(
            self,
            monotonic_fn=lambda: time.monotonic(),
        )
        self._management_api = engine_management.EngineManagementAPI(self)
        self._stream_orchestrator = EngineStreamOrchestrator(
            self,
            repeated_chunk_abort_threshold=_STREAM_REPEATED_CHUNK_ABORT_THRESHOLD,
        )

    @staticmethod
    def _resolve_system_prompt(config: AppSettings) -> str:
        """Return the configured Ollama system prompt."""
        return get_system_prompt(config)

    @staticmethod
    def _finalize_stream_response(full_response: str) -> str:
        """Normalize a stream or fallback response and ensure it is usable."""
        if not full_response.strip():
            raise RuntimeError("empty_response_from_llm")
        normalized = sanitize_model_output(full_response).strip()
        if not normalized:
            raise RuntimeError("empty_response_from_llm")
        return normalized

    # Message handling (hierarchical routing)

    async def process_message(
        self,
        chat_id: int,
        text: str,
        model_override: str | None = None,
        *,
        images: list[bytes] | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Process a user message and return the response.

        Routing order: Skill -> Tier 1 (instant) -> Tier 2 (intent) -> Tier 3
        (cache) -> Tier 4 (LLM)
        """
        t0 = time.monotonic()
        async with self._track_request(chat_id, stream=False):
            try:
                routing = await self._decide_routing(
                    chat_id, text, model_override, images=images, metadata=metadata,
                )

                if routing.tier is RoutingTier.SKILL:
                    skill = routing.skill
                    if skill is None:
                        raise RuntimeError("routing_decision_invalid: missing skill")
                    self._logger.info("skill_triggered", chat_id=chat_id, skill=skill.name)
                    messages = await self._build_context(chat_id, text, skill=skill)
                    content, usage, _target_model = await self._run_skill_chat(
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
                content = await self._maybe_review_full_response(
                    chat_id=chat_id,
                    text=text,
                    response=content,
                    raw_response=chat_response.content,
                    intent=routing.intent,
                    prepared_full=prepared_full,
                    images=images,
                    anomaly_reasons=anomaly_reasons,
                )
                await self._persist_turn(chat_id, text, content)

                await self._maybe_store_semantic_cache(
                    chat_id=chat_id,
                    text=text,
                    response=content,
                    images=images,
                    model_override=model_override,
                    intent=routing.intent,
                    metadata=metadata,
                )

                self._log_request(
                    t0, chat_id, RoutingTier.FULL, usage, len(prepared_full.messages),
                    intent=routing.intent,
                    rag_trace=prepared_full.rag_result.trace if prepared_full.rag_result else None,
                )

                # Trigger background summary refresh.
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
        """Process a message in streaming mode and yield chunks in order."""
        async for chunk in self._stream_orchestrator.process_message_stream(
            chat_id=chat_id,
            text=text,
            model_override=model_override,
            images=images,
            metadata=metadata,
        ):
            yield chunk
