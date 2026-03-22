"""Engine mixins that provide explicit module-backed method boundaries."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from core import engine_context, engine_delegates, engine_management, engine_routing

if TYPE_CHECKING:
    from core.engine_background import BackgroundSummaryController
    from core.engine_management import EngineManagementAPI
    from core.engine_tracking import EngineTrackingOperations


class EngineManagementMixin:
    """Public Engine API backed by the extracted management modules."""

    _management_api: EngineManagementAPI

    async def rollback_last_turn(self, chat_id: int) -> int:
        return await self._management_api.rollback_last_turn(chat_id)

    async def classify_intent(self, text: str) -> str | None:
        return await self._management_api.classify_intent(text)

    async def route_request(
        self,
        text: str,
        images: list[bytes] | None = None,
        metadata: dict | None = None,
    ) -> dict[str, Any]:
        return await self._management_api.route_request(
            text,
            images=images,
            metadata=metadata,
        )

    async def retrieve(
        self,
        text: str,
        metadata: dict | None = None,
    ) -> dict[str, Any]:
        return await self._management_api.retrieve(text, metadata=metadata)

    async def generate(
        self,
        text: str,
        images: list[bytes] | None = None,
        metadata: dict | None = None,
    ) -> dict[str, Any]:
        return await self._management_api.generate(
            text,
            images=images,
            metadata=metadata,
        )

    async def analyze_all_corpus(
        self,
        query: str,
        *,
        progress_callback=None,
    ) -> dict[str, Any]:
        return await self._management_api.analyze_all_corpus(
            query,
            progress_callback=progress_callback,
        )

    async def reindex_rag_corpus(
        self,
        kb_dirs: list[str] | None = None,
    ) -> dict[str, Any]:
        return await self._management_api.reindex_rag_corpus(kb_dirs)

    def consume_last_stream_meta(
        self,
        chat_id: int,
    ) -> dict[str, Any] | None:
        return self._management_api.consume_last_stream_meta(chat_id)

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
        return await self._management_api.execute_skill(
            skill_name,
            parameters,
            chat_id=chat_id,
            model_override=model_override,
            model_role_override=model_role_override,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )

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
        timeout_is_hard: bool = False,
        system_prompt_override: str | None = None,
    ) -> str:
        return await self._management_api.process_prompt(
            prompt,
            chat_id=chat_id,
            response_format=response_format,
            max_tokens=max_tokens,
            temperature=temperature,
            model_override=model_override,
            model_role=model_role,
            timeout=timeout,
            timeout_is_hard=timeout_is_hard,
            system_prompt_override=system_prompt_override,
        )

    async def change_model(self, model: str) -> dict:
        return await self._management_api.change_model(model)

    async def list_models(self) -> list[dict]:
        return await self._management_api.list_models()

    def get_current_model(self) -> str:
        return self._management_api.get_current_model()

    async def reload_skills(self, *, strict: bool = False) -> int:
        return await self._management_api.reload_skills(strict=strict)

    def list_skills(self, lang: str = "ko") -> list[dict]:
        return self._management_api.list_skills(lang=lang)

    def get_last_skill_load_errors(self) -> list[str]:
        return self._management_api.get_last_skill_load_errors()

    async def get_memory_stats(self, chat_id: int) -> dict:
        return await self._management_api.get_memory_stats(chat_id)

    async def clear_conversation(self, chat_id: int) -> int:
        return await self._management_api.clear_conversation(chat_id)

    async def export_conversation_markdown(
        self,
        chat_id: int,
        output_dir: Path,
    ) -> Path:
        return await self._management_api.export_conversation_markdown(
            chat_id,
            output_dir,
        )

    async def get_status(self) -> dict:
        return await self._management_api.get_status()

    def _build_optimization_tier_details(self) -> dict[str, dict[str, Any]]:
        return self._management_api.build_optimization_tier_details()

    def _build_rag_tier_detail(self) -> dict[str, Any]:
        return self._management_api.build_rag_tier_detail()

    def _manual_tier_detail(
        self,
        *,
        name: str,
        configured: bool,
        enabled: bool,
        degraded: bool,
        reason: str | None = None,
    ) -> dict[str, Any]:
        return self._management_api.manual_tier_detail(
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
        return self._management_api.make_tier_detail(
            name=name,
            configured=configured,
            instance=instance,
            unavailable_reason=unavailable_reason,
            enabled_attr=enabled_attr,
            disabled_reason=disabled_reason,
        )

    @staticmethod
    def _format_uptime(seconds: float) -> str:
        return engine_management.EngineManagementAPI.format_uptime(seconds)


class EngineDelegateMixin:
    """Internal Engine helpers backed by the extracted delegate modules."""

    _tracking_ops: EngineTrackingOperations
    _background_summary_controller: BackgroundSummaryController

    def _track_request(
        self,
        chat_id: int,
        *,
        stream: bool,
    ) -> AbstractAsyncContextManager[None]:
        return self._tracking_ops.track_request(chat_id, stream=stream)

    async def _classify_route(self, text: str) -> Any | None:
        return await engine_routing.classify_route(cast(Any, self), text)

    async def _decide_routing(
        self,
        chat_id: int,
        text: str,
        model_override: str | None = None,
        *,
        images: list[bytes] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        return await engine_routing.decide_routing(
            cast(Any, self),
            chat_id,
            text,
            model_override=model_override,
            images=images,
            metadata=metadata,
            decision_factory=self._routing_decision_factory(),
        )

    def _set_stream_meta(self, chat_id: int, **kwargs: Any) -> None:
        self._tracking_ops.set_stream_meta(chat_id, **kwargs)

    def _cleanup_stream_meta(self, now: float | None = None) -> None:
        self._tracking_ops.cleanup_stream_meta(now=now)

    def _build_cache_context(
        self,
        model_override: str | None,
        intent: str | None,
        chat_id: int,
    ) -> Any:
        return engine_delegates._build_cache_context(
            self,
            model_override,
            intent,
            chat_id,
        )

    @staticmethod
    def _is_cache_response_acceptable(query: str, response: str) -> bool:
        return engine_delegates._is_cache_response_acceptable(query, response)

    def _log_request(self, t0: float, chat_id: int, tier: Any, usage=None, history_count: int = 0, **kwargs: Any) -> None:
        self._tracking_ops.log_request(
            t0,
            chat_id,
            tier,
            usage,
            history_count,
            **kwargs,
        )

    @staticmethod
    def _inject_rag_context(messages: list[dict[str, str]], rag_result: Any) -> list[dict[str, str]]:
        return engine_delegates._inject_rag_context(messages, rag_result)

    @staticmethod
    def _inject_extra_context(messages: list[dict[str, str]], context: str) -> list[dict[str, str]]:
        return engine_delegates._inject_extra_context(messages, context)

    def _trigger_background_summary(self, chat_id: int) -> None:
        self._background_summary_controller.trigger_summary(chat_id)

    def _handle_summary_task_done(self, task: Any) -> None:
        self._background_summary_controller.handle_summary_task_done(task)

    def _handle_background_task_error(self, task: Any) -> None:
        self._background_summary_controller.handle_background_task_error(task)

    @staticmethod
    async def _emit_full_scan_progress(callback, payload: dict[str, Any]) -> None:
        await engine_delegates._emit_full_scan_progress(callback, payload)

    @staticmethod
    def _build_full_scan_segments(chunks: list[Any], *, max_chars: int) -> list[Any]:
        return engine_delegates._build_full_scan_segments(chunks, max_chars=max_chars)

    @staticmethod
    def _pack_blocks_for_reduction(blocks: list[str], *, max_chars: int) -> list[str]:
        return engine_delegates._pack_blocks_for_reduction(blocks, max_chars=max_chars)

    async def _prepare_request(
        self,
        chat_id: int,
        text: str,
        *,
        stream: bool,
        strategy: Any | None = None,
    ) -> Any:
        return await engine_delegates._prepare_request(
            self,
            chat_id,
            text,
            stream=stream,
            strategy=strategy,
        )

    @staticmethod
    def _resolve_inference_timeout(
        *,
        base_timeout: int,
        intent: str | None,
        model_role: str | None,
        has_images: bool = False,
    ) -> int:
        return engine_delegates._resolve_inference_timeout(
            base_timeout=base_timeout,
            intent=intent,
            model_role=model_role,
            has_images=has_images,
        )

    async def _prepare_full_request(self, **kwargs: Any) -> Any:
        return await engine_delegates._prepare_full_request(self, **kwargs)

    @staticmethod
    def _extract_json_payload(text: str) -> dict[str, Any] | None:
        return engine_delegates._extract_json_payload(text)

    async def _maybe_store_semantic_cache(self, **kwargs: Any) -> Any:
        return await engine_delegates._maybe_store_semantic_cache(self, **kwargs)

    async def _maybe_review_full_response(self, **kwargs: Any) -> Any:
        return await engine_delegates._maybe_review_full_response(self, **kwargs)

    @staticmethod
    def _is_summarize_skill(skill: Any) -> bool:
        return engine_delegates._is_summarize_skill(skill)

    @staticmethod
    def _extract_skill_user_input(messages: list[dict[str, str]]) -> str:
        return engine_delegates._extract_skill_user_input(messages)

    def _should_use_chunked_summary(self, **kwargs: Any) -> bool:
        return engine_delegates._should_use_chunked_summary(self, **kwargs)

    @staticmethod
    def _split_text_for_summary(text: str) -> list[str]:
        return engine_delegates._split_text_for_summary(text)

    async def _run_skill_chat(self, **kwargs: Any) -> Any:
        return await engine_delegates._run_skill_chat(self, **kwargs)

    async def _run_chunked_summary_pipeline(self, **kwargs: Any) -> Any:
        return await engine_delegates._run_chunked_summary_pipeline(self, **kwargs)

    async def _prepare_target_model(self, **kwargs: Any) -> Any:
        return await engine_delegates._prepare_target_model(self, **kwargs)

    def _resolve_model_for_role(self, role: str) -> str | None:
        return engine_delegates._resolve_model_for_role(self, role)

    async def _persist_turn(self, chat_id: int, text: str, content: str, *, skill: Any = None) -> None:
        await self._tracking_ops.persist_turn(
            chat_id,
            text,
            content,
            skill=skill,
        )

    async def _persist_failed_turn(
        self,
        chat_id: int,
        user_text: str,
        *,
        error: Exception,
        tier: str | None,
        skill: Any = None,
    ) -> None:
        await self._tracking_ops.persist_failed_turn(
            chat_id,
            user_text,
            error=error,
            tier=tier,
            skill=skill,
        )

    async def _build_context(self, chat_id: int, text: str, *, skill: Any = None, strategy: Any | None = None) -> Any:
        return await engine_delegates._build_context(
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
        skill: Any = None,
        strategy: Any | None = None,
    ) -> Any:
        return await engine_delegates._build_base_context(
            self,
            chat_id,
            skill=skill,
            strategy=strategy,
        )

    def _sanitize_history_for_prompt(self, history: list[dict[str, str]]) -> list[dict[str, str]]:
        return engine_delegates._sanitize_history_for_prompt(self, history)

    async def _inject_preferences(self, system: str, chat_id: int) -> str:
        return await engine_delegates._inject_preferences(self, system, chat_id)

    async def _inject_guidelines(
        self,
        system: str,
        chat_id: int,
    ) -> str:
        return await engine_delegates._inject_guidelines(self, system, chat_id)

    async def _inject_dicl_examples(
        self,
        system: str,
        *,
        chat_id: int,
        text: str,
        include_dicl: bool,
        skill: Any = None,
    ) -> str:
        return await engine_delegates._inject_dicl_examples(
            self,
            system,
            chat_id=chat_id,
            text=text,
            include_dicl=include_dicl,
            skill=skill,
        )

    @staticmethod
    def _inject_intent_suffix(system: str, strategy: Any | None) -> str:
        return engine_delegates._inject_intent_suffix(system, strategy)

    @staticmethod
    def _normalize_language(lang: str | None) -> str:
        return engine_delegates._normalize_language(lang or "")

    def _inject_language_policy(
        self,
        system_prompt: str,
        language_override: str | None = None,
    ) -> str:
        if language_override is None:
            return engine_delegates._inject_language_policy(cast(Any, self), system_prompt)
        return engine_context.inject_language_policy(
            cast(Any, self),
            system_prompt,
            language_override=language_override,
        )

    @staticmethod
    def _routing_decision_factory():
        from core.engine_types import _RoutingDecision

        return _RoutingDecision

    @staticmethod
    def _assemble_messages(
        system_prompt: str,
        history: list[dict[str, str]],
        user_text: str,
        skill: Any = None,
    ) -> list[dict[str, str]]:
        return engine_delegates._assemble_messages(
            system_prompt,
            history,
            user_text,
            skill,
        )
