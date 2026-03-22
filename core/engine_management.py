"""Public engine management helpers extracted from ``core.engine``."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from core import engine_rag, engine_status, engine_tracking
from core.text_utils import sanitize_model_output


class EngineManagementAPI:
    """Own public Engine management operations behind a stable API surface."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    async def rollback_last_turn(self, chat_id: int) -> int:
        """Rollback the most recent streaming turn."""
        return await self._engine._memory.delete_last_turn(chat_id)

    async def classify_intent(self, text: str) -> str | None:
        route = await self._engine._classify_route(text)
        return route.intent if route else None

    async def route_request(
        self,
        text: str,
        images: list[bytes] | None = None,
        metadata: dict | None = None,
    ) -> dict[str, Any]:
        _ = (text, images, metadata)
        return {
            "selected_model": self._engine._llm_client.default_model,
            "trigger": "single_model",
        }

    async def retrieve(
        self,
        text: str,
        metadata: dict | None = None,
    ) -> dict[str, Any]:
        if self._engine._rag_pipeline is None:
            return {
                "candidates": [],
                "contexts": [],
                "rag_trace_partial": {
                    "rag_used": False,
                    "error": "rag_pipeline_disabled",
                },
            }

        rag_result = await self._engine._rag_pipeline.execute(text, metadata)
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
        routing_decision = await self.route_request(text, images=images, metadata=metadata)
        target_model = routing_decision["selected_model"]

        user_text = text.strip()
        if not user_text and images:
            user_text = "Analyze this image."

        system_prompt = self._engine._inject_language_policy(self._engine._system_prompt)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

        rag_trace: dict[str, Any] = {"rag_used": False}
        if (
            self._engine._rag_pipeline is not None
            and self._engine._rag_pipeline.should_trigger_rag(text, metadata)
        ):
            rag_result = await self._engine._rag_pipeline.execute(text, metadata)
            if rag_result.contexts:
                messages = self._engine._inject_rag_context(messages, rag_result)
            rag_trace = rag_result.trace.to_dict()

        chat_response = await self._engine._llm_client.chat(
            messages=messages,
            model=target_model,
            timeout=self._engine._config.bot.response_timeout,
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
            self._engine,
            query,
            progress_callback=progress_callback,
        )

    async def reindex_rag_corpus(
        self,
        kb_dirs: list[str] | None = None,
    ) -> dict[str, Any]:
        return await engine_rag.reindex_rag_corpus(self._engine, kb_dirs)

    def consume_last_stream_meta(self, chat_id: int) -> dict[str, Any] | None:
        tracking_ops = getattr(self._engine, "_tracking_ops", None)
        if tracking_ops is not None:
            return tracking_ops.consume_last_stream_meta(chat_id)
        return engine_tracking.consume_last_stream_meta(
            self._engine,
            chat_id,
            monotonic_fn=time.monotonic,
        )

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
        skill = self._engine._skills.get_skill(skill_name)
        if not skill:
            return f"Skill '{skill_name}' not found."

        input_text = parameters.get("input_text", parameters.get("query", ""))
        skill_system = self._engine._inject_language_policy(skill.system_prompt)
        messages = [
            {"role": "system", "content": skill_system},
            {"role": "user", "content": input_text},
        ]

        content, _, _ = await self._engine._run_skill_chat(
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
            await self._engine._memory.add_message(
                chat_id,
                "assistant",
                content,
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
        timeout_is_hard: bool = False,
        system_prompt_override: str | None = None,
    ) -> str:
        _ = chat_id
        base = (
            system_prompt_override
            if system_prompt_override is not None
            else self._engine._system_prompt
        )
        system_prompt = self._engine._inject_language_policy(base)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        base_timeout = int(timeout or self._engine._config.bot.response_timeout)
        if timeout_is_hard:
            effective_timeout = base_timeout
        else:
            effective_timeout = self._engine._resolve_inference_timeout(
                base_timeout=base_timeout,
                intent=None,
                model_role=model_role,
                has_images=False,
            )
        target_model, _ = await self._engine._prepare_target_model(
            model=model_override,
            role=model_role,
            timeout=effective_timeout,
        )
        chat_response = await self._engine._llm_client.chat(
            messages=messages,
            model=target_model,
            timeout=effective_timeout,
            response_format=response_format,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return sanitize_model_output(chat_response.content)

    async def change_model(self, model: str) -> dict:
        models = await self._engine._llm_client.list_models()
        available_names = [m["name"] for m in models]

        if model not in available_names:
            return {
                "success": False,
                "error": f"Model '{model}' not found.",
                "available": available_names,
            }

        old_model = self._engine._llm_client.default_model
        self._engine._llm_client.default_model = model

        if self._engine._semantic_cache is not None:
            await self._engine._semantic_cache.invalidate()

        self._engine._logger.info("model_changed", old_model=old_model, new_model=model)
        return {"success": True, "old_model": old_model, "new_model": model}

    async def list_models(self) -> list[dict]:
        return await self._engine._llm_client.list_models()

    def get_current_model(self) -> str:
        return self._engine._llm_client.default_model

    async def reload_skills(self, *, strict: bool = False) -> int:
        return await self._engine._skills.reload_skills(strict=strict)

    def list_skills(self, lang: str = "ko") -> list[dict]:
        return self._engine._skills.list_skills(lang=lang)

    def get_last_skill_load_errors(self) -> list[str]:
        return self._engine._skills.get_last_load_errors()

    async def get_memory_stats(self, chat_id: int) -> dict:
        return await self._engine._memory.get_memory_stats(chat_id)

    async def clear_conversation(self, chat_id: int) -> int:
        deleted = await self._engine._memory.clear_conversation(chat_id)
        if self._engine._semantic_cache is not None:
            await self._engine._semantic_cache.invalidate(chat_id=chat_id)
        return deleted

    async def export_conversation_markdown(
        self,
        chat_id: int,
        output_dir: str | Path,
    ) -> Path:
        return await self._engine._memory.export_conversation_markdown(chat_id, output_dir)

    async def get_status(self) -> dict[str, Any]:
        status_service = getattr(self._engine, "_status_service", None)
        if status_service is not None:
            return await status_service.get_status()
        return await engine_status.get_status(self._engine)

    def build_optimization_tier_details(self) -> dict[str, dict[str, Any]]:
        status_service = getattr(self._engine, "_status_service", None)
        if status_service is not None:
            return status_service.build_optimization_tier_details()
        return engine_status.build_optimization_tier_details(self._engine)

    def build_rag_tier_detail(self) -> dict[str, Any]:
        status_service = getattr(self._engine, "_status_service", None)
        if status_service is not None:
            return status_service.build_rag_tier_detail()
        return engine_status.build_rag_tier_detail(self._engine)

    def manual_tier_detail(
        self,
        *,
        name: str,
        configured: bool,
        enabled: bool,
        degraded: bool,
        reason: str | None = None,
    ) -> dict[str, Any]:
        status_service = getattr(self._engine, "_status_service", None)
        if status_service is not None:
            return status_service.manual_tier_detail(
                name=name,
                configured=configured,
                enabled=enabled,
                degraded=degraded,
                reason=reason,
            )
        return engine_status.manual_tier_detail(
            self._engine,
            name=name,
            configured=configured,
            enabled=enabled,
            degraded=degraded,
            reason=reason,
        )

    def make_tier_detail(
        self,
        *,
        name: str,
        configured: bool,
        instance: Any,
        unavailable_reason: str,
        enabled_attr: str | None = None,
        disabled_reason: str = "disabled",
    ) -> dict[str, Any]:
        status_service = getattr(self._engine, "_status_service", None)
        if status_service is not None:
            return status_service.make_tier_detail(
                name=name,
                configured=configured,
                instance=instance,
                unavailable_reason=unavailable_reason,
                enabled_attr=enabled_attr,
                disabled_reason=disabled_reason,
            )
        return engine_status.make_tier_detail(
            self._engine,
            name=name,
            configured=configured,
            instance=instance,
            unavailable_reason=unavailable_reason,
            enabled_attr=enabled_attr,
            disabled_reason=disabled_reason,
        )

    @staticmethod
    def format_uptime(seconds: float) -> str:
        return engine_status.format_uptime(seconds)


def _api(engine: Any) -> EngineManagementAPI:
    existing = getattr(engine, "_management_api", None)
    if isinstance(existing, EngineManagementAPI):
        return existing
    return EngineManagementAPI(engine)


async def rollback_last_turn(self: Any, chat_id: int) -> int:
    return await _api(self).rollback_last_turn(chat_id)


async def classify_intent(self: Any, text: str) -> str | None:
    return await _api(self).classify_intent(text)


async def route_request(
    self: Any,
    text: str,
    images: list[bytes] | None = None,
    metadata: dict | None = None,
) -> dict[str, Any]:
    return await _api(self).route_request(text, images=images, metadata=metadata)


async def retrieve(
    self: Any,
    text: str,
    metadata: dict | None = None,
) -> dict[str, Any]:
    return await _api(self).retrieve(text, metadata=metadata)


async def generate(
    self: Any,
    text: str,
    images: list[bytes] | None = None,
    metadata: dict | None = None,
) -> dict[str, Any]:
    return await _api(self).generate(text, images=images, metadata=metadata)


async def analyze_all_corpus(
    self: Any,
    query: str,
    *,
    progress_callback: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
) -> dict[str, Any]:
    return await _api(self).analyze_all_corpus(query, progress_callback=progress_callback)


async def reindex_rag_corpus(
    self: Any,
    kb_dirs: list[str] | None = None,
) -> dict[str, Any]:
    return await _api(self).reindex_rag_corpus(kb_dirs)


def consume_last_stream_meta(
    self: Any,
    chat_id: int,
) -> dict[str, Any] | None:
    return _api(self).consume_last_stream_meta(chat_id)


async def execute_skill(
    self: Any,
    skill_name: str,
    parameters: dict,
    chat_id: int | None = None,
    model_override: str | None = None,
    model_role_override: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    timeout: int | None = None,
) -> str:
    return await _api(self).execute_skill(
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
    self: Any,
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
    return await _api(self).process_prompt(
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


async def change_model(self: Any, model: str) -> dict:
    return await _api(self).change_model(model)


async def list_models(self: Any) -> list[dict]:
    return await _api(self).list_models()


def get_current_model(self: Any) -> str:
    return _api(self).get_current_model()


async def reload_skills(self: Any, *, strict: bool = False) -> int:
    return await _api(self).reload_skills(strict=strict)


def list_skills(self: Any, lang: str = "ko") -> list[dict]:
    return _api(self).list_skills(lang=lang)


def get_last_skill_load_errors(self: Any) -> list[str]:
    return _api(self).get_last_skill_load_errors()


async def get_memory_stats(self: Any, chat_id: int) -> dict:
    return await _api(self).get_memory_stats(chat_id)


async def clear_conversation(self: Any, chat_id: int) -> int:
    return await _api(self).clear_conversation(chat_id)


async def export_conversation_markdown(
    self: Any,
    chat_id: int,
    output_dir: str | Path,
) -> Path:
    return await _api(self).export_conversation_markdown(chat_id, output_dir)


async def get_status(self: Any) -> dict[str, Any]:
    return await _api(self).get_status()


def _build_optimization_tier_details(self: Any) -> dict[str, dict[str, Any]]:
    return _api(self).build_optimization_tier_details()


def _build_rag_tier_detail(self: Any) -> dict[str, Any]:
    return _api(self).build_rag_tier_detail()


def _manual_tier_detail(
    self: Any,
    *,
    name: str,
    configured: bool,
    enabled: bool,
    degraded: bool,
    reason: str | None = None,
) -> dict[str, Any]:
    return _api(self).manual_tier_detail(
        name=name,
        configured=configured,
        enabled=enabled,
        degraded=degraded,
        reason=reason,
    )


def _make_tier_detail(
    self: Any,
    *,
    name: str,
    configured: bool,
    instance: Any,
    unavailable_reason: str,
    enabled_attr: str | None = None,
    disabled_reason: str = "disabled",
) -> dict[str, Any]:
    return _api(self).make_tier_detail(
        name=name,
        configured=configured,
        instance=instance,
        unavailable_reason=unavailable_reason,
        enabled_attr=enabled_attr,
        disabled_reason=disabled_reason,
    )


def _format_uptime(seconds: float) -> str:
    return EngineManagementAPI.format_uptime(seconds)
