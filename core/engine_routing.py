from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from core.async_utils import run_in_thread
from core.text_utils import detect_output_anomalies, sanitize_model_output

if TYPE_CHECKING:
    from core.engine import Engine

_REASONING_TIMEOUT_SECONDS = 3600
_REASONING_INTENTS = {"complex", "code"}
_REASONING_MODEL_ROLES = {"reasoning", "coding", "vision"}


async def classify_route(engine: Engine, text: str) -> Any | None:
    """인텐트 분류를 이벤트 루프 밖 스레드에서 수행한다."""
    if engine._intent_router is None:
        return None
    return await run_in_thread(engine._intent_router.classify, text)


async def decide_routing(
    engine: Engine,
    chat_id: int,
    text: str,
    model_override: str | None = None,
    *,
    images: list[bytes] | None = None,
    decision_factory: Callable[..., Any],
) -> Any:
    """LLM 호출 전 라우팅 판정(스킬/즉시/인텐트/캐시)을 공통 처리한다."""
    skill = engine._skills.match_trigger(text)
    if skill is not None:
        return decision_factory(tier="skill", skill=skill)

    if engine._instant_responder is not None:
        instant = engine._instant_responder.match(text)
        if instant is not None:
            return decision_factory(tier="instant", instant=instant)

    route = await classify_route(engine, text)
    intent = route.intent if route else None

    if (
        engine._semantic_cache is not None
        and not images
        and engine._semantic_cache.is_cacheable(text)
    ):
        cache_ctx = build_cache_context(engine, model_override, intent, chat_id)
        cached = await engine._semantic_cache.get(text, context=cache_ctx)
        if cached is not None:
            if not is_cache_response_acceptable(text, cached.response):
                engine._logger.info(
                    "semantic_cache_entry_rejected",
                    chat_id=chat_id,
                    cache_id=cached.cache_id,
                )
                try:
                    await engine._semantic_cache.invalidate_by_id(cached.cache_id)
                except Exception as exc:
                    engine._logger.debug(
                        "semantic_cache_rejected_entry_invalidate_failed",
                        chat_id=chat_id,
                        cache_id=cached.cache_id,
                        error=str(exc),
                    )
            else:
                return decision_factory(tier="cache", route=route, cached=cached)

    return decision_factory(tier="full", route=route)


def build_cache_context(
    engine: Engine,
    model_override: str | None,
    intent: str | None,
    chat_id: int,
) -> Any:
    from core.semantic_cache import CacheContext

    return CacheContext(
        model=model_override or engine._llm_client.default_model,
        prompt_ver=engine._config.lemonade.prompt_version,
        intent=intent,
        scope="user",
        chat_id=chat_id,
    )


def is_cache_response_acceptable(query: str, response: str) -> bool:
    _ = query
    cleaned = sanitize_model_output(response).strip()
    if not cleaned:
        return False
    return not detect_output_anomalies(response, cleaned)


def resolve_inference_timeout(
    *,
    base_timeout: int,
    intent: str | None,
    model_role: str | None,
    has_images: bool = False,
) -> int:
    timeout = max(1, int(base_timeout))
    if has_images:
        return max(timeout, _REASONING_TIMEOUT_SECONDS)
    role = (model_role or "").strip().lower()
    intent_name = (intent or "").strip().lower()
    if role in _REASONING_MODEL_ROLES or intent_name in _REASONING_INTENTS:
        return max(timeout, _REASONING_TIMEOUT_SECONDS)
    return timeout
