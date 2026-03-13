from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from core.async_utils import run_in_thread
from core.config import get_prompt_version
from core.constants import (
    REASONING_INTENTS,
    REASONING_MODEL_ROLES,
    REASONING_TIMEOUT_SECONDS,
)
from core.enums import RoutingTier
from core.text_utils import detect_output_anomalies, sanitize_model_output

if TYPE_CHECKING:
    from core.engine import Engine


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
    metadata: dict[str, Any] | None = None,
    decision_factory: Callable[..., Any],
) -> Any:
    """계층형 라우팅 판정을 수행한다.

    순서: Skill → Instant → SemanticCache → Full LLM.
    각 계층에서 매칭되면 즉시 decision_factory로 결과를 생성하여 반환한다.

    Args:
        engine: 엔진 인스턴스.
        chat_id: 텔레그램 채팅 ID.
        text: 사용자 입력 텍스트.
        model_override: 모델 오버라이드 (캐시 컨텍스트에 사용).
        images: 첨부 이미지 (있으면 캐시 우회).
        metadata: 요청 단위 제어 메타데이터.
        decision_factory: RoutingDecision 생성 팩토리.

    Returns:
        라우팅 판정 결과 (decision_factory가 생성한 객체).
    """
    skill = engine._skills.match_trigger(text)
    if skill is not None:
        return decision_factory(tier=RoutingTier.SKILL, skill=skill)

    if engine._instant_responder is not None:
        instant = engine._instant_responder.match(text)
        if instant is not None:
            return decision_factory(tier=RoutingTier.INSTANT, instant=instant)

    route = await classify_route(engine, text)
    intent = route.intent if route else None
    skip_semantic_cache = bool(metadata and metadata.get("skip_semantic_cache"))

    if (
        not skip_semantic_cache
        and
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
                return decision_factory(tier=RoutingTier.CACHE, route=route, cached=cached)

    return decision_factory(tier=RoutingTier.FULL, route=route)


def build_cache_context(
    engine: Engine,
    model_override: str | None,
    intent: str | None,
    chat_id: int,
) -> Any:
    from core.semantic_cache import CacheContext

    return CacheContext(
        model=model_override or engine._llm_client.default_model,
        prompt_ver=get_prompt_version(engine._config),
        intent=intent,
        scope="user",
        chat_id=chat_id,
    )


def is_cache_response_acceptable(query: str, response: str) -> bool:
    _ = query
    cleaned = sanitize_model_output(response).strip()
    if not cleaned:
        return False
    if len(cleaned) == 1:
        return False
    return not detect_output_anomalies(response, cleaned)


def resolve_inference_timeout(
    *,
    base_timeout: int,
    intent: str | None,
    model_role: str | None,
    has_images: bool = False,
) -> int:
    """추론 타임아웃을 결정한다. 이미지·추론 모델·복잡 인텐트 시 확장된 타임아웃을 반환한다."""
    timeout = max(1, int(base_timeout))
    if has_images:
        return max(timeout, REASONING_TIMEOUT_SECONDS)
    role = (model_role or "").strip().lower()
    intent_name = (intent or "").strip().lower()
    if role in REASONING_MODEL_ROLES or intent_name in REASONING_INTENTS:
        return max(timeout, REASONING_TIMEOUT_SECONDS)
    return timeout
