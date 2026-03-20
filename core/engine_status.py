from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.engine import Engine


async def get_status(engine: Engine) -> dict[str, Any]:
    """Return overall system status."""
    llm_health = await engine._llm_client.health_check()
    uptime_seconds = time.monotonic() - engine._start_time

    status = {
        "uptime_seconds": int(uptime_seconds),
        "uptime_human": format_uptime(uptime_seconds),
        "llm": llm_health,
        "skills_loaded": engine._skills.skill_count,
        "current_model": engine._llm_client.default_model,
    }

    tier_details = build_optimization_tier_details(engine)
    status["optimization_tiers"] = {
        name: bool(detail["enabled"]) for name, detail in tier_details.items()
    }
    status["optimization_tier_details"] = tier_details
    degraded = {
        name: detail for name, detail in tier_details.items() if detail["degraded"]
    }
    status["degraded_components"] = degraded
    status["degraded"] = bool(degraded)

    if engine._instant_responder is not None:
        status["instant_responder_rules"] = engine._instant_responder.rules_count
    if engine._semantic_cache is not None:
        status["semantic_cache"] = await engine._semantic_cache.get_stats()
    if engine._intent_router is not None:
        status["intent_router_routes"] = engine._intent_router.routes_count

    return status


def build_optimization_tier_details(engine: Engine) -> dict[str, dict[str, Any]]:
    """Build enabled/degraded status for optimization components."""
    return {
        "instant_responder": make_tier_detail(
            engine,
            name="instant_responder",
            configured=engine._config.instant_responder.enabled,
            instance=engine._instant_responder,
            unavailable_reason="init_failed",
        ),
        "semantic_cache": make_tier_detail(
            engine,
            name="semantic_cache",
            configured=engine._config.semantic_cache.enabled,
            instance=engine._semantic_cache,
            enabled_attr="enabled",
            unavailable_reason="init_failed",
            disabled_reason="encoder_unavailable",
        ),
        "intent_router": make_tier_detail(
            engine,
            name="intent_router",
            configured=engine._config.intent_router.enabled,
            instance=engine._intent_router,
            enabled_attr="enabled",
            unavailable_reason="init_failed",
            disabled_reason="router_disabled",
        ),
        "context_compressor": make_tier_detail(
            engine,
            name="context_compressor",
            configured=engine._config.context_compressor.enabled,
            instance=engine._context_compressor,
            unavailable_reason="init_failed",
        ),
        "rag_pipeline": build_rag_tier_detail(engine),
    }


def build_rag_tier_detail(engine: Engine) -> dict[str, Any]:
    """Compute the degraded state for the RAG pipeline."""
    name = "rag_pipeline"
    if not engine._config.rag.enabled:
        return manual_tier_detail(
            engine,
            name=name,
            configured=False,
            enabled=False,
            degraded=False,
        )
    if engine._rag_pipeline is None:
        return manual_tier_detail(
            engine,
            name=name,
            configured=True,
            enabled=False,
            degraded=True,
            reason="init_failed",
        )
    has_reranker = bool(getattr(engine._rag_pipeline, "has_reranker", False))
    if engine._config.rag.rerank_enabled and not has_reranker:
        return manual_tier_detail(
            engine,
            name=name,
            configured=True,
            enabled=True,
            degraded=True,
            reason="reranker_unavailable",
        )
    return manual_tier_detail(
        engine,
        name=name,
        configured=True,
        enabled=True,
        degraded=False,
    )


def manual_tier_detail(
    engine: Engine,
    *,
    name: str,
    configured: bool,
    enabled: bool,
    degraded: bool,
    reason: str | None = None,
) -> dict[str, Any]:
    """Build tier detail for components with dynamic status."""
    now = time.monotonic()
    if not configured:
        engine._degraded_since.pop(name, None)
        return {
            "configured": False,
            "enabled": False,
            "degraded": False,
            "reason": None,
            "degraded_for_seconds": None,
        }
    if degraded:
        since = engine._degraded_since.setdefault(name, now)
        return {
            "configured": True,
            "enabled": enabled,
            "degraded": True,
            "reason": reason or "degraded",
            "degraded_for_seconds": int(now - since),
        }
    engine._degraded_since.pop(name, None)
    return {
        "configured": True,
        "enabled": enabled,
        "degraded": False,
        "reason": None,
        "degraded_for_seconds": None,
    }


def make_tier_detail(
    engine: Engine,
    *,
    name: str,
    configured: bool,
    instance: Any,
    unavailable_reason: str,
    enabled_attr: str | None = None,
    disabled_reason: str = "disabled",
) -> dict[str, Any]:
    """Compute status for a single component."""
    now = time.monotonic()
    if not configured:
        engine._degraded_since.pop(name, None)
        return {
            "configured": False,
            "enabled": False,
            "degraded": False,
            "reason": None,
            "degraded_for_seconds": None,
        }

    if instance is None:
        since = engine._degraded_since.setdefault(name, now)
        return {
            "configured": True,
            "enabled": False,
            "degraded": True,
            "reason": unavailable_reason,
            "degraded_for_seconds": int(now - since),
        }

    if enabled_attr is not None and not bool(getattr(instance, enabled_attr, True)):
        since = engine._degraded_since.setdefault(name, now)
        return {
            "configured": True,
            "enabled": False,
            "degraded": True,
            "reason": disabled_reason,
            "degraded_for_seconds": int(now - since),
        }

    engine._degraded_since.pop(name, None)
    return {
        "configured": True,
        "enabled": True,
        "degraded": False,
        "reason": None,
        "degraded_for_seconds": None,
    }


def format_uptime(seconds: float) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}시간 {minutes}분 {secs}초"
    if minutes > 0:
        return f"{minutes}분 {secs}초"
    return f"{secs}초"
