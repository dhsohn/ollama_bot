from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.engine import Engine


class EngineStatusService:
    """Build status snapshots and degradation details for an Engine."""

    def __init__(
        self,
        engine: Engine,
        *,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._engine = engine
        self._monotonic_fn = monotonic_fn

    async def get_status(self) -> dict[str, Any]:
        """Return overall system status."""
        engine = self._engine
        llm_health = await engine._llm_client.health_check()
        uptime_seconds = self._monotonic_fn() - engine._start_time

        status = {
            "uptime_seconds": int(uptime_seconds),
            "uptime_human": self.format_uptime(uptime_seconds),
            "llm": llm_health,
            "skills_loaded": engine._skills.skill_count,
            "current_model": engine._llm_client.default_model,
        }

        tier_details = self.build_optimization_tier_details()
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

    def build_optimization_tier_details(self) -> dict[str, dict[str, Any]]:
        """Build enabled/degraded status for optimization components."""
        engine = self._engine
        return {
            "instant_responder": self.make_tier_detail(
                name="instant_responder",
                configured=engine._config.instant_responder.enabled,
                instance=engine._instant_responder,
                unavailable_reason="init_failed",
            ),
            "semantic_cache": self.make_tier_detail(
                name="semantic_cache",
                configured=engine._config.semantic_cache.enabled,
                instance=engine._semantic_cache,
                enabled_attr="enabled",
                unavailable_reason="init_failed",
                disabled_reason="encoder_unavailable",
            ),
            "intent_router": self.make_tier_detail(
                name="intent_router",
                configured=engine._config.intent_router.enabled,
                instance=engine._intent_router,
                enabled_attr="enabled",
                unavailable_reason="init_failed",
                disabled_reason="router_disabled",
            ),
            "context_compressor": self.make_tier_detail(
                name="context_compressor",
                configured=engine._config.context_compressor.enabled,
                instance=engine._context_compressor,
                unavailable_reason="init_failed",
            ),
            "rag_pipeline": self.build_rag_tier_detail(),
        }

    def build_rag_tier_detail(self) -> dict[str, Any]:
        """Compute the degraded state for the RAG pipeline."""
        engine = self._engine
        name = "rag_pipeline"
        if not engine._config.rag.enabled:
            return self.manual_tier_detail(
                name=name,
                configured=False,
                enabled=False,
                degraded=False,
            )
        if engine._rag_pipeline is None:
            return self.manual_tier_detail(
                name=name,
                configured=True,
                enabled=False,
                degraded=True,
                reason="init_failed",
            )
        has_reranker = bool(getattr(engine._rag_pipeline, "has_reranker", False))
        if engine._config.rag.rerank_enabled and not has_reranker:
            return self.manual_tier_detail(
                name=name,
                configured=True,
                enabled=True,
                degraded=True,
                reason="reranker_unavailable",
            )
        return self.manual_tier_detail(
            name=name,
            configured=True,
            enabled=True,
            degraded=False,
        )

    def manual_tier_detail(
        self,
        *,
        name: str,
        configured: bool,
        enabled: bool,
        degraded: bool,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Build tier detail for components with dynamic status."""
        engine = self._engine
        now = self._monotonic_fn()
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
        self,
        *,
        name: str,
        configured: bool,
        instance: Any,
        unavailable_reason: str,
        enabled_attr: str | None = None,
        disabled_reason: str = "disabled",
    ) -> dict[str, Any]:
        """Compute status for a single component."""
        engine = self._engine
        now = self._monotonic_fn()
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

    @staticmethod
    def format_uptime(seconds: float) -> str:
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}시간 {minutes}분 {secs}초"
        if minutes > 0:
            return f"{minutes}분 {secs}초"
        return f"{secs}초"


def _service(
    engine: Engine,
    *,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> EngineStatusService:
    existing = getattr(engine, "_status_service", None)
    if isinstance(existing, EngineStatusService):
        return existing
    return EngineStatusService(engine, monotonic_fn=monotonic_fn)


async def get_status(engine: Engine) -> dict[str, Any]:
    return await _service(engine).get_status()


def build_optimization_tier_details(engine: Engine) -> dict[str, dict[str, Any]]:
    return _service(engine).build_optimization_tier_details()


def build_rag_tier_detail(engine: Engine) -> dict[str, Any]:
    return _service(engine).build_rag_tier_detail()


def manual_tier_detail(
    engine: Engine,
    *,
    name: str,
    configured: bool,
    enabled: bool,
    degraded: bool,
    reason: str | None = None,
) -> dict[str, Any]:
    return _service(engine).manual_tier_detail(
        name=name,
        configured=configured,
        enabled=enabled,
        degraded=degraded,
        reason=reason,
    )


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
    return _service(engine).make_tier_detail(
        name=name,
        configured=configured,
        instance=instance,
        unavailable_reason=unavailable_reason,
        enabled_attr=enabled_attr,
        disabled_reason=disabled_reason,
    )


def format_uptime(seconds: float) -> str:
    return EngineStatusService.format_uptime(seconds)
