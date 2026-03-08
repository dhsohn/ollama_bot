"""engine_status 모듈 테스트."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.engine_status import (
    build_optimization_tier_details,
    build_rag_tier_detail,
    format_uptime,
    get_status,
    make_tier_detail,
    manual_tier_detail,
)


def _make_engine(**overrides):
    """테스트용 Engine mock을 생성한다."""
    engine = MagicMock()
    engine._start_time = time.monotonic() - 120
    engine._degraded_since = {}
    engine._llm_client = AsyncMock()
    engine._llm_client.health_check = AsyncMock(return_value={"status": "ok"})
    engine._llm_client.default_model = "test-model"
    engine._skills = MagicMock()
    engine._skills.skill_count = 3
    engine._instant_responder = None
    engine._semantic_cache = None
    engine._intent_router = None
    engine._context_compressor = None
    engine._rag_pipeline = None
    engine._config = MagicMock()
    engine._config.instant_responder.enabled = False
    engine._config.semantic_cache.enabled = False
    engine._config.intent_router.enabled = False
    engine._config.context_compressor.enabled = False
    engine._config.rag.enabled = False
    engine._config.rag.rerank_enabled = False
    for key, value in overrides.items():
        setattr(engine, key, value)
    return engine


class TestFormatUptime:
    def test_seconds_only(self) -> None:
        assert format_uptime(45) == "45초"

    def test_minutes_and_seconds(self) -> None:
        assert format_uptime(125) == "2분 5초"

    def test_hours_minutes_seconds(self) -> None:
        assert format_uptime(3661) == "1시간 1분 1초"

    def test_zero(self) -> None:
        assert format_uptime(0) == "0초"


class TestManualTierDetail:
    def test_not_configured(self) -> None:
        engine = _make_engine()
        engine._degraded_since["test"] = time.monotonic()
        result = manual_tier_detail(engine, name="test", configured=False, enabled=False, degraded=False)
        assert result["configured"] is False
        assert result["degraded"] is False
        assert "test" not in engine._degraded_since

    def test_degraded(self) -> None:
        engine = _make_engine()
        result = manual_tier_detail(engine, name="comp", configured=True, enabled=True, degraded=True, reason="broken")
        assert result["degraded"] is True
        assert result["reason"] == "broken"
        assert result["degraded_for_seconds"] is not None
        assert "comp" in engine._degraded_since

    def test_healthy(self) -> None:
        engine = _make_engine()
        engine._degraded_since["comp"] = time.monotonic() - 10
        result = manual_tier_detail(engine, name="comp", configured=True, enabled=True, degraded=False)
        assert result["degraded"] is False
        assert "comp" not in engine._degraded_since


class TestMakeTierDetail:
    def test_not_configured(self) -> None:
        engine = _make_engine()
        result = make_tier_detail(engine, name="x", configured=False, instance=None, unavailable_reason="init_failed")
        assert result["configured"] is False
        assert result["degraded"] is False

    def test_instance_none_is_degraded(self) -> None:
        engine = _make_engine()
        result = make_tier_detail(engine, name="x", configured=True, instance=None, unavailable_reason="init_failed")
        assert result["degraded"] is True
        assert result["reason"] == "init_failed"

    def test_enabled_attr_false_is_degraded(self) -> None:
        engine = _make_engine()
        instance = SimpleNamespace(enabled=False)
        result = make_tier_detail(
            engine, name="x", configured=True, instance=instance,
            unavailable_reason="init_failed", enabled_attr="enabled", disabled_reason="encoder_unavailable",
        )
        assert result["degraded"] is True
        assert result["reason"] == "encoder_unavailable"

    def test_healthy_instance(self) -> None:
        engine = _make_engine()
        instance = SimpleNamespace(enabled=True)
        result = make_tier_detail(
            engine, name="x", configured=True, instance=instance,
            unavailable_reason="init_failed", enabled_attr="enabled",
        )
        assert result["degraded"] is False
        assert result["enabled"] is True


class TestBuildRagTierDetail:
    def test_rag_disabled(self) -> None:
        engine = _make_engine()
        engine._config.rag.enabled = False
        result = build_rag_tier_detail(engine)
        assert result["configured"] is False

    def test_rag_pipeline_none_is_degraded(self) -> None:
        engine = _make_engine()
        engine._config.rag.enabled = True
        engine._rag_pipeline = None
        result = build_rag_tier_detail(engine)
        assert result["degraded"] is True
        assert result["reason"] == "init_failed"

    def test_rag_reranker_missing_is_degraded(self) -> None:
        engine = _make_engine()
        engine._config.rag.enabled = True
        engine._config.rag.rerank_enabled = True
        engine._rag_pipeline = SimpleNamespace(has_reranker=False)
        result = build_rag_tier_detail(engine)
        assert result["degraded"] is True
        assert result["reason"] == "reranker_unavailable"

    def test_rag_healthy(self) -> None:
        engine = _make_engine()
        engine._config.rag.enabled = True
        engine._config.rag.rerank_enabled = True
        engine._rag_pipeline = SimpleNamespace(has_reranker=True)
        result = build_rag_tier_detail(engine)
        assert result["degraded"] is False
        assert result["enabled"] is True


class TestBuildOptimizationTierDetails:
    def test_all_disabled(self) -> None:
        engine = _make_engine()
        details = build_optimization_tier_details(engine)
        assert "instant_responder" in details
        assert "semantic_cache" in details
        assert "intent_router" in details
        assert "context_compressor" in details
        assert "rag_pipeline" in details
        for detail in details.values():
            assert detail["configured"] is False


class TestGetStatus:
    @pytest.mark.asyncio
    async def test_returns_status_dict(self) -> None:
        engine = _make_engine()
        engine._instant_responder = SimpleNamespace(rules_count=5)
        engine._config.instant_responder.enabled = True
        cache_mock = AsyncMock()
        cache_mock.get_stats = AsyncMock(return_value={"hit_rate": 0.5})
        engine._semantic_cache = cache_mock
        engine._config.semantic_cache.enabled = True
        engine._intent_router = SimpleNamespace(routes_count=3, enabled=True)
        engine._config.intent_router.enabled = True

        status = await get_status(engine)

        assert "uptime_seconds" in status
        assert "uptime_human" in status
        assert status["skills_loaded"] == 3
        assert status["instant_responder_rules"] == 5
        assert status["semantic_cache"] == {"hit_rate": 0.5}
        assert status["intent_router_routes"] == 3
