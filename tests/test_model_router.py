"""ModelRouter 단위 테스트."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio

from core.config import ModelRoutingConfig
from core.model_router import ModelRouter, RoutingDecision


@pytest.fixture
def config():
    return ModelRoutingConfig(
        enabled=True,
        anchors_path="config/routing_anchors.yaml",
        threshold=0.45,
        margin=0.05,
        embedding_cache_size=100,
    )


@pytest.fixture
def mock_registry():
    reg = MagicMock()
    available = {
        "embedding",
        "reranker",
        "vision",
        "low_cost",
        "reasoning",
        "coding",
    }

    def _is_available(role: str) -> bool:
        return role in available

    def _get_model(role: str) -> str | None:
        if role not in available:
            return None
        return f"model-{role}"

    def _resolve(role: str) -> tuple[str, str, bool]:
        model = _get_model(role)
        if model is not None:
            return model, role, False
        if role == "low_cost":
            fallback = _get_model("reasoning")
            if fallback is not None:
                return fallback, "reasoning", True
        raise ValueError("no model")

    reg.is_available.side_effect = _is_available
    reg.get_model.side_effect = _get_model
    reg.resolve_model.side_effect = _resolve
    return reg


@pytest.fixture
def mock_client():
    client = AsyncMock()
    # 임베딩: 10차원 랜덤 벡터 반환
    client.embed = AsyncMock(
        side_effect=lambda texts, model=None: [
            np.random.randn(10).tolist() for _ in texts
        ]
    )
    client.default_model = "default-model"
    client.chat = AsyncMock()
    client.chat.return_value = MagicMock(content="CHEAP")
    return client


class TestModelRouterCodeDetection:

    @pytest.fixture
    def router(self, config, mock_registry, mock_client):
        return ModelRouter(config, mock_registry, mock_client, "embed-model")

    @pytest.mark.asyncio
    async def test_image_routes_to_vision(self, router):
        decision = await router.route("describe this", images=[b"fake_img"])
        assert decision.selected_role == "vision"
        assert decision.trigger == "image"

    @pytest.mark.asyncio
    async def test_code_fence_routes_to_coding(self, router):
        text = "```python\ndef hello():\n    pass\n```\n설명해줘"
        decision = await router.route(text)
        assert decision.selected_role == "coding"
        assert decision.trigger == "code_regex"

    @pytest.mark.asyncio
    async def test_stack_trace_routes_to_coding(self, router):
        text = 'Traceback (most recent call last):\n  File "main.py", line 10'
        decision = await router.route(text)
        assert decision.selected_role == "coding"
        assert decision.trigger == "code_regex"

    @pytest.mark.asyncio
    async def test_error_type_routes_to_coding(self, router):
        text = "TypeError: unsupported operand type(s) for +: 'int' and 'str'"
        decision = await router.route(text)
        assert decision.selected_role == "coding"
        assert decision.trigger == "code_regex"

    @pytest.mark.asyncio
    async def test_file_extension_routes_to_coding(self, router):
        text = "main.py 에서 import 에러가 나요"
        decision = await router.route(text)
        assert decision.selected_role == "coding"
        assert decision.trigger == "code_regex"

    @pytest.mark.asyncio
    async def test_code_keywords_routes_to_coding(self, router):
        text = "이 함수를 리팩토링하고 테스트 추가해줘"
        decision = await router.route(text)
        assert decision.selected_role == "coding"
        assert decision.trigger == "code_regex"


class TestModelRouterSemanticFallback:

    @pytest.mark.asyncio
    async def test_fallback_to_classifier(self, config, mock_registry, mock_client):
        """임베딩 불확실 시 LLM 분류기 폴백."""
        router = ModelRouter(config, mock_registry, mock_client, "embed-model")
        # 초기화 안 함 → 시맨틱 라우팅 불가 → 분류기 사용
        decision = await router.route("안녕하세요")
        assert decision.trigger == "classifier"
        assert decision.classifier_used is True

    @pytest.mark.asyncio
    async def test_fallback_to_low_cost(self, config, mock_registry, mock_client):
        """분류기도 실패 시 low_cost 폴백."""
        mock_client.chat.side_effect = Exception("LLM down")
        router = ModelRouter(config, mock_registry, mock_client, "embed-model")
        decision = await router.route("안녕하세요")
        assert decision.selected_role == "low_cost"
        assert decision.trigger == "fallback"

    @pytest.mark.asyncio
    async def test_no_low_cost_classifier_forces_reasoning(
        self, config, mock_registry, mock_client,
    ):
        """low_cost 분류 모델이 없으면 reasoning으로 강제 라우팅한다."""
        mock_registry.is_available.side_effect = lambda role: role in {
            "embedding", "reasoning", "coding", "vision", "reranker",
        }
        mock_registry.get_model.side_effect = (
            lambda role: f"model-{role}" if role in {"reasoning", "coding", "vision"} else None
        )
        mock_registry.resolve_model.side_effect = (
            lambda role: ("model-reasoning", "reasoning", True)
            if role in {"reasoning", "low_cost"}
            else (f"model-{role}", role, False)
        )

        router = ModelRouter(config, mock_registry, mock_client, "embed-model")
        decision = await router.route("이 문제를 단계별로 분석해줘")
        assert decision.selected_role == "reasoning"
        assert decision.trigger == "classifier_unavailable"
        assert "low_cost_classifier_unavailable" in decision.degradation_reasons

    @pytest.mark.asyncio
    async def test_margin_uses_second_highest_score_not_minimum(
        self, config, mock_registry, mock_client,
    ):
        """마진 계산은 최저점이 아니라 2등 점수를 사용해야 한다."""
        router = ModelRouter(config, mock_registry, mock_client, "embed-model")
        router._initialized = True
        router._anchor_embeddings = {"cheap": np.ones((1, 10), dtype=np.float32)}

        with patch.object(
            router,
            "_get_embedding",
            new=AsyncMock(return_value=np.ones(10, dtype=np.float32)),
        ), patch.object(
            router,
            "_compute_anchor_scores",
            return_value={"cheap": 0.60, "reasoning": 0.56, "other": 0.05},
        ):
            decision = await router.route("일반 질문")

        # 0.60 - 0.56 = 0.04 < margin(0.05) 이므로 semantic 확정이 아닌 classifier로 내려가야 함.
        assert decision.trigger == "classifier"
        assert decision.classifier_used is True


class TestRoutingDecision:

    def test_to_dict(self):
        d = RoutingDecision(
            selected_model="model-a",
            selected_role="vision",
            trigger="image",
            confidence=0.95,
            anchor_scores={"cheap": 0.3, "reasoning": 0.1},
        )
        result = d.to_dict()
        assert result["selected_model"] == "model-a"
        assert result["trigger"] == "image"
        assert result["confidence"] == 0.95
