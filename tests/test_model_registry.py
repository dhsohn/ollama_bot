"""ModelRegistry 단위 테스트."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from core.config import ModelRegistryConfig
from core.model_registry import ModelRegistry


@pytest.fixture
def config():
    return ModelRegistryConfig(
        embedding_model="test-embed",
        reranker_model="test-rerank",
        vision_model="test-vision",
        low_cost_model="test-cheap",
        reasoning_model="test-reason",
        coding_model="test-code",
    )


@pytest.fixture
def mock_client():
    client = AsyncMock()
    client.check_model_availability = AsyncMock(return_value={
        "test-embed": True,
        "test-rerank": True,
        "test-vision": True,
        "test-cheap": True,
        "test-reason": True,
        "test-code": True,
    })
    return client


@pytest_asyncio.fixture
async def registry(config, mock_client):
    reg = ModelRegistry(config, mock_client)
    await reg.initialize()
    return reg


class TestModelRegistry:

    @pytest.mark.asyncio
    async def test_initialize_all_available(self, registry):
        assert registry.is_available("embedding")
        assert registry.is_available("vision")
        assert registry.is_available("low_cost")
        assert registry.is_available("reasoning")
        assert registry.is_available("coding")

    @pytest.mark.asyncio
    async def test_get_model_available(self, registry):
        assert registry.get_model("vision") == "test-vision"
        assert registry.get_model("low_cost") == "test-cheap"

    @pytest.mark.asyncio
    async def test_get_model_unavailable(self, config):
        client = AsyncMock()
        client.check_model_availability = AsyncMock(return_value={
            "test-embed": False,
            "test-rerank": False,
            "test-vision": False,
            "test-cheap": False,
            "test-reason": False,
            "test-code": False,
        })
        reg = ModelRegistry(config, client)
        await reg.initialize()
        assert reg.get_model("vision") is None
        assert reg.get_model("low_cost") is None

    @pytest.mark.asyncio
    async def test_fallback_chain(self, config):
        client = AsyncMock()
        client.check_model_availability = AsyncMock(return_value={
            "test-embed": True,
            "test-rerank": False,
            "test-vision": False,
            "test-cheap": False,
            "test-reason": True,
            "test-code": False,
        })
        reg = ModelRegistry(config, client)
        await reg.initialize()

        # low_cost 미가용 → reasoning 폴백
        assert reg.get_fallback("low_cost") == "test-reason"

        # coding 미가용 → reasoning 폴백
        assert reg.get_fallback("coding") == "test-reason"

        # vision 미가용 → 폴백 없음
        assert reg.get_fallback("vision") is None

    @pytest.mark.asyncio
    async def test_resolve_model_with_fallback(self, config):
        client = AsyncMock()
        client.check_model_availability = AsyncMock(return_value={
            "test-embed": True,
            "test-rerank": False,
            "test-vision": True,
            "test-cheap": False,
            "test-reason": True,
            "test-code": False,
        })
        reg = ModelRegistry(config, client)
        await reg.initialize()

        model, role, fallback = reg.resolve_model("low_cost")
        assert model == "test-reason"
        assert role == "reasoning"
        assert fallback is True

        model, role, fallback = reg.resolve_model("vision")
        assert model == "test-vision"
        assert role == "vision"
        assert fallback is False

    @pytest.mark.asyncio
    async def test_resolve_model_no_fallback_raises(self, config):
        client = AsyncMock()
        client.check_model_availability = AsyncMock(return_value={
            "test-embed": False,
            "test-rerank": False,
            "test-vision": False,
            "test-cheap": False,
            "test-reason": False,
            "test-code": False,
        })
        reg = ModelRegistry(config, client)
        await reg.initialize()

        with pytest.raises(ValueError, match="No available model"):
            reg.resolve_model("vision")

    @pytest.mark.asyncio
    async def test_get_status(self, registry):
        status = registry.get_status()
        assert "embedding" in status
        assert "vision" in status
        assert status["embedding"]["available"] is True

    @pytest.mark.asyncio
    async def test_refresh_availability(self, registry, mock_client):
        mock_client.check_model_availability.return_value = {
            "test-embed": True,
            "test-rerank": False,
            "test-vision": True,
            "test-cheap": True,
            "test-reason": True,
            "test-code": True,
        }
        await registry.refresh_availability()
        assert not registry.is_available("reranker")
