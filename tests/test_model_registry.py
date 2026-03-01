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
        default_model="test-default",
        embedding_model="test-embed",
        reranker_model="test-rerank",
    )


@pytest.fixture
def mock_client():
    client = AsyncMock()
    client.check_model_availability = AsyncMock(return_value={
        "test-embed": True,
        "test-rerank": True,
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
        assert registry.is_available("default")
        assert registry.is_available("embedding")
        assert registry.is_available("reranker")

    @pytest.mark.asyncio
    async def test_get_model_available(self, registry):
        assert registry.get_model("default") == "test-default"
        assert registry.get_model("embedding") == "test-embed"
        assert registry.get_model("reranker") == "test-rerank"

    @pytest.mark.asyncio
    async def test_get_model_unavailable(self, config):
        client = AsyncMock()
        client.check_model_availability = AsyncMock(return_value={
            "test-embed": False,
            "test-rerank": False,
        })
        reg = ModelRegistry(config, client)
        await reg.initialize()
        assert reg.get_model("embedding") is None
        assert reg.get_model("reranker") is None
        # default는 항상 가용
        assert reg.get_model("default") == "test-default"

    @pytest.mark.asyncio
    async def test_resolve_model(self, registry):
        model, role, fallback = registry.resolve_model("default")
        assert model == "test-default"
        assert role == "default"
        assert fallback is False

        model, role, fallback = registry.resolve_model("embedding")
        assert model == "test-embed"
        assert role == "embedding"
        assert fallback is False

    @pytest.mark.asyncio
    async def test_resolve_model_unavailable_raises(self, config):
        client = AsyncMock()
        client.check_model_availability = AsyncMock(return_value={
            "test-embed": False,
            "test-rerank": False,
        })
        reg = ModelRegistry(config, client)
        await reg.initialize()

        with pytest.raises(ValueError, match="No available model"):
            reg.resolve_model("embedding")

    @pytest.mark.asyncio
    async def test_get_model_name(self, registry):
        assert registry.get_model_name("default") == "test-default"
        assert registry.get_model_name("embedding") == "test-embed"
        assert registry.get_model_name("reranker") == "test-rerank"

    @pytest.mark.asyncio
    async def test_get_model_name_unknown_role(self, registry):
        with pytest.raises(ValueError, match="Unknown model role"):
            registry.get_model_name("nonexistent")

    @pytest.mark.asyncio
    async def test_get_status(self, registry):
        status = registry.get_status()
        assert "default" in status
        assert "embedding" in status
        assert "reranker" in status
        assert status["default"]["available"] is True
        assert status["embedding"]["name"] == "test-embed"

    @pytest.mark.asyncio
    async def test_refresh_availability(self, registry, mock_client):
        mock_client.check_model_availability.return_value = {
            "test-embed": True,
            "test-rerank": False,
        }
        await registry.refresh_availability()
        assert registry.is_available("embedding")
        assert not registry.is_available("reranker")
        # default는 refresh와 무관하게 항상 가용
        assert registry.is_available("default")
