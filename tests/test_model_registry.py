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
    async def test_initialize_marks_unavailable_models(self, config):
        client = AsyncMock()
        client.check_model_availability = AsyncMock(return_value={
            "test-embed": False,
            "test-rerank": False,
        })
        reg = ModelRegistry(config, client)
        await reg.initialize()
        assert reg.is_available("default")
        assert not reg.is_available("embedding")
        assert not reg.is_available("reranker")

    @pytest.mark.asyncio
    async def test_initialize_checks_only_retrieval_models(self, registry, mock_client):
        mock_client.check_model_availability.assert_awaited_once_with(
            ["test-embed", "test-rerank"],
        )
