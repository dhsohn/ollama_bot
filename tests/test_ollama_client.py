"""Ollama 클라이언트 테스트."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config import OllamaConfig
from core.ollama_client import OllamaClient, OllamaClientError, ModelNotFoundError


@pytest.fixture
def ollama_config() -> OllamaConfig:
    return OllamaConfig(
        host="http://localhost:11434",
        model="test-model",
        temperature=0.7,
        max_tokens=2048,
        system_prompt="Test prompt",
    )


@pytest.fixture
def ollama_client(ollama_config: OllamaConfig) -> OllamaClient:
    return OllamaClient(ollama_config)


class TestChat:
    @pytest.mark.asyncio
    async def test_chat_returns_response(self, ollama_client: OllamaClient) -> None:
        mock_response = MagicMock()
        mock_response.message.content = "Hello!"

        mock_async_client = AsyncMock()
        mock_async_client.chat = AsyncMock(return_value=mock_response)
        ollama_client._client = mock_async_client

        result = await ollama_client.chat(
            messages=[{"role": "user", "content": "Hi"}]
        )
        assert result == "Hello!"

    @pytest.mark.asyncio
    async def test_chat_timeout_retries(self, ollama_client: OllamaClient) -> None:
        mock_async_client = AsyncMock()
        mock_async_client.chat = AsyncMock(side_effect=asyncio.TimeoutError())
        ollama_client._client = mock_async_client

        with pytest.raises(OllamaClientError, match="failed after"):
            await ollama_client.chat(
                messages=[{"role": "user", "content": "Hi"}],
                timeout=1,
            )

        # 3번 시도 (초기 1회 + 재시도 2회)
        assert mock_async_client.chat.call_count == 3

    @pytest.mark.asyncio
    async def test_chat_uses_explicit_zero_options(self, ollama_client: OllamaClient) -> None:
        mock_response = MagicMock()
        mock_response.message.content = "Hello!"

        mock_async_client = AsyncMock()
        mock_async_client.chat = AsyncMock(return_value=mock_response)
        ollama_client._client = mock_async_client

        await ollama_client.chat(
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.0,
            max_tokens=0,
        )

        call_kwargs = mock_async_client.chat.call_args.kwargs
        assert call_kwargs["options"]["temperature"] == 0.0
        assert call_kwargs["options"]["num_predict"] == 0


class TestInitialize:
    @pytest.mark.asyncio
    async def test_initialize_success(self, ollama_client: OllamaClient) -> None:
        mock_model = MagicMock()
        mock_model.model = "test-model"
        mock_response = MagicMock()
        mock_response.models = [mock_model]

        mock_async_client = AsyncMock()
        mock_async_client.list = AsyncMock(return_value=mock_response)

        with patch("core.ollama_client.AsyncClient", return_value=mock_async_client):
            await ollama_client.initialize()

        assert ollama_client._client is not None

    @pytest.mark.asyncio
    async def test_initialize_missing_default_model_raises(
        self, ollama_client: OllamaClient
    ) -> None:
        mock_model = MagicMock()
        mock_model.model = "other-model"
        mock_response = MagicMock()
        mock_response.models = [mock_model]

        mock_async_client = AsyncMock()
        mock_async_client.list = AsyncMock(return_value=mock_response)

        with patch("core.ollama_client.AsyncClient", return_value=mock_async_client):
            with pytest.raises(ModelNotFoundError, match="Default model"):
                await ollama_client.initialize()


class TestChatStream:
    @pytest.mark.asyncio
    async def test_chat_stream_returns_chunks(self, ollama_client: OllamaClient) -> None:
        class _ChunkStream:
            def __init__(self) -> None:
                self._chunks = ["A", "B"]

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self._chunks:
                    raise StopAsyncIteration
                content = self._chunks.pop(0)
                chunk = MagicMock()
                chunk.message.content = content
                return chunk

        mock_async_client = AsyncMock()
        mock_async_client.chat = AsyncMock(return_value=_ChunkStream())
        ollama_client._client = mock_async_client

        chunks = []
        async for chunk in ollama_client.chat_stream(
            messages=[{"role": "user", "content": "Hi"}],
            timeout=1,
        ):
            chunks.append(chunk)

        assert chunks == ["A", "B"]

    @pytest.mark.asyncio
    async def test_chat_stream_timeout_raises(self, ollama_client: OllamaClient) -> None:
        class _SlowStream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                await asyncio.sleep(0.2)
                chunk = MagicMock()
                chunk.message.content = "slow"
                return chunk

        mock_async_client = AsyncMock()
        mock_async_client.chat = AsyncMock(return_value=_SlowStream())
        ollama_client._client = mock_async_client

        with pytest.raises(OllamaClientError, match="streaming request failed"):
            async for _ in ollama_client.chat_stream(
                messages=[{"role": "user", "content": "Hi"}],
                timeout=0.01,
            ):
                pass

    @pytest.mark.asyncio
    async def test_chat_stream_uses_explicit_zero_options(self, ollama_client: OllamaClient) -> None:
        class _ChunkStream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        mock_async_client = AsyncMock()
        mock_async_client.chat = AsyncMock(return_value=_ChunkStream())
        ollama_client._client = mock_async_client

        async for _ in ollama_client.chat_stream(
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.0,
            max_tokens=0,
            timeout=1,
        ):
            pass

        call_kwargs = mock_async_client.chat.call_args.kwargs
        assert call_kwargs["options"]["temperature"] == 0.0
        assert call_kwargs["options"]["num_predict"] == 0


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check_ok(self, ollama_client: OllamaClient) -> None:
        mock_model = MagicMock()
        mock_model.model = "test-model"

        mock_response = MagicMock()
        mock_response.models = [mock_model]

        mock_async_client = AsyncMock()
        mock_async_client.list = AsyncMock(return_value=mock_response)
        ollama_client._client = mock_async_client

        result = await ollama_client.health_check()
        assert result["status"] == "ok"
        assert result["models_count"] == 1

    @pytest.mark.asyncio
    async def test_health_check_error(self, ollama_client: OllamaClient) -> None:
        mock_async_client = AsyncMock()
        mock_async_client.list = AsyncMock(side_effect=ConnectionError("refused"))
        ollama_client._client = mock_async_client

        result = await ollama_client.health_check()
        assert result["status"] == "error"


class TestListModels:
    @pytest.mark.asyncio
    async def test_list_models(self, ollama_client: OllamaClient) -> None:
        mock_model = MagicMock()
        mock_model.model = "test-model"
        mock_model.size = 1024 * 1024 * 100
        mock_model.modified_at = None

        mock_response = MagicMock()
        mock_response.models = [mock_model]

        mock_async_client = AsyncMock()
        mock_async_client.list = AsyncMock(return_value=mock_response)
        ollama_client._client = mock_async_client

        models = await ollama_client.list_models()
        assert len(models) == 1
        assert models[0]["name"] == "test-model"
