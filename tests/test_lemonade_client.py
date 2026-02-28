"""Lemonade 클라이언트 테스트."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

from core.config import LemonadeConfig, OllamaConfig
from core.lemonade_client import LemonadeClient, LemonadeClientError
from core.llm_types import ChatStreamState


@pytest.fixture
def lemonade_config() -> LemonadeConfig:
    return LemonadeConfig(
        host="http://localhost:8000",
        model="test-model",
        base_path="/api/v1",
        timeout_seconds=10,
    )


@pytest.fixture
def ollama_fallback() -> OllamaConfig:
    return OllamaConfig(
        host="http://localhost:11434",
        model="test-model",
        temperature=0.7,
        max_tokens=512,
        system_prompt="fallback prompt",
    )


class TestInitialize:
    @pytest.mark.asyncio
    async def test_initialize_success(
        self,
        lemonade_config: LemonadeConfig,
        ollama_fallback: OllamaConfig,
    ) -> None:
        def _handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/v1/models"
            return httpx.Response(
                200,
                json={"data": [{"id": "test-model"}]},
            )

        transport = httpx.MockTransport(_handler)
        mock_client = httpx.AsyncClient(
            base_url=lemonade_config.host,
            transport=transport,
        )
        client = LemonadeClient(lemonade_config, fallback_ollama=ollama_fallback)

        with patch("core.lemonade_client.httpx.AsyncClient", return_value=mock_client):
            await client.initialize()
        try:
            assert client.default_model == "test-model"
            assert client.system_prompt == "fallback prompt"
        finally:
            await client.close()


class TestChat:
    @pytest.mark.asyncio
    async def test_chat_returns_content_and_usage(
        self,
        lemonade_config: LemonadeConfig,
        ollama_fallback: OllamaConfig,
    ) -> None:
        captured_payload: dict | None = None

        def _handler(request: httpx.Request) -> httpx.Response:
            nonlocal captured_payload
            if request.url.path == "/api/v1/chat/completions":
                captured_payload = json.loads(request.content.decode("utf-8"))
                return httpx.Response(
                    200,
                    json={
                        "choices": [{"message": {"content": "hello"}}],
                        "usage": {
                            "prompt_tokens": 11,
                            "completion_tokens": 22,
                            "total_tokens": 33,
                        },
                    },
                )
            return httpx.Response(404, json={"error": "not found"})

        client = LemonadeClient(lemonade_config, fallback_ollama=ollama_fallback)
        client._client = httpx.AsyncClient(
            base_url=lemonade_config.host,
            transport=httpx.MockTransport(_handler),
        )

        try:
            response = await client.chat(
                messages=[{"role": "user", "content": "hi"}],
                response_format="json",
            )
            assert response.content == "hello"
            assert response.usage is not None
            assert response.usage.prompt_eval_count == 11
            assert response.usage.eval_count == 22
            assert response.usage.total_duration == 0
            assert captured_payload is not None
            assert captured_payload["response_format"] == {"type": "json_object"}
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_chat_schema_response_format_is_downgraded(
        self,
        lemonade_config: LemonadeConfig,
        ollama_fallback: OllamaConfig,
    ) -> None:
        captured_payload: dict | None = None

        def _handler(request: httpx.Request) -> httpx.Response:
            nonlocal captured_payload
            if request.url.path == "/api/v1/chat/completions":
                captured_payload = json.loads(request.content.decode("utf-8"))
                return httpx.Response(
                    200,
                    json={"choices": [{"message": {"content": "{\"ok\":true}"}}]},
                )
            return httpx.Response(404, json={"error": "not found"})

        client = LemonadeClient(lemonade_config, fallback_ollama=ollama_fallback)
        client._client = httpx.AsyncClient(
            base_url=lemonade_config.host,
            transport=httpx.MockTransport(_handler),
        )

        schema = {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        }
        try:
            await client.chat(
                messages=[{"role": "user", "content": "json으로 응답해"}],
                response_format=schema,
            )
            assert captured_payload is not None
            assert captured_payload["response_format"] == {"type": "json_object"}
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_chat_does_not_fallback_to_reasoning_content(
        self,
        lemonade_config: LemonadeConfig,
        ollama_fallback: OllamaConfig,
    ) -> None:
        def _handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/chat/completions":
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {"message": {"reasoning_content": "internal reasoning"}}
                        ],
                    },
                )
            return httpx.Response(404, json={"error": "not found"})

        client = LemonadeClient(lemonade_config, fallback_ollama=ollama_fallback)
        client._client = httpx.AsyncClient(
            base_url=lemonade_config.host,
            transport=httpx.MockTransport(_handler),
        )

        try:
            response = await client.chat(messages=[{"role": "user", "content": "hi"}])
            assert response.content == ""
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_chat_stream_parses_sse(
        self,
        lemonade_config: LemonadeConfig,
        ollama_fallback: OllamaConfig,
    ) -> None:
        stream_body = "\n\n".join(
            [
                'data: {"choices":[{"delta":{"content":"A"}}]}',
                'data: {"choices":[{"delta":{"content":"B"}}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}',
                "data: [DONE]",
                "",
            ]
        )

        def _handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/chat/completions":
                return httpx.Response(
                    200,
                    text=stream_body,
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(404, json={"error": "not found"})

        client = LemonadeClient(lemonade_config, fallback_ollama=ollama_fallback)
        client._client = httpx.AsyncClient(
            base_url=lemonade_config.host,
            transport=httpx.MockTransport(_handler),
        )
        state = ChatStreamState()

        try:
            chunks: list[str] = []
            async for chunk in client.chat_stream(
                messages=[{"role": "user", "content": "hi"}],
                stream_state=state,
            ):
                chunks.append(chunk)
            assert chunks == ["A", "B"]
            assert state.usage is not None
            assert state.usage.prompt_eval_count == 1
            assert state.usage.eval_count == 2
            assert state.usage.total_duration == 0
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_chat_stream_stops_on_finish_reason_without_done(
        self,
        lemonade_config: LemonadeConfig,
        ollama_fallback: OllamaConfig,
    ) -> None:
        stream_body = "\n\n".join(
            [
                'data: {"choices":[{"delta":{"content":"A"}}]}',
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
                "",
            ]
        )

        def _handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/chat/completions":
                return httpx.Response(
                    200,
                    text=stream_body,
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(404, json={"error": "not found"})

        client = LemonadeClient(lemonade_config, fallback_ollama=ollama_fallback)
        client._client = httpx.AsyncClient(
            base_url=lemonade_config.host,
            transport=httpx.MockTransport(_handler),
        )

        try:
            chunks: list[str] = []
            async for chunk in client.chat_stream(
                messages=[{"role": "user", "content": "hi"}],
                stream_state=ChatStreamState(),
            ):
                chunks.append(chunk)
            assert chunks == ["A"]
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_chat_stream_does_not_emit_reasoning_content_only(
        self,
        lemonade_config: LemonadeConfig,
        ollama_fallback: OllamaConfig,
    ) -> None:
        stream_body = "\n\n".join(
            [
                'data: {"choices":[{"delta":{"reasoning_content":"내부 사고"}}]}',
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
                "",
            ]
        )

        def _handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/chat/completions":
                return httpx.Response(
                    200,
                    text=stream_body,
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(404, json={"error": "not found"})

        client = LemonadeClient(lemonade_config, fallback_ollama=ollama_fallback)
        client._client = httpx.AsyncClient(
            base_url=lemonade_config.host,
            transport=httpx.MockTransport(_handler),
        )

        try:
            chunks: list[str] = []
            async for chunk in client.chat_stream(
                messages=[{"role": "user", "content": "hi"}],
                stream_state=ChatStreamState(),
            ):
                chunks.append(chunk)
            assert chunks == []
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_chat_stream_raises_when_repeating_content_stalls(
        self,
        lemonade_config: LemonadeConfig,
        ollama_fallback: OllamaConfig,
    ) -> None:
        repeating_lines = ['data: {"choices":[{"delta":{"content":"가"}}]}'] * 205
        stream_body = "\n\n".join(repeating_lines + [""])

        def _handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/chat/completions":
                return httpx.Response(
                    200,
                    text=stream_body,
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(404, json={"error": "not found"})

        client = LemonadeClient(lemonade_config, fallback_ollama=ollama_fallback)
        client._client = httpx.AsyncClient(
            base_url=lemonade_config.host,
            transport=httpx.MockTransport(_handler),
        )

        try:
            with pytest.raises(LemonadeClientError, match="streaming request failed"):
                async for _ in client.chat_stream(
                    messages=[{"role": "user", "content": "hi"}],
                    stream_state=ChatStreamState(),
                ):
                    pass
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_chat_stream_normalizes_cumulative_message_content(
        self,
        lemonade_config: LemonadeConfig,
        ollama_fallback: OllamaConfig,
    ) -> None:
        stream_body = "\n\n".join(
            [
                'data: {"choices":[{"message":{"content":"가"}}]}',
                'data: {"choices":[{"message":{"content":"가가"}}]}',
                'data: {"choices":[{"message":{"content":"가가가"},"finish_reason":"stop"}]}',
                "",
            ]
        )

        def _handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/chat/completions":
                return httpx.Response(
                    200,
                    text=stream_body,
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(404, json={"error": "not found"})

        client = LemonadeClient(lemonade_config, fallback_ollama=ollama_fallback)
        client._client = httpx.AsyncClient(
            base_url=lemonade_config.host,
            transport=httpx.MockTransport(_handler),
        )

        try:
            chunks: list[str] = []
            async for chunk in client.chat_stream(
                messages=[{"role": "user", "content": "hi"}],
                stream_state=ChatStreamState(),
            ):
                chunks.append(chunk)
            assert chunks == ["가", "가", "가"]
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_chat_stream_raises_when_fallback_content_stalls(
        self,
        lemonade_config: LemonadeConfig,
        ollama_fallback: OllamaConfig,
    ) -> None:
        repeating_lines = ['data: {"choices":[{"message":{"content":"가"}}]}'] * 205
        stream_body = "\n\n".join(repeating_lines + [""])

        def _handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/chat/completions":
                return httpx.Response(
                    200,
                    text=stream_body,
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(404, json={"error": "not found"})

        client = LemonadeClient(lemonade_config, fallback_ollama=ollama_fallback)
        client._client = httpx.AsyncClient(
            base_url=lemonade_config.host,
            transport=httpx.MockTransport(_handler),
        )

        try:
            with pytest.raises(LemonadeClientError, match="streaming request failed"):
                async for _ in client.chat_stream(
                    messages=[{"role": "user", "content": "hi"}],
                    stream_state=ChatStreamState(),
                ):
                    pass
        finally:
            await client.close()
