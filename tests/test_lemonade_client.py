"""Lemonade 클라이언트 테스트."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from core.config import LemonadeConfig
from core.lemonade_client import LemonadeClient, LemonadeClientError
from core.lemonade_errors import LemonadeModelNotFoundError
from core.llm_types import ChatResponse, ChatStreamState


@pytest.fixture
def lemonade_config() -> LemonadeConfig:
    return LemonadeConfig(
        host="http://localhost:8000",
        default_model="test-model",
        system_prompt="test prompt",
        base_path="/api/v1",
        timeout_seconds=10,
    )


class TestInitialize:
    @pytest.mark.asyncio
    async def test_initialize_success(
        self,
        lemonade_config: LemonadeConfig,
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
        client = LemonadeClient(lemonade_config)

        with patch("core.lemonade_client.httpx.AsyncClient", return_value=mock_client):
            await client.initialize()
        try:
            assert client.default_model == "test-model"
            assert client.system_prompt == "test prompt"
        finally:
            await client.close()


class TestChat:
    @pytest.mark.asyncio
    async def test_chat_returns_content_and_usage(
        self,
        lemonade_config: LemonadeConfig,
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

        client = LemonadeClient(lemonade_config)
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

        client = LemonadeClient(lemonade_config)
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

        client = LemonadeClient(lemonade_config)
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

        client = LemonadeClient(lemonade_config)
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

        client = LemonadeClient(lemonade_config)
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

        client = LemonadeClient(lemonade_config)
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
    ) -> None:
        repeating_lines = ['data: {"choices":[{"delta":{"content":"가"}}]}'] * 205
        stream_body = "\n\n".join([*repeating_lines, ""])

        def _handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/chat/completions":
                return httpx.Response(
                    200,
                    text=stream_body,
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(404, json={"error": "not found"})

        client = LemonadeClient(lemonade_config)
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

        client = LemonadeClient(lemonade_config)
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
    ) -> None:
        repeating_lines = ['data: {"choices":[{"message":{"content":"가"}}]}'] * 205
        stream_body = "\n\n".join([*repeating_lines, ""])

        def _handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/chat/completions":
                return httpx.Response(
                    200,
                    text=stream_body,
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(404, json={"error": "not found"})

        client = LemonadeClient(lemonade_config)
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
    async def test_chat_stream_logs_raw_sse_lines_when_enabled(
        self,
        lemonade_config: LemonadeConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LEMONADE_DEBUG_STREAM_SSE", "1")
        monkeypatch.setenv("LEMONADE_DEBUG_STREAM_MAX_LINES", "10")

        stream_body = "\n\n".join(
            [
                'data: {"choices":[{"delta":{"content":"A"}}]}',
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

        client = LemonadeClient(lemonade_config)
        client._logger = MagicMock()
        client._client = httpx.AsyncClient(
            base_url=lemonade_config.host,
            transport=httpx.MockTransport(_handler),
        )

        try:
            async for _ in client.chat_stream(
                messages=[{"role": "user", "content": "hi"}],
                stream_state=ChatStreamState(),
            ):
                pass

            raw_calls = [
                call
                for call in client._logger.info.call_args_list
                if call.args and call.args[0] == "lemonade_stream_sse_line"
            ]
            assert len(raw_calls) == 2
            assert raw_calls[0].kwargs["line_no"] == 1
            assert raw_calls[0].kwargs["raw_preview"].startswith('{"choices"')
            assert raw_calls[1].kwargs["raw_preview"] == "[DONE]"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_chat_stream_logs_payload_compare_patterns_when_enabled(
        self,
        lemonade_config: LemonadeConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LEMONADE_DEBUG_STREAM_COMPARE", "1")
        monkeypatch.setenv("LEMONADE_DEBUG_STREAM_MAX_LINES", "10")

        stream_body = "\n\n".join(
            [
                'data: {"choices":[{"message":{"content":"가"}}]}',
                'data: {"choices":[{"message":{"content":"가가"}}]}',
                'data: {"choices":[{"delta":{"content":"나"}}]}',
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

        client = LemonadeClient(lemonade_config)
        client._logger = MagicMock()
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

            assert chunks == ["가", "가", "나"]

            compare_calls = [
                call
                for call in client._logger.info.call_args_list
                if call.args and call.args[0] == "lemonade_stream_payload_compare"
            ]
            selected_paths = [call.kwargs["selected_path"] for call in compare_calls]
            assert "message_content" in selected_paths
            assert "message_prefix_delta" in selected_paths
            assert "delta_content" in selected_paths
            assert "finish_reason_only" in selected_paths

            prefix_delta_call = next(
                call for call in compare_calls
                if call.kwargs["selected_path"] == "message_prefix_delta"
            )
            assert prefix_delta_call.kwargs["snapshot_prefix_match"] is True
            assert prefix_delta_call.kwargs["emitted_preview"] == "가"
        finally:
            await client.close()


class TestDelegatedOperations:
    @pytest.mark.asyncio
    async def test_list_models_and_get_model_info(
        self,
        lemonade_config: LemonadeConfig,
    ) -> None:
        def _handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/models":
                return httpx.Response(
                    200,
                    json={"data": [{"id": "alpha"}, {"id": "beta"}]},
                )
            return httpx.Response(404, json={"error": "not found"})

        client = LemonadeClient(lemonade_config)
        client._client = httpx.AsyncClient(
            base_url=lemonade_config.host,
            transport=httpx.MockTransport(_handler),
        )

        try:
            assert await client.list_models() == [
                {"name": "alpha", "size": None, "modified_at": None},
                {"name": "beta", "size": None, "modified_at": None},
            ]
            assert await client.get_model_info("beta") == {
                "model": "beta",
                "modelfile": None,
                "parameters": None,
            }
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_get_model_info_raises_for_missing_model(
        self,
        lemonade_config: LemonadeConfig,
    ) -> None:
        def _handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/models":
                return httpx.Response(200, json={"data": [{"id": "alpha"}]})
            return httpx.Response(404, json={"error": "not found"})

        client = LemonadeClient(lemonade_config)
        client._client = httpx.AsyncClient(
            base_url=lemonade_config.host,
            transport=httpx.MockTransport(_handler),
        )

        try:
            with pytest.raises(LemonadeModelNotFoundError, match="missing"):
                await client.get_model_info("missing")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_health_check_attempts_recovery_on_failure(
        self,
        lemonade_config: LemonadeConfig,
    ) -> None:
        client = LemonadeClient(lemonade_config)
        client._client = AsyncMock()
        client._list_model_names = AsyncMock(side_effect=RuntimeError("down"))
        client.recover_connection = AsyncMock(return_value=True)

        try:
            health = await client.health_check(attempt_recovery=True)
            assert health == {
                "status": "error",
                "host": lemonade_config.host,
                "error": "down",
                "recovery_attempted": True,
                "recovered": True,
            }
            client.recover_connection.assert_awaited_once_with(force=True)
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_embed_returns_sorted_vectors(
        self,
        lemonade_config: LemonadeConfig,
    ) -> None:
        def _handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/embeddings":
                payload = json.loads(request.content.decode("utf-8"))
                assert payload["model"] == "test-model"
                assert payload["input"] == ["hello", "world"]
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {"index": 1, "embedding": [0.2, 0.3]},
                            {"index": 0, "embedding": [0.0, 0.1]},
                        ]
                    },
                )
            return httpx.Response(404, json={"error": "not found"})

        client = LemonadeClient(lemonade_config)
        client._client = httpx.AsyncClient(
            base_url=lemonade_config.host,
            transport=httpx.MockTransport(_handler),
        )

        try:
            assert await client.embed(["hello", "world"]) == [
                [0.0, 0.1],
                [0.2, 0.3],
            ]
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_prepare_model_loads_missing_model(
        self,
        lemonade_config: LemonadeConfig,
    ) -> None:
        load_payloads: list[dict[str, str]] = []

        def _handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/health":
                return httpx.Response(200, json={"all_models_loaded": []})
            if request.url.path == "/api/v1/load":
                load_payloads.append(json.loads(request.content.decode("utf-8")))
                return httpx.Response(200, json={"ok": True})
            return httpx.Response(404, json={"error": "not found"})

        client = LemonadeClient(lemonade_config)
        client._client = httpx.AsyncClient(
            base_url=lemonade_config.host,
            transport=httpx.MockTransport(_handler),
        )

        try:
            await client.prepare_model(model="heavy-model", role="reasoning")
            assert load_payloads == [{"model_name": "heavy-model"}]
            assert "heavy-model" in client._loaded_models
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_rerank_uses_endpoint_when_available(
        self,
        lemonade_config: LemonadeConfig,
    ) -> None:
        def _handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/rerank":
                payload = json.loads(request.content.decode("utf-8"))
                assert payload["top_n"] == 2
                return httpx.Response(
                    200,
                    json={
                        "results": [
                            {"index": 0, "relevance_score": 0.2},
                            {"index": 1, "relevance_score": 0.8},
                        ]
                    },
                )
            return httpx.Response(404, json={"error": "not found"})

        client = LemonadeClient(lemonade_config)
        client._client = httpx.AsyncClient(
            base_url=lemonade_config.host,
            transport=httpx.MockTransport(_handler),
        )

        try:
            assert await client.rerank(
                query="query",
                documents=["a", "b"],
                top_n=2,
            ) == [
                {"index": 1, "score": 0.8},
                {"index": 0, "score": 0.2},
            ]
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_rerank_falls_back_to_chat_when_endpoint_missing(
        self,
        lemonade_config: LemonadeConfig,
    ) -> None:
        def _handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/rerank":
                return httpx.Response(404, json={"error": "not available"})
            return httpx.Response(404, json={"error": "not found"})

        client = LemonadeClient(lemonade_config)
        client._client = httpx.AsyncClient(
            base_url=lemonade_config.host,
            transport=httpx.MockTransport(_handler),
        )
        client.chat = AsyncMock(
            return_value=ChatResponse(
                content='[{"index": 1, "score": 0.9}, {"index": 0, "score": 0.1}]',
                usage=None,
            )
        )

        try:
            assert await client.rerank(
                query="query",
                documents=["a", "b"],
                top_n=1,
            ) == [{"index": 1, "score": 0.9}]
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_check_model_availability_returns_false_on_lookup_error(
        self,
        lemonade_config: LemonadeConfig,
    ) -> None:
        client = LemonadeClient(lemonade_config)
        client._client = AsyncMock()
        client._list_model_names = AsyncMock(side_effect=RuntimeError("down"))

        try:
            assert await client.check_model_availability(["alpha", "beta"]) == {
                "alpha": False,
                "beta": False,
            }
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_recover_connection_replaces_client(
        self,
        lemonade_config: LemonadeConfig,
    ) -> None:
        def _handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/models":
                return httpx.Response(200, json={"data": [{"id": "recovered"}]})
            return httpx.Response(404, json={"error": "not found"})

        previous_client = httpx.AsyncClient(
            base_url=lemonade_config.host,
            transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={"data": []})),
        )
        candidate_client = httpx.AsyncClient(
            base_url=lemonade_config.host,
            transport=httpx.MockTransport(_handler),
        )

        client = LemonadeClient(lemonade_config)
        client._client = previous_client
        client._auto_reconnect_enabled = True

        with patch(
            "core.lemonade_delegates.httpx.AsyncClient",
            return_value=candidate_client,
        ):
            assert await client.recover_connection(force=True) is True
            assert client._client is candidate_client

        await client.close()
