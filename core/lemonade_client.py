"""Lemonade(OpenAI-compatible) API 클라이언트.

Lemonade Server의 OpenAI 호환 엔드포인트를 사용해
채팅, 스트리밍, 헬스체크, 모델 조회를 제공한다.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from core.config import LemonadeConfig, OllamaConfig
from core.logging_setup import get_logger
from core.ollama_client import ChatResponse, ChatStreamState, ChatUsage


class LemonadeClientError(Exception):
    """Lemonade 통신 실패."""


class LemonadeModelNotFoundError(LemonadeClientError):
    """요청한 기본 모델이 서버 모델 목록에 없음."""


class LemonadeClient:
    """Lemonade OpenAI-compatible 클라이언트."""

    def __init__(
        self,
        config: LemonadeConfig,
        *,
        fallback_ollama: OllamaConfig | None = None,
    ) -> None:
        self._host = config.host.rstrip("/")
        base_path = config.base_path.strip()
        if base_path:
            self._base_path = "/" + base_path.strip("/")
        else:
            self._base_path = ""
        self._api_key = config.api_key
        self._default_model = config.model or (
            fallback_ollama.model if fallback_ollama is not None else ""
        )
        self._temperature = (
            fallback_ollama.temperature if fallback_ollama is not None else 0.7
        )
        self._max_tokens = (
            fallback_ollama.max_tokens if fallback_ollama is not None else 1024
        )
        self._system_prompt = (
            fallback_ollama.system_prompt
            if fallback_ollama is not None
            else "You are a helpful assistant."
        )
        self._timeout_default = config.timeout_seconds
        self._client: httpx.AsyncClient | None = None
        self._auto_reconnect_enabled = False
        self._is_healthy = True
        self._last_connection_error: str | None = None
        self._next_reconnect_at = 0.0
        self._reconnect_cooldown_seconds = 15.0
        self._reconnect_lock = asyncio.Lock()
        self._logger = get_logger("lemonade_client")

    @property
    def default_model(self) -> str:
        return self._default_model

    @default_model.setter
    def default_model(self, model: str) -> None:
        self._default_model = model

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def host(self) -> str:
        return self._host

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _endpoint(self, path: str) -> str:
        return f"{self._base_path}{path}"

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("LemonadeClient가 아직 초기화되지 않았습니다.")
        return self._client

    def _mark_healthy(self) -> None:
        self._is_healthy = True
        self._last_connection_error = None
        self._next_reconnect_at = 0.0

    def _mark_unhealthy(self, error: Exception | str) -> None:
        self._is_healthy = False
        self._last_connection_error = str(error)
        self._next_reconnect_at = time.monotonic() + self._reconnect_cooldown_seconds

    async def initialize(self) -> None:
        """클라이언트를 생성하고 연결/모델 상태를 확인한다."""
        client = httpx.AsyncClient(
            base_url=self._host,
            headers=self._headers(),
        )
        self._client = client
        self._auto_reconnect_enabled = True
        try:
            models = await self._list_model_names(client)
            self._logger.info(
                "lemonade_connected",
                host=self._host,
                models_count=len(models),
            )
            if self._default_model and self._default_model not in models:
                raise LemonadeModelNotFoundError(
                    f"Default model '{self._default_model}' not found on {self._host}. "
                    f"Available models: {', '.join(models) if models else '(none)'}"
                )
            self._mark_healthy()
        except Exception as exc:
            self._mark_unhealthy(exc)
            await client.aclose()
            self._client = None
            if isinstance(exc, LemonadeClientError):
                raise
            raise LemonadeClientError(
                f"Failed to connect to Lemonade at {self._host}: {exc}"
            ) from exc

    async def close(self) -> None:
        """클라이언트 리소스를 정리한다."""
        if self._client is not None:
            await self._client.aclose()
        self._client = None
        self._auto_reconnect_enabled = False
        self._is_healthy = False

    async def recover_connection(self, *, force: bool = False) -> bool:
        """연결 장애 시 재연결을 시도한다."""
        if not self._auto_reconnect_enabled:
            return False

        now = time.monotonic()
        if not force and now < self._next_reconnect_at:
            return False

        async with self._reconnect_lock:
            now = time.monotonic()
            if not force and now < self._next_reconnect_at:
                return False

            candidate = httpx.AsyncClient(
                base_url=self._host,
                headers=self._headers(),
            )
            try:
                models = await self._list_model_names(candidate)
            except Exception as exc:
                self._mark_unhealthy(exc)
                self._logger.warning("lemonade_reconnect_failed", error=str(exc))
                await candidate.aclose()
                return False

            if self._default_model and self._default_model not in models:
                error = LemonadeModelNotFoundError(
                    f"Default model '{self._default_model}' not found on {self._host}"
                )
                self._mark_unhealthy(error)
                self._logger.warning(
                    "lemonade_reconnect_default_model_missing",
                    model=self._default_model,
                    available=models,
                )
                await candidate.aclose()
                return False

            previous = self._client
            self._client = candidate
            if previous is not None:
                await previous.aclose()
            self._mark_healthy()
            self._logger.info(
                "lemonade_reconnected",
                host=self._host,
                models_count=len(models),
            )
            return True

    async def _list_model_names(self, client: httpx.AsyncClient) -> list[str]:
        response = await client.get(
            self._endpoint("/models"),
            timeout=self._timeout_default,
        )
        response.raise_for_status()
        payload = response.json()

        if isinstance(payload, dict):
            entries = payload.get("data", [])
        elif isinstance(payload, list):
            entries = payload
        else:
            entries = []

        names: list[str] = []
        for item in entries:
            if isinstance(item, dict):
                model_name = item.get("id") or item.get("model") or item.get("name")
                if isinstance(model_name, str) and model_name:
                    names.append(model_name)
            elif isinstance(item, str) and item:
                names.append(item)
        return names

    @staticmethod
    def _extract_content(choice: dict[str, Any]) -> str:
        message = choice.get("message") if isinstance(choice, dict) else None
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                chunks: list[str] = []
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        chunks.append(part["text"])
                return "".join(chunks)
        return ""

    @staticmethod
    def _usage_from_payload(payload: dict[str, Any]) -> ChatUsage | None:
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return None
        return ChatUsage(
            prompt_eval_count=int(usage.get("prompt_tokens", 0) or 0),
            eval_count=int(usage.get("completion_tokens", 0) or 0),
            eval_duration=0,
            total_duration=int(usage.get("total_tokens", 0) or 0),
        )

    def _build_chat_payload(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int | None,
        response_format: str | dict | None,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self._temperature if temperature is None else temperature,
            "max_tokens": self._max_tokens if max_tokens is None else max_tokens,
            "stream": stream,
        }
        if response_format is not None:
            if response_format == "json":
                payload["response_format"] = {"type": "json_object"}
            else:
                payload["response_format"] = response_format
        return payload

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int = 60,
        response_format: str | dict | None = None,
    ) -> ChatResponse:
        """비스트리밍 채팅 요청."""
        target_model = model or self._default_model
        payload = self._build_chat_payload(
            model=target_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            stream=False,
        )

        async def _do_chat() -> ChatResponse:
            client = self._require_client()
            response = await client.post(
                self._endpoint("/chat/completions"),
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            body = response.json()
            choices = body.get("choices", [])
            if not choices:
                raise LemonadeClientError("lemonade_chat_failed: missing choices")
            content = self._extract_content(choices[0])
            usage = self._usage_from_payload(body)
            return ChatResponse(content=content, usage=usage)

        return await self._retry_with_backoff(_do_chat)

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int = 60,
        stream_state: ChatStreamState | None = None,
    ) -> AsyncGenerator[str, None]:
        """스트리밍 채팅 요청."""
        client = self._require_client()
        target_model = model or self._default_model
        payload = self._build_chat_payload(
            model=target_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=None,
            stream=True,
        )
        state = stream_state or ChatStreamState()
        state.usage = None

        try:
            async with client.stream(
                "POST",
                self._endpoint("/chat/completions"),
                json=payload,
                timeout=timeout,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    data_line = line.strip()
                    if not data_line or not data_line.startswith("data:"):
                        continue
                    data_raw = data_line[5:].strip()
                    if data_raw == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_raw)
                    except json.JSONDecodeError:
                        continue

                    usage = self._usage_from_payload(chunk)
                    if usage is not None:
                        state.usage = usage

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    first_choice = choices[0]
                    delta = first_choice.get("delta", {})
                    content = delta.get("content") if isinstance(delta, dict) else None
                    if isinstance(content, str) and content:
                        yield content
                        continue

                    # 일부 구현은 delta 대신 message.content를 전달한다.
                    fallback_content = self._extract_content(first_choice)
                    if fallback_content:
                        yield fallback_content
            self._mark_healthy()
        except (httpx.HTTPError, asyncio.TimeoutError, OSError) as exc:
            self._mark_unhealthy(exc)
            await self.recover_connection()
            raise LemonadeClientError(
                f"Lemonade streaming request failed for model '{target_model}': {exc}"
            ) from exc

    async def list_models(self) -> list[dict]:
        """모델 목록을 반환한다."""
        client = self._require_client()
        try:
            names = await self._list_model_names(client)
            self._mark_healthy()
        except Exception as exc:
            self._mark_unhealthy(exc)
            await self.recover_connection()
            raise
        return [{"name": name, "size": None, "modified_at": None} for name in names]

    async def get_model_info(self, model: str) -> dict:
        """모델 정보를 반환한다."""
        models = await self.list_models()
        names = {m["name"] for m in models}
        if model not in names:
            raise LemonadeModelNotFoundError(f"Model '{model}' not found")
        return {"model": model, "modelfile": None, "parameters": None}

    async def health_check(self, *, attempt_recovery: bool = False) -> dict:
        """서버 상태를 확인한다."""
        client = self._require_client()
        try:
            models = await self._list_model_names(client)
            self._mark_healthy()
            return {
                "status": "ok",
                "host": self._host,
                "models_count": len(models),
                "models": models,
                "default_model": self._default_model,
                "default_model_available": (
                    True if not self._default_model else self._default_model in models
                ),
            }
        except Exception as exc:
            self._mark_unhealthy(exc)
            recovered = False
            if attempt_recovery:
                recovered = await self.recover_connection(force=True)
            return {
                "status": "error",
                "host": self._host,
                "error": str(exc),
                "recovery_attempted": attempt_recovery,
                "recovered": recovered,
            }

    # ── Embeddings ──

    async def embed(
        self,
        texts: list[str],
        model: str | None = None,
        timeout: int | None = None,
    ) -> list[list[float]]:
        """임베딩 벡터를 반환한다. POST /embeddings"""
        client = self._require_client()
        target_model = model or self._default_model
        payload: dict[str, Any] = {"model": target_model, "input": texts}
        try:
            response = await client.post(
                self._endpoint("/embeddings"),
                json=payload,
                timeout=timeout or self._timeout_default,
            )
            response.raise_for_status()
        except (httpx.HTTPError, asyncio.TimeoutError, OSError) as exc:
            self._mark_unhealthy(exc)
            raise LemonadeClientError(
                f"Embedding request failed for model '{target_model}': {exc}"
            ) from exc
        self._mark_healthy()
        body = response.json()
        data = body.get("data", [])
        return [
            item["embedding"]
            for item in sorted(data, key=lambda x: x.get("index", 0))
        ]

    # ── Rerank ──

    async def rerank(
        self,
        query: str,
        documents: list[str],
        model: str | None = None,
        top_n: int | None = None,
        timeout: int | None = None,
    ) -> list[dict[str, Any]]:
        """리랭크 점수를 반환한다.

        전용 /rerank endpoint를 우선 시도하고 없으면 chat 기반 폴백.
        반환: [{"index": int, "score": float}, ...]  점수 내림차순 정렬.
        """
        client = self._require_client()
        target_model = model or self._default_model
        effective_timeout = timeout or self._timeout_default

        # 1) 전용 /rerank endpoint 시도
        try:
            payload: dict[str, Any] = {
                "model": target_model,
                "query": query,
                "documents": documents,
            }
            if top_n is not None:
                payload["top_n"] = top_n
            response = await client.post(
                self._endpoint("/rerank"),
                json=payload,
                timeout=effective_timeout,
            )
            response.raise_for_status()
            body = response.json()
            results = body.get("results", body.get("data", []))
            scored = []
            for item in results:
                scored.append({
                    "index": int(item.get("index", 0)),
                    "score": float(item.get("relevance_score", item.get("score", 0.0))),
                })
            scored.sort(key=lambda x: x["score"], reverse=True)
            self._mark_healthy()
            return scored
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (404, 405):
                self._logger.debug("rerank_endpoint_not_available", status=exc.response.status_code)
            else:
                raise LemonadeClientError(
                    f"Rerank request failed: {exc}"
                ) from exc
        except (httpx.HTTPError, asyncio.TimeoutError, OSError) as exc:
            self._mark_unhealthy(exc)
            raise LemonadeClientError(f"Rerank request failed: {exc}") from exc

        # 2) chat 기반 폴백 (마지막 수단)
        return await self._rerank_via_chat(
            query, documents, target_model, top_n, effective_timeout,
        )

    async def _rerank_via_chat(
        self,
        query: str,
        documents: list[str],
        model: str,
        top_n: int | None,
        timeout: int,
    ) -> list[dict[str, Any]]:
        """chat/completions를 사용한 리랭크 폴백."""
        doc_list = "\n".join(
            f"[{i}] {doc[:500]}" for i, doc in enumerate(documents)
        )
        prompt = (
            f"Query: {query}\n\n"
            f"Documents:\n{doc_list}\n\n"
            "위 각 문서의 query 관련성을 0.0~1.0 점수로 평가하세요.\n"
            "JSON 배열만 출력: [{\"index\": 0, \"score\": 0.8}, ...]\n"
            "다른 텍스트 없이 JSON만 출력하세요."
        )
        try:
            chat_resp = await self.chat(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=256,
                timeout=timeout,
                response_format="json",
            )
            import json as _json
            parsed = _json.loads(chat_resp.content)
            if isinstance(parsed, list):
                scored = [
                    {"index": int(item.get("index", 0)), "score": float(item.get("score", 0.0))}
                    for item in parsed
                ]
            elif isinstance(parsed, dict) and "results" in parsed:
                scored = [
                    {"index": int(item.get("index", 0)), "score": float(item.get("score", 0.0))}
                    for item in parsed["results"]
                ]
            else:
                raise LemonadeClientError("Unexpected rerank chat response format")
            scored.sort(key=lambda x: x["score"], reverse=True)
            if top_n is not None:
                scored = scored[:top_n]
            return scored
        except (LemonadeClientError, json.JSONDecodeError, KeyError, ValueError) as exc:
            raise LemonadeClientError(f"Chat-based rerank failed: {exc}") from exc

    # ── Model availability ──

    async def check_model_availability(self, model_names: list[str]) -> dict[str, bool]:
        """주어진 모델 이름들의 가용성을 확인한다."""
        client = self._require_client()
        try:
            available = await self._list_model_names(client)
        except Exception:
            return {name: False for name in model_names}
        available_set = set(available)
        return {name: name in available_set for name in model_names}

    # ── Internal helpers ──

    async def _retry_with_backoff(
        self,
        coro_factory,
        max_retries: int = 2,
    ) -> ChatResponse:
        backoff_delays = [1, 3]
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                response = await coro_factory()
                self._mark_healthy()
                return response
            except (httpx.HTTPError, asyncio.TimeoutError, OSError, LemonadeClientError) as exc:
                last_error = exc
                self._mark_unhealthy(exc)
                if attempt < max_retries:
                    await self.recover_connection(force=True)
                    delay = backoff_delays[min(attempt, len(backoff_delays) - 1)]
                    self._logger.warning(
                        "lemonade_retry",
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(exc),
                    )
                    await asyncio.sleep(delay)

        raise LemonadeClientError(
            f"Lemonade request failed after {max_retries + 1} attempts: {last_error}"
        )

