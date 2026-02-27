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
        # 기본 모델 고정 설정을 제거한다. (비어 있으면 요청 단위 모델/서버 기본 모델 사용)
        self._default_model = (config.model or "").strip()
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
        self._model_load_timeout_default = max(
            config.model_load_timeout_seconds,
            config.timeout_seconds,
        )
        self._heavy_model_load_timeout_default = max(
            config.heavy_model_load_timeout_seconds,
            self._model_load_timeout_default,
        )
        self._client: httpx.AsyncClient | None = None
        self._auto_reconnect_enabled = False
        self._is_healthy = True
        self._last_connection_error: str | None = None
        self._next_reconnect_at = 0.0
        self._reconnect_cooldown_seconds = 15.0
        self._reconnect_lock = asyncio.Lock()
        self._prepare_model_lock = asyncio.Lock()
        self._loaded_models: set[str] = set()
        self._loaded_models_checked_at = 0.0
        self._loaded_models_ttl_seconds = 10.0
        self._heavy_roles = {"reasoning", "coding", "vision"}
        self._logger = get_logger("lemonade_client")

    @staticmethod
    def _format_exception(exc: Exception) -> str:
        """로그/에러 메시지에 사용할 예외 문자열을 정규화한다."""
        message = str(exc).strip()
        if message:
            return f"{exc.__class__.__name__}: {message}"
        return exc.__class__.__name__

    @property
    def default_model(self) -> str:
        return self._default_model

    @default_model.setter
    def default_model(self, model: str) -> None:
        self._default_model = model.strip()

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
        if isinstance(error, Exception):
            self._last_connection_error = self._format_exception(error)
        else:
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
                self._logger.warning(
                    "lemonade_default_model_not_listed_on_init",
                    model=self._default_model,
                    available=models,
                )
            self._mark_healthy()
        except Exception as exc:
            self._mark_unhealthy(exc)
            await client.aclose()
            self._client = None
            if isinstance(exc, LemonadeClientError):
                raise
            error_text = self._format_exception(exc)
            raise LemonadeClientError(
                f"Failed to connect to Lemonade at {self._host}: {error_text}"
            ) from exc

    async def close(self) -> None:
        """클라이언트 리소스를 정리한다."""
        if self._client is not None:
            await self._client.aclose()
        self._client = None
        self._auto_reconnect_enabled = False
        self._is_healthy = False
        self._loaded_models.clear()
        self._loaded_models_checked_at = 0.0

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
                self._logger.warning(
                    "lemonade_reconnect_failed",
                    error=self._format_exception(exc),
                )
                await candidate.aclose()
                return False

            if self._default_model and self._default_model not in models:
                self._logger.warning(
                    "lemonade_default_model_not_listed_on_reconnect",
                    model=self._default_model,
                    available=models,
                )

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
    def _compact_text(value: str, *, max_chars: int = 280) -> str:
        compact = " ".join(value.split())
        if len(compact) <= max_chars:
            return compact
        return compact[:max_chars] + "..."

    @staticmethod
    def _parse_loaded_models(payload: Any) -> set[str]:
        if not isinstance(payload, dict):
            return set()
        entries = payload.get("all_models_loaded", [])
        names: set[str] = set()
        if not isinstance(entries, list):
            return names
        for item in entries:
            if isinstance(item, dict):
                name = (
                    item.get("model_name")
                    or item.get("id")
                    or item.get("model")
                    or item.get("name")
                )
                if isinstance(name, str) and name:
                    names.add(name)
            elif isinstance(item, str) and item:
                names.add(item)
        return names

    async def _refresh_loaded_models(self, *, force: bool = False) -> set[str]:
        now = time.monotonic()
        if (
            not force
            and self._loaded_models
            and now - self._loaded_models_checked_at < self._loaded_models_ttl_seconds
        ):
            return set(self._loaded_models)

        client = self._require_client()
        try:
            response = await client.get(
                self._endpoint("/health"),
                timeout=self._timeout_default,
            )
            response.raise_for_status()
            loaded = self._parse_loaded_models(response.json())
            self._loaded_models = loaded
            self._loaded_models_checked_at = now
            return set(loaded)
        except Exception as exc:
            self._logger.debug(
                "lemonade_loaded_models_refresh_failed",
                error=self._format_exception(exc),
            )
            return set(self._loaded_models)

    async def prepare_model(
        self,
        *,
        model: str | None = None,
        role: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        """요청 전 모델 로드를 보장한다.

        - low_cost는 기본 로드 타임아웃을 사용한다.
        - reasoning/coding/vision은 heavy 로드 타임아웃을 사용한다.
        """
        target_model = (model or self._default_model or "").strip()
        if not target_model:
            return

        loaded = await self._refresh_loaded_models()
        if target_model in loaded:
            return

        role_key = (role or "").strip().lower()
        role_timeout = (
            self._heavy_model_load_timeout_default
            if role_key in self._heavy_roles
            else self._model_load_timeout_default
        )
        effective_timeout = max(role_timeout, int(timeout_seconds or 0))

        async with self._prepare_model_lock:
            loaded = await self._refresh_loaded_models(force=True)
            if target_model in loaded:
                return

            client = self._require_client()
            started = time.monotonic()
            response = await client.post(
                self._endpoint("/load"),
                json={"model_name": target_model},
                timeout=effective_timeout,
            )
            if response.status_code in (404, 405):
                # 일부 서버 구현은 /load endpoint가 없을 수 있다.
                self._logger.debug(
                    "lemonade_model_preload_skipped",
                    model=target_model,
                    role=role_key or None,
                    status_code=response.status_code,
                )
                return
            if response.is_error:
                message = self._compact_text(response.text)
                raise LemonadeClientError(
                    f"Failed to load model '{target_model}' before generation "
                    f"(role={role_key or 'unknown'}, status={response.status_code}): {message}"
                )

            self._loaded_models.add(target_model)
            self._loaded_models_checked_at = time.monotonic()
            self._logger.info(
                "lemonade_model_preloaded",
                model=target_model,
                role=role_key or None,
                elapsed_seconds=round(time.monotonic() - started, 2),
            )

    @staticmethod
    def _extract_content(choice: dict[str, Any], *, allow_reasoning_fallback: bool = True) -> str:
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
            if allow_reasoning_fallback:
                reasoning = message.get("reasoning_content")
                if isinstance(reasoning, str) and reasoning:
                    return reasoning
        return ""

    def _extract_api_error(self, payload: Any) -> str | None:
        if not isinstance(payload, dict):
            return None
        error = payload.get("error")
        if isinstance(error, str) and error.strip():
            return self._compact_text(error)
        if not isinstance(error, dict):
            return None
        message = error.get("message") or error.get("detail") or error.get("error")
        code = error.get("code") or error.get("type")
        parts: list[str] = []
        if isinstance(code, str) and code.strip():
            parts.append(code.strip())
        if isinstance(message, str) and message.strip():
            parts.append(message.strip())
        if not parts:
            return self._compact_text(str(error))
        return self._compact_text(": ".join(parts))

    @staticmethod
    def _usage_from_payload(payload: dict[str, Any]) -> ChatUsage | None:
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return None
        return ChatUsage(
            prompt_eval_count=int(usage.get("prompt_tokens", 0) or 0),
            eval_count=int(usage.get("completion_tokens", 0) or 0),
            eval_duration=0,
            # OpenAI-style usage에는 보통 total_duration이 없으므로 0 또는 동명 필드만 사용한다.
            total_duration=int(usage.get("total_duration", 0) or 0),
        )

    def _build_chat_payload(
        self,
        *,
        model: str | None,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int | None,
        response_format: str | dict | None,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": self._temperature if temperature is None else temperature,
            "max_tokens": self._max_tokens if max_tokens is None else max_tokens,
            "stream": stream,
        }
        if model:
            payload["model"] = model
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
        requested_model = (model or "").strip()
        target_model = requested_model or self._default_model or None
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
            error_text = self._extract_api_error(body)
            if error_text:
                raise LemonadeClientError(
                    f"Lemonade chat request returned API error for model '{target_model}': {error_text}"
                )
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
        requested_model = (model or "").strip()
        target_model = requested_model or self._default_model or None
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
        emitted_content = False
        reasoning_chunks: list[str] = []

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

                    error_text = self._extract_api_error(chunk)
                    if error_text:
                        raise LemonadeClientError(
                            f"Lemonade streaming API error for model '{target_model}': {error_text}"
                        )

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
                        emitted_content = True
                        yield content
                        continue

                    if isinstance(delta, dict) and not emitted_content:
                        reasoning = delta.get("reasoning_content")
                        if isinstance(reasoning, str) and reasoning:
                            reasoning_chunks.append(reasoning)
                            continue

                    # 일부 구현은 delta 대신 message.content를 전달한다.
                    fallback_content = self._extract_content(first_choice)
                    if fallback_content:
                        emitted_content = True
                        yield fallback_content
                if not emitted_content and reasoning_chunks:
                    yield "".join(reasoning_chunks)
            self._mark_healthy()
        except (httpx.HTTPError, asyncio.TimeoutError, OSError) as exc:
            self._mark_unhealthy(exc)
            await self.recover_connection()
            error_text = self._format_exception(exc)
            raise LemonadeClientError(
                f"Lemonade streaming request failed for model '{target_model}': {error_text}"
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
        requested_model = (model or "").strip()
        target_model = requested_model or self._default_model or None
        payload: dict[str, Any] = {"input": texts}
        if target_model:
            payload["model"] = target_model
        try:
            response = await client.post(
                self._endpoint("/embeddings"),
                json=payload,
                timeout=timeout or self._timeout_default,
            )
            response.raise_for_status()
        except (httpx.HTTPError, asyncio.TimeoutError, OSError) as exc:
            self._mark_unhealthy(exc)
            error_text = self._format_exception(exc)
            raise LemonadeClientError(
                f"Embedding request failed for model '{target_model}': {error_text}"
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
        requested_model = (model or "").strip()
        target_model = requested_model or self._default_model or None
        effective_timeout = timeout or self._timeout_default

        # 1) 전용 /rerank endpoint 시도
        try:
            payload: dict[str, Any] = {
                "query": query,
                "documents": documents,
            }
            if target_model:
                payload["model"] = target_model
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
                error_text = self._format_exception(exc)
                raise LemonadeClientError(
                    f"Rerank request failed: {error_text}"
                ) from exc
        except (httpx.HTTPError, asyncio.TimeoutError, OSError) as exc:
            self._mark_unhealthy(exc)
            error_text = self._format_exception(exc)
            raise LemonadeClientError(f"Rerank request failed: {error_text}") from exc

        # 2) chat 기반 폴백 (마지막 수단)
        return await self._rerank_via_chat(
            query, documents, target_model, top_n, effective_timeout,
        )

    async def _rerank_via_chat(
        self,
        query: str,
        documents: list[str],
        model: str | None,
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
                        error=self._format_exception(exc),
                    )
                    await asyncio.sleep(delay)

        last_error_text = (
            self._format_exception(last_error) if last_error is not None else "unknown"
        )
        raise LemonadeClientError(
            f"Lemonade request failed after {max_retries + 1} attempts: {last_error_text}"
        )
