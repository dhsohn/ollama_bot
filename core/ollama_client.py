"""Ollama API 클라이언트.

ollama 라이브러리의 AsyncClient를 래핑하여
타임아웃, 재시도, 모델 관리, 스트리밍 응답을 제공한다.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from ollama import AsyncClient, ResponseError

from core.config import OllamaConfig
from core.logging_setup import get_logger


# ── 응답 메타데이터 ──


@dataclass
class ChatUsage:
    """LLM 호출 사용량 메타데이터."""

    prompt_eval_count: int = 0
    eval_count: int = 0
    eval_duration: int = 0  # nanoseconds
    total_duration: int = 0  # nanoseconds


@dataclass
class ChatResponse:
    """LLM 응답 + 메타데이터."""

    content: str
    usage: ChatUsage | None = None

    def __str__(self) -> str:
        return self.content


@dataclass
class ChatStreamState:
    """요청 단위 스트리밍 메타데이터."""

    usage: ChatUsage | None = None


class OllamaClientError(Exception):
    """Ollama 통신 실패."""


class ModelNotFoundError(OllamaClientError):
    """요청한 모델이 존재하지 않음."""


class OllamaClient:
    """Ollama API 클라이언트. 재시도, 타임아웃, 스트리밍을 지원한다."""

    def __init__(self, config: OllamaConfig) -> None:
        self._host = config.host
        self._default_model = config.model
        self._temperature = config.temperature
        self._max_tokens = config.max_tokens
        self._num_ctx: int = getattr(config, "num_ctx", 8192)
        self._system_prompt = config.system_prompt
        self._client: AsyncClient | None = None
        self._auto_reconnect_enabled = False
        self._is_healthy = True
        self._last_connection_error: str | None = None
        self._next_reconnect_at = 0.0
        self._reconnect_cooldown_seconds = 15.0
        self._reconnect_lock = asyncio.Lock()
        self._logger = get_logger("ollama_client")

    @property
    def default_model(self) -> str:
        return self._default_model

    @default_model.setter
    def default_model(self, model: str) -> None:
        self._default_model = model

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    async def initialize(self) -> None:
        """클라이언트를 생성하고 연결을 확인한다."""
        client = AsyncClient(host=self._host)
        try:
            response = await client.list()
            available = [m.model for m in response.models if m.model is not None]
            self._logger.info(
                "ollama_connected",
                host=self._host,
                models_count=len(available),
            )
            if self._default_model not in available:
                self._logger.error(
                    "default_model_not_found_fail_fast",
                    model=self._default_model,
                    available=available,
                )
                available_text = ", ".join(available) if available else "(none)"
                raise ModelNotFoundError(
                    f"Default model '{self._default_model}' not found on {self._host}. "
                    f"Available models: {available_text}. "
                    f"Pull model first: ollama pull {self._default_model}"
                )
            self._client = client
            self._auto_reconnect_enabled = True
            self._mark_healthy()
        except ModelNotFoundError:
            self._mark_unhealthy("default model missing")
            raise
        except Exception as exc:
            self._logger.error("ollama_connection_failed", error=str(exc))
            self._mark_unhealthy(exc)
            raise OllamaClientError(
                f"Failed to connect to Ollama at {self._host}: {exc}"
            ) from exc

    async def close(self) -> None:
        """클라이언트 리소스를 정리한다."""
        self._client = None
        self._auto_reconnect_enabled = False
        self._is_healthy = False

    def _mark_healthy(self) -> None:
        self._is_healthy = True
        self._last_connection_error = None
        self._next_reconnect_at = 0.0

    def _mark_unhealthy(self, error: Exception | str) -> None:
        self._is_healthy = False
        self._last_connection_error = str(error)
        self._next_reconnect_at = time.monotonic() + self._reconnect_cooldown_seconds

    async def recover_connection(self, *, force: bool = False) -> bool:
        """연결 장애 시 재연결을 시도한다.

        Returns:
            재연결 성공 여부.
        """
        if not self._auto_reconnect_enabled:
            return False
        if self._client is None:
            return False

        now = time.monotonic()
        if not force and now < self._next_reconnect_at:
            return False

        async with self._reconnect_lock:
            now = time.monotonic()
            if not force and now < self._next_reconnect_at:
                return False

            candidate = AsyncClient(host=self._host)
            try:
                response = await candidate.list()
            except Exception as exc:
                self._mark_unhealthy(exc)
                self._logger.warning("ollama_reconnect_failed", error=str(exc))
                return False

            available = [m.model for m in response.models if m.model is not None]
            if self._default_model not in available:
                error = ModelNotFoundError(
                    f"Default model '{self._default_model}' not found on {self._host}"
                )
                self._mark_unhealthy(error)
                self._logger.warning(
                    "ollama_reconnect_default_model_missing",
                    model=self._default_model,
                    available=available,
                )
                return False

            self._client = candidate
            self._mark_healthy()
            self._logger.info(
                "ollama_reconnected",
                host=self._host,
                models_count=len(available),
            )
            return True

    def _require_client(self) -> AsyncClient:
        if self._client is None:
            raise RuntimeError("OllamaClient가 아직 초기화되지 않았습니다.")
        return self._client

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int = 60,
        response_format: str | dict | None = None,
    ) -> ChatResponse:
        """비스트리밍 채팅 요청. 재시도 포함."""
        model = model or self._default_model
        options = {
            "temperature": self._temperature if temperature is None else temperature,
            "num_predict": self._max_tokens if max_tokens is None else max_tokens,
            "num_ctx": self._num_ctx,
        }

        async def _do_chat() -> ChatResponse:
            client = self._require_client()
            kwargs: dict = dict(
                model=model,
                messages=messages,
                options=options,
            )
            if response_format is not None:
                kwargs["format"] = response_format
            response = await asyncio.wait_for(
                client.chat(**kwargs),
                timeout=timeout,
            )
            usage = ChatUsage(
                prompt_eval_count=getattr(response, "prompt_eval_count", 0) or 0,
                eval_count=getattr(response, "eval_count", 0) or 0,
                eval_duration=getattr(response, "eval_duration", 0) or 0,
                total_duration=getattr(response, "total_duration", 0) or 0,
            )
            return ChatResponse(content=response.message.content, usage=usage)

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
        """스트리밍 채팅 요청. 청크를 순차적으로 반환한다."""
        client = self._require_client()
        model = model or self._default_model
        options = {
            "temperature": self._temperature if temperature is None else temperature,
            "num_predict": self._max_tokens if max_tokens is None else max_tokens,
            "num_ctx": self._num_ctx,
        }
        state = stream_state or ChatStreamState()
        state.usage = None

        try:
            response = await asyncio.wait_for(
                client.chat(
                    model=model,
                    messages=messages,
                    options=options,
                    stream=True,
                ),
                timeout=timeout,
            )

            iterator = response.__aiter__()
            while True:
                try:
                    chunk = await asyncio.wait_for(iterator.__anext__(), timeout=timeout)
                except StopAsyncIteration:
                    break
                if chunk.message.content:
                    yield chunk.message.content
                # 마지막 청크(done=True)에서 usage 메타데이터 추출
                if getattr(chunk, "done", False):
                    state.usage = ChatUsage(
                        prompt_eval_count=getattr(chunk, "prompt_eval_count", 0) or 0,
                        eval_count=getattr(chunk, "eval_count", 0) or 0,
                        eval_duration=getattr(chunk, "eval_duration", 0) or 0,
                        total_duration=getattr(chunk, "total_duration", 0) or 0,
                    )
            self._mark_healthy()
        except (ResponseError, asyncio.TimeoutError, OSError) as exc:
            self._mark_unhealthy(exc)
            await self.recover_connection()
            raise OllamaClientError(
                f"Ollama streaming request failed for model '{model}': {exc}"
            ) from exc

    async def list_models(self) -> list[dict]:
        """로컬에 설치된 모델 목록을 반환한다."""
        client = self._require_client()
        try:
            response = await client.list()
            self._mark_healthy()
        except Exception as exc:
            self._mark_unhealthy(exc)
            await self.recover_connection()
            raise
        return [
            {
                "name": m.model,
                "size": m.size,
                "modified_at": str(m.modified_at) if m.modified_at else None,
            }
            for m in response.models
        ]

    async def get_model_info(self, model: str) -> dict:
        """특정 모델의 상세 정보를 반환한다."""
        client = self._require_client()
        try:
            response = await client.show(model)
            self._mark_healthy()
            return {
                "model": model,
                "modelfile": response.modelfile if hasattr(response, "modelfile") else None,
                "parameters": response.parameters if hasattr(response, "parameters") else None,
            }
        except ResponseError as exc:
            self._mark_unhealthy(exc)
            await self.recover_connection()
            raise ModelNotFoundError(f"Model '{model}' not found: {exc}") from exc

    async def health_check(self, *, attempt_recovery: bool = False) -> dict:
        """Ollama 서버 상태를 확인한다."""
        client = self._require_client()
        try:
            response = await client.list()
            models = [m.model for m in response.models]
            self._mark_healthy()
            return {
                "status": "ok",
                "host": self._host,
                "models_count": len(models),
                "models": models,
                "default_model": self._default_model,
                "default_model_available": self._default_model in models,
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

    async def prepare_model(
        self,
        *,
        model: str | None = None,
        role: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        """Ollama는 별도 사전 로드 단계가 없어 no-op 처리한다."""
        _ = (model, role, timeout_seconds)
        return None

    async def _retry_with_backoff(
        self,
        coro_factory,
        max_retries: int = 2,
    ) -> ChatResponse:
        """재시도 래퍼. 지수 백오프 적용 (1초, 3초)."""
        backoff_delays = [1, 3]
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                response = await coro_factory()
                self._mark_healthy()
                return response
            except (ResponseError, asyncio.TimeoutError, OSError) as exc:
                last_error = exc
                self._mark_unhealthy(exc)
                if attempt < max_retries:
                    await self.recover_connection(force=True)
                    delay = backoff_delays[min(attempt, len(backoff_delays) - 1)]
                    self._logger.warning(
                        "ollama_retry",
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(exc),
                    )
                    await asyncio.sleep(delay)

        raise OllamaClientError(
            f"Ollama request failed after {max_retries + 1} attempts: {last_error}"
        )
