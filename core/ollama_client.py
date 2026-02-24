"""Ollama API 클라이언트.

ollama 라이브러리의 AsyncClient를 래핑하여
타임아웃, 재시도, 모델 관리, 스트리밍 응답을 제공한다.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

from ollama import AsyncClient, ResponseError

from core.config import OllamaConfig
from core.logging_setup import get_logger


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
        self._system_prompt = config.system_prompt
        self._client: AsyncClient | None = None
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
        self._client = AsyncClient(host=self._host)
        try:
            response = await self._client.list()
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
        except ModelNotFoundError:
            raise
        except Exception as exc:
            self._logger.error("ollama_connection_failed", error=str(exc))
            raise OllamaClientError(
                f"Failed to connect to Ollama at {self._host}: {exc}"
            ) from exc

    async def close(self) -> None:
        """클라이언트 리소스를 정리한다."""
        self._client = None

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
        format: str | dict | None = None,
    ) -> str:
        """비스트리밍 채팅 요청. 재시도 포함."""
        client = self._require_client()
        model = model or self._default_model
        options = {
            "temperature": self._temperature if temperature is None else temperature,
            "num_predict": self._max_tokens if max_tokens is None else max_tokens,
        }

        async def _do_chat() -> str:
            kwargs: dict = dict(
                model=model,
                messages=messages,
                options=options,
            )
            if format is not None:
                kwargs["format"] = format
            response = await asyncio.wait_for(
                client.chat(**kwargs),
                timeout=timeout,
            )
            return response.message.content

        return await self._retry_with_backoff(_do_chat)

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int = 60,
    ) -> AsyncGenerator[str, None]:
        """스트리밍 채팅 요청. 청크를 순차적으로 반환한다."""
        client = self._require_client()
        model = model or self._default_model
        options = {
            "temperature": self._temperature if temperature is None else temperature,
            "num_predict": self._max_tokens if max_tokens is None else max_tokens,
        }

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
        except (ResponseError, asyncio.TimeoutError, OSError) as exc:
            raise OllamaClientError(
                f"Ollama streaming request failed for model '{model}': {exc}"
            ) from exc

    async def list_models(self) -> list[dict]:
        """로컬에 설치된 모델 목록을 반환한다."""
        client = self._require_client()
        response = await client.list()
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
            return {
                "model": model,
                "modelfile": response.modelfile if hasattr(response, "modelfile") else None,
                "parameters": response.parameters if hasattr(response, "parameters") else None,
            }
        except ResponseError as exc:
            raise ModelNotFoundError(f"Model '{model}' not found: {exc}") from exc

    async def health_check(self) -> dict:
        """Ollama 서버 상태를 확인한다."""
        client = self._require_client()
        try:
            response = await client.list()
            models = [m.model for m in response.models]
            return {
                "status": "ok",
                "host": self._host,
                "models_count": len(models),
                "models": models,
                "default_model": self._default_model,
                "default_model_available": self._default_model in models,
            }
        except Exception as exc:
            return {
                "status": "error",
                "host": self._host,
                "error": str(exc),
            }

    async def _retry_with_backoff(
        self,
        coro_factory,
        max_retries: int = 2,
    ) -> str:
        """재시도 래퍼. 지수 백오프 적용 (1초, 3초)."""
        backoff_delays = [1, 3]
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                return await coro_factory()
            except (ResponseError, asyncio.TimeoutError, OSError) as exc:
                last_error = exc
                if attempt < max_retries:
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
