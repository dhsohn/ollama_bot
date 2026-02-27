"""LLM 클라이언트 공통 인터페이스 프로토콜."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, Protocol

from core.ollama_client import ChatResponse, ChatStreamState


class LLMClientProtocol(Protocol):
    """엔진/런타임에서 공통으로 사용하는 LLM 클라이언트 인터페이스."""

    @property
    def default_model(self) -> str: ...

    @default_model.setter
    def default_model(self, model: str) -> None: ...

    @property
    def system_prompt(self) -> str: ...

    async def initialize(self) -> None: ...

    async def close(self) -> None: ...

    async def prepare_model(
        self,
        *,
        model: str | None = None,
        role: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None: ...

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int = 60,
        response_format: str | dict | None = None,
    ) -> ChatResponse: ...

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int = 60,
        stream_state: ChatStreamState | None = None,
    ) -> AsyncGenerator[str, None]: ...

    async def list_models(self) -> list[dict]: ...

    async def health_check(self, *, attempt_recovery: bool = False) -> dict: ...

    async def recover_connection(self, *, force: bool = False) -> bool: ...


class RetrievalClientProtocol(LLMClientProtocol, Protocol):
    """임베딩/리랭크가 필요한 Lemonade 계열 확장 인터페이스."""

    @property
    def host(self) -> str: ...

    async def embed(
        self,
        texts: list[str],
        model: str | None = None,
        timeout: int | None = None,
    ) -> list[list[float]]: ...

    async def rerank(
        self,
        query: str,
        documents: list[str],
        model: str | None = None,
        top_n: int | None = None,
        timeout: int | None = None,
    ) -> list[dict[str, Any]]: ...

    async def check_model_availability(
        self,
        model_names: list[str],
    ) -> dict[str, bool]: ...
