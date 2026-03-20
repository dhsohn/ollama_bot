"""Shared protocol definitions for LLM clients."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, Protocol

from core.llm_types import ChatResponse, ChatStreamState


class LLMClientProtocol(Protocol):
    """Common LLM client interface used by the engine and runtime."""

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
    """LLM interface that also supports embedding and reranking."""

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
