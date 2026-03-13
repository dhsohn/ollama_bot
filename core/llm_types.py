"""LLM 공용 데이터 타입.

현재 Ollama 클라이언트가 공유하는
응답 메타데이터 및 스트리밍 상태 타입을 정의한다.
"""

from __future__ import annotations

from dataclasses import dataclass


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
