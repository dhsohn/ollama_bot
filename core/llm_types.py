"""Shared LLM data types.

Defines response metadata and streaming-state types shared by the current
Ollama client implementation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChatUsage:
    """Usage metadata for an LLM call."""

    prompt_eval_count: int = 0
    eval_count: int = 0
    eval_duration: int = 0  # nanoseconds
    total_duration: int = 0  # nanoseconds


@dataclass
class ChatResponse:
    """LLM response plus metadata."""

    content: str
    usage: ChatUsage | None = None

    def __str__(self) -> str:
        return self.content


@dataclass
class ChatStreamState:
    """Per-request streaming metadata."""

    usage: ChatUsage | None = None
