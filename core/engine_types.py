"""Shared engine-facing types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from core.enums import RoutingTier
from core.skill_manager import SkillDefinition

if TYPE_CHECKING:
    from core.intent_router import ContextStrategy, RouteResult


class ContextProvider:
    """Generic interface for injecting extra request context."""

    async def get_context(self, text: str) -> str | None:
        raise NotImplementedError


@dataclass
class _PreparedRequest:
    """Precomputed chat request state."""

    skill: SkillDefinition | None
    messages: list[dict[str, str]]
    timeout: int
    max_tokens: int | None = None


@dataclass
class _PreparedFullRequest:
    """Precomputed full-tier request state."""

    messages: list[dict[str, str]]
    timeout: int
    max_tokens: int | None
    target_model: str | None
    rag_result: Any = None
    planner_applied: bool = False
    review_enabled: bool = False
    stream_buffering: bool = False


@dataclass
class _StreamMeta:
    """Last streaming-request metadata."""

    tier: RoutingTier = RoutingTier.FULL
    intent: str | None = None
    cache_id: int | None = None
    usage: Any = None
    model_role: str | None = None
    rag_trace: dict | None = None
    created_at: float = 0.0


@dataclass
class _RoutingDecision:
    """Routing result before an LLM call."""

    tier: RoutingTier = RoutingTier.FULL
    skill: SkillDefinition | None = None
    instant: Any = None
    route: RouteResult | None = None
    cached: Any = None
    rag_result: Any = None

    @property
    def intent(self) -> str | None:
        return self.route.intent if self.route else None

    @property
    def strategy(self) -> ContextStrategy | None:
        return self.route.context_strategy if self.route else None


@dataclass
class _FullScanSegment:
    """Merged RAG full-scan segment."""

    source_path: str
    start_chunk_id: int
    end_chunk_id: int
    text: str
