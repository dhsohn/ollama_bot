"""프로젝트 전역 Enum 정의."""

from __future__ import annotations

from enum import StrEnum


class RoutingTier(StrEnum):
    """엔진 라우팅 계층."""

    SKILL = "skill"
    INSTANT = "instant"
    CACHE = "cache"
    FULL = "full"
