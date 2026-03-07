"""프로젝트 전역 Enum 정의."""

from __future__ import annotations

from enum import Enum


class RoutingTier(str, Enum):
    """엔진 라우팅 계층."""

    SKILL = "skill"
    INSTANT = "instant"
    CACHE = "cache"
    FULL = "full"


class DFTIntent(str, Enum):
    """DFT 자연어 질의 의도 분류."""

    STATS = "stats"
    FAILED = "failed"
    COMPARE = "compare"
    LOWEST_ENERGY = "lowest_energy"
    RECENT = "recent"
    IMAGINARY_FREQ = "imaginary_freq"
    BY_CALCTYPE_FREQ = "by_calctype_freq"
    BY_CALCTYPE_OPT = "by_calctype_opt"
    BY_CALCTYPE_TS = "by_calctype_ts"
    GENERAL = "general"
