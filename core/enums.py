"""Project-wide enum definitions."""

from __future__ import annotations

from enum import StrEnum


class RoutingTier(StrEnum):
    """Engine routing tiers."""

    SKILL = "skill"
    INSTANT = "instant"
    CACHE = "cache"
    FULL = "full"
