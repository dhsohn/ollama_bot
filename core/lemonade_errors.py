"""Shared Lemonade client exceptions."""

from __future__ import annotations


class LemonadeClientError(Exception):
    """Lemonade 통신 실패."""


class LemonadeModelNotFoundError(LemonadeClientError):
    """요청한 모델이 서버 모델 목록에 없음."""
