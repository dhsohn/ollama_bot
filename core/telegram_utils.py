"""텔레그램 핸들러 유틸리티 — 포맷팅, 에러 목록 처리.

TelegramHandler에서 사용되는 순수 유틸리티 함수들을 분리하여
테스트 용이성과 모듈 크기를 개선한다.
"""

from __future__ import annotations

import inspect
from typing import Any

from core.i18n import t


def format_memory_gb(value_mb: object) -> str:
    """MB 단위 값을 GB 문자열로 변환한다."""
    mb = 0.0
    if isinstance(value_mb, bool):
        mb = 0.0
    elif isinstance(value_mb, int | float):
        mb = max(0.0, float(value_mb))
    elif isinstance(value_mb, str):
        try:
            mb = max(0.0, float(value_mb.strip()))
        except ValueError:
            mb = 0.0

    gb = mb / 1024.0
    rounded = round(gb)
    if abs(gb - rounded) < 1e-9:
        return f"{int(rounded)}GB"
    if gb >= 10:
        return f"{gb:.1f}GB"
    return f"{gb:.2f}GB"


def coerce_error_list(value: object) -> list[str]:
    """값을 문자열 리스트로 변환한다."""
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def get_skill_reload_errors(engine: Any) -> list[str]:
    """엔진에서 스킬 로드 에러를 안전하게 추출한다."""
    getter = getattr(engine, "get_last_skill_load_errors", None)
    if not callable(getter):
        return []
    try:
        errors = getter()
    except Exception:
        return []
    if inspect.isawaitable(errors):
        closer = getattr(errors, "close", None)
        if callable(closer):
            closer()
        return []
    return coerce_error_list(errors)


def get_auto_reload_errors(scheduler: Any) -> list[str]:
    """스케줄러에서 자동화 로드 에러를 안전하게 추출한다."""
    if scheduler is None:
        return []
    getter = getattr(scheduler, "get_last_load_errors", None)
    if not callable(getter):
        return []
    try:
        errors = getter()
    except Exception:
        return []
    if inspect.isawaitable(errors):
        closer = getattr(errors, "close", None)
        if callable(closer):
            closer()
        return []
    return coerce_error_list(errors)


def format_reload_warnings(
    errors: list[str], max_items: int = 3, lang: str = "ko",
) -> str:
    """에러 목록을 경고 메시지로 포맷한다."""
    preview = errors[:max_items]
    lines = [t("reload_warnings", lang, count=len(errors))]
    lines.extend(f"- {item}" for item in preview)
    if len(errors) > max_items:
        lines.append(t("reload_more", lang, count=len(errors) - max_items))
    return "\n".join(lines)
