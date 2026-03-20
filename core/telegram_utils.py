"""Telegram handler utilities for formatting and error-list handling.

These pure helper functions are split out from `TelegramHandler` to improve
testability and keep the main module smaller.
"""

from __future__ import annotations

import inspect
from typing import Any

from core.i18n import t


def format_memory_gb(value_mb: object) -> str:
    """Convert a value in MB to a formatted GB string."""
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
    """Convert a value into a list of strings."""
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def get_skill_reload_errors(engine: Any) -> list[str]:
    """Safely extract skill reload errors from the engine."""
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
    """Safely extract automation reload errors from the scheduler."""
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
    """Format an error list as a warning message."""
    preview = errors[:max_items]
    lines = [t("reload_warnings", lang, count=len(errors))]
    lines.extend(f"- {item}" for item in preview)
    if len(errors) > max_items:
        lines.append(t("reload_more", lang, count=len(errors) - max_items))
    return "\n".join(lines)
