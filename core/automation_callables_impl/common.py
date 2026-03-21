"""자동화 callable 공용 상수/유틸리티."""

from __future__ import annotations

import asyncio
import functools
import json
import re
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

ROLE_LABELS = {
    "user": "사용자",
    "assistant": "봇",
    "system": "시스템",
}

PREFERENCES_SCHEMA: dict = {
    "type": "array",
    "maxItems": 10,
    "items": {
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "value": {"type": "string"},
        },
        "required": ["key", "value"],
    },
}

DAILY_SUMMARY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "topics": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 3,
        },
        "decisions": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 5,
        },
        "todos": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 5,
        },
        "notes": {"type": ["string", "null"]},
    },
    "required": ["topics", "decisions", "todos", "notes"],
}

TRIAGE_SCHEMA: dict = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "event": {"type": "string"},
            "severity": {"type": "string", "enum": ["urgent", "warning", "low"]},
            "cause": {"type": "string"},
            "action": {"type": "string"},
            "recurring": {"type": "boolean"},
        },
        "required": ["event", "severity", "cause", "action", "recurring"],
    },
}

SEVERITY_ICONS = {"urgent": "\U0001f534", "warning": "\U0001f7e1", "low": "\U0001f7e2"}

MEMORY_HYGIENE_SCHEMA: dict = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "keep_key": {"type": "string"},
            "delete_key": {"type": "string"},
            "reason": {"type": "string", "enum": ["duplicate", "conflict"]},
        },
        "required": ["keep_key", "delete_key", "reason"],
    },
}

STALE_EVALUATION_SCHEMA: dict = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "stale": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["key", "stale", "reason"],
    },
}

CONSOLIDATION_MERGE_SCHEMA: dict = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "merge_keys": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
            },
            "new_key": {"type": "string"},
            "new_value": {"type": "string"},
        },
        "required": ["merge_keys", "new_key", "new_value"],
    },
}

FEEDBACK_ANALYSIS_SCHEMA: dict = {
    "type": "array",
    "maxItems": 10,
    "items": {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["avoid", "prefer", "style"],
            },
            "guideline": {"type": "string"},
        },
        "required": ["type", "guideline"],
    },
}

SQLITE_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_log_level(entry: dict[str, Any]) -> str:
    """구조화 로그 엔트리에서 level 키를 정규화한다."""
    level = entry.get("log_level", "")
    if not level:
        level = entry.get("level", "")
    return str(level).strip().lower()


def safe_timezone(name: str, logger: Any) -> tzinfo:
    """설정값에서 안전한 타임존 객체를 반환한다."""
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        logger.warning(
            "invalid_timezone_fallback",
            timezone=name,
            fallback="UTC",
        )
        return UTC


def truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def resolve_llm_timeout(
    *,
    timeout: int | None,
    llm_timeout: int | None,
) -> tuple[int | None, bool]:
    """Resolve an explicit per-LLM timeout and whether it should be enforced."""
    if llm_timeout is not None:
        return max(1, int(llm_timeout)), True
    if timeout is not None:
        return max(1, int(timeout)), False
    return None, False


def parse_json_array(text: str) -> list[dict] | None:
    """LLM 응답에서 JSON 배열을 파싱한다. 코드 펜스를 제거하고 폴백 시도."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")

    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(cleaned[start : end + 1])
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def count_recent_errors(
    log_path: Path,
    hours_back: int,
    max_entries: int = 0,
) -> tuple[list[dict], int, int]:
    """로그 파일에서 최근 error/warning 엔트리를 수집한다."""
    entries: list[dict] = []
    error_count = 0
    warning_count = 0

    if not log_path.exists() or not log_path.is_dir():
        return entries, error_count, warning_count

    log_files = sorted(
        log_path.glob("app.log*"),
        key=lambda p: (p.name == "app.log", p.name),
        reverse=True,
    )
    if not log_files:
        return entries, error_count, warning_count

    cutoff = datetime.now(UTC) - timedelta(hours=hours_back)
    collect = max_entries > 0

    for lf in log_files:
        try:
            text = lf.read_text(encoding="utf-8")
        except OSError:
            continue

        for line in reversed(text.splitlines()):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                continue

            level = get_log_level(entry)
            if level not in ("error", "warning"):
                continue

            ts_str = entry.get("timestamp", "")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=UTC)
                    if ts < cutoff:
                        continue
                except (ValueError, TypeError):
                    pass

            if level == "error":
                error_count += 1
            else:
                warning_count += 1

            if collect:
                entries.append(entry)
                if len(entries) >= max_entries:
                    return entries, error_count, warning_count

    return entries, error_count, warning_count


async def count_recent_errors_async(
    log_path: Path,
    hours_back: int,
    max_entries: int = 0,
) -> tuple[list[dict], int, int]:
    """count_recent_errors를 스레드풀에서 실행해 이벤트루프를 보호한다."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        functools.partial(
            count_recent_errors,
            log_path,
            hours_back,
            max_entries,
        ),
    )
