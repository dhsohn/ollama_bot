from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from core.i18n import t

if TYPE_CHECKING:
    from core.telegram_handler import TelegramHandler

_CONTINUATION_TTL_SECONDS = 30 * 60
_CONTINUE_REQUEST_RE = re.compile(
    r"^\s*(continue|more|계속|이어서|이어줘|더\s*보여줘)\s*$",
    re.IGNORECASE,
)


def is_continue_request(text: str) -> bool:
    return bool(_CONTINUE_REQUEST_RE.match(text.strip()))


def cleanup_pending_continuations(
    self: TelegramHandler,
    *,
    monotonic_fn: Callable[[], float],
) -> None:
    if not self._pending_continuation:
        return
    now = monotonic_fn()
    expired_chat_ids = [
        chat_id
        for chat_id, pending in self._pending_continuation.items()
        if now > float(pending.get("expires", 0.0))
    ]
    for chat_id in expired_chat_ids:
        del self._pending_continuation[chat_id]


def take_pending_continuation(
    self: TelegramHandler,
    chat_id: int,
    *,
    monotonic_fn: Callable[[], float],
) -> dict[str, Any] | None:
    cleanup_pending_continuations(self, monotonic_fn=monotonic_fn)
    pending = self._pending_continuation.get(chat_id)
    if pending is None:
        return None
    del self._pending_continuation[chat_id]
    return pending


def set_pending_continuation(
    self: TelegramHandler,
    chat_id: int,
    *,
    root_query: str,
    turn: int,
    monotonic_fn: Callable[[], float],
) -> None:
    cleanup_pending_continuations(self, monotonic_fn=monotonic_fn)
    self._pending_continuation[chat_id] = {
        "root_query": root_query,
        "turn": max(1, turn),
        "expires": monotonic_fn() + _CONTINUATION_TTL_SECONDS,
    }


def build_continuation_prompt(pending: dict[str, Any]) -> str:
    return build_continuation_prompt_for_lang(pending, lang="ko")


def build_continuation_prompt_for_lang(
    pending: dict[str, Any],
    *,
    lang: str,
) -> str:
    root_query = str(pending.get("root_query", "")).strip()
    turn = max(1, int(pending.get("turn", 1)))
    return (
        f"{t('continuation_prompt_intro', lang)}\n"
        f"{t('continuation_prompt_no_repeat', lang)}\n"
        f"{t('continuation_prompt_summary_first', lang)}\n"
        f"{t('continuation_prompt_turn', lang, turn=turn)}\n"
        f"{t('continuation_prompt_root_query', lang)}\n{root_query}"
    ).strip()


def truncate_summary_line(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def extract_summary_points(
    cls: type[TelegramHandler],
    text: str,
    *,
    max_points: int = 3,
    lang: str = "ko",
) -> list[str]:
    content = text.strip()
    long_notice = t("stream_notice_max_total_chars", lang)
    marker = f"\n\n{long_notice}"
    if marker in content:
        content = content.split(marker, 1)[0].strip()
    elif content.startswith(long_notice):
        content = ""

    points: list[str] = []
    seen: set[str] = set()
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```"):
            continue
        line = re.sub(r"^(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if len(line) < 8:
            continue
        key = line.casefold()
        if key in seen:
            continue
        seen.add(key)
        points.append(cls._truncate_summary_line(line, max_chars=140))
        if len(points) >= max_points:
            return points

    collapsed = " ".join(part.strip() for part in content.splitlines() if part.strip())
    if collapsed:
        points.append(cls._truncate_summary_line(collapsed, max_chars=180))
    return points


def build_long_response_followup_message(
    cls: type[TelegramHandler],
    response_text: str,
    *,
    lang: str,
) -> str:
    points = cls._extract_summary_points(response_text, max_points=3, lang=lang)
    followup = t("continuation_manual_followup", lang)
    if points:
        summary = "\n".join(f"- {point}" for point in points)
        return (
            f"{t('continuation_summary_title', lang)}\n"
            f"{summary}\n\n"
            f"{followup}"
        )
    return followup
