from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.telegram_handler import TelegramHandler

_CONTINUATION_TTL_SECONDS = 30 * 60
_CONTINUE_REQUEST_RE = re.compile(
    r"^\s*(continue|more|계속|이어서|이어줘|더\s*보여줘)\s*$",
    re.IGNORECASE,
)
_LONG_RESPONSE_STOP_NOTICE_PREFIX = "⚠️ 응답이 길어서 여기서 끊었습니다."


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
    root_query = str(pending.get("root_query", "")).strip()
    turn = max(1, int(pending.get("turn", 1)))
    return (
        "직전 답변을 이어서 작성해줘.\n"
        "- 이미 설명한 내용은 반복하지 말고 중단 지점부터 이어서 설명해줘.\n"
        "- 먼저 3줄 이내로 지금까지 핵심을 요약해줘.\n"
        "- 답변이 다시 길어지면 마지막 줄에 '계속하려면 계속이라고 입력해주세요.'를 적어줘.\n"
        f"- 이어보기 턴: {turn}\n"
        f"[원 질문]\n{root_query}"
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
) -> list[str]:
    content = text.strip()
    marker = f"\n\n{_LONG_RESPONSE_STOP_NOTICE_PREFIX}"
    if marker in content:
        content = content.split(marker, 1)[0].strip()
    elif content.startswith(_LONG_RESPONSE_STOP_NOTICE_PREFIX):
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
) -> str:
    points = cls._extract_summary_points(response_text, max_points=3)
    if points:
        summary = "\n".join(f"- {point}" for point in points)
        return (
            "📌 지금까지 요약\n"
            f"{summary}\n\n"
            "계속 보려면 /continue 또는 '계속'이라고 입력하세요."
        )
    return "응답이 길어서 여기서 끊었습니다. /continue 또는 '계속'이라고 입력하면 이어서 보여드릴게요."
