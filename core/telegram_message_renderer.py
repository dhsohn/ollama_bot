"""텔레그램 메시지 렌더링 유틸리티."""

from __future__ import annotations

import time
from collections.abc import AsyncIterable, Awaitable, Callable
from dataclasses import dataclass
from html import escape
from typing import Any

from core.text_utils import sanitize_model_output


@dataclass
class StreamResult:
    """stream_and_render의 반환 결과."""

    full_response: str
    last_message: Any  # telegram.Message
    tier: str = "full"
    intent: str | None = None
    cache_id: int | None = None
    usage: Any = None  # ChatUsage


def escape_html(value: object) -> str:
    """HTML parse mode용 최소 이스케이프."""
    return escape(str(value), quote=False)


def split_message(text: str, max_length: int = 4096) -> list[str]:
    """긴 메시지를 단락 기준으로 분할한다."""
    if len(text) <= max_length:
        return [text]

    parts: list[str] = []
    while text:
        if len(text) <= max_length:
            parts.append(text)
            break

        split_at = text.rfind("\n\n", 0, max_length)
        if split_at == -1:
            split_at = text.rfind("\n", 0, max_length)
        if split_at == -1:
            split_at = text.rfind(" ", 0, max_length)
        if split_at == -1:
            split_at = max_length

        parts.append(text[:split_at])
        text = text[split_at:].lstrip()

    return parts


def _calc_edit_interval(elapsed: float) -> float:
    """경과 시간에 따라 동적 편집 간격을 계산한다.

    초반엔 빠르게 편집하여 응답이 오는 느낌을 주고,
    시간이 길어지면 간격을 넓혀 API 호출을 줄인다.
    """
    if elapsed < 3.0:
        return 0.5
    if elapsed < 10.0:
        return 1.0
    return 2.0


async def stream_and_render(
    stream: AsyncIterable[str],
    sent_message: Any,
    reply_text: Callable[[str], Awaitable[Any]],
    split_message_fn: Callable[[str], list[str]],
    *,
    edit_interval: float = 1.0,
    edit_char_threshold: int = 100,
    max_edit_length: int = 4096,
) -> StreamResult:
    """스트리밍 청크를 텔레그램 메시지 편집/분할 전송으로 렌더링한다."""
    full_response = ""
    stream_start = time.monotonic()
    last_edit_time = stream_start
    last_edit_len = 0
    last_msg = sent_message

    async for chunk in stream:
        full_response += chunk

        now = time.monotonic()
        chars_since_edit = len(full_response) - last_edit_len
        elapsed = now - stream_start
        effective_interval = (
            edit_interval if edit_interval < _calc_edit_interval(0)
            else _calc_edit_interval(elapsed)
        )
        time_since_edit = now - last_edit_time

        if (
            time_since_edit >= effective_interval
            and chars_since_edit >= edit_char_threshold
        ):
            display_source = sanitize_model_output(full_response)
            if not display_source.strip():
                continue
            display_text = display_source + " ▌"
            if len(display_text) > max_edit_length:
                # 길이 초과 시 중간 편집은 건너뛰고 최종 분할 전송으로 마무리한다.
                last_edit_time = now
                last_edit_len = len(full_response)
                continue
            try:
                await sent_message.edit_text(display_text)
                last_edit_time = now
                last_edit_len = len(full_response)
            except Exception:
                pass

    if full_response:
        rendered_response = sanitize_model_output(full_response)
        parts = split_message_fn(rendered_response)
        for idx, part in enumerate(parts):
            if idx == 0:
                try:
                    await sent_message.edit_text(part)
                    last_msg = sent_message
                except Exception:
                    last_msg = await reply_text(part)
            else:
                last_msg = await reply_text(part)
    else:
        await sent_message.edit_text("응답을 생성하지 못했습니다.")

    return StreamResult(full_response=sanitize_model_output(full_response), last_message=last_msg)
