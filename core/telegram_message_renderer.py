"""Utilities for rendering Telegram messages."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import AsyncIterable, Awaitable, Callable
from dataclasses import dataclass
from html import escape
from typing import Any

from core.i18n import t
from core.logging_setup import get_logger
from core.text_utils import sanitize_model_output, should_defer_stream_display

_logger = get_logger("telegram_message_renderer")
_MAX_RENDERED_DUPLICATE_CHUNKS = 0


@dataclass
class StreamResult:
    """Return payload from ``stream_and_render``."""

    full_response: str
    last_message: Any  # telegram.Message
    stop_reason: str | None = None
    tier: str = "full"
    intent: str | None = None
    cache_id: int | None = None
    usage: Any = None  # ChatUsage


def escape_html(value: object) -> str:
    """Escape a value for Telegram HTML parse mode."""
    return escape(str(value), quote=False)


def split_message(text: str, max_length: int = 4096) -> list[str]:
    """Split a long message near paragraph boundaries."""
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
    """Adjust edit cadence over time to balance responsiveness and API churn."""
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
    first_chunk_timeout_seconds: float | None = None,
    chunk_timeout_seconds: float = 20.0,
    max_stream_seconds: float = 180.0,
    max_total_chars: int = 262_144,
    max_repeated_chunks: int = 200,
    lang: str = "ko",
) -> StreamResult:
    """Render streaming chunks via Telegram edits and follow-up messages."""
    full_response = ""
    stream_start = time.monotonic()
    last_edit_time = stream_start
    last_edit_len = 0
    last_msg = sent_message
    last_chunk: str | None = None
    repeated_chunk_count = 0
    stop_reason: str | None = None
    iterator = stream.__aiter__()
    received_first_chunk = False

    while True:
        elapsed_before_next = time.monotonic() - stream_start
        if max_stream_seconds > 0 and elapsed_before_next >= max_stream_seconds:
            stop_reason = "stream_timeout"
            break

        timeout_budget = chunk_timeout_seconds
        if not received_first_chunk and first_chunk_timeout_seconds is not None:
            timeout_budget = first_chunk_timeout_seconds
        effective_wait = timeout_budget if timeout_budget > 0 else None
        if max_stream_seconds > 0:
            remaining = max_stream_seconds - elapsed_before_next
            if remaining <= 0:
                stop_reason = "stream_timeout"
                break
            if effective_wait is None:
                effective_wait = remaining
            else:
                effective_wait = min(effective_wait, remaining)

        try:
            if effective_wait is None:
                chunk = await iterator.__anext__()
            else:
                chunk = await asyncio.wait_for(iterator.__anext__(), timeout=effective_wait)
        except StopAsyncIteration:
            break
        except TimeoutError:
            stop_reason = "chunk_timeout"
            break

        if not chunk:
            continue
        received_first_chunk = True

        if chunk == last_chunk:
            repeated_chunk_count += 1
            if max_repeated_chunks > 0 and repeated_chunk_count >= max_repeated_chunks:
                stop_reason = "repeated_chunks"
                break
            # Suppress duplicated chunks so repetition does not flood the chat.
            if repeated_chunk_count > _MAX_RENDERED_DUPLICATE_CHUNKS:
                continue
        else:
            last_chunk = chunk
            repeated_chunk_count = 0

        full_response += chunk
        if max_total_chars > 0 and len(full_response) >= max_total_chars:
            full_response = full_response[:max_total_chars]
            stop_reason = "max_total_chars"
            break

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
            if should_defer_stream_display(full_response, display_source):
                continue
            display_text = display_source + " ▌"
            if len(display_text) > max_edit_length:
                # Skip oversized mid-stream edits and finish with split final messages.
                last_edit_time = now
                last_edit_len = len(full_response)
                continue
            try:
                await sent_message.edit_text(display_text)
                last_edit_time = now
                last_edit_len = len(full_response)
            except Exception:
                pass

    close_stream = getattr(iterator, "aclose", None)
    if callable(close_stream):
        try:
            maybe_close = close_stream()
            if inspect.isawaitable(maybe_close):
                await maybe_close
        except Exception:
            pass

    notice = ""
    if stop_reason == "chunk_timeout":
        notice = t("stream_notice_chunk_timeout", lang)
    elif stop_reason == "stream_timeout":
        notice = t("stream_notice_stream_timeout", lang)
    elif stop_reason == "max_total_chars":
        notice = t("stream_notice_max_total_chars", lang)
    elif stop_reason == "repeated_chunks":
        notice = t("stream_notice_repeated_chunks", lang)
    if stop_reason is not None:
        _logger.warning(
            "stream_render_stopped",
            reason=stop_reason,
            elapsed=round(time.monotonic() - stream_start, 3),
            response_chars=len(full_response),
            repeated_chunk_count=repeated_chunk_count,
        )

    if full_response:
        rendered_response = sanitize_model_output(full_response)
        if notice:
            rendered_response = f"{rendered_response}\n\n{notice}".strip()
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
        await sent_message.edit_text(notice or t("stream_no_response", lang))

    final_response = sanitize_model_output(full_response)
    if notice:
        final_response = f"{final_response}\n\n{notice}".strip()
    return StreamResult(
        full_response=final_response,
        last_message=last_msg,
        stop_reason=stop_reason,
    )
