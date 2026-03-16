"""TelegramHandler 일반 메시지 조립 레이어."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from telegram.ext import ContextTypes

from core import telegram_continuation, telegram_streaming

if TYPE_CHECKING:
    from telegram import Update

    from core.telegram_handler import TelegramHandler


async def handle_message(
    self: TelegramHandler,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    await telegram_streaming.handle_message(self, update, context)


async def handle_message_impl(
    self: TelegramHandler,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    text_override: str | None = None,
    force_continuation: bool = False,
    auto_continuation_turn: int = 0,
    stream_and_render_fn: Callable[..., Any],
    detect_output_anomalies_fn: Callable[[str, str], list[str]],
) -> None:
    await telegram_streaming.handle_message_impl(
        self,
        update,
        context,
        text_override=text_override,
        force_continuation=force_continuation,
        auto_continuation_turn=auto_continuation_turn,
        stream_and_render_fn=stream_and_render_fn,
        detect_output_anomalies_fn=detect_output_anomalies_fn,
    )


def should_auto_trigger_analyze_all(text: str) -> bool:
    return telegram_streaming.should_auto_trigger_analyze_all(text)


async def run_analyze_all_flow(
    self: TelegramHandler,
    *,
    chat: Any,
    message: Any,
    query: str,
    auto_triggered: bool,
) -> None:
    await telegram_streaming.run_analyze_all_flow(
        self,
        chat=chat,
        message=message,
        query=query,
        auto_triggered=auto_triggered,
    )


def is_continue_request(text: str) -> bool:
    return telegram_continuation.is_continue_request(text)


def cleanup_pending_continuations(
    self: TelegramHandler,
    *,
    monotonic_fn: Callable[[], float],
) -> None:
    telegram_continuation.cleanup_pending_continuations(
        self,
        monotonic_fn=monotonic_fn,
    )


def take_pending_continuation(
    self: TelegramHandler,
    chat_id: int,
    *,
    monotonic_fn: Callable[[], float],
) -> dict[str, Any] | None:
    return telegram_continuation.take_pending_continuation(
        self,
        chat_id,
        monotonic_fn=monotonic_fn,
    )


def set_pending_continuation(
    self: TelegramHandler,
    chat_id: int,
    *,
    root_query: str,
    turn: int,
    monotonic_fn: Callable[[], float],
) -> None:
    telegram_continuation.set_pending_continuation(
        self,
        chat_id,
        root_query=root_query,
        turn=turn,
        monotonic_fn=monotonic_fn,
    )


def build_continuation_prompt(pending: dict[str, Any], *, lang: str) -> str:
    return telegram_continuation.build_continuation_prompt_for_lang(
        pending,
        lang=lang,
    )


def truncate_summary_line(text: str, *, max_chars: int) -> str:
    return telegram_continuation.truncate_summary_line(text, max_chars=max_chars)


def extract_summary_points(
    cls: type[TelegramHandler],
    text: str,
    *,
    max_points: int = 3,
    lang: str = "ko",
) -> list[str]:
    return telegram_continuation.extract_summary_points(
        cls,
        text,
        max_points=max_points,
        lang=lang,
    )


def build_long_response_followup_message(
    cls: type[TelegramHandler],
    response_text: str,
    *,
    lang: str,
) -> str:
    return telegram_continuation.build_long_response_followup_message(
        cls,
        response_text,
        lang=lang,
    )


async def keep_typing(chat: Any, stop_event: asyncio.Event) -> None:
    await telegram_streaming.keep_typing(chat, stop_event)
