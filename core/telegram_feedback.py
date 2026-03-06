"""TelegramHandler 피드백/사유 수집 구현."""

from __future__ import annotations

import time
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatType, ParseMode
from telegram.ext import ContextTypes

from core.security import AuthenticationError, RateLimitError

if TYPE_CHECKING:
    from core.telegram_handler import TelegramHandler


async def link_feedback_target(
    self: TelegramHandler,
    chat_id: int,
    result: Any,
) -> None:
    if (
        self._semantic_cache is None
        or result.cache_id is None
        or not result.last_message
    ):
        return
    with suppress(Exception):
        await self._semantic_cache.link_feedback_target(
            chat_id,
            result.last_message.message_id,
            result.cache_id,
        )


async def attach_feedback_controls(
    self: TelegramHandler,
    chat_id: int,
    user_text: str,
    result: Any,
) -> None:
    if not (
        self._feedback_enabled
        and self._config.feedback.show_buttons
        and result.full_response.strip()
        and result.last_message
    ):
        return

    target_msg = result.last_message
    self._cache_preview(chat_id, target_msg.message_id, user_text, result.full_response)
    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("\U0001f44d", callback_data=f"fb:1:{target_msg.message_id}"),
            InlineKeyboardButton("\U0001f44e", callback_data=f"fb:-1:{target_msg.message_id}"),
        ]
    ])
    with suppress(Exception):
        await target_msg.edit_reply_markup(reply_markup=keyboard)


def cleanup_preview_cache(self: TelegramHandler) -> None:
    """프리뷰 캐시의 TTL 만료 항목을 정리하고 크기 제한을 유지한다."""
    self._cleanup_pending_reasons()
    self._cleanup_pending_continuations()
    max_size = self._config.feedback.preview_cache_max_size
    ttl_hours = self._config.feedback.preview_cache_ttl_hours
    if max_size <= 0 or ttl_hours <= 0:
        self._preview_cache.clear()
        return

    now = time.monotonic()
    ttl_seconds = ttl_hours * 3600
    expired = [
        key
        for key, value in self._preview_cache.items()
        if now - value["ts"] > ttl_seconds
    ]
    for key in expired:
        del self._preview_cache[key]

    while len(self._preview_cache) > max_size:
        oldest_key = min(
            self._preview_cache,
            key=lambda key: self._preview_cache[key]["ts"],
        )
        del self._preview_cache[oldest_key]


def cleanup_pending_reasons(self: TelegramHandler) -> None:
    """사유 입력 대기 상태의 만료 항목을 주기적으로 정리한다."""
    if not self._pending_reason:
        return
    now = time.monotonic()
    expired_chat_ids = [
        chat_id
        for chat_id, pending in self._pending_reason.items()
        if now > float(pending.get("expires", 0.0))
    ]
    for chat_id in expired_chat_ids:
        del self._pending_reason[chat_id]


def cache_preview(
    self: TelegramHandler,
    chat_id: int,
    bot_message_id: int,
    user_text: str,
    bot_text: str,
) -> None:
    """프리뷰를 캐시에 저장한다. TTL 초과/크기 초과 시 정리."""
    max_chars = self._config.feedback.preview_max_chars
    max_size = self._config.feedback.preview_cache_max_size
    ttl_hours = self._config.feedback.preview_cache_ttl_hours
    if max_chars <= 0 or max_size <= 0 or ttl_hours <= 0:
        return
    self._cleanup_preview_cache()

    while len(self._preview_cache) >= max_size:
        oldest_key = min(
            self._preview_cache,
            key=lambda key: self._preview_cache[key]["ts"],
        )
        del self._preview_cache[oldest_key]

    self._preview_cache[(chat_id, bot_message_id)] = {
        "user": user_text[:max_chars],
        "bot": bot_text[:max_chars],
        "ts": time.monotonic(),
    }


def parse_feedback_callback_data(data: str | None) -> tuple[int, int] | None:
    if not data:
        return None
    try:
        _, rating_str, msg_id_str = data.split(":")
        return int(rating_str), int(msg_id_str)
    except (ValueError, AttributeError):
        return None


async def authorize_feedback_callback(
    self: TelegramHandler,
    chat_id: int,
    query: Any,
) -> bool:
    try:
        self._authorize_chat_id(chat_id)
    except AuthenticationError:
        await query.answer()
        return False
    except RateLimitError:
        await query.answer("요청이 너무 많습니다. 잠시 후 다시 시도해주세요.", show_alert=True)
        return False
    return True


async def handle_feedback_callback(
    self: TelegramHandler,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    _ = context
    query = update.callback_query
    if not query or not update.effective_chat:
        return
    if self._feedback is None:
        await query.answer()
        return
    if update.effective_chat.type != ChatType.PRIVATE:
        await query.answer("private chat에서만 사용할 수 있습니다.", show_alert=False)
        return

    parsed = self._parse_feedback_callback_data(query.data)
    if parsed is None:
        await query.answer("잘못된 피드백 요청입니다.", show_alert=True)
        return
    rating, bot_message_id = parsed

    chat_id = update.effective_chat.id
    if not await self._authorize_feedback_callback(chat_id, query):
        return

    if rating not in (-1, 1):
        await query.answer("지원하지 않는 피드백 값입니다.", show_alert=True)
        return

    self._cleanup_preview_cache()
    preview = self._preview_cache.get((chat_id, bot_message_id), {})
    is_update = await self._feedback.store_feedback(
        chat_id=chat_id,
        bot_message_id=bot_message_id,
        rating=rating,
        user_preview=preview.get("user"),
        bot_preview=preview.get("bot"),
    )

    pending = self._pending_reason.get(chat_id)
    if (
        pending is not None
        and pending.get("bot_message_id") == bot_message_id
        and (is_update or rating == 1)
    ):
        del self._pending_reason[chat_id]

    if (
        rating == -1
        and self._semantic_cache is not None
        and self._config.semantic_cache.invalidate_on_negative_feedback
    ):
        try:
            linked_cache_id = await self._semantic_cache.get_feedback_cache_id(
                chat_id,
                bot_message_id,
            )
            if linked_cache_id is not None:
                await self._semantic_cache.invalidate_by_id(linked_cache_id)
                self._logger.info(
                    "cache_invalidated_by_feedback",
                    chat_id=chat_id,
                    cache_id=linked_cache_id,
                )
        except Exception as exc:
            self._logger.debug("cache_feedback_invalidation_failed", error=str(exc))

    if rating == -1 and not is_update and self._config.feedback.collect_reason:
        existing_pending = self._pending_reason.get(chat_id)
        replaced_pending = False
        if existing_pending is not None:
            previous_expires = float(existing_pending.get("expires", 0.0))
            previous_bot_message_id = existing_pending.get("bot_message_id")
            del self._pending_reason[chat_id]
            replaced_pending = (
                time.monotonic() <= previous_expires
                and previous_bot_message_id != bot_message_id
            )

        timeout = self._config.feedback.reason_timeout_seconds
        self._pending_reason[chat_id] = {
            "bot_message_id": bot_message_id,
            "expires": time.monotonic() + timeout,
        }
        await query.answer("피드백 감사합니다!", show_alert=False)
        if query.message is not None and hasattr(query.message, "reply_text"):
            if replaced_pending:
                await query.message.reply_text(
                    "이전 사유 입력 요청은 자동 만료되어 최신 요청으로 교체되었어요."
                )
            await query.message.reply_text(
                "어떤 점이 아쉬웠나요? 사유를 입력해주세요.\n"
                "건너뛰려면 /skip 을 입력하세요."
            )
        return

    if is_update:
        await query.answer("피드백을 업데이트했어요.", show_alert=False)
    else:
        await query.answer("피드백 감사합니다!", show_alert=False)


async def cmd_feedback(
    self: TelegramHandler,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    _ = context
    if self._feedback is None:
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            "피드백 기능이 비활성화되어 있습니다."
        )
        return
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    stats = await self._feedback.get_user_stats(chat_id)
    table = (
        "항목        값\n"
        "─" * 18 + "\n"
        f"전체        {stats['total']}건\n"
        f"👍 긍정     {stats['positive']}건\n"
        f"👎 부정     {stats['negative']}건\n"
        f"만족도      {stats['satisfaction_rate']:.0%}"
    )
    text = f"📊 <b>피드백 통계</b>\n\n<pre>{table}</pre>"
    await update.effective_message.reply_text(  # type: ignore[union-attr]
        text,
        parse_mode=ParseMode.HTML,
    )


async def handle_reason_input(
    self: TelegramHandler,
    chat_id: int,
    text: str,
    update: Update,
) -> bool:
    pending = self._pending_reason.get(chat_id)
    if pending is None:
        return False

    if time.monotonic() > pending["expires"]:
        del self._pending_reason[chat_id]
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            "사유 입력 시간이 만료되었습니다."
        )
        return True

    min_chars = self._config.feedback.reason_min_chars
    max_chars = self._config.feedback.reason_max_chars
    reason = self._security.sanitize_input(text).strip()

    if len(reason) < min_chars:
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            f"사유는 최소 {min_chars}자 이상 입력해주세요. 건너뛰려면 /skip"
        )
        return True

    reason = reason[:max_chars]

    updated = False
    if self._feedback is not None:
        updated = await self._feedback.update_reason(
            chat_id=chat_id,
            bot_message_id=pending["bot_message_id"],
            reason=reason,
        )

    del self._pending_reason[chat_id]
    if updated:
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            "사유가 기록되었습니다. 감사합니다!"
        )
    else:
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            "사유를 저장할 대상 피드백을 찾지 못했습니다."
        )
    return True


async def handle_reason_skip(
    self: TelegramHandler,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    _ = context
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    if chat_id in self._pending_reason:
        del self._pending_reason[chat_id]
        await update.effective_message.reply_text("사유 입력을 건너뛰었습니다.")  # type: ignore[union-attr]
    else:
        await update.effective_message.reply_text("건너뛸 사유 요청이 없습니다.")  # type: ignore[union-attr]


async def handle_reason_or_message(
    self: TelegramHandler,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    text = update.effective_message.text  # type: ignore[union-attr]
    if text is None:
        return

    if await self._handle_reason_input(chat_id, text, update):
        return

    await self._handle_message_impl(update, context)
