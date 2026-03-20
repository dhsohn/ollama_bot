"""Telegram handler decorators for auth, rate limiting, and concurrency.

These decorators isolate request pre-processing concerns so `TelegramHandler`
can stay focused on handler logic.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from typing import TYPE_CHECKING

from telegram.constants import ChatType

from core.i18n import t
from core.security import AuthenticationError, GlobalConcurrencyError, RateLimitError

if TYPE_CHECKING:
    from telegram import Update
    from telegram.ext import ContextTypes

    from core.telegram_handler import TelegramHandler


def auth_required(func: Callable) -> Callable:
    """Apply private-chat, authentication, and rate-limit checks."""

    async def wrapper(self: TelegramHandler, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_chat or not update.effective_message:
            return

        chat = update.effective_chat
        message = update.effective_message

        if chat.type != ChatType.PRIVATE:
            self._logger.warning(
                "non_private_chat_blocked",
                chat_id=chat.id,
                chat_type=chat.type,
            )
            lang = self._config.bot.language
            with suppress(Exception):
                await message.reply_text(t("private_chat_only", lang))
            return

        chat_id = chat.id

        try:
            self._authorize_chat_id(chat_id)
        except AuthenticationError:
            self._logger.warning("unauthorized_access", chat_id=chat_id)
            return
        except RateLimitError:
            lang = self._config.bot.language
            await update.effective_message.reply_text(
                t("rate_limited", lang)
            )
            return

        return await func(self, update, context)

    return wrapper


def global_slot_required(func: Callable) -> Callable:
    """Apply the global concurrency slot guard."""

    async def wrapper(self: TelegramHandler, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_chat:
            return

        chat_id = update.effective_chat.id
        try:
            async with self._security.global_slot(chat_id):
                return await func(self, update, context)
        except GlobalConcurrencyError:
            lang = self._config.bot.language
            msg = t("concurrency_limited", lang)
            query = update.callback_query
            if query is not None:
                await query.answer(msg, show_alert=True)
                return
            if update.effective_message is not None:
                await update.effective_message.reply_text(msg)
            return

    return wrapper
