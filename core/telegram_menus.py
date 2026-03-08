"""Telegram inline menus, onboarding flow, and settings handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from core.i18n import t

if TYPE_CHECKING:
    from core.telegram_handler import TelegramHandler


async def get_user_language(self: TelegramHandler, chat_id: int) -> str:
    """Resolve the user's preferred language from memory, falling back to config."""
    try:
        prefs = await self._engine._memory.recall_memory(chat_id, category="preferences")
        for pref in prefs:
            if pref.get("key") == "language":
                val = str(pref.get("value", "")).strip().lower()
                if val in ("ko", "en"):
                    return val
    except Exception:
        pass
    return self._config.bot.language


async def _is_new_user(self: TelegramHandler, chat_id: int) -> bool:
    """Check if this is a first-time user (no conversation history)."""
    try:
        stats = await self._engine.get_memory_stats(chat_id)
        return (
            int(stats.get("conversation_count", 0)) == 0
            and int(stats.get("memory_count", 0)) == 0
        )
    except Exception:
        return True


async def handle_start_with_onboarding(
    self: TelegramHandler,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Enhanced /start with first-time user onboarding."""
    _ = context
    chat_id = update.effective_chat.id  # type: ignore[union-attr]

    if await _is_new_user(self, chat_id):
        lang = self._config.bot.language
        text = t("onboard_welcome", lang, bot_name=self._config.bot.name)
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("한국어", callback_data="onboard:lang:ko"),
                InlineKeyboardButton("English", callback_data="onboard:lang:en"),
            ],
        ])
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            text,
            reply_markup=keyboard,
        )
        return

    lang = await get_user_language(self, chat_id)
    welcome = t("welcome", lang, bot_name=self._config.bot.name)
    keyboard = build_main_menu_keyboard(lang)
    await update.effective_message.reply_text(  # type: ignore[union-attr]
        welcome,
        reply_markup=keyboard,
    )


def build_main_menu_keyboard(lang: str) -> InlineKeyboardMarkup:
    """Build the main inline menu keyboard."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                f"\U0001f527 {t('menu_btn_skills', lang)}",
                callback_data="menu:skills",
            ),
            InlineKeyboardButton(
                f"\U0001f9e0 {t('menu_btn_memory', lang)}",
                callback_data="menu:memory",
            ),
            InlineKeyboardButton(
                f"\U0001f4ca {t('menu_btn_status', lang)}",
                callback_data="menu:status",
            ),
        ],
        [
            InlineKeyboardButton(
                f"\u2753 {t('menu_btn_help', lang)}",
                callback_data="menu:help",
            ),
            InlineKeyboardButton(
                f"\u2699\ufe0f {t('menu_btn_settings', lang)}",
                callback_data="menu:settings",
            ),
            InlineKeyboardButton(
                f"\u23f0 {t('menu_btn_auto', lang)}",
                callback_data="menu:auto",
            ),
        ],
    ])


async def handle_onboard_callback(
    self: TelegramHandler,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle onboarding inline button callbacks."""
    _ = context
    query = update.callback_query
    if not query or not update.effective_chat:
        return
    await query.answer()

    chat_id = update.effective_chat.id
    data = query.data or ""

    if data.startswith("onboard:lang:"):
        lang = data.split(":")[-1]
        if lang not in ("ko", "en"):
            lang = "ko"

        await self._engine._memory.store_memory(
            chat_id, "language", lang, category="preferences",
        )

        response = t("onboard_lang_set", lang)
        done_text = t("onboard_done", lang)
        keyboard = build_main_menu_keyboard(lang)

        if query.message is not None:
            await query.message.edit_text(
                f"{response}\n\n{done_text}",
                reply_markup=keyboard,
            )
        return

    if data == "onboard:done":
        lang = await get_user_language(self, chat_id)
        done_text = t("onboard_done", lang)
        if query.message is not None:
            await query.message.edit_text(done_text)


async def handle_menu_callback(
    self: TelegramHandler,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle main menu inline button callbacks."""
    query = update.callback_query
    if not query or not update.effective_chat:
        return
    await query.answer()

    data = query.data or ""
    action = data.removeprefix("menu:")

    if action == "skills":
        await self._cmd_skills(update, context)
    elif action == "memory":
        await self._cmd_memory(update, context)
    elif action == "status":
        await self._cmd_status(update, context)
    elif action == "help":
        await self._cmd_help(update, context)
    elif action == "auto":
        await self._cmd_auto(update, context)
    elif action == "settings":
        await _show_settings(self, update)
    elif action.startswith("settings:lang:"):
        await _handle_lang_change(self, update, action.split(":")[-1])


async def _show_settings(self: TelegramHandler, update: Update) -> None:
    """Show the settings menu."""
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    lang = await get_user_language(self, chat_id)
    text = f"\u2699\ufe0f <b>{t('settings_title', lang)}</b>\n\n{t('settings_select_language', lang)}"
    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("한국어", callback_data="menu:settings:lang:ko"),
            InlineKeyboardButton("English", callback_data="menu:settings:lang:en"),
        ],
    ])
    query = update.callback_query
    if query and query.message:
        await query.message.reply_text(
            text, parse_mode=ParseMode.HTML, reply_markup=keyboard,
        )
    elif update.effective_message:
        await update.effective_message.reply_text(
            text, parse_mode=ParseMode.HTML, reply_markup=keyboard,
        )


async def _handle_lang_change(
    self: TelegramHandler,
    update: Update,
    lang: str,
) -> None:
    """Handle language change from settings menu."""
    if lang not in ("ko", "en"):
        lang = "ko"
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    await self._engine._memory.store_memory(
        chat_id, "language", lang, category="preferences",
    )
    response = t("onboard_lang_set", lang)
    query = update.callback_query
    if query and query.message:
        await query.message.edit_text(response)
    elif update.effective_message:
        await update.effective_message.reply_text(response)
