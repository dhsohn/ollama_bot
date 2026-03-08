"""Tests for Telegram inline menu and onboarding helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telegram.constants import ParseMode

from core import telegram_menus
from core.i18n import t


def _make_handler(
    *,
    language: str = "ko",
    name: str = "ollama_bot",
    prefs: list[dict[str, str]] | None = None,
    stats: dict[str, int] | Exception | None = None,
) -> SimpleNamespace:
    memory = SimpleNamespace(
        recall_memory=AsyncMock(return_value=prefs or []),
        store_memory=AsyncMock(),
    )
    engine = SimpleNamespace(
        _memory=memory,
        get_memory_stats=AsyncMock(
            side_effect=stats if isinstance(stats, Exception) else None,
            return_value=stats if isinstance(stats, dict) else {
                "conversation_count": 0,
                "memory_count": 0,
            },
        ),
    )
    return SimpleNamespace(
        _engine=engine,
        _config=SimpleNamespace(
            bot=SimpleNamespace(language=language, name=name),
        ),
        _cmd_skills=AsyncMock(),
        _cmd_memory=AsyncMock(),
        _cmd_status=AsyncMock(),
        _cmd_help=AsyncMock(),
        _cmd_auto=AsyncMock(),
    )


def _make_update(
    *,
    chat_id: int = 111,
    callback_data: str | None = None,
    callback_message: object | None = None,
) -> SimpleNamespace:
    effective_message = SimpleNamespace(reply_text=AsyncMock())
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=chat_id),
        effective_message=effective_message,
        callback_query=None,
    )
    if callback_data is not None:
        update.callback_query = SimpleNamespace(
            data=callback_data,
            answer=AsyncMock(),
            message=callback_message,
        )
    return update


def test_get_message_method_returns_callable_or_none() -> None:
    reply_text = AsyncMock()
    message = SimpleNamespace(reply_text=reply_text)

    assert telegram_menus._get_message_method(message, "reply_text") is reply_text
    assert telegram_menus._get_message_method(object(), "reply_text") is None
    assert telegram_menus._get_message_method(None, "reply_text") is None


@pytest.mark.asyncio
async def test_get_user_language_prefers_saved_language() -> None:
    handler = _make_handler(prefs=[{"key": "language", "value": " en "}])

    result = await telegram_menus.get_user_language(handler, 111)

    assert result == "en"


@pytest.mark.asyncio
async def test_get_user_language_falls_back_to_default_on_invalid_value() -> None:
    handler = _make_handler(prefs=[{"key": "language", "value": "jp"}])

    result = await telegram_menus.get_user_language(handler, 111)

    assert result == "ko"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stats", "expected"),
    [
        ({"conversation_count": 0, "memory_count": 0}, True),
        ({"conversation_count": 1, "memory_count": 0}, False),
    ],
)
async def test_is_new_user_depends_on_history(stats: dict[str, int], expected: bool) -> None:
    handler = _make_handler(stats=stats)

    result = await telegram_menus._is_new_user(handler, 111)

    assert result is expected


@pytest.mark.asyncio
async def test_handle_start_with_onboarding_for_new_user() -> None:
    handler = _make_handler(stats={"conversation_count": 0, "memory_count": 0})
    update = _make_update()

    await telegram_menus.handle_start_with_onboarding(handler, update, None)

    update.effective_message.reply_text.assert_awaited_once()
    text = update.effective_message.reply_text.await_args.args[0]
    keyboard = update.effective_message.reply_text.await_args.kwargs["reply_markup"]
    assert text == t("onboard_welcome", "ko", bot_name="ollama_bot")
    assert keyboard.inline_keyboard[0][0].callback_data == "onboard:lang:ko"
    assert keyboard.inline_keyboard[0][1].callback_data == "onboard:lang:en"


@pytest.mark.asyncio
async def test_handle_start_with_onboarding_for_existing_user_uses_saved_language() -> None:
    handler = _make_handler(
        prefs=[{"key": "language", "value": "en"}],
        stats={"conversation_count": 2, "memory_count": 1},
    )
    update = _make_update()

    await telegram_menus.handle_start_with_onboarding(handler, update, None)

    update.effective_message.reply_text.assert_awaited_once()
    text = update.effective_message.reply_text.await_args.args[0]
    keyboard = update.effective_message.reply_text.await_args.kwargs["reply_markup"]
    assert text == t("welcome", "en", bot_name="ollama_bot")
    assert keyboard.inline_keyboard[1][1].callback_data == "menu:settings"


@pytest.mark.asyncio
async def test_handle_onboard_callback_stores_language_and_edits_message() -> None:
    handler = _make_handler()
    callback_message = SimpleNamespace(edit_text=AsyncMock())
    update = _make_update(
        callback_data="onboard:lang:invalid",
        callback_message=callback_message,
    )

    await telegram_menus.handle_onboard_callback(handler, update, None)

    update.callback_query.answer.assert_awaited_once()
    handler._engine._memory.store_memory.assert_awaited_once_with(
        111,
        "language",
        "ko",
        category="preferences",
    )
    callback_message.edit_text.assert_awaited_once()
    text = callback_message.edit_text.await_args.args[0]
    keyboard = callback_message.edit_text.await_args.kwargs["reply_markup"]
    assert t("onboard_lang_set", "ko") in text
    assert t("onboard_done", "ko") in text
    assert keyboard.inline_keyboard[0][0].callback_data == "menu:skills"


@pytest.mark.asyncio
async def test_handle_onboard_callback_done_uses_saved_language() -> None:
    handler = _make_handler(prefs=[{"key": "language", "value": "en"}])
    callback_message = SimpleNamespace(edit_text=AsyncMock())
    update = _make_update(
        callback_data="onboard:done",
        callback_message=callback_message,
    )

    await telegram_menus.handle_onboard_callback(handler, update, None)

    callback_message.edit_text.assert_awaited_once_with(t("onboard_done", "en"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "handler_attr"),
    [
        ("skills", "_cmd_skills"),
        ("memory", "_cmd_memory"),
        ("status", "_cmd_status"),
        ("help", "_cmd_help"),
        ("auto", "_cmd_auto"),
    ],
)
async def test_handle_menu_callback_dispatches_commands(
    action: str,
    handler_attr: str,
) -> None:
    handler = _make_handler()
    update = _make_update(callback_data=f"menu:{action}", callback_message=object())

    await telegram_menus.handle_menu_callback(handler, update, None)

    update.callback_query.answer.assert_awaited_once()
    getattr(handler, handler_attr).assert_awaited_once_with(update, None)


@pytest.mark.asyncio
async def test_handle_menu_callback_routes_to_show_settings(monkeypatch) -> None:
    handler = _make_handler()
    update = _make_update(callback_data="menu:settings", callback_message=object())
    show_settings = AsyncMock()
    monkeypatch.setattr(telegram_menus, "_show_settings", show_settings)

    await telegram_menus.handle_menu_callback(handler, update, None)

    show_settings.assert_awaited_once_with(handler, update)


@pytest.mark.asyncio
async def test_handle_menu_callback_routes_to_language_change(monkeypatch) -> None:
    handler = _make_handler()
    update = _make_update(
        callback_data="menu:settings:lang:en",
        callback_message=object(),
    )
    handle_lang_change = AsyncMock()
    monkeypatch.setattr(telegram_menus, "_handle_lang_change", handle_lang_change)

    await telegram_menus.handle_menu_callback(handler, update, None)

    handle_lang_change.assert_awaited_once_with(handler, update, "en")


@pytest.mark.asyncio
async def test_show_settings_uses_callback_message_when_accessible() -> None:
    handler = _make_handler(prefs=[{"key": "language", "value": "en"}])
    callback_message = SimpleNamespace(reply_text=AsyncMock())
    update = _make_update(
        callback_data="menu:settings",
        callback_message=callback_message,
    )

    await telegram_menus._show_settings(handler, update)

    callback_message.reply_text.assert_awaited_once()
    kwargs = callback_message.reply_text.await_args.kwargs
    assert kwargs["parse_mode"] == ParseMode.HTML
    assert kwargs["reply_markup"].inline_keyboard[0][1].callback_data == "menu:settings:lang:en"


@pytest.mark.asyncio
async def test_show_settings_falls_back_to_effective_message() -> None:
    handler = _make_handler()
    update = _make_update(
        callback_data="menu:settings",
        callback_message=object(),
    )

    await telegram_menus._show_settings(handler, update)

    update.effective_message.reply_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_lang_change_uses_callback_message_edit_when_accessible() -> None:
    handler = _make_handler()
    callback_message = SimpleNamespace(edit_text=AsyncMock())
    update = _make_update(
        callback_data="menu:settings:lang:ko",
        callback_message=callback_message,
    )

    await telegram_menus._handle_lang_change(handler, update, "invalid")

    handler._engine._memory.store_memory.assert_awaited_once_with(
        111,
        "language",
        "ko",
        category="preferences",
    )
    callback_message.edit_text.assert_awaited_once_with(t("onboard_lang_set", "ko"))


@pytest.mark.asyncio
async def test_handle_lang_change_falls_back_to_effective_message() -> None:
    handler = _make_handler()
    update = _make_update(
        callback_data="menu:settings:lang:en",
        callback_message=object(),
    )

    await telegram_menus._handle_lang_change(handler, update, "en")

    update.effective_message.reply_text.assert_awaited_once_with(
        t("onboard_lang_set", "en")
    )
