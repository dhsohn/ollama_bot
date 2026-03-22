"""Delivery helpers for :mod:`core.auto_scheduler`."""

from __future__ import annotations

import asyncio
import functools
from typing import Any

from core.engine_context import normalize_language


async def deliver_output(scheduler: Any, auto: Any, result: str) -> None:
    """Deliver automation output to Telegram and/or save it to a file."""
    output = auto.output

    if output.send_to_telegram and scheduler._telegram:
        lang = normalize_language(scheduler._config.bot.language)
        use_html = "<b>" in result or "<pre>" in result
        parse_mode = "HTML" if use_html else None
        header = (
            f"⏰ <b>{'Automation' if lang == 'en' else '자동화'}: {auto.name}</b>\n\n"
            if use_html
            else f"⏰ {'Automation' if lang == 'en' else '자동화'}: {auto.name}\n\n"
        )
        for user_id in scheduler._config.security.allowed_users:
            try:
                await scheduler._telegram.send_message(
                    user_id, header + result, parse_mode=parse_mode,
                )
            except Exception as exc:
                scheduler._logger.error(
                    "auto_telegram_send_failed",
                    user_id=user_id,
                    error=str(exc),
                )

    if output.save_to_file:
        await _save_output_to_file(
            scheduler,
            output_path_template=output.save_to_file,
            result=result,
        )


async def _save_output_to_file(
    scheduler: Any,
    *,
    output_path_template: str,
    result: str,
) -> None:
    """Persist automation output to disk."""
    try:
        now = scheduler._current_datetime()
        file_path = output_path_template.replace(
            "{date}",
            now.strftime("%Y%m%d"),
        )
        validated_path = scheduler._security.validate_path(
            file_path,
            base_dir=scheduler._config.data_dir,
        )
        validated_path.parent.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            functools.partial(validated_path.write_text, result, encoding="utf-8"),
        )
        scheduler._logger.info("auto_output_saved", path=str(validated_path))
    except Exception as exc:
        scheduler._logger.error(
            "auto_file_save_failed",
            path=output_path_template,
            error=str(exc),
        )


async def deliver_failure_notice(
    scheduler: Any,
    auto: Any,
    error: Exception | None,
) -> None:
    """Send an automation failure notice to Telegram."""
    if scheduler._telegram is None:
        return

    from html import escape as _h

    now = scheduler._current_datetime().strftime("%Y-%m-%d %H:%M:%S %Z")
    message = (
        f"⚠️ <b>자동화 실패: {_h(auto.name)}</b>\n"
        f"- 시각: {now}\n"
        f"- 원인: <code>{_h(format_exception(error))}</code>\n"
        f"- 재시도: {auto.retry.max_attempts}회 모두 실패"
    )

    for user_id in scheduler._config.security.allowed_users:
        try:
            await scheduler._telegram.send_message(
                user_id, message, parse_mode="HTML",
            )
        except Exception as exc:
            scheduler._logger.error(
                "auto_failure_notice_send_failed",
                user_id=user_id,
                error=format_exception(exc),
            )


def format_exception(exc: Exception | None) -> str:
    """Normalize an exception into a string that includes its class name."""
    if exc is None:
        return "unknown"
    message = str(exc).strip()
    if message:
        return f"{exc.__class__.__name__}: {message}"
    return exc.__class__.__name__
