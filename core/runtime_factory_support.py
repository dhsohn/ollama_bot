"""Supporting helpers shared by the runtime factory."""

from __future__ import annotations

import fcntl
import os
from contextlib import AsyncExitStack, suppress
from pathlib import Path
from typing import Any, TextIO

import aiosqlite

from core.config import (
    AppSettings,
    OllamaConfig,
)
from core.ollama_client import OllamaClient


class StartupError(RuntimeError):
    """Carry a startup-failure message that can be shown to the operator."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def _create_retrieval_client(config: AppSettings) -> OllamaClient:
    """Create an Ollama client dedicated to retrieval tasks."""
    retrieval_config = OllamaConfig(
        host=config.ollama.host,
        model=config.ollama.embedding_model,
    )
    return OllamaClient(retrieval_config)


async def _open_sqlite_db(path: Path) -> aiosqlite.Connection:
    """Open a SQLite connection configured for WAL mode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(path))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.commit()
    return db


def validate_required_settings(config: AppSettings, logger: Any) -> None:
    """Validate required runtime settings."""
    if not config.ollama.chat_model.strip():
        logger.error("ollama_chat_model_not_set")
        raise StartupError(
            "Error: ollama.chat_model is empty.\n"
            "Set a chat model in config/config.yaml under ollama.chat_model."
        )

    if (
        not config.telegram.bot_token
        or config.telegram.bot_token == "your_telegram_bot_token_here"
    ):
        logger.error("telegram_bot_token_not_set")
        raise StartupError(
            "Error: telegram.bot_token is not configured.\n"
            "Set a valid bot token in config/config.yaml under telegram.bot_token."
        )

    if not config.security.allowed_users:
        logger.error("allowed_users_not_set")
        raise StartupError(
            "Error: telegram.allowed_users is empty.\n"
            "Set the allowed user IDs in config/config.yaml under telegram.allowed_users."
        )


def _release_runtime_lock(lock_file: TextIO) -> None:
    with suppress(OSError):
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    lock_file.close()


def _acquire_runtime_lock(
    config: AppSettings,
    cleanup_stack: AsyncExitStack,
    logger: Any,
) -> None:
    """Acquire a process lock so only one bot uses the same data_dir."""
    lock_dir = Path(config.data_dir)
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{config.bot.name}.runtime.lock"
    lock_file = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_file.seek(0)
        owner_pid = lock_file.read().strip()
        lock_file.close()
        owner_hint = f" (pid={owner_pid})" if owner_pid else ""
        raise StartupError(
            "Error: an ollama_bot instance is already running.\n"
            f"- lock: {lock_path}{owner_hint}\n"
            "Stop the existing process or inspect the systemd service before retrying."
        ) from exc

    cleanup_stack.callback(_release_runtime_lock, lock_file)
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(str(os.getpid()))
    lock_file.flush()
    logger.info("runtime_lock_acquired", path=str(lock_path), pid=os.getpid())


def _record_degraded_component(
    logger: Any,
    degraded_components: list[dict[str, str]],
    *,
    component: str,
    error: str,
) -> None:
    error_text = error.strip() or "unknown_error"
    degraded_components.append({"component": component, "error": error_text})
    logger.warning(
        "startup_component_degraded",
        component=component,
        error=error_text,
    )


def handle_optional_component_failure(
    config: AppSettings,
    logger: Any,
    degraded_components: list[dict[str, str]],
    *,
    component: str,
    error: Exception,
) -> None:
    error_text = str(error).strip() or type(error).__name__
    _record_degraded_component(
        logger,
        degraded_components,
        component=component,
        error=error_text,
    )
    if config.strict_startup:
        raise StartupError(
            "Error: strict_startup=true stops startup when an optional component fails to initialize.\n"
            f"- component: {component}\n"
            f"- reason: {error_text}"
        )


def log_degraded_startup_summary(
    logger: Any,
    degraded_components: list[dict[str, str]],
) -> None:
    if not degraded_components:
        return
    logger.warning(
        "startup_degraded_summary",
        degraded_count=len(degraded_components),
        degraded_components=degraded_components,
    )
