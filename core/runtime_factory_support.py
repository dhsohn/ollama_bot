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
    """초기화 실패 시 사용자 노출용 메시지를 전달한다."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def _create_retrieval_client(config: AppSettings) -> OllamaClient:
    """Ollama 기반 retrieval 전용 클라이언트를 생성한다."""
    retrieval_config = OllamaConfig(
        host=config.ollama.host,
        model=config.ollama.embedding_model,
    )
    return OllamaClient(retrieval_config)


async def _open_sqlite_db(path: Path) -> aiosqlite.Connection:
    """WAL 모드 SQLite 연결을 생성한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(path))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.commit()
    return db


def validate_required_settings(config: AppSettings, logger: Any) -> None:
    """필수 런타임 설정을 검사한다."""
    if not config.ollama.chat_model.strip():
        logger.error("ollama_chat_model_not_set")
        raise StartupError(
            "오류: ollama.chat_model이 비어 있습니다.\n"
            "config/config.yaml의 ollama.chat_model에 채팅 모델을 설정하세요."
        )

    if (
        not config.telegram.bot_token
        or config.telegram.bot_token == "your_telegram_bot_token_here"
    ):
        logger.error("telegram_bot_token_not_set")
        raise StartupError(
            "오류: telegram.bot_token이 설정되지 않았습니다.\n"
            "config/config.yaml의 telegram.bot_token에 유효한 봇 토큰을 입력하세요."
        )

    if not config.security.allowed_users:
        logger.error("allowed_users_not_set")
        raise StartupError(
            "오류: telegram.allowed_users가 비어 있습니다.\n"
            "config/config.yaml의 telegram.allowed_users에 허용할 사용자 ID를 설정하세요."
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
    """동일 data_dir에서 봇이 중복 실행되지 않도록 프로세스 락을 건다."""
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
            "오류: 이미 실행 중인 ollama_bot 인스턴스가 있습니다.\n"
            f"- lock: {lock_path}{owner_hint}\n"
            "기존 프로세스를 종료하거나 systemd 서비스 상태를 확인한 뒤 다시 시작하세요."
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
            "오류: strict_startup=true로 설정되어 선택 컴포넌트 초기화 실패 시 시작을 중단합니다.\n"
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
