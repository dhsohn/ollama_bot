"""runtime_factory 런타임 락 테스트."""

from __future__ import annotations

from contextlib import AsyncExitStack
from unittest.mock import MagicMock

import pytest

from core.config import AppSettings
from core.runtime_factory import StartupError, _acquire_runtime_lock


@pytest.mark.asyncio
async def test_runtime_lock_blocks_second_instance(tmp_path) -> None:
    config = AppSettings(data_dir=str(tmp_path))
    first_stack = AsyncExitStack()
    second_stack = AsyncExitStack()

    try:
        _acquire_runtime_lock(config, first_stack, MagicMock())

        with pytest.raises(StartupError, match="이미 실행 중인 ollama_bot 인스턴스"):
            _acquire_runtime_lock(config, second_stack, MagicMock())
    finally:
        await second_stack.aclose()
        await first_stack.aclose()


@pytest.mark.asyncio
async def test_runtime_lock_can_be_reacquired_after_cleanup(tmp_path) -> None:
    config = AppSettings(data_dir=str(tmp_path))
    first_stack = AsyncExitStack()
    second_stack = AsyncExitStack()

    try:
        _acquire_runtime_lock(config, first_stack, MagicMock())
        await first_stack.aclose()

        _acquire_runtime_lock(config, second_stack, MagicMock())
    finally:
        await second_stack.aclose()
