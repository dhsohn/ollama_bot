"""ContextCompressor 테스트."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from core.context_compressor import ContextCompressor
from core.ollama_client import ChatResponse


@pytest.mark.asyncio
async def test_maybe_refresh_summary_advances_archive_pointer_incrementally() -> None:
    ollama = AsyncMock()
    ollama.chat = AsyncMock(return_value=ChatResponse(content="updated summary"))

    memory = AsyncMock()
    summary = {"summary": "old", "last_archive_id": 200, "message_count": 200}
    new_archived = [
        {"id": idx, "role": "user", "content": f"msg-{idx}", "timestamp": "2026-01-01 00:00:00"}
        for idx in range(201, 401)
    ]
    memory.get_summary = AsyncMock(side_effect=[summary, summary])
    memory.get_archived_messages = AsyncMock(side_effect=[new_archived, new_archived])
    memory.store_summary = AsyncMock()

    compressor = ContextCompressor(
        ollama=ollama,
        memory=memory,
        recent_keep=10,
        summary_refresh_interval=10,
    )

    refreshed = await compressor.maybe_refresh_summary(111)

    assert refreshed is True
    memory.store_summary.assert_awaited_once()
    args = memory.store_summary.await_args.args
    assert args[0] == 111
    assert args[1] == "updated summary"
    assert args[2] == 400  # 마지막으로 처리한 archive id
    assert args[3] == 400  # 누적 메시지 수


@pytest.mark.asyncio
async def test_maybe_refresh_summary_skips_when_new_archive_is_small() -> None:
    ollama = AsyncMock()
    memory = AsyncMock()
    memory.get_summary = AsyncMock(return_value={"summary": "old", "last_archive_id": 10, "message_count": 10})
    memory.get_archived_messages = AsyncMock(return_value=[
        {"id": 11, "role": "user", "content": "new", "timestamp": "2026-01-01 00:00:00"}
    ])
    memory.store_summary = AsyncMock()

    compressor = ContextCompressor(
        ollama=ollama,
        memory=memory,
        recent_keep=10,
        summary_refresh_interval=5,
    )

    refreshed = await compressor.maybe_refresh_summary(111)

    assert refreshed is False
    memory.store_summary.assert_not_called()
