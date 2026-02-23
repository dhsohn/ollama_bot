"""메모리 매니저 테스트."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import pytest_asyncio

from core.config import MemoryConfig
from core.memory import MemoryManager


@pytest_asyncio.fixture
async def memory_manager(tmp_path: Path) -> MemoryManager:
    manager = MemoryManager(
        config=MemoryConfig(conversation_retention_days=1),
        data_dir=str(tmp_path),
        max_conversation_length=2,
    )
    await manager.initialize()
    yield manager
    await manager.close()


class TestConversationOrder:
    @pytest.mark.asyncio
    async def test_get_conversation_stays_in_insert_order_with_same_timestamp(
        self,
        memory_manager: MemoryManager,
    ) -> None:
        chat_id = 101
        await memory_manager.add_message(chat_id, "user", "u1")
        await memory_manager.add_message(chat_id, "assistant", "a1")
        await memory_manager.add_message(chat_id, "user", "u2")

        assert memory_manager._db is not None
        await memory_manager._db.execute(
            "UPDATE conversations SET timestamp = '2026-01-01 00:00:00' WHERE chat_id = ?",
            (chat_id,),
        )
        await memory_manager._db.commit()

        history = await memory_manager.get_conversation(chat_id, limit=10)
        assert [m["content"] for m in history] == ["u1", "a1", "u2"]

    @pytest.mark.asyncio
    async def test_prune_over_limit_removes_oldest_by_id(
        self,
        memory_manager: MemoryManager,
    ) -> None:
        chat_id = 202
        messages = [f"m{i}" for i in range(6)]
        for i, content in enumerate(messages):
            role = "user" if i % 2 == 0 else "assistant"
            await memory_manager.add_message(chat_id, role, content)

        history = await memory_manager.get_conversation(chat_id, limit=10)
        # max_conversation_length=2 => 최대 4개(user/assistant 2쌍) 유지
        assert [m["content"] for m in history] == ["m2", "m3", "m4", "m5"]

    @pytest.mark.asyncio
    async def test_get_conversation_in_range_filters_time_window(
        self,
        memory_manager: MemoryManager,
    ) -> None:
        chat_id = 404
        await memory_manager.add_message(chat_id, "user", "yesterday-message")
        await memory_manager.add_message(chat_id, "assistant", "today-message")

        assert memory_manager._db is not None
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)
        today = now

        await memory_manager._db.execute(
            "UPDATE conversations SET timestamp = ? WHERE chat_id = ? AND content = ?",
            (yesterday.strftime("%Y-%m-%d %H:%M:%S"), chat_id, "yesterday-message"),
        )
        await memory_manager._db.execute(
            "UPDATE conversations SET timestamp = ? WHERE chat_id = ? AND content = ?",
            (today.strftime("%Y-%m-%d %H:%M:%S"), chat_id, "today-message"),
        )
        await memory_manager._db.commit()

        start_at = datetime(
            year=yesterday.year,
            month=yesterday.month,
            day=yesterday.day,
            tzinfo=timezone.utc,
        )
        end_at = start_at + timedelta(days=1)
        rows = await memory_manager.get_conversation_in_range(
            chat_id=chat_id,
            start_at=start_at,
            end_at=end_at,
        )

        assert len(rows) == 1
        assert rows[0]["content"] == "yesterday-message"


class TestRetentionPrune:
    @pytest.mark.asyncio
    async def test_prune_old_conversations_deletes_expired_rows(
        self,
        memory_manager: MemoryManager,
    ) -> None:
        chat_id = 303
        await memory_manager.add_message(chat_id, "user", "old-message")

        assert memory_manager._db is not None
        await memory_manager._db.execute(
            "UPDATE conversations SET timestamp = '2000-01-01 00:00:00' WHERE chat_id = ?",
            (chat_id,),
        )
        await memory_manager._db.commit()

        deleted = await memory_manager.prune_old_conversations()
        assert deleted >= 1
        history = await memory_manager.get_conversation(chat_id, limit=10)
        assert history == []
