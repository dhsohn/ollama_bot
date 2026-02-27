"""메모리 매니저 테스트."""

from __future__ import annotations

import asyncio
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
        max_conversation_length=10,
    )
    await manager.initialize()
    yield manager
    await manager.close()


class TestPing:
    @pytest.mark.asyncio
    async def test_ping_returns_true(
        self,
        memory_manager: MemoryManager,
    ) -> None:
        result = await memory_manager.ping()
        assert result is True

    @pytest.mark.asyncio
    async def test_ping_fails_before_initialize(self) -> None:
        manager = MemoryManager(
            config=MemoryConfig(),
            data_dir="/tmp/nonexistent",
            max_conversation_length=10,
        )
        with pytest.raises(RuntimeError):
            await manager.ping()


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
        memory_manager._max_conversation_messages = 2
        messages = [f"m{i}" for i in range(6)]
        for i, content in enumerate(messages):
            role = "user" if i % 2 == 0 else "assistant"
            await memory_manager.add_message(chat_id, role, content)

        history = await memory_manager.get_conversation(chat_id, limit=10)
        # max_conversation_length=2 => 최근 2개 메시지만 유지
        assert [m["content"] for m in history] == ["m4", "m5"]

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


class TestDeleteMemoriesByCategory:
    @pytest.mark.asyncio
    async def test_delete_by_category(self, memory_manager: MemoryManager) -> None:
        chat_id = 505
        await memory_manager.store_memory(chat_id, "k1", "v1", category="preferences")
        await memory_manager.store_memory(chat_id, "k2", "v2", category="general")

        deleted = await memory_manager.delete_memories_by_category(chat_id, "preferences")
        assert deleted == 1
        remaining = await memory_manager.recall_memory(chat_id)
        assert len(remaining) == 1
        assert remaining[0]["key"] == "k2"

    @pytest.mark.asyncio
    async def test_delete_by_category_fails_before_initialize(self) -> None:
        manager = MemoryManager(
            config=MemoryConfig(),
            data_dir="/tmp/nonexistent",
            max_conversation_length=10,
        )
        with pytest.raises(RuntimeError):
            await manager.delete_memories_by_category(1, "preferences")


class TestMemoryTransactions:
    @pytest.mark.asyncio
    async def test_store_memory_without_nested_begin_conflict(
        self,
        memory_manager: MemoryManager,
    ) -> None:
        # 외부 write가 트랜잭션을 열어둔 상태를 만든다.
        assert memory_manager._db is not None
        await memory_manager._db.execute(
            "INSERT INTO conversations (chat_id, role, content, metadata) VALUES (?, ?, ?, ?)",
            (999, "user", "pending-write", None),
        )

        # 기존 구현(BEGIN IMMEDIATE)에서는 여기서 nested transaction 오류가 날 수 있었다.
        await memory_manager.store_memory(999, "k1", "v1", category="preferences")

        # store_memory 커밋 시 pending-write도 함께 정상 커밋되어야 한다.
        history = await memory_manager.get_conversation(999, limit=10)
        assert any(item["content"] == "pending-write" for item in history)

    @pytest.mark.asyncio
    async def test_concurrent_writes_are_serialized(
        self,
        memory_manager: MemoryManager,
    ) -> None:
        async def _add(idx: int) -> None:
            await memory_manager.add_message(777, "user", f"m{idx}")

        async def _store(idx: int) -> None:
            await memory_manager.store_memory(777, f"k{idx}", f"v{idx}")

        tasks = []
        for idx in range(30):
            tasks.append(asyncio.create_task(_add(idx)))
            tasks.append(asyncio.create_task(_store(idx)))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        errors = [result for result in results if isinstance(result, Exception)]
        assert errors == []
