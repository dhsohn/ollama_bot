"""피드백 매니저 테스트."""

from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest
import pytest_asyncio

from core.feedback_manager import FeedbackManager


@pytest_asyncio.fixture
async def feedback_db(tmp_path: Path):
    """테스트용 DB 연결 + FeedbackManager."""
    db_path = tmp_path / "test.db"
    db = await aiosqlite.connect(str(db_path))
    db.row_factory = aiosqlite.Row
    fm = FeedbackManager(db)
    await fm.initialize_schema()
    yield fm, db
    await db.close()


class TestInitializeSchema:
    @pytest.mark.asyncio
    async def test_table_created(self, feedback_db) -> None:
        fm, db = feedback_db
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='message_feedback'"
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None


class TestStoreFeedback:
    @pytest.mark.asyncio
    async def test_new_feedback_returns_false(self, feedback_db) -> None:
        fm, db = feedback_db
        is_update = await fm.store_feedback(
            chat_id=111, bot_message_id=1, rating=1, user_preview="hi", bot_preview="hello"
        )
        assert is_update is False

    @pytest.mark.asyncio
    async def test_upsert_returns_true(self, feedback_db) -> None:
        fm, db = feedback_db
        await fm.store_feedback(chat_id=111, bot_message_id=1, rating=1)
        is_update = await fm.store_feedback(chat_id=111, bot_message_id=1, rating=-1)
        assert is_update is True

    @pytest.mark.asyncio
    async def test_upsert_changes_rating(self, feedback_db) -> None:
        fm, db = feedback_db
        await fm.store_feedback(chat_id=111, bot_message_id=1, rating=1)
        await fm.store_feedback(chat_id=111, bot_message_id=1, rating=-1)
        stats = await fm.get_user_stats(111)
        assert stats["negative"] == 1
        assert stats["positive"] == 0


class TestGetUserStats:
    @pytest.mark.asyncio
    async def test_empty_stats(self, feedback_db) -> None:
        fm, db = feedback_db
        stats = await fm.get_user_stats(999)
        assert stats["total"] == 0
        assert stats["satisfaction_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_stats_calculation(self, feedback_db) -> None:
        fm, db = feedback_db
        await fm.store_feedback(111, 1, 1)
        await fm.store_feedback(111, 2, 1)
        await fm.store_feedback(111, 3, -1)
        stats = await fm.get_user_stats(111)
        assert stats["total"] == 3
        assert stats["positive"] == 2
        assert stats["negative"] == 1
        assert abs(stats["satisfaction_rate"] - 2 / 3) < 0.01


class TestGetGlobalStats:
    @pytest.mark.asyncio
    async def test_global_stats(self, feedback_db) -> None:
        fm, db = feedback_db
        await fm.store_feedback(111, 1, 1)
        await fm.store_feedback(222, 2, -1)
        stats = await fm.get_global_stats()
        assert stats["total"] == 2
        assert stats["positive"] == 1
        assert stats["negative"] == 1


class TestGetRecentFeedback:
    @pytest.mark.asyncio
    async def test_returns_matching_rating(self, feedback_db) -> None:
        fm, db = feedback_db
        await fm.store_feedback(111, 1, 1, "q1", "a1")
        await fm.store_feedback(111, 2, -1, "q2", "a2")
        await fm.store_feedback(111, 3, 1, "q3", "a3")
        negatives = await fm.get_recent_feedback(111, rating=-1, limit=10)
        assert len(negatives) == 1
        assert negatives[0]["user_preview"] == "q2"

    @pytest.mark.asyncio
    async def test_limit_respected(self, feedback_db) -> None:
        fm, db = feedback_db
        for i in range(5):
            await fm.store_feedback(111, i, 1)
        positives = await fm.get_recent_feedback(111, rating=1, limit=3)
        assert len(positives) == 3

    @pytest.mark.asyncio
    async def test_sorted_by_updated_at(self, feedback_db) -> None:
        fm, db = feedback_db
        await fm.store_feedback(111, 1, 1, "q1", "a1")
        await fm.store_feedback(111, 2, 1, "q2", "a2")

        await db.execute(
            "UPDATE message_feedback SET updated_at = datetime('now', '-10 minutes') "
            "WHERE chat_id = 111 AND bot_message_id = 1"
        )
        await db.execute(
            "UPDATE message_feedback SET updated_at = datetime('now', '-5 minutes') "
            "WHERE chat_id = 111 AND bot_message_id = 2"
        )
        await db.commit()

        await fm.store_feedback(111, 1, 1)
        positives = await fm.get_recent_feedback(111, rating=1, limit=2)
        assert len(positives) == 2
        assert positives[0]["bot_message_id"] == 1


class TestCountFeedback:
    @pytest.mark.asyncio
    async def test_count(self, feedback_db) -> None:
        fm, db = feedback_db
        assert await fm.count_feedback(111) == 0
        await fm.store_feedback(111, 1, 1)
        await fm.store_feedback(111, 2, -1)
        assert await fm.count_feedback(111) == 2


class TestPruneOldFeedback:
    @pytest.mark.asyncio
    async def test_prune_old_entries(self, feedback_db) -> None:
        fm, db = feedback_db
        await fm.store_feedback(111, 1, 1)
        # 오래된 날짜로 백데이트
        await db.execute(
            "UPDATE message_feedback SET created_at = datetime('now', '-100 days') "
            "WHERE chat_id = 111 AND bot_message_id = 1"
        )
        await db.commit()
        await fm.store_feedback(111, 2, 1)  # 최신

        pruned = await fm.prune_old_feedback(90)
        assert pruned == 1
        assert await fm.count_feedback(111) == 1
