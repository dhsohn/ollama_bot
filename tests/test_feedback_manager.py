"""피드백 매니저 테스트."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
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

    @pytest.mark.asyncio
    async def test_preview_redacts_pii_on_store(self, feedback_db) -> None:
        fm, db = feedback_db
        await fm.store_feedback(
            chat_id=111,
            bot_message_id=9,
            rating=1,
            user_preview="연락처는 test@example.com 입니다.",
            bot_preview="전화번호는 010-1234-5678 입니다.",
        )

        rows = await fm.get_recent_feedback(111, rating=1, limit=1)
        assert len(rows) == 1
        assert "[REDACTED]" in (rows[0]["user_preview"] or "")
        assert "[REDACTED]" in (rows[0]["bot_preview"] or "")
        assert "test@example.com" not in (rows[0]["user_preview"] or "")
        assert "010-1234-5678" not in (rows[0]["bot_preview"] or "")


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


# ── V2 테스트 ──


class TestSchemaMigration:
    @pytest.mark.asyncio
    async def test_v2_migration_adds_reason_column(self, feedback_db) -> None:
        fm, db = feedback_db
        # reason 컬럼이 존재하는지 확인
        async with db.execute("PRAGMA table_info(message_feedback)") as cursor:
            columns = [row[1] for row in await cursor.fetchall()]
        assert "reason" in columns

    @pytest.mark.asyncio
    async def test_v3_migration_creates_auto_evaluation_table(self, feedback_db) -> None:
        fm, db = feedback_db
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='auto_evaluation'"
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None

    @pytest.mark.asyncio
    async def test_migrations_idempotent(self, feedback_db) -> None:
        fm, db = feedback_db
        # 두 번째 호출도 에러 없이 통과해야 함
        await fm.initialize_schema()
        async with db.execute("PRAGMA table_info(message_feedback)") as cursor:
            columns = [row[1] for row in await cursor.fetchall()]
        assert "reason" in columns

    @pytest.mark.asyncio
    async def test_schema_migrations_table_tracks_versions(self, feedback_db) -> None:
        fm, db = feedback_db
        async with db.execute("SELECT version FROM schema_migrations ORDER BY version") as cursor:
            versions = [row[0] for row in await cursor.fetchall()]
        assert 2 in versions
        assert 3 in versions

    @pytest.mark.asyncio
    async def test_migration_creates_backup_for_legacy_db(self, tmp_path: Path) -> None:
        db_path = tmp_path / "legacy.db"
        db = await aiosqlite.connect(str(db_path))
        try:
            await db.execute(
                "CREATE TABLE message_feedback ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "chat_id INTEGER NOT NULL, "
                "bot_message_id INTEGER NOT NULL, "
                "rating INTEGER NOT NULL CHECK(rating IN (-1, 1)), "
                "user_message_preview TEXT, "
                "bot_response_preview TEXT, "
                "created_at DATETIME DEFAULT CURRENT_TIMESTAMP, "
                "updated_at DATETIME DEFAULT CURRENT_TIMESTAMP, "
                "UNIQUE(chat_id, bot_message_id)"
                ")"
            )
            await db.commit()

            fm = FeedbackManager(db)
            await fm.initialize_schema()
        finally:
            await db.close()

        backups = list(tmp_path.glob("legacy.db.pre_migration_*.bak"))
        assert len(backups) == 1


class TestUpdateReason:
    @pytest.mark.asyncio
    async def test_update_reason_success(self, feedback_db) -> None:
        fm, db = feedback_db
        await fm.store_feedback(111, 1, -1, "q", "a")
        result = await fm.update_reason(111, 1, "응답이 부정확했어요")
        assert result is True

        fb_list = await fm.get_recent_feedback(111, rating=-1)
        assert fb_list[0]["reason"] == "응답이 부정확했어요"

    @pytest.mark.asyncio
    async def test_update_reason_not_found(self, feedback_db) -> None:
        fm, db = feedback_db
        result = await fm.update_reason(111, 999, "없는 피드백")
        assert result is False

    @pytest.mark.asyncio
    async def test_reason_none_without_update(self, feedback_db) -> None:
        fm, db = feedback_db
        await fm.store_feedback(111, 1, -1, "q", "a")
        fb_list = await fm.get_recent_feedback(111, rating=-1)
        assert fb_list[0]["reason"] is None


class TestSearchPositiveExamples:
    @pytest.mark.asyncio
    async def test_search_finds_matching(self, feedback_db) -> None:
        fm, db = feedback_db
        await fm.store_feedback(111, 1, 1, "파이썬 리스트 정렬", "sorted() 함수를 사용하세요")
        await fm.store_feedback(111, 2, 1, "자바스크립트 배열", "Array.sort() 를 사용하세요")
        await fm.store_feedback(111, 3, -1, "파이썬 에러", "에러 메시지를 확인하세요")  # 부정

        results = await fm.search_positive_examples(111, ["파이썬"], limit=5, min_preview_length=5)
        assert len(results) >= 1
        assert any("파이썬" in (r.get("user_preview") or "") for r in results)

    @pytest.mark.asyncio
    async def test_search_returns_empty_for_no_keywords(self, feedback_db) -> None:
        fm, db = feedback_db
        results = await fm.search_positive_examples(111, [], limit=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_deduplicates_previews(self, feedback_db) -> None:
        fm, db = feedback_db
        # 동일한 bot_preview로 두 건 저장
        await fm.store_feedback(111, 1, 1, "파이썬 질문1", "같은 답변입니다")
        await fm.store_feedback(111, 2, 1, "파이썬 질문2", "같은 답변입니다")

        results = await fm.search_positive_examples(111, ["파이썬"], limit=5, min_preview_length=5)
        bot_previews = [r["bot_preview"] for r in results]
        assert len(bot_previews) == len(set(bot_previews))


class TestAutoEvaluation:
    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, feedback_db) -> None:
        fm, db = feedback_db
        await fm.store_auto_evaluation(
            chat_id=111,
            bot_message_id=1,
            user_input="안녕",
            bot_response="안녕하세요",
            score=4,
            explanation="적절한 인사",
        )
        low = await fm.get_low_score_evaluations(chat_id=111, max_score=5)
        assert len(low) == 1
        assert low[0]["score"] == 4
        assert low[0]["explanation"] == "적절한 인사"

    @pytest.mark.asyncio
    async def test_get_low_score_filters(self, feedback_db) -> None:
        fm, db = feedback_db
        await fm.store_auto_evaluation(111, 1, "q1", "a1", score=5)
        await fm.store_auto_evaluation(111, 2, "q2", "a2", score=2)
        await fm.store_auto_evaluation(111, 3, "q3", "a3", score=1)

        low = await fm.get_low_score_evaluations(chat_id=111, max_score=2)
        assert len(low) == 2
        assert all(e["score"] <= 2 for e in low)

    @pytest.mark.asyncio
    async def test_get_low_score_excludes_manual_feedback(self, feedback_db) -> None:
        fm, db = feedback_db
        await fm.store_feedback(111, 10, -1, "q", "a")
        await fm.store_auto_evaluation(111, 10, "q", "a", score=1)

        low = await fm.get_low_score_evaluations(chat_id=111, max_score=2)
        assert low == []

    @pytest.mark.asyncio
    async def test_count_today(self, feedback_db) -> None:
        fm, db = feedback_db
        assert await fm.count_today_evaluations() == 0
        await fm.store_auto_evaluation(111, 1, "q", "a", score=3)
        assert await fm.count_today_evaluations() == 1

    @pytest.mark.asyncio
    async def test_count_today_with_bounds_and_chat(self, feedback_db) -> None:
        fm, db = feedback_db
        await fm.store_auto_evaluation(111, 1, "q1", "a1", score=3)
        await fm.store_auto_evaluation(222, 1, "q2", "a2", score=4)

        await db.execute(
            "UPDATE auto_evaluation SET created_at = datetime('now', '-2 days') "
            "WHERE chat_id = 111 AND bot_message_id = 1"
        )
        await db.commit()

        start = datetime.now(timezone.utc).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        end = start + timedelta(days=1)
        start_utc = start.strftime("%Y-%m-%d %H:%M:%S")
        end_utc = end.strftime("%Y-%m-%d %H:%M:%S")

        assert (
            await fm.count_today_evaluations(
                chat_id=111,
                start_utc=start_utc,
                end_utc=end_utc,
            )
            == 0
        )
        assert (
            await fm.count_today_evaluations(
                chat_id=222,
                start_utc=start_utc,
                end_utc=end_utc,
            )
            == 1
        )

    @pytest.mark.asyncio
    async def test_prune_auto_evaluations(self, feedback_db) -> None:
        fm, db = feedback_db
        await fm.store_auto_evaluation(111, 1, "q", "a", score=3)
        # 오래된 날짜로 백데이트
        await db.execute(
            "UPDATE auto_evaluation SET created_at = datetime('now', '-100 days') "
            "WHERE chat_id = 111 AND bot_message_id = 1"
        )
        await db.commit()
        await fm.store_auto_evaluation(111, 2, "q2", "a2", score=4)

        pruned = await fm.prune_old_auto_evaluations(90)
        assert pruned == 1
        assert await fm.count_today_evaluations() == 1


class TestExportKtoDataset:
    @pytest.mark.asyncio
    async def test_export_basic(self, feedback_db) -> None:
        fm, db = feedback_db
        await fm.store_feedback(111, 1, 1, "좋은 질문이었어요" * 3, "좋은 답변이었습니다" * 3)
        await fm.store_feedback(111, 2, -1, "나쁜 질문이었어요" * 3, "나쁜 답변이었습니다" * 3)

        dataset = await fm.export_kto_dataset(chat_id=111, min_preview_length=5)
        assert len(dataset) == 2
        labels = {d["label"] for d in dataset}
        assert True in labels
        assert False in labels

    @pytest.mark.asyncio
    async def test_export_filters_short_previews(self, feedback_db) -> None:
        fm, db = feedback_db
        await fm.store_feedback(111, 1, 1, "짧", "짧")  # 너무 짧음
        await fm.store_feedback(111, 2, 1, "충분히 긴 질문입니다", "충분히 긴 답변입니다")

        dataset = await fm.export_kto_dataset(chat_id=111, min_preview_length=5)
        assert len(dataset) == 1

    @pytest.mark.asyncio
    async def test_export_empty(self, feedback_db) -> None:
        fm, db = feedback_db
        dataset = await fm.export_kto_dataset(chat_id=111)
        assert dataset == []
