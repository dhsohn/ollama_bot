"""사용자 피드백 관리 모듈.

봇 응답에 대한 thumbs-up/down 피드백을 저장·조회·통계한다.
"""

from __future__ import annotations

import aiosqlite

_FEEDBACK_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS message_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    bot_message_id INTEGER NOT NULL,
    rating INTEGER NOT NULL CHECK(rating IN (-1, 1)),
    user_message_preview TEXT,
    bot_response_preview TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(chat_id, bot_message_id)
);
CREATE INDEX IF NOT EXISTS idx_feedback_chat_created
    ON message_feedback(chat_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_rating
    ON message_feedback(rating);
"""


class FeedbackManager:
    """메시지 피드백 저장/조회/통계를 관리한다."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    async def initialize_schema(self) -> None:
        """피드백 테이블을 생성한다."""
        await self._db.executescript(_FEEDBACK_SCHEMA_SQL)
        await self._db.commit()

    async def store_feedback(
        self,
        chat_id: int,
        bot_message_id: int,
        rating: int,
        user_preview: str | None = None,
        bot_preview: str | None = None,
    ) -> bool:
        """피드백을 저장(upsert)한다. 재평가이면 True를 반환한다."""
        await self._db.execute("BEGIN IMMEDIATE")
        try:
            async with self._db.execute(
                "SELECT 1 FROM message_feedback "
                "WHERE chat_id = ? AND bot_message_id = ? LIMIT 1",
                (chat_id, bot_message_id),
            ) as cursor:
                exists = await cursor.fetchone() is not None

            if exists:
                await self._db.execute(
                    "UPDATE message_feedback "
                    "SET rating = ?, "
                    "    user_message_preview = COALESCE(?, user_message_preview), "
                    "    bot_response_preview = COALESCE(?, bot_response_preview), "
                    "    updated_at = CURRENT_TIMESTAMP "
                    "WHERE chat_id = ? AND bot_message_id = ?",
                    (rating, user_preview, bot_preview, chat_id, bot_message_id),
                )
            else:
                try:
                    await self._db.execute(
                        "INSERT INTO message_feedback "
                        "(chat_id, bot_message_id, rating, user_message_preview, bot_response_preview) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (chat_id, bot_message_id, rating, user_preview, bot_preview),
                    )
                except aiosqlite.IntegrityError:
                    # 동시 요청으로 UNIQUE 충돌 시 update 경로로 전환한다.
                    exists = True
                    await self._db.execute(
                        "UPDATE message_feedback "
                        "SET rating = ?, "
                        "    user_message_preview = COALESCE(?, user_message_preview), "
                        "    bot_response_preview = COALESCE(?, bot_response_preview), "
                        "    updated_at = CURRENT_TIMESTAMP "
                        "WHERE chat_id = ? AND bot_message_id = ?",
                        (rating, user_preview, bot_preview, chat_id, bot_message_id),
                    )

            await self._db.commit()
            return exists
        except Exception:
            await self._db.rollback()
            raise

    async def get_user_stats(self, chat_id: int) -> dict:
        """사용자별 피드백 통계를 반환한다."""
        return await self._fetch_stats(chat_id=chat_id)

    async def get_global_stats(self) -> dict:
        """전체 피드백 통계를 반환한다."""
        return await self._fetch_stats(chat_id=None)

    async def _fetch_stats(self, chat_id: int | None) -> dict:
        """피드백 통계를 조회한다."""
        if chat_id is None:
            query = (
                "SELECT "
                "  COUNT(*) AS total, "
                "  SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) AS positive, "
                "  SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END) AS negative "
                "FROM message_feedback"
            )
            params: tuple = ()
        else:
            query = (
                "SELECT "
                "  COUNT(*) AS total, "
                "  SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) AS positive, "
                "  SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END) AS negative "
                "FROM message_feedback WHERE chat_id = ?"
            )
            params = (chat_id,)

        async with self._db.execute(
            query,
            params,
        ) as cursor:
            row = await cursor.fetchone()

        return self._row_to_stats(row)

    @staticmethod
    def _row_to_stats(row) -> dict:
        """집계 row를 통계 dict로 변환한다."""
        total = row[0] if row else 0
        positive = row[1] if row and row[1] else 0
        negative = row[2] if row and row[2] else 0
        rate = positive / total if total > 0 else 0.0
        return {
            "total": total,
            "positive": positive,
            "negative": negative,
            "satisfaction_rate": rate,
        }

    async def get_recent_feedback(
        self,
        chat_id: int,
        rating: int,
        limit: int = 10,
    ) -> list[dict]:
        """최근 피드백을 조회한다."""
        async with self._db.execute(
            "SELECT bot_message_id, rating, user_message_preview, bot_response_preview, created_at "
            "FROM message_feedback "
            "WHERE chat_id = ? AND rating = ? "
            "ORDER BY updated_at DESC LIMIT ?",
            (chat_id, rating, limit),
        ) as cursor:
            rows = await cursor.fetchall()

        return [
            {
                "bot_message_id": row[0],
                "rating": row[1],
                "user_preview": row[2],
                "bot_preview": row[3],
                "created_at": row[4],
            }
            for row in rows
        ]

    async def count_feedback(self, chat_id: int) -> int:
        """사용자의 피드백 총 건수를 반환한다."""
        async with self._db.execute(
            "SELECT COUNT(*) FROM message_feedback WHERE chat_id = ?",
            (chat_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return row[0] if row else 0

    async def prune_old_feedback(self, retention_days: int) -> int:
        """retention_days보다 오래된 피드백을 삭제한다."""
        cursor = await self._db.execute(
            "DELETE FROM message_feedback WHERE created_at < datetime('now', ?)",
            (f"-{retention_days} days",),
        )
        await self._db.commit()
        return cursor.rowcount
