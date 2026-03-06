"""사용자 피드백 관리 모듈.

봇 응답에 대한 thumbs-up/down 피드백을 저장·조회·통계한다.
"""

from __future__ import annotations

import aiosqlite

from core.db_migrations import MigrationRunner, MigrationStep
from core.logging_setup import get_logger

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

_PREVIEW_MAX_CHARS = 500
_TABLE_INFO_WHITELIST = frozenset({"message_feedback"})


def _sanitize_preview(text: str | None) -> str | None:
    """미리보기 텍스트 길이를 제한한다."""
    if text is None:
        return None
    return text[:_PREVIEW_MAX_CHARS]


async def _has_column(db: aiosqlite.Connection, table: str, column: str) -> bool:
    """PRAGMA table_info로 특정 컬럼 존재 여부를 확인한다."""
    if table not in _TABLE_INFO_WHITELIST:
        raise ValueError(f"unsupported table for pragma table_info: {table}")
    async with db.execute(f"PRAGMA table_info({table})") as cursor:
        rows = await cursor.fetchall()
    return any(row[1] == column for row in rows)


async def _apply_feedback_v1(db: aiosqlite.Connection) -> None:
    await db.executescript(_FEEDBACK_SCHEMA_SQL)


async def _apply_feedback_v2(db: aiosqlite.Connection) -> None:
    if not await _has_column(db, "message_feedback", "reason"):
        await db.execute(
            "ALTER TABLE message_feedback ADD COLUMN reason TEXT"
        )


async def _apply_feedback_v3(db: aiosqlite.Connection) -> None:
    pass  # auto_evaluation 테이블 제거됨 (기존 DB 호환 위해 마이그레이션 스텝 유지)


class FeedbackManager:
    """메시지 피드백 저장/조회/통계를 관리한다."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db
        self._logger = get_logger("feedback_manager")

    async def initialize_schema(self) -> None:
        """피드백 테이블을 생성하고 마이그레이션을 적용한다."""
        runner = MigrationRunner(self._db, self._logger, db_label="feedback")
        await runner.run(
            [
                # memory/semantic_cache와 같은 DB를 공유할 수 있으므로
                # 버전 번호는 컴포넌트별로 충돌하지 않게 분리한다.
                MigrationStep(101, _apply_feedback_v1, "create_message_feedback"),
                MigrationStep(102, _apply_feedback_v2, "add_feedback_reason_column"),
                MigrationStep(103, _apply_feedback_v3, "create_auto_evaluation"),
            ],
            backup_tables={"message_feedback"},
        )

    async def store_feedback(
        self,
        chat_id: int,
        bot_message_id: int,
        rating: int,
        user_preview: str | None = None,
        bot_preview: str | None = None,
    ) -> bool:
        """피드백을 저장(upsert)한다. 재평가이면 True를 반환한다."""
        safe_user_preview = _sanitize_preview(user_preview)
        safe_bot_preview = _sanitize_preview(bot_preview)

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
                    (
                        rating,
                        safe_user_preview,
                        safe_bot_preview,
                        chat_id,
                        bot_message_id,
                    ),
                )
            else:
                try:
                    await self._db.execute(
                        "INSERT INTO message_feedback "
                        "(chat_id, bot_message_id, rating, user_message_preview, bot_response_preview) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            chat_id,
                            bot_message_id,
                            rating,
                            safe_user_preview,
                            safe_bot_preview,
                        ),
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
                        (
                            rating,
                            safe_user_preview,
                            safe_bot_preview,
                            chat_id,
                            bot_message_id,
                        ),
                    )

            await self._db.commit()
            return exists
        except Exception:
            await self._db.rollback()
            raise

    async def update_reason(
        self,
        chat_id: int,
        bot_message_id: int,
        reason: str,
    ) -> bool:
        """피드백에 사유를 추가한다. 해당 피드백이 없으면 False를 반환한다."""
        cursor = await self._db.execute(
            "UPDATE message_feedback "
            "SET reason = ?, updated_at = CURRENT_TIMESTAMP "
            "WHERE chat_id = ? AND bot_message_id = ?",
            (reason, chat_id, bot_message_id),
        )
        await self._db.commit()
        return cursor.rowcount > 0

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
            "SELECT bot_message_id, rating, user_message_preview, "
            "bot_response_preview, reason, created_at "
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
                "reason": row[4],
                "created_at": row[5],
            }
            for row in rows
        ]

    async def search_positive_examples(
        self,
        chat_id: int,
        keywords: list[str],
        limit: int = 2,
        min_preview_length: int = 20,
        recent_days: int = 180,
    ) -> list[dict]:
        """긍정 피드백 중 키워드 매칭이 높은 예시를 반환한다."""
        if not keywords:
            return []

        # LIKE 기반 매칭: 각 키워드가 user 또는 bot preview에 포함된 횟수를 점수로 사용
        score_clauses = []
        kw_params: list[str] = []
        for keyword in keywords:
            pattern = f"%{keyword}%"
            score_clauses.append(
                "(CASE WHEN user_message_preview LIKE ? THEN 1 ELSE 0 END "
                "+ CASE WHEN bot_response_preview LIKE ? THEN 1 ELSE 0 END)"
            )
            kw_params.extend([pattern, pattern])

        score_expr = " + ".join(score_clauses)

        query = (
            f"WITH scored AS ( "
            f"SELECT user_message_preview, bot_response_preview, updated_at, "
            f"({score_expr}) AS relevance "
            f"FROM message_feedback "
            f"WHERE chat_id = ? AND rating = 1 "
            f"AND created_at >= datetime('now', ?) "
            f"AND LENGTH(COALESCE(bot_response_preview, '')) >= ? "
            f") "
            f"SELECT user_message_preview, bot_response_preview, relevance "
            f"FROM scored "
            f"WHERE relevance > 0 "
            f"ORDER BY relevance DESC, updated_at DESC "
            f"LIMIT ?"
        )

        final_params = (
            *kw_params,
            chat_id,
            f"-{recent_days} days",
            min_preview_length,
            limit * 3,
        )

        async with self._db.execute(query, final_params) as cursor:
            rows = await cursor.fetchall()

        # 중복 preview 제거
        seen: set[str] = set()
        results: list[dict] = []
        for row in rows:
            bot_preview = row[1] or ""
            if bot_preview in seen:
                continue
            seen.add(bot_preview)
            results.append({
                "user_preview": row[0],
                "bot_preview": bot_preview,
            })
            if len(results) >= limit:
                break

        return results

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
