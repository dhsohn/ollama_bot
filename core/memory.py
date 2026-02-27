"""대화 기록 및 장기 메모리 관리 모듈.

SQLite(aiosqlite) 백엔드로 대화 히스토리와 장기 메모리를 관리한다.
"""

from __future__ import annotations

import asyncio
import functools
import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiosqlite

from core.config import MemoryConfig
from core.db_migrations import MigrationRunner, MigrationStep
from core.logging_setup import get_logger

_MEMORY_SCHEMA_V1_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT
);
CREATE INDEX IF NOT EXISTS idx_conversations_chat_id
    ON conversations(chat_id);
CREATE INDEX IF NOT EXISTS idx_conversations_chat_id_id
    ON conversations(chat_id, id);
CREATE INDEX IF NOT EXISTS idx_conversations_timestamp
    ON conversations(timestamp);

CREATE TABLE IF NOT EXISTS conversations_archive (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id       INTEGER NOT NULL,
    role          TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content       TEXT NOT NULL,
    message_id    INTEGER NOT NULL,
    timestamp     DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_archive_chat_id_id
    ON conversations_archive(chat_id, id);

CREATE TABLE IF NOT EXISTS context_summaries (
    chat_id        INTEGER NOT NULL,
    summary        TEXT NOT NULL,
    last_archive_id INTEGER NOT NULL,
    message_count  INTEGER NOT NULL,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(chat_id)
);

CREATE TABLE IF NOT EXISTS long_term_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_memory_chat_id
    ON long_term_memory(chat_id);
CREATE INDEX IF NOT EXISTS idx_memory_key
    ON long_term_memory(key);
"""
_MEMORY_SCHEMA_V2_SQL = """
DELETE FROM long_term_memory AS target WHERE EXISTS (
  SELECT 1 FROM long_term_memory AS newer
  WHERE newer.chat_id = target.chat_id AND newer.key = target.key
    AND (newer.updated_at > target.updated_at
      OR (newer.updated_at = target.updated_at AND newer.id > target.id))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_chat_key
    ON long_term_memory(chat_id, key);
"""

_SQLITE_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def _utc_now_sql() -> str:
    """SQLite CURRENT_TIMESTAMP와 동일한 UTC 문자열 포맷."""
    return datetime.now(timezone.utc).strftime(_SQLITE_TIMESTAMP_FORMAT)


async def _apply_memory_v1(db: aiosqlite.Connection) -> None:
    await db.executescript(_MEMORY_SCHEMA_V1_SQL)


async def _apply_memory_v2(db: aiosqlite.Connection) -> None:
    await db.executescript(_MEMORY_SCHEMA_V2_SQL)


class MemoryManager:
    """대화 히스토리와 장기 메모리를 관리한다."""

    def __init__(
        self,
        config: MemoryConfig,
        data_dir: str,
        max_conversation_length: int = 50,
        archive_enabled: bool = False,
    ) -> None:
        self._db_path = Path(data_dir) / "memory" / "ollama_bot.db"
        self._max_long_term = config.max_long_term_entries
        self._retention_days = config.conversation_retention_days
        self._max_conversation_messages = max_conversation_length
        self._archive_enabled = archive_enabled
        self._db: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()
        self._logger = get_logger("memory")

    async def initialize(self) -> None:
        """데이터베이스를 열고 테이블을 생성한다."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        runner = MigrationRunner(self._db, self._logger, db_label="memory")
        await runner.run(
            [
                MigrationStep(1, _apply_memory_v1, "create_memory_tables"),
                MigrationStep(2, _apply_memory_v2, "dedupe_memory_and_add_unique_index"),
            ],
            backup_tables={"conversations", "long_term_memory", "context_summaries"},
        )
        self._logger.info("memory_initialized", db_path=str(self._db_path))

    @property
    def db(self) -> aiosqlite.Connection:
        """내부 DB 커넥션을 반환한다 (외부 모듈 공유용)."""
        if self._db is None:
            raise RuntimeError("MemoryManager가 아직 초기화되지 않았습니다.")
        return self._db

    def _require_db(self) -> aiosqlite.Connection:
        """초기화된 DB 커넥션을 반환한다."""
        if self._db is None:
            raise RuntimeError("MemoryManager가 아직 초기화되지 않았습니다.")
        return self._db

    @asynccontextmanager
    async def _write_transaction(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """단일 커넥션 쓰기를 직렬화하고 원자적으로 커밋/롤백한다."""
        db = self._require_db()
        async with self._write_lock:
            try:
                yield db
                await db.commit()
            except Exception:
                try:
                    await db.rollback()
                except Exception:
                    pass
                raise

    async def close(self) -> None:
        """데이터베이스 연결을 닫는다."""
        if self._db:
            await self._db.close()
            self._db = None
            self._logger.info("memory_closed")

    # ── 대화 기록 ──

    async def add_message(
        self,
        chat_id: int,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> None:
        """대화 턴을 저장한다. 오래된 항목은 자동 정리된다.

        INSERT와 정리를 같은 트랜잭션으로 묶어 디스크 I/O를 줄인다.
        """
        meta_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
        async with self._write_transaction() as db:
            await db.execute(
                "INSERT INTO conversations (chat_id, role, content, metadata) "
                "VALUES (?, ?, ?, ?)",
                (chat_id, role, content, meta_json),
            )

            limit = self._max_conversation_messages
            if limit > 0:
                # 트림 전 아카이브에 복사 (삭제 대상만)
                if self._archive_enabled:
                    await db.execute(
                        "INSERT INTO conversations_archive (chat_id, role, content, message_id, timestamp) "
                        "SELECT chat_id, role, content, id, timestamp FROM conversations "
                        "WHERE chat_id = ? AND id IN ("
                        "  SELECT id FROM conversations WHERE chat_id = ? "
                        "  ORDER BY id DESC LIMIT -1 OFFSET ?"
                        ")",
                        (chat_id, chat_id, limit),
                    )
                await db.execute(
                    "DELETE FROM conversations WHERE chat_id = ? AND id IN ("
                    "  SELECT id FROM conversations WHERE chat_id = ? "
                    "  ORDER BY id DESC LIMIT -1 OFFSET ?"
                    ")",
                    (chat_id, chat_id, limit),
                )
            else:
                await db.execute(
                    "DELETE FROM conversations WHERE chat_id = ?",
                    (chat_id,),
                )

    async def get_conversation(
        self,
        chat_id: int,
        limit: int | None = None,
    ) -> list[dict[str, str]]:
        """최근 대화 메시지를 시간순으로 반환한다."""
        db = self._require_db()
        if limit is None:
            limit = self._max_conversation_messages
        if limit <= 0:
            return []

        async with db.execute(
            "SELECT role, content FROM ("
            "  SELECT id, role, content FROM conversations "
            "  WHERE chat_id = ? ORDER BY id DESC LIMIT ?"
            ") ORDER BY id ASC",
            (chat_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
        return [{"role": row[0], "content": row[1]} for row in rows]

    async def get_conversation_in_range(
        self,
        chat_id: int,
        start_at: datetime,
        end_at: datetime,
        limit: int | None = None,
    ) -> list[dict[str, str]]:
        """주어진 시간 구간의 대화 메시지를 조회한다.

        start_at/end_at은 timezone-aware datetime이어야 하며,
        [start_at, end_at) 범위로 조회한다.
        """
        db = self._require_db()
        if start_at.tzinfo is None or end_at.tzinfo is None:
            raise ValueError("start_at and end_at must be timezone-aware datetimes")
        if end_at <= start_at:
            raise ValueError("end_at must be later than start_at")

        start_text = start_at.astimezone(timezone.utc).strftime(_SQLITE_TIMESTAMP_FORMAT)
        end_text = end_at.astimezone(timezone.utc).strftime(_SQLITE_TIMESTAMP_FORMAT)

        query = (
            "SELECT id, role, content, timestamp FROM conversations "
            "WHERE chat_id = ? AND timestamp >= ? AND timestamp < ? "
            "ORDER BY id ASC"
        )
        params: list = [chat_id, start_text, end_text]
        if limit is not None:
            if limit <= 0:
                return []
            query += " LIMIT ?"
            params.append(limit)

        async with db.execute(query, params) as cursor:
            rows = await cursor.fetchall()

        return [
            {
                "role": row[1],
                "content": row[2],
                "timestamp": row[3],
            }
            for row in rows
        ]

    async def clear_conversation(self, chat_id: int) -> int:
        """특정 채팅의 대화 기록을 삭제한다 (archive/summary 포함)."""
        deleted = 0
        async with self._write_transaction() as db:
            cursor = await db.execute(
                "DELETE FROM conversations WHERE chat_id = ?", (chat_id,)
            )
            await db.execute(
                "DELETE FROM conversations_archive WHERE chat_id = ?", (chat_id,)
            )
            await db.execute(
                "DELETE FROM context_summaries WHERE chat_id = ?", (chat_id,)
            )
            deleted = cursor.rowcount
        self._logger.info("conversation_cleared", chat_id=chat_id, deleted=deleted)
        return deleted

    # ── 장기 메모리 ──

    async def store_memory(
        self,
        chat_id: int,
        key: str,
        value: str,
        category: str = "general",
    ) -> None:
        """장기 메모리에 항목을 저장한다 (upsert)."""
        now = _utc_now_sql()
        async with self._write_transaction() as db:
            async with db.execute(
                "SELECT 1 FROM long_term_memory WHERE chat_id = ? AND key = ? LIMIT 1",
                (chat_id, key),
            ) as cursor:
                exists = await cursor.fetchone() is not None

            if not exists and self._max_long_term > 0:
                # 신규 키 삽입 전에 사용자별 최대 항목 수를 유지한다.
                async with db.execute(
                    "SELECT COUNT(*) FROM long_term_memory WHERE chat_id = ?",
                    (chat_id,),
                ) as cursor:
                    row = await cursor.fetchone()
                    count = row[0] if row else 0

                if count >= self._max_long_term:
                    await db.execute(
                        "DELETE FROM long_term_memory WHERE id IN ("
                        "  SELECT id FROM long_term_memory WHERE chat_id = ? "
                        "  ORDER BY updated_at ASC, id ASC LIMIT 1"
                        ")",
                        (chat_id,),
                    )

            await db.execute(
                "INSERT INTO long_term_memory "
                "(chat_id, key, value, category, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(chat_id, key) DO UPDATE SET "
                "value = excluded.value, "
                "category = excluded.category, "
                "updated_at = excluded.updated_at",
                (chat_id, key, value, category, now, now),
            )

    async def recall_memory(
        self,
        chat_id: int,
        key: str | None = None,
        category: str | None = None,
    ) -> list[dict]:
        """장기 메모리를 검색한다."""
        db = self._require_db()
        query = "SELECT key, value, category, updated_at FROM long_term_memory WHERE chat_id = ?"
        params: list = [chat_id]

        if key:
            query += " AND key = ?"
            params.append(key)
        if category:
            query += " AND category = ?"
            params.append(category)

        query += " ORDER BY updated_at DESC"

        async with db.execute(query, params) as cursor:
            rows = await cursor.fetchall()

        return [
            {
                "key": row[0],
                "value": row[1],
                "category": row[2],
                "updated_at": row[3],
            }
            for row in rows
        ]

    async def delete_memory(self, chat_id: int, key: str) -> bool:
        """특정 메모리 항목을 삭제한다."""
        deleted = False
        async with self._write_transaction() as db:
            cursor = await db.execute(
                "DELETE FROM long_term_memory WHERE chat_id = ? AND key = ?",
                (chat_id, key),
            )
            deleted = cursor.rowcount > 0
        return deleted

    async def delete_memories_by_category(self, chat_id: int, category: str) -> int:
        """지정된 chat_id/category에 해당하는 장기 메모리를 모두 삭제한다."""
        deleted = 0
        async with self._write_transaction() as db:
            cursor = await db.execute(
                "DELETE FROM long_term_memory WHERE chat_id = ? AND category = ?",
                (chat_id, category),
            )
            deleted = cursor.rowcount
        return deleted

    async def get_memory_stats(self, chat_id: int) -> dict:
        """메모리 통계를 반환한다."""
        db = self._require_db()
        stats: dict = {"chat_id": chat_id}

        async with db.execute(
            "SELECT COUNT(*) FROM conversations WHERE chat_id = ?",
            (chat_id,),
        ) as cursor:
            row = await cursor.fetchone()
            stats["conversation_count"] = row[0] if row else 0

        async with db.execute(
            "SELECT COUNT(*) FROM long_term_memory WHERE chat_id = ?",
            (chat_id,),
        ) as cursor:
            row = await cursor.fetchone()
            stats["memory_count"] = row[0] if row else 0

        async with db.execute(
            "SELECT MIN(timestamp) FROM conversations WHERE chat_id = ?",
            (chat_id,),
        ) as cursor:
            row = await cursor.fetchone()
            stats["oldest_conversation"] = row[0] if row and row[0] else None

        return stats

    async def ping(self) -> bool:
        """DB 연결이 유효한지 확인한다."""
        db = self._require_db()
        async with db.execute("SELECT 1") as cursor:
            row = await cursor.fetchone()
        return row is not None and row[0] == 1

    # ── 유지보수 ──

    async def prune_old_conversations(self) -> int:
        """보관 기간이 지난 대화를 삭제한다 (archive/summary 포함)."""
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=self._retention_days)
        ).strftime(_SQLITE_TIMESTAMP_FORMAT)
        deleted = 0
        archive_deleted = 0
        summary_deleted = 0
        async with self._write_transaction() as db:
            cursor = await db.execute(
                "DELETE FROM conversations WHERE timestamp < ?", (cutoff,)
            )
            archive_cursor = await db.execute(
                "DELETE FROM conversations_archive WHERE timestamp < ?", (cutoff,)
            )
            summary_cursor = await db.execute(
                "DELETE FROM context_summaries WHERE created_at < ?", (cutoff,)
            )
            # 대화/아카이브가 모두 없는 채팅의 요약은 orphan으로 정리한다.
            orphan_cursor = await db.execute(
                "DELETE FROM context_summaries "
                "WHERE chat_id NOT IN (SELECT DISTINCT chat_id FROM conversations) "
                "AND chat_id NOT IN (SELECT DISTINCT chat_id FROM conversations_archive)"
            )
            deleted = cursor.rowcount
            archive_deleted = archive_cursor.rowcount
            summary_deleted = summary_cursor.rowcount + orphan_cursor.rowcount
        if deleted or archive_deleted or summary_deleted:
            self._logger.info(
                "conversations_pruned",
                deleted=deleted,
                archive_deleted=archive_deleted,
                summary_deleted=summary_deleted,
            )
        return deleted

    # ── 아카이브 / 요약 (Phase 5: 컨텍스트 압축) ──

    async def get_archived_messages(
        self,
        chat_id: int,
        after_id: int = 0,
        limit: int = 200,
    ) -> list[dict]:
        """아카이브에서 메시지를 조회한다."""
        db = self._require_db()
        async with db.execute(
            "SELECT id, role, content, timestamp FROM conversations_archive "
            "WHERE chat_id = ? AND id > ? ORDER BY id ASC LIMIT ?",
            (chat_id, after_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
        return [
            {"id": row[0], "role": row[1], "content": row[2], "timestamp": row[3]}
            for row in rows
        ]

    async def get_summary(self, chat_id: int) -> dict | None:
        """캐시된 요약을 조회한다."""
        db = self._require_db()
        async with db.execute(
            "SELECT summary, last_archive_id, message_count, created_at "
            "FROM context_summaries WHERE chat_id = ?",
            (chat_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "summary": row[0],
            "last_archive_id": row[1],
            "message_count": row[2],
            "created_at": row[3],
        }

    async def store_summary(
        self,
        chat_id: int,
        summary: str,
        last_archive_id: int,
        message_count: int,
    ) -> None:
        """요약을 저장/갱신한다."""
        async with self._write_transaction() as db:
            await db.execute(
                "INSERT INTO context_summaries (chat_id, summary, last_archive_id, message_count) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(chat_id) DO UPDATE SET "
                "summary = excluded.summary, "
                "last_archive_id = excluded.last_archive_id, "
                "message_count = excluded.message_count, "
                "created_at = CURRENT_TIMESTAMP",
                (chat_id, summary, last_archive_id, message_count),
            )

    async def export_conversation_markdown(
        self, chat_id: int, output_dir: Path
    ) -> Path:
        """대화를 마크다운 파일로 내보낸다."""
        db = self._require_db()
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"chat_{chat_id}_{timestamp}.md"

        async with db.execute(
            "SELECT role, content, timestamp FROM conversations "
            "WHERE chat_id = ? ORDER BY id ASC",
            (chat_id,),
        ) as cursor:
            rows = await cursor.fetchall()

        lines = [f"# 대화 기록 (Chat ID: {chat_id})\n\n"]
        for row in rows:
            role_label = {"user": "사용자", "assistant": "봇", "system": "시스템"}.get(
                row[0], row[0]
            )
            lines.append(f"### {role_label} ({row[2]})\n{row[1]}\n\n")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            functools.partial(filepath.write_text, "".join(lines), encoding="utf-8"),
        )
        self._logger.info("conversation_exported", path=str(filepath))
        return filepath
