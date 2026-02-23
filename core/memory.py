"""대화 기록 및 장기 메모리 관리 모듈.

SQLite(aiosqlite) 백엔드로 대화 히스토리와 장기 메모리를 관리한다.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiosqlite

from core.config import MemoryConfig
from core.logging_setup import get_logger

_SCHEMA_SQL = """
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

_SQLITE_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def _utc_now_sql() -> str:
    """SQLite CURRENT_TIMESTAMP와 동일한 UTC 문자열 포맷."""
    return datetime.now(timezone.utc).strftime(_SQLITE_TIMESTAMP_FORMAT)


class MemoryManager:
    """대화 히스토리와 장기 메모리를 관리한다."""

    def __init__(
        self,
        config: MemoryConfig,
        data_dir: str,
        max_conversation_length: int = 50,
    ) -> None:
        self._db_path = Path(data_dir) / "memory" / "ollama_bot.db"
        self._max_long_term = config.max_long_term_entries
        self._retention_days = config.conversation_retention_days
        self._max_conversation = max_conversation_length
        self._db: aiosqlite.Connection | None = None
        self._logger = get_logger("memory")

    async def initialize(self) -> None:
        """데이터베이스를 열고 테이블을 생성한다."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.executescript(_SCHEMA_SQL)
        await self._db.commit()
        self._logger.info("memory_initialized", db_path=str(self._db_path))

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
        """대화 턴을 저장한다. 오래된 항목은 자동 정리된다."""
        assert self._db is not None
        meta_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
        await self._db.execute(
            "INSERT INTO conversations (chat_id, role, content, metadata) "
            "VALUES (?, ?, ?, ?)",
            (chat_id, role, content, meta_json),
        )
        await self._db.commit()

        # 대화 버퍼 초과 시 오래된 항목 삭제
        async with self._db.execute(
            "SELECT COUNT(*) FROM conversations WHERE chat_id = ?",
            (chat_id,),
        ) as cursor:
            row = await cursor.fetchone()
            count = row[0] if row else 0

        limit = self._max_conversation * 2
        if count > limit:
            await self._db.execute(
                "DELETE FROM conversations WHERE id IN ("
                "  SELECT id FROM conversations WHERE chat_id = ? "
                "  ORDER BY id ASC LIMIT ?"
                ")",
                (chat_id, count - limit),
            )
            await self._db.commit()

    async def get_conversation(
        self,
        chat_id: int,
        limit: int | None = None,
    ) -> list[dict[str, str]]:
        """최근 대화 메시지를 시간순으로 반환한다."""
        assert self._db is not None
        limit = limit or self._max_conversation
        async with self._db.execute(
            "SELECT role, content FROM ("
            "  SELECT id, role, content FROM conversations "
            "  WHERE chat_id = ? ORDER BY id DESC LIMIT ?"
            ") ORDER BY id ASC",
            (chat_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
        return [{"role": row[0], "content": row[1]} for row in rows]

    async def clear_conversation(self, chat_id: int) -> int:
        """특정 채팅의 대화 기록을 삭제한다."""
        assert self._db is not None
        cursor = await self._db.execute(
            "DELETE FROM conversations WHERE chat_id = ?", (chat_id,)
        )
        await self._db.commit()
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
        assert self._db is not None
        now = _utc_now_sql()

        # 기존 항목이 있으면 업데이트
        async with self._db.execute(
            "SELECT id FROM long_term_memory "
            "WHERE chat_id = ? AND key = ?",
            (chat_id, key),
        ) as cursor:
            existing = await cursor.fetchone()

        if existing:
            await self._db.execute(
                "UPDATE long_term_memory SET value = ?, category = ?, "
                "updated_at = ? WHERE id = ?",
                (value, category, now, existing[0]),
            )
        else:
            # 최대 항목 수 체크
            async with self._db.execute(
                "SELECT COUNT(*) FROM long_term_memory WHERE chat_id = ?",
                (chat_id,),
            ) as cursor:
                row = await cursor.fetchone()
                count = row[0] if row else 0

            if count >= self._max_long_term:
                # 가장 오래된 항목 삭제
                await self._db.execute(
                    "DELETE FROM long_term_memory WHERE id IN ("
                    "  SELECT id FROM long_term_memory WHERE chat_id = ? "
                    "  ORDER BY updated_at ASC LIMIT 1"
                    ")",
                    (chat_id,),
                )

            await self._db.execute(
                "INSERT INTO long_term_memory "
                "(chat_id, key, value, category, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (chat_id, key, value, category, now, now),
            )

        await self._db.commit()

    async def recall_memory(
        self,
        chat_id: int,
        key: str | None = None,
        category: str | None = None,
    ) -> list[dict]:
        """장기 메모리를 검색한다."""
        assert self._db is not None
        query = "SELECT key, value, category, updated_at FROM long_term_memory WHERE chat_id = ?"
        params: list = [chat_id]

        if key:
            query += " AND key = ?"
            params.append(key)
        if category:
            query += " AND category = ?"
            params.append(category)

        query += " ORDER BY updated_at DESC"

        async with self._db.execute(query, params) as cursor:
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
        assert self._db is not None
        cursor = await self._db.execute(
            "DELETE FROM long_term_memory WHERE chat_id = ? AND key = ?",
            (chat_id, key),
        )
        await self._db.commit()
        return cursor.rowcount > 0

    async def get_memory_stats(self, chat_id: int) -> dict:
        """메모리 통계를 반환한다."""
        assert self._db is not None
        stats: dict = {"chat_id": chat_id}

        async with self._db.execute(
            "SELECT COUNT(*) FROM conversations WHERE chat_id = ?",
            (chat_id,),
        ) as cursor:
            row = await cursor.fetchone()
            stats["conversation_count"] = row[0] if row else 0

        async with self._db.execute(
            "SELECT COUNT(*) FROM long_term_memory WHERE chat_id = ?",
            (chat_id,),
        ) as cursor:
            row = await cursor.fetchone()
            stats["memory_count"] = row[0] if row else 0

        async with self._db.execute(
            "SELECT MIN(timestamp) FROM conversations WHERE chat_id = ?",
            (chat_id,),
        ) as cursor:
            row = await cursor.fetchone()
            stats["oldest_conversation"] = row[0] if row and row[0] else None

        return stats

    # ── 유지보수 ──

    async def prune_old_conversations(self) -> int:
        """보관 기간이 지난 대화를 삭제한다."""
        assert self._db is not None
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=self._retention_days)
        ).strftime(_SQLITE_TIMESTAMP_FORMAT)
        cursor = await self._db.execute(
            "DELETE FROM conversations WHERE timestamp < ?", (cutoff,)
        )
        await self._db.commit()
        deleted = cursor.rowcount
        if deleted:
            self._logger.info("conversations_pruned", deleted=deleted)
        return deleted

    async def export_conversation_markdown(
        self, chat_id: int, output_dir: Path
    ) -> Path:
        """대화를 마크다운 파일로 내보낸다."""
        assert self._db is not None
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"chat_{chat_id}_{timestamp}.md"

        async with self._db.execute(
            "SELECT role, content, timestamp FROM conversations "
            "WHERE chat_id = ? ORDER BY timestamp ASC",
            (chat_id,),
        ) as cursor:
            rows = await cursor.fetchall()

        lines = [f"# 대화 기록 (Chat ID: {chat_id})\n\n"]
        for row in rows:
            role_label = {"user": "사용자", "assistant": "봇", "system": "시스템"}.get(
                row[0], row[0]
            )
            lines.append(f"### {role_label} ({row[2]})\n{row[1]}\n\n")

        filepath.write_text("".join(lines), encoding="utf-8")
        self._logger.info("conversation_exported", path=str(filepath))
        return filepath
