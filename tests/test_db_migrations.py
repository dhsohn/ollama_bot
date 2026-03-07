"""DB 마이그레이션 프레임워크 테스트."""

from __future__ import annotations

import logging
from pathlib import Path

import aiosqlite
import pytest
import pytest_asyncio

from core.db_migrations import MigrationRunner, MigrationStep
from core.memory import _apply_memory_v1, _apply_memory_v2

_MEMORY_TABLES = {"conversations", "conversations_archive", "context_summaries", "long_term_memory"}


class _FakeLogger:
    """MigrationRunner가 요구하는 최소 로거."""

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


_logger = _FakeLogger()


async def _table_names(db: aiosqlite.Connection) -> set[str]:
    async with db.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ) as cur:
        rows = await cur.fetchall()
    return {row[0] for row in rows}


async def _index_names(db: aiosqlite.Connection) -> set[str]:
    async with db.execute(
        "SELECT name FROM sqlite_master WHERE type = 'index' AND name NOT LIKE 'sqlite_%'"
    ) as cur:
        rows = await cur.fetchall()
    return {row[0] for row in rows}


# ---------------------------------------------------------------------------
# V1 schema creation from scratch
# ---------------------------------------------------------------------------


class TestV1SchemaCreation:
    @pytest.mark.asyncio
    async def test_v1_creates_all_tables(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        async with aiosqlite.connect(str(db_path)) as db:
            runner = MigrationRunner(db, _logger, db_label="test")
            await runner.run(
                [MigrationStep(1, _apply_memory_v1, "create_memory_tables")],
            )
            tables = await _table_names(db)

        assert _MEMORY_TABLES.issubset(tables)
        assert "schema_migrations" in tables

    @pytest.mark.asyncio
    async def test_v1_records_applied_version(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        async with aiosqlite.connect(str(db_path)) as db:
            runner = MigrationRunner(db, _logger, db_label="test")
            await runner.run(
                [MigrationStep(1, _apply_memory_v1, "v1")],
            )
            async with db.execute("SELECT version FROM schema_migrations") as cur:
                rows = await cur.fetchall()
        assert {row[0] for row in rows} == {1}

    @pytest.mark.asyncio
    async def test_v1_creates_expected_indexes(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        async with aiosqlite.connect(str(db_path)) as db:
            runner = MigrationRunner(db, _logger, db_label="test")
            await runner.run(
                [MigrationStep(1, _apply_memory_v1, "v1")],
            )
            indexes = await _index_names(db)

        assert "idx_conversations_chat_id" in indexes
        assert "idx_memory_chat_id" in indexes


# ---------------------------------------------------------------------------
# V1 -> V2 migration
# ---------------------------------------------------------------------------


class TestV1ToV2Migration:
    @pytest.mark.asyncio
    async def test_v2_adds_unique_index(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        async with aiosqlite.connect(str(db_path)) as db:
            runner = MigrationRunner(db, _logger, db_label="test")
            await runner.run(
                [
                    MigrationStep(1, _apply_memory_v1, "v1"),
                    MigrationStep(2, _apply_memory_v2, "v2"),
                ],
            )
            indexes = await _index_names(db)

        assert "idx_memory_chat_key" in indexes

    @pytest.mark.asyncio
    async def test_v2_deduplicates_memory_rows(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        async with aiosqlite.connect(str(db_path)) as db:
            runner = MigrationRunner(db, _logger, db_label="test")
            # Apply V1 only first.
            await runner.run(
                [MigrationStep(1, _apply_memory_v1, "v1")],
            )
            # Insert duplicate keys (same chat_id + key, different timestamps).
            await db.execute(
                "INSERT INTO long_term_memory (chat_id, key, value, updated_at) VALUES (1, 'color', 'red', '2025-01-01 00:00:00')"
            )
            await db.execute(
                "INSERT INTO long_term_memory (chat_id, key, value, updated_at) VALUES (1, 'color', 'blue', '2025-06-01 00:00:00')"
            )
            await db.execute(
                "INSERT INTO long_term_memory (chat_id, key, value, updated_at) VALUES (1, 'food', 'pizza', '2025-01-01 00:00:00')"
            )
            await db.commit()

            # Now apply V2 — it should deduplicate, keeping 'blue'.
            runner2 = MigrationRunner(db, _logger, db_label="test")
            await runner2.run(
                [
                    MigrationStep(1, _apply_memory_v1, "v1"),
                    MigrationStep(2, _apply_memory_v2, "v2"),
                ],
            )
            async with db.execute(
                "SELECT key, value FROM long_term_memory WHERE chat_id = 1 ORDER BY key"
            ) as cur:
                rows = await cur.fetchall()

        result = {row[0]: row[1] for row in rows}
        assert result == {"color": "blue", "food": "pizza"}

    @pytest.mark.asyncio
    async def test_v2_records_both_versions(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        async with aiosqlite.connect(str(db_path)) as db:
            runner = MigrationRunner(db, _logger, db_label="test")
            await runner.run(
                [
                    MigrationStep(1, _apply_memory_v1, "v1"),
                    MigrationStep(2, _apply_memory_v2, "v2"),
                ],
            )
            async with db.execute("SELECT version FROM schema_migrations ORDER BY version") as cur:
                rows = await cur.fetchall()

        assert [row[0] for row in rows] == [1, 2]


# ---------------------------------------------------------------------------
# Idempotent migration (running twice must not fail)
# ---------------------------------------------------------------------------


class TestIdempotentMigration:
    @pytest.mark.asyncio
    async def test_running_all_migrations_twice_is_safe(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        steps = [
            MigrationStep(1, _apply_memory_v1, "v1"),
            MigrationStep(2, _apply_memory_v2, "v2"),
        ]
        async with aiosqlite.connect(str(db_path)) as db:
            runner = MigrationRunner(db, _logger, db_label="test")
            await runner.run(steps)
            # Second run — must not raise.
            await runner.run(steps)

            async with db.execute("SELECT version FROM schema_migrations ORDER BY version") as cur:
                rows = await cur.fetchall()

        assert [row[0] for row in rows] == [1, 2]

    @pytest.mark.asyncio
    async def test_idempotent_with_separate_connections(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        steps = [
            MigrationStep(1, _apply_memory_v1, "v1"),
            MigrationStep(2, _apply_memory_v2, "v2"),
        ]
        for _ in range(2):
            async with aiosqlite.connect(str(db_path)) as db:
                runner = MigrationRunner(db, _logger, db_label="test")
                await runner.run(steps)

        async with aiosqlite.connect(str(db_path)) as db:
            tables = await _table_names(db)
            assert _MEMORY_TABLES.issubset(tables)


# ---------------------------------------------------------------------------
# Existing data survives migration
# ---------------------------------------------------------------------------


class TestDataSurvivedMigration:
    @pytest.mark.asyncio
    async def test_conversations_survive_v2_migration(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        async with aiosqlite.connect(str(db_path)) as db:
            runner = MigrationRunner(db, _logger, db_label="test")
            await runner.run(
                [MigrationStep(1, _apply_memory_v1, "v1")],
            )
            # Seed conversation data.
            await db.execute(
                "INSERT INTO conversations (chat_id, role, content) VALUES (42, 'user', 'hello world')"
            )
            await db.execute(
                "INSERT INTO conversations (chat_id, role, content) VALUES (42, 'assistant', 'hi there')"
            )
            await db.commit()

            # Apply V2 migration on top.
            runner2 = MigrationRunner(db, _logger, db_label="test")
            await runner2.run(
                [
                    MigrationStep(1, _apply_memory_v1, "v1"),
                    MigrationStep(2, _apply_memory_v2, "v2"),
                ],
            )

            async with db.execute(
                "SELECT role, content FROM conversations WHERE chat_id = 42 ORDER BY id"
            ) as cur:
                rows = await cur.fetchall()

        assert [(r[0], r[1]) for r in rows] == [("user", "hello world"), ("assistant", "hi there")]

    @pytest.mark.asyncio
    async def test_unique_memory_entries_survive_v2(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        async with aiosqlite.connect(str(db_path)) as db:
            runner = MigrationRunner(db, _logger, db_label="test")
            await runner.run(
                [MigrationStep(1, _apply_memory_v1, "v1")],
            )
            # Insert non-duplicate memory entries.
            await db.execute(
                "INSERT INTO long_term_memory (chat_id, key, value) VALUES (1, 'name', 'Alice')"
            )
            await db.execute(
                "INSERT INTO long_term_memory (chat_id, key, value) VALUES (1, 'lang', 'Python')"
            )
            await db.execute(
                "INSERT INTO long_term_memory (chat_id, key, value) VALUES (2, 'name', 'Bob')"
            )
            await db.commit()

            # Apply V2.
            runner2 = MigrationRunner(db, _logger, db_label="test")
            await runner2.run(
                [
                    MigrationStep(1, _apply_memory_v1, "v1"),
                    MigrationStep(2, _apply_memory_v2, "v2"),
                ],
            )

            async with db.execute(
                "SELECT chat_id, key, value FROM long_term_memory ORDER BY chat_id, key"
            ) as cur:
                rows = await cur.fetchall()

        assert [(r[0], r[1], r[2]) for r in rows] == [
            (1, "lang", "Python"),
            (1, "name", "Alice"),
            (2, "name", "Bob"),
        ]

    @pytest.mark.asyncio
    async def test_context_summaries_survive_migration(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        async with aiosqlite.connect(str(db_path)) as db:
            runner = MigrationRunner(db, _logger, db_label="test")
            await runner.run(
                [MigrationStep(1, _apply_memory_v1, "v1")],
            )
            await db.execute(
                "INSERT INTO context_summaries (chat_id, summary, last_archive_id, message_count) "
                "VALUES (10, 'test summary', 100, 50)"
            )
            await db.commit()

            runner2 = MigrationRunner(db, _logger, db_label="test")
            await runner2.run(
                [
                    MigrationStep(1, _apply_memory_v1, "v1"),
                    MigrationStep(2, _apply_memory_v2, "v2"),
                ],
            )

            async with db.execute(
                "SELECT summary, message_count FROM context_summaries WHERE chat_id = 10"
            ) as cur:
                row = await cur.fetchone()

        assert row is not None
        assert row[0] == "test summary"
        assert row[1] == 50
