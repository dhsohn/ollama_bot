"""공통 SQLite 스키마 마이그레이션 프레임워크."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
from typing import Any, Awaitable, Callable, Iterable

import aiosqlite

_MIGRATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

MigrationCallable = Callable[[aiosqlite.Connection], Awaitable[None]]


@dataclass(frozen=True)
class MigrationStep:
    """단일 스키마 마이그레이션 단계."""

    version: int
    apply: MigrationCallable
    description: str = ""


class MigrationRunner:
    """DB별 스키마 마이그레이션을 실행한다."""

    def __init__(
        self,
        db: aiosqlite.Connection,
        logger: Any,
        *,
        db_label: str,
    ) -> None:
        self._db = db
        self._logger = logger
        self._db_label = db_label

    async def run(
        self,
        steps: Iterable[MigrationStep],
        *,
        backup_tables: set[str] | None = None,
    ) -> set[int]:
        """마이그레이션을 순서대로 적용한다."""
        ordered = sorted(steps, key=lambda item: item.version)
        if not ordered:
            return set()

        had_migrations_table = await self._table_exists("schema_migrations")
        had_target_table = False
        if backup_tables:
            for table in backup_tables:
                if await self._table_exists(table):
                    had_target_table = True
                    break

        await self._db.executescript(_MIGRATIONS_TABLE_SQL)
        await self._db.commit()

        applied = await self._get_applied_versions()
        pending = [step for step in ordered if step.version not in applied]

        if pending and (had_migrations_table or had_target_table):
            await self._backup_before_migration([step.version for step in pending])

        for step in pending:
            await step.apply(self._db)
            await self._db.execute(
                "INSERT OR IGNORE INTO schema_migrations (version) VALUES (?)",
                (step.version,),
            )
            await self._db.commit()
            self._logger.info(
                "db_migration_applied",
                db=self._db_label,
                version=step.version,
                description=step.description or None,
            )

        return {step.version for step in ordered}

    async def _table_exists(self, table: str) -> bool:
        async with self._db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
            (table,),
        ) as cursor:
            return await cursor.fetchone() is not None

    async def _get_applied_versions(self) -> set[int]:
        async with self._db.execute("SELECT version FROM schema_migrations") as cursor:
            rows = await cursor.fetchall()
        return {int(row[0]) for row in rows}

    async def _resolve_main_db_path(self) -> str | None:
        async with self._db.execute("PRAGMA database_list") as cursor:
            rows = await cursor.fetchall()
        for row in rows:
            name = row[1]
            db_path = row[2]
            if name != "main":
                continue
            if not db_path or db_path == ":memory:":
                return None
            return str(db_path)
        return None

    async def _backup_before_migration(self, pending_versions: list[int]) -> None:
        db_path = await self._resolve_main_db_path()
        if db_path is None:
            self._logger.info(
                "schema_backup_skipped",
                db=self._db_label,
                reason="in_memory_or_unknown_db",
                pending_versions=pending_versions,
            )
            return

        source = Path(db_path)
        if not source.exists():
            self._logger.warning(
                "schema_backup_skipped",
                db=self._db_label,
                reason="db_file_missing",
                db_path=str(source),
                pending_versions=pending_versions,
            )
            return

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = source.with_name(f"{source.name}.pre_migration_{stamp}.bak")

        await self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        await self._db.commit()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            shutil.copy2,
            str(source),
            str(backup),
        )
        self._logger.info(
            "schema_backup_created",
            db=self._db_label,
            source=str(source),
            backup=str(backup),
            pending_versions=pending_versions,
        )
