"""시뮬레이션 작업 큐 — SQLite 영속 계층.

작업 상태, 리소스 요청, 재시도 카운트 등을 관리한다.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite

from core.logging_setup import get_logger

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS sim_jobs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id        TEXT    NOT NULL UNIQUE,
    tool          TEXT    NOT NULL,
    status        TEXT    NOT NULL DEFAULT 'queued',
    priority      INTEGER NOT NULL DEFAULT 100,
    cores         INTEGER NOT NULL,
    memory_mb     INTEGER NOT NULL,
    input_file    TEXT    NOT NULL,
    output_file   TEXT,
    work_dir      TEXT,
    cli_command   TEXT,
    submitted_by  INTEGER NOT NULL,
    submitted_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at    TIMESTAMP,
    completed_at  TIMESTAMP,
    pid           INTEGER,
    exit_code     INTEGER,
    retry_count   INTEGER NOT NULL DEFAULT 0,
    max_retries   INTEGER NOT NULL DEFAULT 2,
    retry_delay_s INTEGER NOT NULL DEFAULT 30,
    error_message TEXT,
    label         TEXT    DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_sim_jobs_status    ON sim_jobs(status);
CREATE INDEX IF NOT EXISTS idx_sim_jobs_priority  ON sim_jobs(priority, submitted_at);
CREATE INDEX IF NOT EXISTS idx_sim_jobs_tool      ON sim_jobs(tool);
CREATE INDEX IF NOT EXISTS idx_sim_jobs_submitted ON sim_jobs(submitted_by);
"""


@dataclass
class SimJob:
    """새 작업 생성용 데이터 클래스."""

    job_id: str
    tool: str
    input_file: str
    submitted_by: int
    cores: int
    memory_mb: int
    priority: int = 100
    max_retries: int = 2
    retry_delay_s: int = 30
    label: str = ""


class SimJobStore:
    """시뮬레이션 작업의 비동기 SQLite 영속 계층."""

    def __init__(self) -> None:
        self._db: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()
        self._logger = get_logger("sim_job_store")

    async def initialize(self, db_path: str) -> None:
        """데이터베이스를 열고 스키마를 생성한다."""
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.executescript(_SCHEMA_SQL)
        await self._db.commit()
        self._logger.info("sim_job_store_initialized", db_path=db_path)

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    def _require_db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("SimJobStore가 아직 초기화되지 않았습니다.")
        return self._db

    # ── 쓰기 ──

    async def insert_job(self, job: SimJob) -> str:
        """새 작업을 삽입하고 job_id를 반환한다."""
        db = self._require_db()
        async with self._write_lock:
            await db.execute(
                """INSERT INTO sim_jobs
                   (job_id, tool, status, priority, cores, memory_mb,
                    input_file, submitted_by, max_retries, retry_delay_s, label)
                   VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job.job_id, job.tool, job.priority,
                    job.cores, job.memory_mb, job.input_file,
                    job.submitted_by, job.max_retries, job.retry_delay_s,
                    job.label,
                ),
            )
            await db.commit()
        self._logger.info("sim_job_inserted", job_id=job.job_id, tool=job.tool)
        return job.job_id

    async def update_status(self, job_id: str, status: str, **kwargs: Any) -> bool:
        """작업 상태와 추가 필드를 원자적으로 업데이트한다."""
        db = self._require_db()
        sets = ["status = ?"]
        params: list[Any] = [status]

        allowed_fields = {
            "pid", "exit_code", "error_message", "output_file",
            "work_dir", "cli_command", "started_at", "completed_at",
            "priority",
        }
        for key, value in kwargs.items():
            if key not in allowed_fields:
                continue
            if value == "CURRENT_TIMESTAMP":
                sets.append(f"{key} = CURRENT_TIMESTAMP")
            else:
                sets.append(f"{key} = ?")
                params.append(value)

        params.append(job_id)
        sql = f"UPDATE sim_jobs SET {', '.join(sets)} WHERE job_id = ?"

        async with self._write_lock:
            cursor = await db.execute(sql, params)
            await db.commit()
        return cursor.rowcount > 0

    async def increment_retry(self, job_id: str) -> bool:
        """retry_count를 증가시키고 상태를 'queued'로 되돌린다."""
        db = self._require_db()
        async with self._write_lock:
            cursor = await db.execute(
                """UPDATE sim_jobs
                   SET retry_count = retry_count + 1,
                       status = 'queued',
                       pid = NULL,
                       exit_code = NULL,
                       started_at = NULL
                   WHERE job_id = ?""",
                (job_id,),
            )
            await db.commit()
        return cursor.rowcount > 0

    async def cancel_job(self, job_id: str) -> bool:
        """작업을 취소한다. pending/queued/running 상태에서만 가능."""
        db = self._require_db()
        async with self._write_lock:
            cursor = await db.execute(
                """UPDATE sim_jobs
                   SET status = 'cancelled', completed_at = CURRENT_TIMESTAMP
                   WHERE job_id = ? AND status IN ('queued', 'running')""",
                (job_id,),
            )
            await db.commit()
        return cursor.rowcount > 0

    # ── 읽기 ──

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        db = self._require_db()
        async with db.execute(
            "SELECT * FROM sim_jobs WHERE job_id = ?", (job_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def get_next_queued(self, limit: int = 1) -> list[dict[str, Any]]:
        """우선순위가 가장 높은 대기 중 작업을 반환한다."""
        db = self._require_db()
        async with db.execute(
            """SELECT * FROM sim_jobs
               WHERE status = 'queued'
               ORDER BY priority ASC, submitted_at ASC
               LIMIT ?""",
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def get_running_jobs(self) -> list[dict[str, Any]]:
        db = self._require_db()
        async with db.execute(
            "SELECT * FROM sim_jobs WHERE status = 'running'"
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def list_jobs(
        self,
        *,
        status: str | None = None,
        submitted_by: int | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        conditions: list[str] = []
        params: list[Any] = []

        if status and status != "all":
            conditions.append("status = ?")
            params.append(status)
        if submitted_by is not None:
            conditions.append("submitted_by = ?")
            params.append(submitted_by)

        where = " AND ".join(conditions) if conditions else "1=1"
        params.append(min(limit, 50))

        async with self._require_db().execute(
            f"SELECT * FROM sim_jobs WHERE {where} ORDER BY submitted_at DESC LIMIT ?",
            params,
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def get_queue_stats(self) -> dict[str, Any]:
        """상태별 작업 수를 반환한다."""
        db = self._require_db()
        stats: dict[str, Any] = {}
        async with db.execute(
            "SELECT status, COUNT(*) as cnt FROM sim_jobs GROUP BY status"
        ) as cursor:
            async for row in cursor:
                stats[row["status"]] = row["cnt"]
        return stats
