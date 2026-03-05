"""SimJobStore 스키마/마이그레이션 테스트."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from core.sim_job_store import SimJobStore

_LEGACY_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS sim_jobs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id        TEXT    NOT NULL UNIQUE,
    tool          TEXT    NOT NULL,
    status        TEXT    NOT NULL DEFAULT 'queued',
    priority      INTEGER NOT NULL DEFAULT 100,
    cores         INTEGER NOT NULL DEFAULT 0,
    memory_mb     INTEGER NOT NULL DEFAULT 0,
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
"""


@pytest.mark.asyncio
async def test_initialize_migrates_legacy_schema_and_status(tmp_path: Path) -> None:
    db_path = tmp_path / "sim_jobs_legacy.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_LEGACY_SCHEMA_SQL)
        conn.execute(
            """
            INSERT INTO sim_jobs (
                job_id, tool, status, input_file, submitted_by, pid, cli_command
            ) VALUES (?, ?, 'running_external', ?, ?, ?, NULL)
            """,
            ("legacy-job-1", "orca_auto", "/tmp/STRUC_LEGACY", 1, 43210),
        )
        conn.commit()
    finally:
        conn.close()

    store = SimJobStore()
    await store.initialize(str(db_path))
    try:
        job = await store.get_job("legacy-job-1")
        assert job is not None
        assert job["status"] == "running"
        assert str(job["cli_command"]).startswith("delegated:external-43210")
    finally:
        await store.close()

    conn = sqlite3.connect(db_path)
    try:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(sim_jobs)")]
        assert "cores" not in columns
        assert "memory_mb" not in columns

        versions = [row[0] for row in conn.execute("SELECT version FROM schema_migrations")]
        assert 201 in versions
        assert 202 in versions
    finally:
        conn.close()
