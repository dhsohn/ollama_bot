"""SimJobScheduler 입력 경로 해석 테스트."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import SimQueueConfig, SimToolConfig
from core.sim_job_store import SimJobStore
from core.sim_resource_manager import ResourceManager
from core.sim_scheduler import SimJobScheduler


async def _build_scheduler(tmp_path: Path, tool: str) -> tuple[SimJobScheduler, SimJobStore]:
    store = SimJobStore()
    await store.initialize(str(tmp_path / "sim_jobs.db"))
    resources = ResourceManager(total_cores=16, total_memory_mb=131072, max_concurrent=4)
    config = SimQueueConfig(
        enabled=True,
        tools={
            tool: SimToolConfig(
                enabled=True,
                executable="echo",
                cli_template="{executable} {input_file}",
            ),
        },
    )
    scheduler = SimJobScheduler(config=config, store=store, resources=resources)
    return scheduler, store


@pytest.mark.asyncio
async def test_submit_job_resolves_input_from_tool_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tool = "orca_auto"
    base_dir = tmp_path / "orca_runs"
    input_dir = base_dir / "STRUC1"
    input_dir.mkdir(parents=True)
    monkeypatch.setenv("SIM_INPUT_DIR_ORCA_AUTO", str(base_dir))

    scheduler, store = await _build_scheduler(tmp_path, tool)
    try:
        job_id = await scheduler.submit_job(
            tool=tool,
            input_file="STRUC1",
            submitted_by=123,
        )
        job = await store.get_job(job_id)
        assert job is not None
        assert job["input_file"] == str(input_dir.resolve())
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_submit_job_resolves_input_from_global_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tool = "crest"
    base_dir = tmp_path / "sim_inputs"
    input_dir = base_dir / "jobA"
    input_dir.mkdir(parents=True)
    monkeypatch.setenv("SIM_INPUT_DIR", str(base_dir))

    scheduler, store = await _build_scheduler(tmp_path, tool)
    try:
        job_id = await scheduler.submit_job(
            tool=tool,
            input_file="jobA",
            submitted_by=456,
        )
        job = await store.get_job(job_id)
        assert job is not None
        assert job["input_file"] == str(input_dir.resolve())
    finally:
        await store.close()
