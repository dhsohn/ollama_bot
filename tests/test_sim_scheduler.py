"""SimJobScheduler 입력 경로 해석 테스트."""

from __future__ import annotations

import os
import signal
from pathlib import Path
from unittest.mock import AsyncMock

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


@pytest.mark.asyncio
async def test_detects_external_running_jobs_from_process_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimQueueConfig(
        enabled=True,
        tools={
            "orca_auto": SimToolConfig(
                enabled=True,
                executable="./bin/orca_auto",
                cli_template="{executable} run-inp --reaction-dir '{input_file}' > {output_file}",
            ),
            "crest": SimToolConfig(
                enabled=True,
                executable="crest",
                cli_template="{executable} {input_file} > {output_file}",
            ),
        },
    )
    store = SimJobStore()
    await store.initialize(str(tmp_path / "sim_jobs.db"))
    resources = ResourceManager(total_cores=16, total_memory_mb=131072, max_concurrent=4)
    scheduler = SimJobScheduler(config=config, store=store, resources=resources)

    class _FakePsProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            lines = [
                f"{os.getpid()} 10 python -m apps.ollama_bot.main",
                "1001 120 /home/test/orca_auto/.venv/bin/python3 -m core.cli run-inp --reaction-dir /tmp/STRUC1",
                "1002 30 crest /tmp/mol.xyz",
            ]
            return "\n".join(lines).encode(), b""

    async def _fake_create_subprocess_exec(*_args: object, **_kwargs: object) -> _FakePsProcess:
        return _FakePsProcess()

    monkeypatch.setattr(
        "core.sim_scheduler.asyncio.create_subprocess_exec",
        _fake_create_subprocess_exec,
    )

    try:
        jobs = await scheduler.get_external_running_jobs()
        assert len(jobs) == 2
        assert jobs[0]["job_id"] == "external-1001"
        assert jobs[0]["tool"] == "orca_auto"
        assert jobs[0]["input_file"] == "/tmp/STRUC1"
        assert "--reaction-dir /tmp/STRUC1" in jobs[0]["cli_command"]
        assert jobs[0]["cores"] == 4
        assert jobs[0]["memory_mb"] == 8192
        assert jobs[0]["resource_source"] == "config_default"
        assert jobs[1]["job_id"] == "external-1002"
        assert jobs[1]["tool"] == "crest"
        assert jobs[1]["cores"] == 4
        assert jobs[1]["memory_mb"] == 8192
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_queue_status_includes_external_running_count(tmp_path: Path) -> None:
    scheduler, store = await _build_scheduler(tmp_path, "orca_auto")
    scheduler.get_external_running_jobs = AsyncMock(  # type: ignore[method-assign]
        return_value=[
            {"pid": 101, "cores": 4, "memory_mb": 8192},
            {"pid": 102, "cores": 8, "memory_mb": 16384},
        ]
    )
    try:
        status = await scheduler.get_queue_status()
        assert status["external_running"] == 2
        assert status["running_total"] == 2
        assert status["allocated_external_cores"] == 12
        assert status["allocated_external_memory_mb"] == 24576
        assert status["allocated_total_cores"] == 12
        assert status["allocated_total_memory_mb"] == 24576
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_cancel_external_job_terminates_with_sigterm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler, store = await _build_scheduler(tmp_path, "orca_auto")
    scheduler.get_external_running_jobs = AsyncMock(  # type: ignore[method-assign]
        return_value=[{"pid": 43210}]
    )

    state = {"alive": True}
    signals: list[int] = []

    def _fake_kill(pid: int, sig: int) -> None:
        assert pid == 43210
        if sig == 0:
            if not state["alive"]:
                raise ProcessLookupError
            return
        signals.append(sig)
        if sig == signal.SIGTERM:
            state["alive"] = False

    async def _fast_sleep(_seconds: float) -> None:
        return

    monkeypatch.setattr("core.sim_scheduler.os.kill", _fake_kill)
    monkeypatch.setattr("core.sim_scheduler.asyncio.sleep", _fast_sleep)

    try:
        success = await scheduler.cancel_external_job(43210, grace_seconds=0.1)
        assert success is True
        assert signal.SIGTERM in signals
        assert signal.SIGKILL not in signals
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_cancel_external_job_escalates_to_sigkill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler, store = await _build_scheduler(tmp_path, "orca_auto")
    scheduler.get_external_running_jobs = AsyncMock(  # type: ignore[method-assign]
        return_value=[{"pid": 54321}]
    )

    state = {"alive": True}
    signals: list[int] = []

    def _fake_kill(pid: int, sig: int) -> None:
        assert pid == 54321
        if sig == 0:
            if not state["alive"]:
                raise ProcessLookupError
            return
        signals.append(sig)
        if sig == signal.SIGKILL:
            state["alive"] = False

    async def _fast_sleep(_seconds: float) -> None:
        return

    monkeypatch.setattr("core.sim_scheduler.os.kill", _fake_kill)
    monkeypatch.setattr("core.sim_scheduler.asyncio.sleep", _fast_sleep)

    try:
        success = await scheduler.cancel_external_job(54321, grace_seconds=0.1)
        assert success is True
        assert signal.SIGTERM in signals
        assert signal.SIGKILL in signals
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_cancel_external_job_rejects_non_external_pid(tmp_path: Path) -> None:
    scheduler, store = await _build_scheduler(tmp_path, "orca_auto")
    scheduler.get_external_running_jobs = AsyncMock(return_value=[])  # type: ignore[method-assign]
    try:
        success = await scheduler.cancel_external_job(98765)
        assert success is False
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_detects_external_jobs_from_lock_files_when_ps_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = "orca_auto"
    root = tmp_path / "orca_runs"
    job_dir = root / "STRUC2"
    job_dir.mkdir(parents=True)
    (job_dir / "run.lock").write_text(
        '{"pid": 77777, "started_at": "2026-03-04T10:58:50+00:00"}',
        encoding="utf-8",
    )
    (job_dir / "run_state.json").write_text(
        (
            "{"
            '"reaction_dir":"/host/orca_runs/STRUC2",'
            '"selected_inp":"/host/orca_runs/STRUC2/input.inp",'
            '"status":"running",'
            '"started_at":"2026-03-04T10:58:50+00:00"'
            "}"
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SIM_INPUT_DIR_ORCA_AUTO", str(root))

    scheduler, store = await _build_scheduler(tmp_path, tool)

    async def _raise_missing_ps(*_args: object, **_kwargs: object) -> object:
        raise FileNotFoundError

    monkeypatch.setattr(
        "core.sim_scheduler.asyncio.create_subprocess_exec",
        _raise_missing_ps,
    )

    try:
        jobs = await scheduler.get_external_running_jobs()
        assert len(jobs) == 1
        assert jobs[0]["job_id"] == "external-77777"
        assert jobs[0]["tool"] == "orca_auto"
        assert jobs[0]["status"] == "running"
        assert jobs[0]["input_file"] == "/host/orca_runs/STRUC2"
        assert jobs[0]["source"] == "lockfile"
        assert jobs[0]["cores"] == 4
        assert jobs[0]["memory_mb"] == 8192
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_cancel_external_job_returns_false_for_unreachable_lockfile_pid(
    tmp_path: Path,
) -> None:
    scheduler, store = await _build_scheduler(tmp_path, "orca_auto")
    scheduler.get_external_running_jobs = AsyncMock(  # type: ignore[method-assign]
        return_value=[{"pid": 45678, "source": "lockfile"}]
    )
    scheduler._is_pid_alive = lambda _pid: False  # type: ignore[method-assign]
    try:
        success = await scheduler.cancel_external_job(45678)
        assert success is False
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_get_external_running_jobs_uses_agent_when_enabled(
    tmp_path: Path,
) -> None:
    scheduler, store = await _build_scheduler(tmp_path, "orca_auto")
    scheduler._config.external_agent_enabled = True
    scheduler._fetch_external_jobs_from_agent = AsyncMock(  # type: ignore[method-assign]
        return_value=[{"job_id": "external-88888", "pid": 88888}]
    )
    scheduler._scan_lockfile_external_jobs = lambda **_kwargs: [  # type: ignore[method-assign]
        {"job_id": "external-99999", "pid": 99999}
    ]
    try:
        jobs = await scheduler.get_external_running_jobs()
        assert len(jobs) == 1
        assert jobs[0]["job_id"] == "external-88888"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_cancel_external_job_uses_agent_path_when_available(
    tmp_path: Path,
) -> None:
    scheduler, store = await _build_scheduler(tmp_path, "orca_auto")
    scheduler._config.external_agent_enabled = True
    scheduler.get_external_running_jobs = AsyncMock(  # type: ignore[method-assign]
        return_value=[{"pid": 24680, "source": "agent"}]
    )
    scheduler._cancel_external_job_via_agent = AsyncMock(  # type: ignore[method-assign]
        return_value=True
    )
    scheduler._is_pid_alive = lambda _pid: False  # type: ignore[method-assign]
    try:
        success = await scheduler.cancel_external_job(24680)
        assert success is True
    finally:
        await store.close()
