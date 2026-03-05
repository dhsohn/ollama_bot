"""SimJobScheduler 입력 경로 해석 테스트."""

from __future__ import annotations

import os
import signal
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from core.config import SimQueueConfig, SimToolConfig
from core.sim_job_store import SimJob, SimJobStore
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
async def test_submit_job_queues_job_for_host_agent_dispatch(
    tmp_path: Path,
) -> None:
    tool = "orca_auto"
    base_dir = tmp_path / "orca_runs"
    input_dir = base_dir / "STRUC_EXT"
    input_dir.mkdir(parents=True)
    os.environ["SIM_INPUT_DIR_ORCA_AUTO"] = str(base_dir)

    scheduler, store = await _build_scheduler(tmp_path, tool)
    scheduler._submit_external_job_via_agent = AsyncMock(  # type: ignore[method-assign]
        return_value="external-12345"
    )
    try:
        job_id = await scheduler.submit_job(
            tool=tool,
            input_file="STRUC_EXT",
            submitted_by=123,
        )
        assert len(job_id) == 32
        scheduler._submit_external_job_via_agent.assert_not_awaited()
        job = await store.get_job(job_id)
        assert job is not None
        assert job["status"] == "queued"
        assert job["input_file"] == str(input_dir.resolve())
        assert job["pid"] is None
        assert job["cli_command"] is None
    finally:
        os.environ.pop("SIM_INPUT_DIR_ORCA_AUTO", None)
        await store.close()


@pytest.mark.asyncio
async def test_launch_job_builds_absolute_output_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    store = SimJobStore()
    await store.initialize(str(tmp_path / "sim_jobs.db"))
    resources = ResourceManager(total_cores=16, total_memory_mb=131072, max_concurrent=4)
    config = SimQueueConfig(
        enabled=True,
        job_work_dir="data/sim_jobs",
        tools={
            "orca_auto": SimToolConfig(
                enabled=True,
                executable="echo",
                cli_template="{executable} run-inp --reaction-dir '{input_file}' > {output_file} 2>&1",
                output_extension=".out",
            ),
        },
    )
    scheduler = SimJobScheduler(config=config, store=store, resources=resources)
    input_dir = tmp_path / "orca_runs" / "mj1"
    input_dir.mkdir(parents=True)
    job = {
        "job_id": "job-abs-out-1",
        "tool": "orca_auto",
        "input_file": str(input_dir),
        "cores": 4,
        "memory_mb": 16384,
    }

    captured: dict[str, str] = {}

    async def _fake_create_subprocess_shell(
        cmd: str,
        **kwargs: object,
    ) -> object:
        captured["cmd"] = cmd
        captured["cwd"] = str(kwargs.get("cwd") or "")
        raise RuntimeError("intentional failure")

    monkeypatch.setattr(
        "core.sim_scheduler.asyncio.create_subprocess_shell",
        _fake_create_subprocess_shell,
    )

    try:
        await scheduler._launch_job(job)
        expected_work_dir = (tmp_path / "data" / "sim_jobs" / job["job_id"]).resolve()
        expected_output = (expected_work_dir / "mj1.out").resolve()
        assert captured["cwd"] == str(expected_work_dir)
        assert str(expected_output) in captured["cmd"]
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
        assert status["external_memory_rss_mb"] == 0
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_dispatch_waits_when_external_jobs_exhaust_resources(
    tmp_path: Path,
) -> None:
    scheduler, store = await _build_scheduler(tmp_path, "orca_auto")
    scheduler._config.total_cores = 16
    scheduler._config.total_memory_mb = 131072
    scheduler._config.max_concurrent_jobs = 4

    tool = scheduler._config.tools["orca_auto"]
    tool.min_cores = 2
    tool.default_cores = 4
    tool.min_memory_mb = 16384
    tool.default_memory_mb = 32768
    tool.max_cores = 16
    tool.max_memory_mb = 65536

    job_id = await store.insert_job(
        SimJob(
            job_id="job-ext-block-1",
            tool="orca_auto",
            input_file="/tmp/STRUC_EXT_BLOCK",
            submitted_by=1,
            cores=4,
            memory_mb=32768,
        )
    )

    scheduler.get_external_running_jobs = AsyncMock(  # type: ignore[method-assign]
        return_value=[{"cores": 4, "memory_mb": 32768} for _ in range(4)]
    )
    try:
        await scheduler._dispatch_pending_jobs()
        job = await store.get_job(job_id)
        assert job is not None
        assert job["status"] == "queued"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_dispatch_runs_when_external_jobs_leave_capacity(
    tmp_path: Path,
) -> None:
    scheduler, store = await _build_scheduler(tmp_path, "orca_auto")
    scheduler._config.total_cores = 16
    scheduler._config.total_memory_mb = 131072
    scheduler._config.max_concurrent_jobs = 4

    tool = scheduler._config.tools["orca_auto"]
    tool.min_cores = 2
    tool.default_cores = 4
    tool.min_memory_mb = 16384
    tool.default_memory_mb = 32768
    tool.max_cores = 16
    tool.max_memory_mb = 65536

    job_id = await store.insert_job(
        SimJob(
            job_id="job-ext-run-1",
            tool="orca_auto",
            input_file="/tmp/STRUC_EXT_RUN",
            submitted_by=1,
            cores=4,
            memory_mb=32768,
        )
    )

    scheduler.get_external_running_jobs = AsyncMock(  # type: ignore[method-assign]
        return_value=[{"cores": 4, "memory_mb": 32768} for _ in range(2)]
    )
    scheduler._submit_external_job_via_agent = AsyncMock(  # type: ignore[method-assign]
        return_value="external-77777"
    )
    try:
        await scheduler._dispatch_pending_jobs()
        job = await store.get_job(job_id)
        assert job is not None
        assert job["status"] == "running"
        assert job["pid"] == 77777
        assert job["cli_command"] == "delegated:external-77777"
        scheduler._submit_external_job_via_agent.assert_awaited_once()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_compute_dispatch_resources_scales_up_when_queue_is_shallow(
    tmp_path: Path,
) -> None:
    scheduler, store = await _build_scheduler(tmp_path, "orca_auto")
    scheduler._config.max_concurrent_jobs = 8
    scheduler._config.total_cores = 16
    scheduler._config.total_memory_mb = 131072
    tool = scheduler._config.tools["orca_auto"]
    tool.default_cores = 2
    tool.default_memory_mb = 16384
    tool.max_cores = 16
    tool.max_memory_mb = 65536

    try:
        job = {"tool": "orca_auto", "cores": 2, "memory_mb": 16384}
        cores, memory_mb = scheduler._compute_dispatch_resources(
            job,
            queued_count=1,
            resource_status={
                "running_jobs": 0,
                "available_cores": 16,
                "available_memory_mb": 131072,
            },
        )
        assert cores == 16
        assert memory_mb == 65536
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_compute_dispatch_resources_keeps_baseline_when_backlog_is_high(
    tmp_path: Path,
) -> None:
    scheduler, store = await _build_scheduler(tmp_path, "orca_auto")
    scheduler._config.max_concurrent_jobs = 8
    scheduler._config.total_cores = 16
    scheduler._config.total_memory_mb = 131072
    tool = scheduler._config.tools["orca_auto"]
    tool.default_cores = 2
    tool.default_memory_mb = 16384
    tool.max_cores = 16
    tool.max_memory_mb = 65536

    try:
        job = {"tool": "orca_auto", "cores": 2, "memory_mb": 16384}
        cores, memory_mb = scheduler._compute_dispatch_resources(
            job,
            queued_count=5,
            resource_status={
                "running_jobs": 3,
                "available_cores": 10,
                "available_memory_mb": 65536,
            },
        )
        assert cores == 2
        assert memory_mb == 16384
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_compute_dispatch_resources_can_scale_down_to_tool_min(
    tmp_path: Path,
) -> None:
    scheduler, store = await _build_scheduler(tmp_path, "orca_auto")
    scheduler._config.max_concurrent_jobs = 8
    scheduler._config.total_cores = 8
    scheduler._config.total_memory_mb = 65536
    tool = scheduler._config.tools["orca_auto"]
    tool.min_cores = 1
    tool.default_cores = 2
    tool.min_memory_mb = 8192
    tool.default_memory_mb = 16384
    tool.max_cores = 16
    tool.max_memory_mb = 65536

    try:
        job = {"tool": "orca_auto", "cores": 2, "memory_mb": 16384}
        cores, memory_mb = scheduler._compute_dispatch_resources(
            job,
            queued_count=8,
            resource_status={
                "running_jobs": 0,
                "available_cores": 8,
                "available_memory_mb": 65536,
            },
        )
        assert cores == 1
        assert memory_mb == 8192
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_compute_dispatch_resources_returns_baseline_when_disabled(
    tmp_path: Path,
) -> None:
    scheduler, store = await _build_scheduler(tmp_path, "orca_auto")
    scheduler._config.adaptive_allocation_enabled = False
    scheduler._config.max_concurrent_jobs = 8
    scheduler._config.total_cores = 16
    scheduler._config.total_memory_mb = 131072
    tool = scheduler._config.tools["orca_auto"]
    tool.default_cores = 2
    tool.default_memory_mb = 16384
    tool.max_cores = 16
    tool.max_memory_mb = 65536

    try:
        job = {"tool": "orca_auto", "cores": 2, "memory_mb": 16384}
        cores, memory_mb = scheduler._compute_dispatch_resources(
            job,
            queued_count=1,
            resource_status={
                "running_jobs": 0,
                "available_cores": 16,
                "available_memory_mb": 131072,
            },
        )
        assert cores == 2
        assert memory_mb == 16384
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_submit_job_clamps_requested_resources_to_tool_min(
    tmp_path: Path,
) -> None:
    scheduler, store = await _build_scheduler(tmp_path, "orca_auto")
    tool = scheduler._config.tools["orca_auto"]
    tool.min_cores = 2
    tool.default_cores = 4
    tool.max_cores = 16
    tool.min_memory_mb = 8192
    tool.default_memory_mb = 16384
    tool.max_memory_mb = 65536

    input_dir = tmp_path / "orca_runs" / "STRUCX"
    input_dir.mkdir(parents=True)
    os.environ["SIM_INPUT_DIR_ORCA_AUTO"] = str(input_dir.parent)

    try:
        job_id = await scheduler.submit_job(
            tool="orca_auto",
            input_file="STRUCX",
            submitted_by=1,
            cores=1,
            memory_mb=4096,
        )
        job = await store.get_job(job_id)
        assert job is not None
        assert job["cores"] == 2
        assert job["memory_mb"] == 8192
    finally:
        os.environ.pop("SIM_INPUT_DIR_ORCA_AUTO", None)
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
async def test_get_external_running_jobs_uses_agent_when_available(
    tmp_path: Path,
) -> None:
    scheduler, store = await _build_scheduler(tmp_path, "orca_auto")
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


@pytest.mark.asyncio
async def test_sync_external_job_states_marks_missing_job_completed(
    tmp_path: Path,
) -> None:
    scheduler, store = await _build_scheduler(tmp_path, "orca_auto")
    job_id = await store.insert_job(
        SimJob(
            job_id="job-external-sync-1",
            tool="orca_auto",
            input_file="/tmp/STRUC_SYNC",
            submitted_by=1,
            cores=4,
            memory_mb=16384,
        )
    )
    await store.update_status(
        job_id,
        "running",
        pid=12345,
        started_at="2026-03-05 00:00:00",
        cli_command="delegated:external-12345",
    )
    scheduler.get_external_running_jobs = AsyncMock(return_value=[])  # type: ignore[method-assign]
    try:
        await scheduler._sync_external_job_states()
        job = await store.get_job(job_id)
        assert job is not None
        assert job["status"] == "completed"
        assert "외부 작업 종료 감지" in str(job["error_message"])
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_sync_external_job_states_marks_missing_job_failed_from_output(
    tmp_path: Path,
) -> None:
    scheduler, store = await _build_scheduler(tmp_path, "orca_auto")
    input_dir = tmp_path / "orca_runs" / "mj1"
    input_dir.mkdir(parents=True)
    output_file = input_dir / "mj1.out"
    output_file.write_text(
        "2026-03-05 05:53:38,956 [ERROR] core.commands.run_inp: "
        "Reaction directory must be under allowed root: /home/daehyupsohn/orca_runs. "
        "got=/app/kb/orca_runs/mj1\n",
        encoding="utf-8",
    )

    job_id = await store.insert_job(
        SimJob(
            job_id="job-external-sync-fail-1",
            tool="orca_auto",
            input_file=str(input_dir),
            submitted_by=1,
            cores=4,
            memory_mb=16384,
        )
    )
    await store.update_status(
        job_id,
        "running",
        pid=33333,
        started_at="2026-03-05 00:00:00",
        cli_command="delegated:external-33333",
        output_file=str(output_file),
    )
    scheduler.get_external_running_jobs = AsyncMock(return_value=[])  # type: ignore[method-assign]
    try:
        await scheduler._sync_external_job_states()
        job = await store.get_job(job_id)
        assert job is not None
        assert job["status"] == "failed"
        assert "외부 실행 실패" in str(job["error_message"])
        assert "allowed root" in str(job["error_message"]).lower()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_get_queue_status_includes_delegated_running_from_db(
    tmp_path: Path,
) -> None:
    scheduler, store = await _build_scheduler(tmp_path, "orca_auto")
    job_id = await store.insert_job(
        SimJob(
            job_id="job-external-status-1",
            tool="orca_auto",
            input_file="/tmp/STRUC_STATUS",
            submitted_by=1,
            cores=4,
            memory_mb=32768,
        )
    )
    await store.update_status(
        job_id,
        "running",
        pid=54321,
        started_at="2026-03-05 00:00:00",
        cli_command="delegated:external-54321",
    )
    scheduler.get_external_running_jobs = AsyncMock(return_value=[])  # type: ignore[method-assign]
    try:
        status = await scheduler.get_queue_status()
        assert status["external_running"] == 1
        assert status["running_total"] == 1
        assert status["allocated_external_cores"] == 4
        assert status["allocated_external_memory_mb"] == 32768
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_cancel_job_delegated_running_uses_external_cancel(
    tmp_path: Path,
) -> None:
    scheduler, store = await _build_scheduler(tmp_path, "orca_auto")
    job_id = await store.insert_job(
        SimJob(
            job_id="job-external-cancel-1",
            tool="orca_auto",
            input_file="/tmp/STRUC_CANCEL",
            submitted_by=1,
            cores=4,
            memory_mb=32768,
        )
    )
    await store.update_status(
        job_id,
        "running",
        pid=67890,
        started_at="2026-03-05 00:00:00",
        cli_command="delegated:external-67890",
    )
    scheduler.cancel_external_job = AsyncMock(return_value=True)  # type: ignore[method-assign]
    try:
        success = await scheduler.cancel_job(job_id)
        assert success is True
        scheduler.cancel_external_job.assert_awaited_once_with(67890)
        job = await store.get_job(job_id)
        assert job is not None
        assert job["status"] == "cancelled"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_recover_orphaned_jobs_does_not_requeue_delegated_running(
    tmp_path: Path,
) -> None:
    scheduler, store = await _build_scheduler(tmp_path, "orca_auto")
    job_id = await store.insert_job(
        SimJob(
            job_id="job-recover-delegated-1",
            tool="orca_auto",
            input_file="/tmp/STRUC_RECOVER",
            submitted_by=1,
            cores=4,
            memory_mb=16384,
        )
    )
    await store.update_status(
        job_id,
        "running",
        pid=55555,
        started_at="2026-03-05 00:00:00",
        cli_command="delegated:external-55555",
    )

    scheduler._resources.sync_from_db = AsyncMock()  # type: ignore[method-assign]
    try:
        await scheduler._recover_orphaned_jobs()
        job = await store.get_job(job_id)
        assert job is not None
        assert job["status"] == "running"
        scheduler._resources.sync_from_db.assert_awaited_once_with([])
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_to_agent_input_path_strips_container_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SIM_INPUT_DIR_ORCA_AUTO 접두사를 제거해 상대 경로를 반환한다."""
    monkeypatch.setenv("SIM_INPUT_DIR_ORCA_AUTO", "/app/kb/orca_runs")
    scheduler, store = await _build_scheduler(tmp_path, "orca_auto")
    try:
        # 단순 경로
        assert scheduler._to_agent_input_path("orca_auto", "/app/kb/orca_runs/mj1") == "mj1"
        # 중첩 경로
        assert scheduler._to_agent_input_path("orca_auto", "/app/kb/orca_runs/proj/mj1") == "proj/mj1"
        # 매칭 안 되면 원본 반환
        assert scheduler._to_agent_input_path("orca_auto", "/other/path/mj1") == "/other/path/mj1"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_to_agent_input_path_falls_back_to_global_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """도구별 환경변수가 없으면 SIM_INPUT_DIR로 폴백한다."""
    monkeypatch.delenv("SIM_INPUT_DIR_ORCA_AUTO", raising=False)
    monkeypatch.setenv("SIM_INPUT_DIR", "/app/kb/sim_inputs")
    scheduler, store = await _build_scheduler(tmp_path, "orca_auto")
    try:
        assert scheduler._to_agent_input_path("orca_auto", "/app/kb/sim_inputs/mj1") == "mj1"
    finally:
        await store.close()
