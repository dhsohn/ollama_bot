"""시뮬레이션 작업 스케줄러 — 큐에서 작업을 꺼내 subprocess로 실행한다.

동시 실행 슬롯 확인 → 디스패치 → 프로세스 모니터링 → 재시도/완료 처리를 수행하고,
텔레그램으로 상태 알림을 보낸다.
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import shutil
import signal
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from core.config import SimQueueConfig
from core.logging_setup import get_logger
from core.sim_external_tracker import ExternalRunningSnapshot, SimExternalTracker
from core.sim_job_store import SimJob, SimJobStore
from core.sim_resource_manager import ResourceManager

JobCompletionHook = Callable[[dict[str, Any]], Awaitable[None]]


class _TelegramLike(Protocol):
    """send_message를 제공하는 최소 인터페이스."""

    async def send_message(self, chat_id: int, text: str) -> None: ...


@dataclass(frozen=True)
class SubmitResult:
    """submit_job 반환값."""
    job_id: str
    cancelled_job_id: str | None = None


class SimJobScheduler:
    """비동기 스케줄링 루프로 시뮬레이션 작업을 관리한다."""

    def __init__(
        self,
        config: SimQueueConfig,
        store: SimJobStore,
        resources: ResourceManager,
    ) -> None:
        self._config = config
        self._store = store
        self._resources = resources
        self._logger = get_logger("sim_scheduler")
        self._external_tracker = SimExternalTracker(
            config=self._config,
            store=self._store,
            logger=self._logger,
        )

        self._telegram: _TelegramLike | None = None
        self._allowed_users: list[int] = []
        self._completion_hooks: list[JobCompletionHook] = []

        self._running_processes: dict[str, asyncio.subprocess.Process] = {}
        self._monitor_tasks: dict[str, asyncio.Task[None]] = {}
        self._scheduler_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._known_external_jobs: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _tool_env_suffix(tool: str) -> str:
        """도구명을 환경변수 suffix 형식으로 변환한다."""
        return re.sub(r"[^A-Z0-9]+", "_", tool.upper()).strip("_")

    @staticmethod
    def _is_pid_alive(pid: int) -> bool:
        """PID가 현재 살아있는지 확인한다."""
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # 권한 부족 시에도 프로세스는 존재한다고 본다.
            return True
        except OSError:
            return False

    @staticmethod
    def _is_delegated_job(job: dict[str, Any]) -> bool:
        return SimExternalTracker.is_delegated_job(job)

    def _find_ongoing_sim_pid(self, job: dict[str, Any]) -> int | None:
        """wrapper 종료 후 실제 시뮬레이션이 계속 실행 중인지 lockfile로 확인한다."""
        for dir_raw in (job.get("work_dir"), job.get("input_file")):
            if not dir_raw:
                continue
            dir_path = Path(str(dir_raw)).expanduser()
            if not dir_path.is_dir():
                continue
            for src in ("run.lock", "run_state.json"):
                data = SimExternalTracker._load_json_file(dir_path / src)
                pid = self._safe_int(data.get("pid"))
                if pid and pid > 0 and self._is_pid_alive(pid):
                    return pid
        return None

    async def _list_tracked_delegated_jobs(self) -> list[dict[str, Any]]:
        """DB에서 위임 실행 중인 작업을 반환한다."""
        return await self._external_tracker.list_tracked_delegated_jobs()

    async def _sync_external_job_states(self) -> None:
        await self._external_tracker.sync_external_job_states(
            notify_completed=self._notify_job_completed,
            notify_failed=self._notify_delegated_job_failed,
        )

    async def _check_new_external_jobs(self) -> None:
        """외부 실행 작업의 등장/소멸을 감지하여 텔레그램으로 알림을 보낸다."""
        external_jobs = await self.get_external_running_jobs()
        current_ids = {job["job_id"] for job in external_jobs}

        # 사라진 작업 → 완료/실패 판정 후 알림
        disappeared_ids = set(self._known_external_jobs) - current_ids
        for job_id in disappeared_ids:
            prev_job = self._known_external_jobs.pop(job_id)
            tool = prev_job.get("tool", "unknown")
            input_file = prev_job.get("input_file", "-")
            status, error = self._external_tracker._infer_missing_delegated_terminal_state(prev_job)
            if status == "failed":
                await self._notify(
                    f"[SIM] 외부 시뮬레이션 실패\n"
                    f"도구: {tool}\n"
                    f"입력: {input_file}\n"
                    f"오류: {error or '알 수 없음'}"
                )
            else:
                await self._notify(
                    f"[SIM] 외부 시뮬레이션 완료\n"
                    f"도구: {tool}\n"
                    f"입력: {input_file}"
                )

        # 새 작업 감지 알림
        new_detected = False
        for job in external_jobs:
            job_id = job["job_id"]
            if job_id in self._known_external_jobs:
                continue
            self._known_external_jobs[job_id] = job
            new_detected = True
            tool = job.get("tool", "unknown")
            input_file = job.get("input_file", "-")
            source = job.get("source", "")
            source_label = "lockfile" if source == "lockfile" else "process"
            await self._notify(
                f"[SIM] 외부 시뮬레이션 감지 ({source_label})\n"
                f"도구: {tool}\n"
                f"입력: {input_file}"
            )

        # 새 외부 작업 감지 시 동시 실행 한도 초과 경고
        if new_detected:
            resource_status = await self._resources.get_status()
            internal_running = int(resource_status.get("running_jobs", 0))
            external_snapshot = await self._external_running_snapshot()
            total_running = internal_running + external_snapshot.external_running
            max_jobs = int(self._config.max_concurrent_jobs)
            if total_running > max_jobs:
                await self._notify(
                    f"⚠️ [SIM] 동시 실행 한도 초과\n"
                    f"현재 실행: {total_running}개 (내부 {internal_running} + 외부 {external_snapshot.external_running})\n"
                    f"최대 한도: {max_jobs}개"
                )

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    async def _external_running_snapshot(self) -> ExternalRunningSnapshot:
        tracked_external_jobs = await self._list_tracked_delegated_jobs()
        tracked_external_pids = {
            pid
            for pid in (self._safe_int(job.get("pid")) for job in tracked_external_jobs)
            if pid is not None
        }
        external_jobs = await self.get_external_running_jobs()
        untracked_count = sum(
            1
            for job in external_jobs
            if (pid := self._safe_int(job.get("pid"))) is not None
            and pid not in tracked_external_pids
        )
        external_running = len(tracked_external_jobs) + untracked_count
        return ExternalRunningSnapshot(
            external_running=external_running,
            untracked_count=untracked_count,
        )

    def _resolve_input_path(self, tool: str, input_file: str) -> Path:
        """입력 경로를 실제 경로로 해석한다.

        우선 전달된 경로를 그대로 확인하고, 없으면 .env의
        SIM_INPUT_DIR_<TOOL> 또는 SIM_INPUT_DIR 하위에서 찾는다.
        """
        raw_path = Path(input_file).expanduser()
        if raw_path.exists():
            return raw_path.resolve()

        # 절대경로가 없으면 그대로 실패시킨다.
        if raw_path.is_absolute():
            raise FileNotFoundError(f"입력 경로를 찾을 수 없음: {input_file}")

        tool_suffix = self._tool_env_suffix(tool)
        env_keys = [f"SIM_INPUT_DIR_{tool_suffix}", "SIM_INPUT_DIR"]
        tried: list[str] = []

        for key in env_keys:
            root_raw = os.environ.get(key, "").strip()
            if not root_raw:
                continue
            root = Path(root_raw).expanduser()
            if not root.is_absolute():
                root = (Path.cwd() / root).resolve()
            candidate = (root / raw_path).resolve()
            tried.append(f"{key}:{candidate}")
            if candidate.exists():
                return candidate

        hint = (
            f"입력 경로를 찾을 수 없음: {input_file}. "
            f".env에 SIM_INPUT_DIR_{tool_suffix}=<기본경로> 또는 "
            "SIM_INPUT_DIR=<기본경로> 설정 후 "
            f"'/sim submit {tool} {input_file}' 형태로 사용하세요."
        )
        if tried:
            hint += f" (확인한 후보: {', '.join(tried)})"
        raise FileNotFoundError(hint)

    # ── 의존성 주입 ──

    def set_telegram(self, telegram: _TelegramLike) -> None:
        self._telegram = telegram

    def set_allowed_users(self, users: list[int]) -> None:
        self._allowed_users = list(users)

    def add_completion_hook(self, hook: JobCompletionHook) -> None:
        """작업 완료 시 호출될 콜백을 등록한다."""
        self._completion_hooks.append(hook)

    # ── 생명주기 ──

    async def start(self) -> None:
        """스케줄링 루프를 시작하고 orphan 작업을 복구한다."""
        await self._recover_orphaned_jobs()
        self._stop_event.clear()
        self._scheduler_task = asyncio.create_task(
            self._scheduling_loop(),
            name="sim_scheduler_loop",
        )
        self._logger.info("sim_scheduler_started")

    async def stop(self) -> None:
        """루프를 중지하고 실행 중인 프로세스를 정리한다."""
        self._stop_event.set()

        for _job_id, proc in list(self._running_processes.items()):
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=10)
            except (TimeoutError, ProcessLookupError):
                proc.kill()

        monitor_tasks = list(self._monitor_tasks.values())
        for task in monitor_tasks:
            task.cancel()
        if monitor_tasks:
            await asyncio.gather(*monitor_tasks, return_exceptions=True)

        self._running_processes.clear()
        self._monitor_tasks.clear()

        if self._scheduler_task is not None:
            self._scheduler_task.cancel()
            await asyncio.gather(self._scheduler_task, return_exceptions=True)

        self._logger.info("sim_scheduler_stopped")

    # ── 스케줄링 루프 ──

    async def _scheduling_loop(self) -> None:
        interval = self._config.queue_check_interval_seconds
        while not self._stop_event.is_set():
            try:
                await self._sync_external_job_states()
                await self._check_new_external_jobs()
                await self._dispatch_pending_jobs()
            except Exception as exc:
                self._logger.error("sim_scheduler_loop_error", error=str(exc))
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=interval,
                )
                break
            except TimeoutError:
                pass


    async def _dispatch_pending_jobs(self) -> None:
        """대기 중인 작업을 큐 순서대로 하나씩 디스패치한다."""
        candidates = await self._store.get_next_queued(limit=1)
        if not candidates:
            return
        job = candidates[0]

        # 내부 실행 중 작업 수
        resource_status = await self._resources.get_status()
        internal_running = int(resource_status.get("running_jobs", 0))

        # 외부 실행 중 작업 수 (tracked + untracked)
        external_snapshot = await self._external_running_snapshot()
        running_now = internal_running + external_snapshot.external_running

        if running_now >= int(self._config.max_concurrent_jobs):
            self._logger.info(
                "sim_job_waiting_for_slot",
                job_id=str(job.get("job_id") or ""),
                running_jobs=running_now,
                max_concurrent=int(self._config.max_concurrent_jobs),
            )
            return
        await self._launch_job(job)

    @staticmethod
    def _resolve_path(raw: str) -> Path:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            found = shutil.which(raw)
            p = Path(found).resolve() if found else (Path.cwd() / p).resolve()
        return p

    def _prepare_orca_auto_runtime_config(self, executable: str) -> str:
        """orca_auto 실행용 임시 설정 파일을 생성한다."""
        exe_path = self._resolve_path(executable)
        repo_root = exe_path.parent.parent
        source_cfg = repo_root / "config" / "orca_auto.yaml"
        if not source_cfg.exists():
            raise ValueError(f"orca_auto_config_not_found:{source_cfg}")

        try:
            payload = yaml.safe_load(source_cfg.read_text(encoding="utf-8")) or {}
        except OSError as exc:
            raise ValueError(f"orca_auto_config_read_failed:{exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"orca_auto_config_invalid:{source_cfg}")

        runtime = payload.get("runtime")
        if not isinstance(runtime, dict):
            runtime = {}
            payload["runtime"] = runtime

        allowed_root_raw = os.environ.get("SIM_INPUT_DIR_ORCA_AUTO", "").strip()
        if not allowed_root_raw:
            allowed_root_raw = os.environ.get("SIM_INPUT_DIR", "").strip() or "kb/orca_runs"
        organized_root_raw = os.environ.get("SIM_OUTPUT_DIR_ORCA_AUTO", "").strip()
        if not organized_root_raw:
            organized_root_raw = os.environ.get("SIM_OUTPUT_DIR", "").strip() or "kb/orca_outputs"

        allowed_root = self._resolve_path(allowed_root_raw)
        organized_root = self._resolve_path(organized_root_raw)
        if not allowed_root.is_dir():
            raise ValueError(f"allowed_root_not_found:{allowed_root}")
        organized_root.mkdir(parents=True, exist_ok=True)
        runtime["allowed_root"] = str(allowed_root)
        runtime["organized_root"] = str(organized_root)

        paths = payload.get("paths")
        if not isinstance(paths, dict):
            paths = {}
            payload["paths"] = paths

        override_orca = os.environ.get("ORCA_AUTO_ORCA_EXECUTABLE", "").strip()
        resolved_orca: Path | None = None
        if override_orca:
            resolved_orca = self._resolve_path(override_orca)
        else:
            existing_orca = str(paths.get("orca_executable") or "").strip()
            if existing_orca:
                candidate = self._resolve_path(existing_orca)
                if candidate.exists():
                    resolved_orca = candidate
            if resolved_orca is None:
                fallback = Path.home() / "opt" / "orca" / "orca"
                if fallback.exists():
                    resolved_orca = fallback

        if resolved_orca is None or not resolved_orca.exists():
            raise ValueError("orca_executable_not_found")
        if not os.access(resolved_orca, os.X_OK):
            raise ValueError(f"orca_executable_not_executable:{resolved_orca}")
        paths["orca_executable"] = str(resolved_orca)

        tmp_dir = Path("/tmp/sim_scheduler")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_cfg = tmp_dir / f"orca_auto_runtime_{int(time.time() * 1000)}_{os.getpid()}.yaml"
        tmp_cfg.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
        return str(tmp_cfg)

    async def _launch_job(self, job: dict[str, Any]) -> None:
        """CLI 커맨드를 빌드하고 subprocess를 실행한다."""
        job_id = job["job_id"]
        tool_name = job["tool"]
        tool_config = self._config.tools.get(tool_name)

        if not await self._resources.acquire():
            self._logger.info("sim_job_slot_unavailable", job_id=job_id)
            return

        if not tool_config or not tool_config.enabled:
            error_msg = f"알 수 없거나 비활성화된 도구: {tool_name}"
            await self._store.update_status(
                job_id, "failed",
                error_message=error_msg,
            )
            await self._resources.release()
            await self._notify(
                f"[SIM] 작업 {job_id[:8]} 실패\n{error_msg}"
            )
            return

        input_path = Path(job["input_file"]).expanduser().resolve()
        work_dir = input_path if input_path.is_dir() else input_path.parent
        output_file = str((work_dir / (input_path.stem + tool_config.output_extension)).resolve())

        exe_env_key = f"SIM_TOOL_EXECUTABLE_{self._tool_env_suffix(tool_name)}"
        executable = os.environ.get(exe_env_key, "").strip() or tool_config.executable
        if executable.startswith(("~", ".", "/")):
            executable = str(Path(executable).expanduser().resolve())

        safe_input = shlex.quote(job["input_file"])
        safe_output = shlex.quote(output_file)
        safe_executable = shlex.quote(executable)
        cmd = tool_config.cli_template.format(
            executable=safe_executable,
            input_file=safe_input,
            output_file=safe_output,
        )
        if tool_config.command_prefix.strip():
            prefix = tool_config.command_prefix.format(
                executable=safe_executable,
                input_file=safe_input,
                output_file=safe_output,
            )
            cmd = f"{prefix} {cmd}"

        env = dict(os.environ)
        for key, val_template in tool_config.env_vars.items():
            env[key] = val_template

        if tool_name == "orca_auto":
            try:
                env["ORCA_AUTO_CONFIG"] = self._prepare_orca_auto_runtime_config(executable)
            except ValueError as exc:
                self._logger.warning("sim_orca_auto_config_failed", error=str(exc))
                await self._store.update_status(
                    job_id, "failed", error_message=str(exc),
                )
                await self._resources.release()
                await self._notify(
                    f"[SIM] 작업 {job_id[:8]} 실패\n{exc}"
                )
                return

        self._logger.info("sim_job_launching", job_id=job_id, tool=tool_name, cmd=cmd)

        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                cwd=str(work_dir),
                env=env,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            self._running_processes[job_id] = proc
            await self._store.update_status(
                job_id, "running",
                pid=proc.pid,
                output_file=output_file,
                work_dir=str(work_dir),
                cli_command=cmd,
                started_at="CURRENT_TIMESTAMP",
            )
            await self._notify_job_started(job)

            task = asyncio.create_task(
                self._monitor_process(job_id, proc),
                name=f"sim_monitor_{job_id}",
            )
            self._monitor_tasks[job_id] = task

        except Exception as exc:
            self._logger.error("sim_job_launch_failed", job_id=job_id, error=str(exc))
            await self._store.update_status(
                job_id, "failed", error_message=str(exc),
            )
            await self._resources.release()
            await self._notify(
                f"[SIM] 작업 {job_id[:8]} 실행 실패\n{exc}"
            )

    async def _monitor_process(
        self,
        job_id: str,
        proc: asyncio.subprocess.Process,
    ) -> None:
        """프로세스 종료를 대기하고 성공/실패/재시도를 처리한다."""
        try:
            exit_code = await proc.wait()
        except asyncio.CancelledError:
            self._monitor_tasks.pop(job_id, None)
            await asyncio.shield(self._resources.release())
            return

        self._running_processes.pop(job_id, None)
        self._monitor_tasks.pop(job_id, None)
        await asyncio.shield(self._resources.release())

        job = await self._store.get_job(job_id)
        if exit_code == 0:
            # wrapper가 먼저 종료되고 실제 시뮬레이션이 계속 실행 중인지 확인
            if job:
                await asyncio.sleep(0.5)  # lockfile 기록 대기
                delegated_pid = self._find_ongoing_sim_pid(job)
                if delegated_pid is not None:
                    self._logger.info(
                        "sim_job_delegated",
                        job_id=job_id,
                        wrapper_pid=proc.pid,
                        delegated_pid=delegated_pid,
                    )
                    await self._store.update_status(
                        job_id, "running",
                        cli_command=f"delegated:{job.get('cli_command', '')}",
                        pid=delegated_pid,
                    )
                    return

            await self._store.update_status(
                job_id, "completed",
                exit_code=exit_code,
                completed_at="CURRENT_TIMESTAMP",
            )
            if job:
                await self._notify_job_completed(job)
            await self._run_completion_hooks(job_id)
        else:
            if job and job["retry_count"] < job["max_retries"]:
                retry_num = job["retry_count"] + 1
                self._logger.warning(
                    "sim_job_failed_will_retry",
                    job_id=job_id, exit_code=exit_code,
                    retry=retry_num, max_retries=job["max_retries"],
                )
                await asyncio.sleep(job["retry_delay_s"])
                await self._store.increment_retry(job_id)
                if job:
                    await self._notify_job_retrying(job, exit_code, retry_num)
            else:
                await self._store.update_status(
                    job_id, "failed",
                    exit_code=exit_code,
                    completed_at="CURRENT_TIMESTAMP",
                    error_message=f"프로세스 종료 코드: {exit_code}",
                )
                if job:
                    await self._notify_job_failed(job, exit_code)

    # ── 복구 ──

    async def _recover_orphaned_jobs(self) -> None:
        """시작 시 DB에 running이지만 프로세스가 없는 작업을 복구한다."""
        running_jobs = await self._store.get_running_jobs()
        for job in running_jobs:
            if self._is_delegated_job(job):
                # 위임 작업은 host PID namespace에 있으므로 로컬 kill(0)로 판단하지 않는다.
                continue
            pid = job.get("pid")
            alive = False
            if pid:
                try:
                    os.kill(pid, 0)
                    alive = True
                except (OSError, ProcessLookupError):
                    pass

            if not alive:
                await self._store.requeue_job(job["job_id"])
                self._logger.info("sim_job_recovered_to_queue", job_id=job["job_id"])

        actual_running = await self._store.get_running_jobs()
        internal_running = [j for j in actual_running if not self._is_delegated_job(j)]
        await self._resources.sync_from_db(internal_running)

    # ── 완료 훅 ──

    async def _run_completion_hooks(self, job_id: str) -> None:
        """등록된 완료 콜백을 실행한다."""
        if not self._completion_hooks:
            return
        job = await self._store.get_job(job_id)
        if not job:
            return
        for hook in self._completion_hooks:
            try:
                await hook(job)
            except Exception as exc:
                self._logger.warning(
                    "sim_completion_hook_failed",
                    job_id=job_id, hook=getattr(hook, "__name__", "?"),
                    error=str(exc),
                )

    # ── 알림 ──

    async def _notify(self, text: str) -> None:
        if self._telegram is None:
            return
        for user_id in self._allowed_users:
            try:
                await self._telegram.send_message(user_id, text)
            except Exception as exc:
                self._logger.warning(
                    "sim_notify_failed", user_id=user_id, error=str(exc),
                )

    async def _notify_job_started(self, job: dict[str, Any]) -> None:
        label = f" ({job.get('label')})" if job.get("label") else ""
        await self._notify(
            f"[SIM] 작업 {job['job_id'][:8]}{label} 시작\n"
            f"도구: {job['tool']}"
        )

    async def _notify_job_completed(self, job: dict[str, Any]) -> None:
        label = f" ({job.get('label')})" if job.get("label") else ""
        await self._notify(
            f"[SIM] 작업 {job['job_id'][:8]}{label} 완료"
        )

    async def _notify_delegated_job_failed(
        self, job: dict[str, Any], error_message: str,
    ) -> None:
        label = f" ({job.get('label')})" if job.get("label") else ""
        await self._notify(
            f"[SIM] 작업 {job['job_id'][:8]}{label} 실패\n"
            f"{error_message}"
        )

    async def _notify_job_failed(self, job: dict[str, Any], exit_code: int) -> None:
        label = f" ({job.get('label')})" if job.get("label") else ""
        await self._notify(
            f"[SIM] 작업 {job['job_id'][:8]}{label} 실패 (종료코드: {exit_code})\n"
            f"재시도 횟수 소진."
        )

    async def _notify_job_retrying(
        self, job: dict[str, Any], exit_code: int, attempt: int,
    ) -> None:
        label = f" ({job.get('label')})" if job.get("label") else ""
        await self._notify(
            f"[SIM] 작업 {job['job_id'][:8]}{label} 실패 (종료코드: {exit_code}), "
            f"재시도 중 ({attempt}/{job.get('max_retries', '?')})"
        )

    # ── 공개 API (텔레그램 핸들러용) ──

    async def submit_job(
        self,
        tool: str,
        input_file: str,
        submitted_by: int,
        *,
        priority: int = 100,
        max_retries: int | None = None,
        label: str = "",
    ) -> SubmitResult:
        """작업을 검증하고 큐에 등록한다."""
        tool_config = self._config.tools.get(tool)
        if not tool_config or not tool_config.enabled:
            raise ValueError(f"알 수 없거나 비활성화된 도구: {tool}")

        resolved_input = self._resolve_input_path(tool, input_file)

        cancelled_job_id: str | None = None
        existing = await self._store.find_active_job_by_input(tool, str(resolved_input))
        if existing:
            ex_status = existing["status"]
            ex_id = existing["job_id"]
            if ex_status == "running":
                raise ValueError(
                    f"이미 동일한 입력으로 실행 중인 작업이 있습니다: "
                    f"{ex_id[:8]}"
                )
            # queued 상태면 자동 취소 후 새 작업으로 교체
            await self._store.cancel_job(ex_id)
            cancelled_job_id = ex_id
            self._logger.info(
                "sim_job_duplicate_cancelled",
                cancelled_job_id=ex_id, tool=tool,
            )

        # external 작업(ps/lockfile)과도 중복 검사
        if not existing or existing["status"] != "running":
            resolved_str = str(resolved_input)
            for ext_job in await self.get_external_running_jobs():
                ext_input = ext_job.get("input_file", "")
                if not ext_input or ext_input == "-":
                    continue
                try:
                    ext_resolved = str(Path(ext_input).resolve())
                except OSError:
                    ext_resolved = ext_input
                if ext_resolved == resolved_str:
                    raise ValueError(
                        f"이미 동일한 입력으로 외부에서 실행 중인 작업이 있습니다: "
                        f"{ext_job['job_id']}"
                    )

        effective_retries = min(
            max_retries if max_retries is not None else self._config.default_retry_count,
            self._config.max_retry_count,
        )

        job = SimJob(
            job_id=uuid.uuid4().hex,
            tool=tool,
            input_file=str(resolved_input),
            submitted_by=submitted_by,
            priority=priority,
            max_retries=effective_retries,
            retry_delay_s=self._config.retry_delay_seconds,
            label=label,
        )

        job_id = await self._store.insert_job(job)
        self._logger.info("sim_job_submitted", job_id=job_id, tool=tool)
        return SubmitResult(job_id=job_id, cancelled_job_id=cancelled_job_id)

    async def cancel_job(self, job_id: str) -> bool:
        """작업을 취소한다. running이면 프로세스도 종료한다."""
        job = await self._store.get_job(job_id)
        if not job:
            return False
        if job["status"] in ("completed", "cancelled", "failed"):
            return False

        if job["status"] == "running" and self._is_delegated_job(job):
            pid_raw = job.get("pid")
            pid: int | None = None
            if pid_raw is not None:
                with suppress(TypeError, ValueError):
                    pid = int(pid_raw)
            if pid is None or pid <= 0:
                return False
            cancelled = await self.cancel_external_job(pid)
            if cancelled:
                await self._store.update_status(
                    job_id,
                    "cancelled",
                    completed_at="CURRENT_TIMESTAMP",
                )
            return cancelled

        proc = self._running_processes.get(job_id)
        if proc:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=10)
            except (TimeoutError, ProcessLookupError):
                proc.kill()
            self._running_processes.pop(job_id, None)
            task = self._monitor_tasks.pop(job_id, None)
            if task:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        return await self._store.cancel_job(job_id)

    async def cancel_external_job(
        self,
        pid: int,
        *,
        grace_seconds: float = 10.0,
    ) -> bool:
        """큐 외부에서 실행 중인 시뮬레이션 프로세스를 종료한다."""
        if pid <= 0 or pid == os.getpid():
            return False

        external_jobs = await self.get_external_running_jobs()
        matched_job = next(
            (
                j for j in external_jobs
                if isinstance(j.get("pid"), int) and int(j["pid"]) == pid
            ),
            None,
        )
        if matched_job is None:
            return False

        if not self._is_pid_alive(pid):
            self._logger.info(
                "sim_external_cancel_unreachable_pid",
                pid=pid,
                source=str(matched_job.get("source") or "unknown"),
            )
            return False

        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return False
        except (PermissionError, OSError) as exc:
            self._logger.warning(
                "sim_external_cancel_failed",
                pid=pid,
                signal="SIGTERM",
                error=str(exc),
            )
            return False

        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(grace_seconds, 0.1)
        while loop.time() < deadline:
            if not self._is_pid_alive(pid):
                return True
            await asyncio.sleep(0.2)

        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return True
        except (PermissionError, OSError) as exc:
            self._logger.warning(
                "sim_external_cancel_failed",
                pid=pid,
                signal="SIGKILL",
                error=str(exc),
            )
            return False

        for _ in range(10):
            if not self._is_pid_alive(pid):
                return True
            await asyncio.sleep(0.1)
        return not self._is_pid_alive(pid)

    async def clear_finished(self) -> int:
        """완료/실패/취소된 작업을 DB에서 삭제한다."""
        return await self._store.delete_finished_jobs()

    async def get_queue_status(self) -> dict[str, Any]:
        """큐 통계 + 실행 현황을 반환한다."""
        queue_stats = await self._store.get_queue_stats()
        resource_status = await self._resources.get_status()
        external_snapshot = await self._external_running_snapshot()
        queue_running = int(queue_stats.get("running", 0))

        return {
            **queue_stats,
            **resource_status,
            "external_running": external_snapshot.external_running,
            "running_total": queue_running + external_snapshot.untracked_count,
        }

    async def get_external_running_jobs(self) -> list[dict[str, Any]]:
        """큐 DB에 없는 실행 중 외부 시뮬레이션 프로세스를 탐지한다."""
        return await self._external_tracker.get_external_running_jobs()

    async def list_jobs(self, **kwargs: Any) -> list[dict[str, Any]]:
        return await self._store.list_jobs(**kwargs)

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        return await self._store.get_job(job_id)

    async def reprioritize(self, job_id: str, new_priority: int) -> bool:
        """대기 중인 작업의 우선순위를 변경한다."""
        job = await self._store.get_job(job_id)
        if not job or job["status"] != "queued":
            return False
        return await self._store.update_status(
            job_id, "queued", priority=new_priority,
        )

    def get_tools(self) -> dict[str, dict[str, Any]]:
        """설정된 도구 목록을 반환한다."""
        result: dict[str, dict[str, Any]] = {}
        for name, tc in self._config.tools.items():
            result[name] = {
                "enabled": tc.enabled,
                "executable": tc.executable,
                "command_prefix": tc.command_prefix,
            }
        return result
