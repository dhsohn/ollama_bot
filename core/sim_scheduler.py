"""시뮬레이션 작업 스케줄러 — 큐에서 작업을 꺼내 subprocess로 실행한다.

리소스 확인 → 디스패치 → 프로세스 모니터링 → 재시도/완료 처리를 수행하고,
텔레그램으로 상태 알림을 보낸다.
"""

from __future__ import annotations

import asyncio
import os
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from core.config import SimQueueConfig
from core.logging_setup import get_logger
from core.sim_job_store import SimJob, SimJobStore
from core.sim_resource_manager import ResourceManager

if TYPE_CHECKING:
    from core.dft_index import DFTIndex


class _TelegramLike(Protocol):
    """send_message를 제공하는 최소 인터페이스."""

    async def send_message(self, chat_id: int, text: str) -> None: ...


class SimJobScheduler:
    """비동기 스케줄링 루프로 시뮬레이션 작업을 관리한다."""

    def __init__(
        self,
        config: SimQueueConfig,
        store: SimJobStore,
        resources: ResourceManager,
        dft_index: DFTIndex | None = None,
    ) -> None:
        self._config = config
        self._store = store
        self._resources = resources
        self._dft_index = dft_index
        self._logger = get_logger("sim_scheduler")

        self._telegram: _TelegramLike | None = None
        self._allowed_users: list[int] = []

        self._running_processes: dict[str, asyncio.subprocess.Process] = {}
        self._monitor_tasks: dict[str, asyncio.Task[None]] = {}
        self._scheduler_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    @staticmethod
    def _tool_env_suffix(tool: str) -> str:
        """도구명을 환경변수 suffix 형식으로 변환한다."""
        return re.sub(r"[^A-Z0-9]+", "_", tool.upper()).strip("_")

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

    def set_dft_index(self, dft_index: DFTIndex) -> None:
        self._dft_index = dft_index

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

        for task in list(self._monitor_tasks.values()):
            task.cancel()
        if self._monitor_tasks:
            await asyncio.gather(
                *self._monitor_tasks.values(), return_exceptions=True,
            )

        for job_id, proc in list(self._running_processes.items()):
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=10)
            except (asyncio.TimeoutError, ProcessLookupError):
                proc.kill()

        if self._scheduler_task is not None:
            self._scheduler_task.cancel()
            await asyncio.gather(self._scheduler_task, return_exceptions=True)

        self._logger.info("sim_scheduler_stopped")

    # ── 스케줄링 루프 ──

    async def _scheduling_loop(self) -> None:
        interval = self._config.queue_check_interval_seconds
        while not self._stop_event.is_set():
            try:
                await self._dispatch_pending_jobs()
            except Exception as exc:
                self._logger.error("sim_scheduler_loop_error", error=str(exc))
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=interval,
                )
                break
            except asyncio.TimeoutError:
                pass

    async def _dispatch_pending_jobs(self) -> None:
        """리소스가 허용하는 만큼 대기 중인 작업을 디스패치한다."""
        while True:
            candidates = await self._store.get_next_queued(limit=1)
            if not candidates:
                break
            job = candidates[0]
            if not await self._resources.can_allocate(job["cores"], job["memory_mb"]):
                break
            success = await self._resources.allocate(job["cores"], job["memory_mb"])
            if not success:
                break
            await self._launch_job(job)

    async def _launch_job(self, job: dict[str, Any]) -> None:
        """CLI 커맨드를 빌드하고 subprocess를 실행한다."""
        job_id = job["job_id"]
        tool_name = job["tool"]
        tool_config = self._config.tools.get(tool_name)

        if not tool_config or not tool_config.enabled:
            await self._store.update_status(
                job_id, "failed",
                error_message=f"알 수 없거나 비활성화된 도구: {tool_name}",
            )
            await self._resources.release(job["cores"], job["memory_mb"])
            return

        work_dir = Path(self._config.job_work_dir) / job_id
        work_dir.mkdir(parents=True, exist_ok=True)

        input_path = Path(job["input_file"])
        output_file = str(work_dir / (input_path.stem + tool_config.output_extension))
        executable = tool_config.executable
        if executable.startswith(("~", ".", "/")):
            executable = str(Path(executable).expanduser().resolve())

        cmd = tool_config.cli_template.format(
            executable=executable,
            input_file=job["input_file"],
            output_file=output_file,
            cores=job["cores"],
            memory_mb=job["memory_mb"],
        )
        if tool_config.command_prefix.strip():
            prefix = tool_config.command_prefix.format(
                executable=executable,
                input_file=job["input_file"],
                output_file=output_file,
                cores=job["cores"],
                memory_mb=job["memory_mb"],
            )
            cmd = f"{prefix} {cmd}"

        env = dict(os.environ)
        for key, val_template in tool_config.env_vars.items():
            env[key] = val_template.format(
                cores=job["cores"],
                memory_mb=job["memory_mb"],
            )

        self._logger.info(
            "sim_job_launching",
            job_id=job_id, tool=tool_name, cmd=cmd,
            cores=job["cores"], memory_mb=job["memory_mb"],
        )

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
                self._monitor_process(job_id, proc, job),
                name=f"sim_monitor_{job_id}",
            )
            self._monitor_tasks[job_id] = task

        except Exception as exc:
            self._logger.error("sim_job_launch_failed", job_id=job_id, error=str(exc))
            await self._store.update_status(
                job_id, "failed", error_message=str(exc),
            )
            await self._resources.release(job["cores"], job["memory_mb"])

    async def _monitor_process(
        self,
        job_id: str,
        proc: asyncio.subprocess.Process,
        job: dict[str, Any],
    ) -> None:
        """프로세스 종료를 대기하고 성공/실패/재시도를 처리한다."""
        try:
            exit_code = await proc.wait()
        except asyncio.CancelledError:
            self._running_processes.pop(job_id, None)
            self._monitor_tasks.pop(job_id, None)
            await self._resources.release(job["cores"], job["memory_mb"])
            return

        self._running_processes.pop(job_id, None)
        self._monitor_tasks.pop(job_id, None)
        await self._resources.release(job["cores"], job["memory_mb"])

        if exit_code == 0:
            await self._store.update_status(
                job_id, "completed",
                exit_code=exit_code,
                completed_at="CURRENT_TIMESTAMP",
            )
            await self._notify_job_completed(job)
            await self._try_index_output(job_id)
        else:
            current = await self._store.get_job(job_id)
            if current and current["retry_count"] < current["max_retries"]:
                retry_num = current["retry_count"] + 1
                self._logger.warning(
                    "sim_job_failed_will_retry",
                    job_id=job_id, exit_code=exit_code,
                    retry=retry_num, max_retries=current["max_retries"],
                )
                await asyncio.sleep(current["retry_delay_s"])
                await self._store.increment_retry(job_id)
                await self._notify_job_retrying(job, exit_code, retry_num)
            else:
                await self._store.update_status(
                    job_id, "failed",
                    exit_code=exit_code,
                    completed_at="CURRENT_TIMESTAMP",
                    error_message=f"프로세스 종료 코드: {exit_code}",
                )
                await self._notify_job_failed(job, exit_code)

    # ── 복구 ──

    async def _recover_orphaned_jobs(self) -> None:
        """시작 시 DB에 running이지만 프로세스가 없는 작업을 복구한다."""
        running_jobs = await self._store.get_running_jobs()
        for job in running_jobs:
            pid = job.get("pid")
            alive = False
            if pid:
                try:
                    os.kill(pid, 0)
                    alive = True
                except (OSError, ProcessLookupError):
                    pass

            if not alive:
                if job["retry_count"] < job["max_retries"]:
                    await self._store.increment_retry(job["job_id"])
                    self._logger.info("sim_job_recovered_to_queue", job_id=job["job_id"])
                else:
                    await self._store.update_status(
                        job["job_id"], "failed",
                        error_message="재시작 후 orphan 상태, 재시도 횟수 소진",
                    )
                    self._logger.warning("sim_job_orphan_failed", job_id=job["job_id"])

        actual_running = await self._store.get_running_jobs()
        await self._resources.sync_from_db(actual_running)

    # ── DFT 통합 ──

    async def _try_index_output(self, job_id: str) -> None:
        """ORCA 완료 시 DFT 인덱스에 자동 등록한다."""
        if self._dft_index is None:
            return
        job = await self._store.get_job(job_id)
        if not job or not job.get("output_file"):
            return
        if job.get("tool") not in ("orca_auto",):
            return
        try:
            success = await self._dft_index.upsert_single(job["output_file"])
            if success:
                self._logger.info(
                    "sim_job_output_indexed",
                    job_id=job_id, output_file=job["output_file"],
                )
        except Exception as exc:
            self._logger.warning(
                "sim_job_output_index_failed",
                job_id=job_id, error=str(exc),
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
            f"도구: {job['tool']} | 코어: {job['cores']} | 메모리: {job['memory_mb']}MB"
        )

    async def _notify_job_completed(self, job: dict[str, Any]) -> None:
        label = f" ({job.get('label')})" if job.get("label") else ""
        await self._notify(
            f"[SIM] 작업 {job['job_id'][:8]}{label} 완료"
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
        cores: int | None = None,
        memory_mb: int | None = None,
        priority: int = 100,
        max_retries: int | None = None,
        label: str = "",
    ) -> str:
        """작업을 검증하고 큐에 등록한다. job_id를 반환한다."""
        tool_config = self._config.tools.get(tool)
        if not tool_config or not tool_config.enabled:
            raise ValueError(f"알 수 없거나 비활성화된 도구: {tool}")

        resolved_input = self._resolve_input_path(tool, input_file)

        effective_cores = min(
            cores or tool_config.default_cores,
            tool_config.max_cores,
        )
        effective_memory = min(
            memory_mb or tool_config.default_memory_mb,
            tool_config.max_memory_mb,
        )
        effective_retries = min(
            max_retries if max_retries is not None else self._config.default_retry_count,
            self._config.max_retry_count,
        )

        if effective_cores > self._config.total_cores:
            raise ValueError(
                f"요청 코어 수({effective_cores})가 "
                f"전체 가용량({self._config.total_cores})을 초과합니다."
            )
        if effective_memory > self._config.total_memory_mb:
            raise ValueError(
                f"요청 메모리({effective_memory}MB)가 "
                f"전체 가용량({self._config.total_memory_mb}MB)을 초과합니다."
            )

        job = SimJob(
            job_id=uuid.uuid4().hex,
            tool=tool,
            input_file=str(resolved_input),
            submitted_by=submitted_by,
            cores=effective_cores,
            memory_mb=effective_memory,
            priority=priority,
            max_retries=effective_retries,
            retry_delay_s=self._config.retry_delay_seconds,
            label=label,
        )

        job_id = await self._store.insert_job(job)
        self._logger.info(
            "sim_job_submitted",
            job_id=job_id, tool=tool,
            cores=effective_cores, memory_mb=effective_memory,
        )
        return job_id

    async def cancel_job(self, job_id: str) -> bool:
        """작업을 취소한다. running이면 프로세스도 종료한다."""
        job = await self._store.get_job(job_id)
        if not job:
            return False
        if job["status"] in ("completed", "cancelled", "failed"):
            return False

        proc = self._running_processes.get(job_id)
        if proc:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=10)
            except (asyncio.TimeoutError, ProcessLookupError):
                proc.kill()
            self._running_processes.pop(job_id, None)
            task = self._monitor_tasks.pop(job_id, None)
            if task:
                task.cancel()
            await self._resources.release(job["cores"], job["memory_mb"])

        return await self._store.cancel_job(job_id)

    async def get_queue_status(self) -> dict[str, Any]:
        """큐 통계 + 리소스 현황을 합쳐 반환한다."""
        queue_stats = await self._store.get_queue_stats()
        resource_status = await self._resources.get_status()
        return {**queue_stats, **resource_status}

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
                "default_cores": tc.default_cores,
                "default_memory_mb": tc.default_memory_mb,
                "max_cores": tc.max_cores,
                "max_memory_mb": tc.max_memory_mb,
            }
        return result
