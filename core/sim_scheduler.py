"""시뮬레이션 작업 스케줄러 — 큐에서 작업을 꺼내 subprocess로 실행한다.

리소스 확인 → 디스패치 → 프로세스 모니터링 → 재시도/완료 처리를 수행하고,
텔레그램으로 상태 알림을 보낸다.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import time

import yaml

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

    _IGNORED_PROCESS_TOKENS = {
        "python",
        "python3",
        "bash",
        "sh",
        "conda",
        "run",
        "base",
        "env",
        "bin",
        "usr",
        "local",
        "tmp",
    }
    _INPUT_HINT_PATTERNS = (
        re.compile(r"--reaction-dir(?:=|\s+)(?P<value>'[^']+'|\"[^\"]+\"|\S+)"),
        re.compile(r"(?:^|\s)--input(?:=|\s+)(?P<value>'[^']+'|\"[^\"]+\"|\S+)"),
        re.compile(r"(?:^|\s)--input-file(?:=|\s+)(?P<value>'[^']+'|\"[^\"]+\"|\S+)"),
        re.compile(r"(?:^|\s)-i(?:=|\s+)(?P<value>'[^']+'|\"[^\"]+\"|\S+)"),
    )

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

    @staticmethod
    def _token_in_command(command_lower: str, token_lower: str) -> bool:
        """명령 문자열에 토큰이 단어 단위로 포함되는지 확인한다."""
        if not token_lower:
            return False
        if token_lower.startswith("-"):
            return re.search(
                rf"(?<!\S){re.escape(token_lower)}(?:=|\s|$)",
                command_lower,
            ) is not None
        if re.fullmatch(r"[a-z0-9_]+", token_lower):
            return re.search(
                rf"(?<![a-z0-9_]){re.escape(token_lower)}(?![a-z0-9_])",
                command_lower,
            ) is not None
        return token_lower in command_lower

    @classmethod
    def _extract_input_hint(cls, command: str) -> str:
        """실행 커맨드에서 입력 경로 힌트를 추출한다."""
        for pattern in cls._INPUT_HINT_PATTERNS:
            match = pattern.search(command)
            if not match:
                continue
            value = match.group("value").strip().strip("'\"")
            if value:
                return value
        return "-"

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
    def _resolve_scan_root(raw_path: str) -> Path:
        root = Path(raw_path).expanduser()
        if root.is_absolute():
            return root
        return (Path.cwd() / root).resolve()

    def _lockfile_scan_roots(self) -> list[tuple[str, Path]]:
        """외부 실행 lock 파일(run.lock) 탐색 루트를 반환한다."""
        candidates: list[tuple[str, Path]] = []
        for tool_name in self._config.tools:
            key = f"SIM_INPUT_DIR_{self._tool_env_suffix(tool_name)}"
            raw = os.environ.get(key, "").strip()
            if not raw:
                continue
            root = self._resolve_scan_root(raw)
            if root.is_dir():
                candidates.append((tool_name, root))

        global_root = os.environ.get("SIM_INPUT_DIR", "").strip()
        if global_root:
            root = self._resolve_scan_root(global_root)
            if root.is_dir():
                fallback_tool = (
                    "orca_auto"
                    if "orca_auto" in self._config.tools
                    else (next(iter(self._config.tools), "external"))
                )
                candidates.append((fallback_tool, root))

        fallback_tool = (
            "orca_auto"
            if "orca_auto" in self._config.tools
            else (next(iter(self._config.tools), "external"))
        )
        for raw in ("kb/orca_runs",):
            root = self._resolve_scan_root(raw)
            if root.is_dir():
                candidates.append((fallback_tool, root))

        deduped: list[tuple[str, Path]] = []
        seen_roots: set[Path] = set()
        for tool_name, root in candidates:
            try:
                resolved = root.resolve()
            except OSError:
                continue
            if resolved in seen_roots:
                continue
            seen_roots.add(resolved)
            deduped.append((tool_name, resolved))
        return deduped

    @staticmethod
    def _load_json_file(path: Path) -> dict[str, Any]:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return {}
        if not raw.strip():
            return {}
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _parse_iso_datetime(raw: str | None) -> datetime | None:
        if not raw:
            return None
        text = raw.strip()
        if not text:
            return None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _is_delegated_job(job: dict[str, Any]) -> bool:
        cli = str(job.get("cli_command") or "")
        return cli.startswith("delegated:")

    async def _list_tracked_delegated_jobs(self) -> list[dict[str, Any]]:
        """DB에서 위임 실행 중인 작업을 반환한다.

        Backward compatibility:
        - 신규: status='running' + cli_command='delegated:*'
        - 레거시: status='running_external'
        """
        running_jobs = [
            j for j in await self._store.get_jobs_by_status("running")
            if self._is_delegated_job(j)
        ]
        legacy_jobs = await self._store.get_jobs_by_status("running_external")
        return running_jobs + legacy_jobs

    def _elapsed_seconds_from_lock(
        self,
        *,
        started_at_raw: str | None,
        lock_path: Path,
    ) -> int:
        started_at = self._parse_iso_datetime(started_at_raw)
        if started_at is not None:
            delta = datetime.now(timezone.utc) - started_at
            return max(0, int(delta.total_seconds()))
        try:
            mtime = lock_path.stat().st_mtime
        except OSError:
            return 0
        return max(0, int(datetime.now(timezone.utc).timestamp() - mtime))

    @staticmethod
    def _tail_text(path: Path, *, max_lines: int = 30, max_chars: int = 2000) -> str:
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return ""
        if max_lines > 0:
            lines = lines[-max_lines:]
        text = "\n".join(lines).strip()
        if len(text) > max_chars:
            text = text[-max_chars:]
        return text

    def _expected_output_file(self, job: dict[str, Any]) -> str | None:
        tool_name = str(job.get("tool") or "")
        input_raw = str(job.get("input_file") or "").strip()
        if not input_raw:
            return None
        output_ext = ".out"
        tool_cfg = self._config.tools.get(tool_name)
        if tool_cfg and tool_cfg.output_extension:
            output_ext = tool_cfg.output_extension

        input_path = Path(input_raw).expanduser()
        if input_path.is_dir():
            return str((input_path / (input_path.stem + output_ext)).resolve())
        if input_path.suffix:
            return str(input_path.with_suffix(output_ext).resolve())
        return str((input_path / (input_path.name + output_ext)).resolve())

    def _infer_missing_delegated_terminal_state(
        self,
        job: dict[str, Any],
    ) -> tuple[str | None, str | None]:
        """실행 목록에서 사라진 위임 작업의 종료 상태를 추론한다."""
        input_raw = str(job.get("input_file") or "").strip()
        if input_raw:
            run_state = self._load_json_file(Path(input_raw).expanduser() / "run_state.json")
            raw_status = str(run_state.get("status") or "").strip().lower()
            if raw_status in {"failed", "error", "cancelled", "aborted"}:
                detail = (
                    str(run_state.get("error") or "")
                    or str(run_state.get("message") or "")
                    or str(run_state.get("failure_reason") or "")
                    or "작업 실패"
                )
                return "failed", detail[:300]
            if raw_status in {"completed", "success", "done"}:
                return "completed", None

        output_raw = str(job.get("output_file") or "").strip()
        if not output_raw:
            expected_output = self._expected_output_file(job)
            if expected_output:
                output_raw = expected_output
        if not output_raw:
            return None, None

        output_path = Path(output_raw).expanduser()
        if not output_path.exists():
            return None, None

        tail = self._tail_text(output_path)
        if not tail:
            return None, None
        tail_lower = tail.lower()
        error_tokens = (
            "[error]",
            "traceback",
            "exception",
            "failed",
            "not found",
            "permission denied",
            "must be under allowed root",
        )
        if any(token in tail_lower for token in error_tokens):
            last_line = next((line.strip() for line in reversed(tail.splitlines()) if line.strip()), "")
            detail = last_line or tail
            detail = detail.replace("\n", " ").strip()
            return "failed", f"실행 실패: {detail[:300]}"
        return None, None

    def _tool_default_resources(self, tool_name: str) -> tuple[int, int]:
        """도구 기본 리소스(코어/메모리)를 반환한다."""
        tool = self._config.tools.get(tool_name)
        if tool is None:
            return 0, 0
        return int(tool.default_cores), int(tool.default_memory_mb)

    def _scan_lockfile_external_jobs(
        self,
        *,
        tracked_pids: set[int],
        seen_external_pids: set[int],
    ) -> list[dict[str, Any]]:
        """run.lock/run_state.json 기반으로 외부 작업을 탐지한다."""
        jobs: list[dict[str, Any]] = []
        for tool_name, root in self._lockfile_scan_roots():
            for lock_path in root.glob("*/run.lock"):
                lock_dir = lock_path.parent
                lock_data = self._load_json_file(lock_path)
                state_data = self._load_json_file(lock_dir / "run_state.json")

                state_status = str(state_data.get("status") or "").strip().lower()
                if state_status and state_status not in {"running", "retrying"}:
                    continue

                pid_raw = lock_data.get("pid")
                pid: int | None = None
                try:
                    if pid_raw is not None:
                        pid = int(pid_raw)
                except (TypeError, ValueError):
                    pid = None

                if pid is not None and pid > 0:
                    if pid in tracked_pids or pid in seen_external_pids:
                        continue
                    seen_external_pids.add(pid)

                input_hint = str(
                    state_data.get("reaction_dir")
                    or lock_dir
                )
                started_at_raw = (
                    str(state_data.get("started_at"))
                    if state_data.get("started_at")
                    else str(lock_data.get("started_at") or "")
                )
                elapsed_seconds = self._elapsed_seconds_from_lock(
                    started_at_raw=started_at_raw,
                    lock_path=lock_path,
                )
                selected_inp = str(state_data.get("selected_inp") or "").strip()
                if selected_inp:
                    cli_command = f"run-inp --reaction-dir {input_hint}"
                else:
                    cli_command = f"lockfile:{lock_path}"

                if pid is not None and pid > 0:
                    job_id = f"external-{pid}"
                else:
                    job_id = f"external-lock-{lock_dir.name}"

                default_cores, default_memory = self._tool_default_resources(tool_name)
                jobs.append(
                    {
                        "job_id": job_id,
                        "tool": tool_name,
                        "status": state_status or "running",
                        "priority": 0,
                        "cores": default_cores,
                        "memory_mb": default_memory,
                        "input_file": input_hint,
                        "output_file": None,
                        "submitted_by": 0,
                        "submitted_at": None,
                        "started_at": started_at_raw or None,
                        "completed_at": None,
                        "pid": pid,
                        "elapsed_seconds": elapsed_seconds,
                        "cli_command": cli_command,
                        "retry_count": 0,
                        "max_retries": 0,
                        "label": "external",
                        "external": True,
                        "source": "lockfile",
                        "resource_source": "config_default",
                    }
                )
        return jobs

    def _build_external_detection_tokens(self) -> dict[str, list[str]]:
        """도구별 외부 프로세스 탐지 토큰 목록을 구성한다."""
        placeholders = {"executable", "input_file", "output_file", "cores", "memory_mb"}
        token_map: dict[str, list[str]] = {}

        for tool_name, tool_config in self._config.tools.items():
            if not tool_config.enabled:
                continue

            tokens: set[str] = {tool_name.lower()}
            executable = tool_config.executable
            exec_name = Path(executable).name.lower()
            if exec_name:
                tokens.add(exec_name)
                exec_stem = Path(exec_name).stem.lower()
                if exec_stem:
                    tokens.add(exec_stem)

            for source in (tool_config.cli_template, tool_config.command_prefix):
                for word in re.findall(r"[A-Za-z0-9_.-]{3,}", source):
                    lw = word.lower()
                    if lw in placeholders or lw in self._IGNORED_PROCESS_TOKENS:
                        continue
                    if lw.startswith("-") and len(lw) <= 3:
                        continue
                    if lw.isdigit():
                        continue
                    tokens.add(lw)

            filtered = sorted(
                (t for t in tokens if t and t not in self._IGNORED_PROCESS_TOKENS),
                key=len,
                reverse=True,
            )
            if filtered:
                token_map[tool_name] = filtered

        return token_map

    def _match_tool_from_command(
        self,
        command: str,
        token_map: dict[str, list[str]],
    ) -> str | None:
        """명령행 문자열로부터 가장 가능성 높은 도구명을 반환한다."""
        command_lower = command.lower()
        best_tool: str | None = None
        best_len = -1

        for tool_name, tokens in token_map.items():
            for token in tokens:
                if not self._token_in_command(command_lower, token):
                    continue
                token_len = len(token)
                if token_len > best_len:
                    best_tool = tool_name
                    best_len = token_len
                break

        return best_tool

    async def _sync_external_job_states(self) -> None:
        """DB의 위임 실행 작업을 외부 에이전트 조회 결과로 동기화한다."""
        tracked_jobs = await self._list_tracked_delegated_jobs()
        if not tracked_jobs:
            return

        live_jobs = await self.get_external_running_jobs()
        live_by_pid: dict[int, dict[str, Any]] = {}
        for item in live_jobs:
            pid_raw = item.get("pid")
            if pid_raw is None:
                continue
            try:
                live_pid = int(pid_raw)
            except (TypeError, ValueError):
                continue
            if live_pid > 0:
                live_by_pid[live_pid] = item

        now = datetime.now(timezone.utc)
        stale_seconds = 30

        for job in tracked_jobs:
            pid_raw = job.get("pid")
            pid = None
            if pid_raw is not None:
                try:
                    pid = int(pid_raw)
                except (TypeError, ValueError):
                    pass

            if pid is not None and pid > 0:
                live = live_by_pid.get(pid)
                if live is not None:
                    updates: dict[str, Any] = {}
                    if not job.get("cli_command") and live.get("cli_command"):
                        updates["cli_command"] = str(live.get("cli_command"))
                    if not job.get("output_file") and live.get("output_file"):
                        updates["output_file"] = str(live.get("output_file"))
                    if updates:
                        await self._store.update_status(
                            str(job["job_id"]),
                            "running",
                            **updates,
                        )
                    continue

            inferred_status, inferred_error = self._infer_missing_delegated_terminal_state(job)
            if inferred_status == "failed":
                error_msg = inferred_error or "실행 실패"
                await self._store.update_status(
                    str(job["job_id"]),
                    "failed",
                    completed_at="CURRENT_TIMESTAMP",
                    error_message=error_msg,
                    pid=None,
                )
                await self._notify_delegated_job_failed(job, error_msg)
                continue
            if inferred_status == "completed":
                await self._store.update_status(
                    str(job["job_id"]),
                    "completed",
                    completed_at="CURRENT_TIMESTAMP",
                    pid=None,
                )
                await self._notify_job_completed(job)
                continue

            started_raw = job.get("started_at")
            started_dt = self._parse_iso_datetime(
                str(started_raw) if started_raw is not None else None,
            )
            if started_dt is not None:
                elapsed = (now - started_dt).total_seconds()
                if elapsed < stale_seconds:
                    continue

            await self._store.update_status(
                str(job["job_id"]),
                "completed",
                completed_at="CURRENT_TIMESTAMP",
                error_message="작업 종료 감지 (성공/실패 미확인)",
                pid=None,
            )
            await self._notify_job_completed(job)

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
                await self._sync_external_job_states()
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

    @staticmethod
    def _round_down_step(value: int, step: int) -> int:
        if value <= 0:
            return 0
        if step <= 1:
            return value
        if value < step:
            return value
        return max(1, (value // step) * step)

    def _compute_dispatch_resources(
        self,
        job: dict[str, Any],
        *,
        queued_count: int,
        resource_status: dict[str, Any],
    ) -> tuple[int, int]:
        """디스패치 직전 동적 리소스(코어/메모리)를 계산한다."""
        tool_name = str(job.get("tool") or "")
        tool_config = self._config.tools.get(tool_name)
        job_cores = max(1, int(job.get("cores", 1)))
        job_memory = max(1, int(job.get("memory_mb", 1)))
        if tool_config is None:
            return job_cores, job_memory

        min_cores = min(
            max(1, int(tool_config.min_cores)),
            int(tool_config.max_cores),
            int(self._config.total_cores),
        )
        min_memory = min(
            max(1, int(tool_config.min_memory_mb)),
            int(tool_config.max_memory_mb),
            int(self._config.total_memory_mb),
        )
        preferred_cores = min(
            max(job_cores, min_cores),
            int(tool_config.max_cores),
            int(self._config.total_cores),
        )
        preferred_memory = min(
            max(job_memory, min_memory),
            int(tool_config.max_memory_mb),
            int(self._config.total_memory_mb),
        )

        if not self._config.adaptive_allocation_enabled:
            return preferred_cores, preferred_memory

        running_jobs = max(0, int(resource_status.get("running_jobs", 0)))
        pending_jobs = max(1, int(queued_count))
        target_parallelism = max(
            1,
            min(int(self._config.max_concurrent_jobs), running_jobs + pending_jobs),
        )

        fair_cores = max(1, int(self._config.total_cores) // target_parallelism)
        fair_memory = max(1, int(self._config.total_memory_mb) // target_parallelism)

        target_cores = min(
            max(fair_cores, min_cores),
            int(tool_config.max_cores),
            int(self._config.total_cores),
        )
        target_memory = min(
            max(fair_memory, min_memory),
            int(tool_config.max_memory_mb),
            int(self._config.total_memory_mb),
        )

        available_cores = max(0, int(resource_status.get("available_cores", 0)))
        available_memory = max(0, int(resource_status.get("available_memory_mb", 0)))
        if available_cores > 0:
            target_cores = min(target_cores, available_cores)
        if available_memory > 0:
            target_memory = min(target_memory, available_memory)

        target_memory = self._round_down_step(
            target_memory,
            int(self._config.adaptive_memory_step_mb),
        )
        target_memory = max(1, target_memory)

        # 안전 하한 아래로는 내리지 않는다.
        # 가용 리소스가 min 이하이면 can_allocate에서 대기한다.
        target_cores = max(min_cores, target_cores)
        target_memory = max(min_memory, target_memory)

        # 실행 시작 시점을 기준으로 동적으로 증/감시키되,
        # 제출 시 요청치(preferred)보다 낮게 내려갈 수 있도록 허용한다.
        if target_cores == preferred_cores and target_memory == preferred_memory:
            return preferred_cores, preferred_memory
        return target_cores, target_memory

    async def _dispatch_pending_jobs(self) -> None:
        """대기 중인 작업을 큐 순서대로 하나씩 디스패치한다."""
        queue_stats = await self._store.get_queue_stats()
        queued_count = int(queue_stats.get("queued", 0))
        if queued_count <= 0:
            return
        candidates = await self._store.get_next_queued(limit=1)
        if not candidates:
            return
        job = candidates[0]
        resource_status = await self._resources.get_status()
        tracked_external_jobs = await self._list_tracked_delegated_jobs()
        tracked_external_pids = {
            int(j["pid"])
            for j in tracked_external_jobs
            if isinstance(j.get("pid"), int)
        }
        tracked_external_cores = sum(int(j.get("cores", 0)) for j in tracked_external_jobs)
        tracked_external_memory = sum(int(j.get("memory_mb", 0)) for j in tracked_external_jobs)
        external_jobs = await self.get_external_running_jobs()
        untracked_external_jobs: list[dict[str, Any]] = []
        for item in external_jobs:
            pid_raw = item.get("pid")
            pid: int | None = None
            if pid_raw is not None:
                try:
                    pid = int(pid_raw)
                except (TypeError, ValueError):
                    pass
            if pid is not None and pid in tracked_external_pids:
                continue
            untracked_external_jobs.append(item)

        external_running = len(tracked_external_jobs) + len(untracked_external_jobs)
        external_alloc_cores = tracked_external_cores + sum(
            int(j.get("cores", 0)) for j in untracked_external_jobs
        )
        external_alloc_memory = tracked_external_memory + sum(
            int(j.get("memory_mb", 0)) for j in untracked_external_jobs
        )

        allocated_cores = int(resource_status.get("allocated_cores", 0))
        allocated_memory = int(resource_status.get("allocated_memory_mb", 0))
        effective_status = dict(resource_status)
        effective_status["running_jobs"] = int(resource_status.get("running_jobs", 0)) + external_running
        effective_status["available_cores"] = max(
            0,
            int(self._config.total_cores) - allocated_cores - external_alloc_cores,
        )
        effective_status["available_memory_mb"] = max(
            0,
            int(self._config.total_memory_mb) - allocated_memory - external_alloc_memory,
        )
        dispatch_cores, dispatch_memory = self._compute_dispatch_resources(
            job,
            queued_count=queued_count,
            resource_status=effective_status,
        )
        dispatch_job = dict(job)
        dispatch_job["cores"] = dispatch_cores
        dispatch_job["memory_mb"] = dispatch_memory
        if (
            int(job.get("cores", 0)) != dispatch_cores
            or int(job.get("memory_mb", 0)) != dispatch_memory
        ):
            self._logger.info(
                "sim_job_resources_adapted",
                job_id=str(job.get("job_id") or ""),
                tool=str(job.get("tool") or ""),
                queued_count=queued_count,
                running_jobs=int(effective_status.get("running_jobs", 0)),
                cores_from=int(job.get("cores", 0)),
                memory_from=int(job.get("memory_mb", 0)),
                cores_to=dispatch_cores,
                memory_to=dispatch_memory,
            )

        if (
            dispatch_cores > int(effective_status.get("available_cores", 0))
            or dispatch_memory > int(effective_status.get("available_memory_mb", 0))
        ):
            self._logger.info(
                "sim_job_waiting_for_resources",
                job_id=str(job.get("job_id") or ""),
                tool=str(job.get("tool") or ""),
                queued_count=queued_count,
                running_jobs=int(effective_status.get("running_jobs", 0)),
                available_cores=int(effective_status.get("available_cores", 0)),
                available_memory_mb=int(effective_status.get("available_memory_mb", 0)),
                required_cores=dispatch_cores,
                required_memory_mb=dispatch_memory,
            )
            return

        running_now = int(effective_status.get("running_jobs", 0))
        if running_now >= int(self._config.max_concurrent_jobs):
            self._logger.info(
                "sim_job_waiting_for_slot",
                job_id=str(job.get("job_id") or ""),
                running_jobs=running_now,
                max_concurrent=int(self._config.max_concurrent_jobs),
            )
            return
        await self._launch_job(dispatch_job)

    @staticmethod
    def _resolve_path(raw: str) -> Path:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
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

        if not tool_config or not tool_config.enabled:
            await self._store.update_status(
                job_id, "failed",
                error_message=f"알 수 없거나 비활성화된 도구: {tool_name}",
            )
            await self._resources.release(job["cores"], job["memory_mb"])
            return

        input_path = Path(job["input_file"]).expanduser().resolve()
        work_dir = input_path if input_path.is_dir() else input_path.parent
        output_file = str((work_dir / (input_path.stem + tool_config.output_extension)).resolve())

        exe_env_key = f"SIM_TOOL_EXECUTABLE_{self._tool_env_suffix(tool_name)}"
        executable = os.environ.get(exe_env_key, "").strip() or tool_config.executable
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

        if tool_name == "orca_auto":
            try:
                env["ORCA_AUTO_CONFIG"] = self._prepare_orca_auto_runtime_config(executable)
            except ValueError as exc:
                self._logger.warning("sim_orca_auto_config_failed", error=str(exc))
                await self._store.update_status(
                    job_id, "failed", error_message=str(exc),
                )
                await self._resources.release(job["cores"], job["memory_mb"])
                return

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
                cores=job["cores"],
                memory_mb=job["memory_mb"],
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
        cores: int | None = None,
        memory_mb: int | None = None,
        priority: int = 100,
        max_retries: int | None = None,
        label: str = "",
    ) -> str:
        """작업을 검증하고 큐에 등록한다."""
        tool_config = self._config.tools.get(tool)
        if not tool_config or not tool_config.enabled:
            raise ValueError(f"알 수 없거나 비활성화된 도구: {tool}")

        resolved_input = self._resolve_input_path(tool, input_file)

        requested_cores = int(cores) if cores is not None else int(tool_config.default_cores)
        effective_cores = min(
            max(requested_cores, int(tool_config.min_cores)),
            int(tool_config.max_cores),
        )
        requested_memory = (
            int(memory_mb) if memory_mb is not None else int(tool_config.default_memory_mb)
        )
        effective_memory = min(
            max(requested_memory, int(tool_config.min_memory_mb)),
            int(tool_config.max_memory_mb),
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

        if (
            (job["status"] == "running" and self._is_delegated_job(job))
            or job["status"] == "running_external"
        ):
            pid_raw = job.get("pid")
            pid: int | None = None
            if pid_raw is not None:
                try:
                    pid = int(pid_raw)
                except (TypeError, ValueError):
                    pass
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
            except (asyncio.TimeoutError, ProcessLookupError):
                proc.kill()
            self._running_processes.pop(job_id, None)
            task = self._monitor_tasks.pop(job_id, None)
            if task:
                task.cancel()
            await self._resources.release(job["cores"], job["memory_mb"])

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

    async def get_queue_status(self) -> dict[str, Any]:
        """큐 통계 + 리소스 현황을 합쳐 반환한다."""
        queue_stats = await self._store.get_queue_stats()
        resource_status = await self._resources.get_status()
        tracked_external_jobs = await self._list_tracked_delegated_jobs()
        tracked_external_pids = {
            int(j["pid"])
            for j in tracked_external_jobs
            if isinstance(j.get("pid"), int)
        }
        tracked_external_cores = sum(int(j.get("cores", 0)) for j in tracked_external_jobs)
        tracked_external_memory_mb = sum(int(j.get("memory_mb", 0)) for j in tracked_external_jobs)

        external_jobs = await self.get_external_running_jobs()
        untracked_external_jobs: list[dict[str, Any]] = []
        for job in external_jobs:
            pid_raw = job.get("pid")
            pid: int | None = None
            if pid_raw is not None:
                try:
                    pid = int(pid_raw)
                except (TypeError, ValueError):
                    pass
            if pid is not None and pid in tracked_external_pids:
                continue
            untracked_external_jobs.append(job)

        queue_running = int(queue_stats.get("running", 0))
        external_running = len(tracked_external_jobs) + len(untracked_external_jobs)
        allocated_external_cores = tracked_external_cores + sum(
            int(j.get("cores", 0)) for j in untracked_external_jobs
        )
        allocated_external_memory_mb = tracked_external_memory_mb + sum(
            int(j.get("memory_mb", 0)) for j in untracked_external_jobs
        )
        external_memory_rss_mb = sum(
            int(j.get("memory_rss_mb", 0) or 0) for j in untracked_external_jobs
        )
        allocated_queue_cores = int(resource_status.get("allocated_cores", 0))
        allocated_queue_memory_mb = int(resource_status.get("allocated_memory_mb", 0))
        return {
            **queue_stats,
            **resource_status,
            "external_running": external_running,
            "running_total": queue_running + len(untracked_external_jobs),
            "allocated_external_cores": allocated_external_cores,
            "allocated_external_memory_mb": allocated_external_memory_mb,
            "allocated_total_cores": allocated_queue_cores + allocated_external_cores,
            "allocated_total_memory_mb": (
                allocated_queue_memory_mb + allocated_external_memory_mb
            ),
            "external_memory_rss_mb": external_memory_rss_mb,
            "running_total_jobs": (
                int(resource_status.get("running_jobs", 0))
                + len(tracked_external_jobs)
                + len(untracked_external_jobs)
            ),
        }

    async def get_external_running_jobs(self) -> list[dict[str, Any]]:
        """큐 DB에 없는 실행 중 외부 시뮬레이션 프로세스를 탐지한다."""
        token_map = self._build_external_detection_tokens()
        tracked_running = await self._store.get_running_jobs()
        tracked_pids = {
            int(job["pid"]) for job in tracked_running
            if isinstance(job.get("pid"), int)
        }

        current_pid = os.getpid()
        external_jobs: list[dict[str, Any]] = []
        seen_external_pids: set[int] = set()
        if token_map:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "ps",
                    "-eo",
                    "pid=,etimes=,args=",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()
            except FileNotFoundError:
                self._logger.info("sim_external_scan_ps_unavailable")
            except Exception as exc:
                self._logger.warning("sim_external_scan_failed", error=str(exc))
            else:
                if proc.returncode != 0:
                    self._logger.warning(
                        "sim_external_scan_failed",
                        returncode=proc.returncode,
                        stderr=stderr.decode(errors="ignore").strip(),
                    )
                else:
                    for raw_line in stdout.decode(errors="ignore").splitlines():
                        line = raw_line.strip()
                        if not line:
                            continue

                        match = re.match(r"^(?P<pid>\d+)\s+(?P<elapsed>\d+)\s+(?P<cmd>.+)$", line)
                        if not match:
                            continue

                        pid = int(match.group("pid"))
                        if pid == current_pid or pid in tracked_pids:
                            continue

                        command = match.group("cmd").strip()
                        tool_name = self._match_tool_from_command(command, token_map)
                        if tool_name is None:
                            continue

                        seen_external_pids.add(pid)
                        elapsed_seconds = int(match.group("elapsed"))
                        input_hint = self._extract_input_hint(command)
                        default_cores, default_memory = self._tool_default_resources(tool_name)
                        external_jobs.append(
                            {
                                "job_id": f"external-{pid}",
                                "tool": tool_name,
                                "status": "running",
                                "priority": 0,
                                "cores": default_cores,
                                "memory_mb": default_memory,
                                "input_file": input_hint,
                                "output_file": None,
                                "submitted_by": 0,
                                "submitted_at": None,
                                "started_at": None,
                                "completed_at": None,
                                "pid": pid,
                                "elapsed_seconds": elapsed_seconds,
                                "cli_command": command,
                                "retry_count": 0,
                                "max_retries": 0,
                                "label": "external",
                                "external": True,
                                "source": "process",
                                "resource_source": "config_default",
                            }
                        )

        external_jobs.extend(
            self._scan_lockfile_external_jobs(
                tracked_pids=tracked_pids,
                seen_external_pids=seen_external_pids,
            )
        )

        external_jobs.sort(key=lambda job: int(job.get("elapsed_seconds", 0)), reverse=True)
        return external_jobs

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
                "min_cores": tc.min_cores,
                "default_cores": tc.default_cores,
                "min_memory_mb": tc.min_memory_mb,
                "default_memory_mb": tc.default_memory_mb,
                "max_cores": tc.max_cores,
                "max_memory_mb": tc.max_memory_mb,
            }
        return result
