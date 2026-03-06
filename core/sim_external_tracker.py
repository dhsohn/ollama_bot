"""외부 시뮬레이션 프로세스 탐지/동기화 전담 모듈."""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from core.config import SimQueueConfig
from core.sim_job_store import SimJobStore


NotifyJobCompleted = Callable[[dict[str, Any]], Awaitable[None]]
NotifyJobFailed = Callable[[dict[str, Any], str], Awaitable[None]]


@dataclass(frozen=True)
class ExternalRunningSnapshot:
    """외부 실행 작업 집계 스냅샷."""

    external_running: int
    untracked_count: int


class SimExternalTracker:
    """큐 외부 실행 감지와 위임 작업 상태 동기화를 담당한다."""

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
        # 위치 인자: .xyz/.inp/.coord 확장자 (crest, xtb 등)
        re.compile(r"(?:^|\s)(?P<value>\S+\.(?:xyz|inp|coord))(?=\s|$)"),
    )

    def __init__(
        self,
        *,
        config: SimQueueConfig,
        store: SimJobStore,
        logger: Any,
    ) -> None:
        self._config = config
        self._store = store
        self._logger = logger

    @staticmethod
    def _tool_env_suffix(tool: str) -> str:
        return re.sub(r"[^A-Z0-9]+", "_", tool.upper()).strip("_")

    @staticmethod
    def _token_in_command(command_lower: str, token_lower: str) -> bool:
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
        for pattern in cls._INPUT_HINT_PATTERNS:
            match = pattern.search(command)
            if not match:
                continue
            value = match.group("value").strip().strip("'\"")
            if value:
                return value
        return "-"

    @staticmethod
    def _resolve_scan_root(raw_path: str) -> Path:
        root = Path(raw_path).expanduser()
        if root.is_absolute():
            return root
        return (Path.cwd() / root).resolve()

    def _lockfile_scan_roots(self) -> list[tuple[str, Path]]:
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
    def _safe_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _is_pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False

    @staticmethod
    def is_delegated_job(job: dict[str, Any]) -> bool:
        cli = str(job.get("cli_command") or "")
        return cli.startswith("delegated:")

    async def list_tracked_delegated_jobs(self) -> list[dict[str, Any]]:
        return [
            job
            for job in await self._store.get_jobs_by_status("running")
            if self.is_delegated_job(job)
        ]

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

    def _scan_lockfile_external_jobs(
        self,
        *,
        tracked_pids: set[int],
        seen_external_pids: set[int],
    ) -> list[dict[str, Any]]:
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

                input_hint = str(state_data.get("reaction_dir") or lock_dir)
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

                jobs.append(
                    {
                        "job_id": job_id,
                        "tool": tool_name,
                        "status": state_status or "running",
                        "priority": 0,
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
                    }
                )
        return jobs

    def _build_external_detection_tokens(self) -> dict[str, list[str]]:
        placeholders = {"executable", "input_file", "output_file"}
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

    async def sync_external_job_states(
        self,
        *,
        notify_completed: NotifyJobCompleted,
        notify_failed: NotifyJobFailed,
    ) -> None:
        tracked_jobs = await self.list_tracked_delegated_jobs()
        if not tracked_jobs:
            return

        live_jobs = await self.get_external_running_jobs()
        live_by_pid: dict[int, dict[str, Any]] = {}
        for item in live_jobs:
            pid = self._safe_int(item.get("pid"))
            if pid is None or pid <= 0:
                continue
            live_by_pid[pid] = item

        now = datetime.now(timezone.utc)
        stale_seconds = 30

        for job in tracked_jobs:
            pid = self._safe_int(job.get("pid"))
            if pid is not None and pid > 0:
                # get_external_running_jobs는 tracked PID를 제외하므로,
                # 프로세스 생존 여부를 직접 확인한다.
                if self._is_pid_alive(pid):
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
                await notify_failed(job, error_msg)
                continue
            if inferred_status == "completed":
                await self._store.update_status(
                    str(job["job_id"]),
                    "completed",
                    completed_at="CURRENT_TIMESTAMP",
                    pid=None,
                )
                await notify_completed(job)
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
            await notify_completed(job)

    async def external_running_snapshot(self) -> ExternalRunningSnapshot:
        tracked_external_jobs = await self.list_tracked_delegated_jobs()
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

    async def get_external_running_jobs(self) -> list[dict[str, Any]]:
        token_map = self._build_external_detection_tokens()
        tracked_running = await self._store.get_running_jobs()
        tracked_pids = {
            pid
            for pid in (self._safe_int(job.get("pid")) for job in tracked_running)
            if pid is not None
        }

        current_pid = os.getpid()
        external_jobs: list[dict[str, Any]] = []
        seen_external_pids: set[int] = set()
        ppid_map: dict[int, int] = {}
        if token_map:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "ps",
                    "-eo",
                    "pid=,ppid=,etimes=,args=",
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
                    parsed_lines: list[tuple[int, int, str]] = []
                    for raw_line in stdout.decode(errors="ignore").splitlines():
                        line = raw_line.strip()
                        if not line:
                            continue
                        match = re.match(
                            r"^(?P<pid>\d+)\s+(?P<ppid>\d+)\s+(?P<elapsed>\d+)\s+(?P<cmd>.+)$",
                            line,
                        )
                        if not match:
                            continue
                        p, pp = int(match.group("pid")), int(match.group("ppid"))
                        ppid_map[p] = pp
                        parsed_lines.append((p, int(match.group("elapsed")), match.group("cmd").strip()))

                    def _is_descendant_of_tracked(pid: int) -> bool:
                        visited: set[int] = set()
                        cur = pid
                        while cur in ppid_map and cur not in visited:
                            visited.add(cur)
                            cur = ppid_map[cur]
                            if cur in tracked_pids:
                                return True
                        return False

                    for pid, elapsed_seconds, command in parsed_lines:
                        if pid == current_pid or pid in tracked_pids:
                            continue
                        if _is_descendant_of_tracked(pid):
                            continue

                        cmd_base = command.split()[0] if command else ""
                        if cmd_base.endswith(("bash", "sh", "zsh", "fish", "dash")):
                            continue
                        tool_name = self._match_tool_from_command(command, token_map)
                        if tool_name is None:
                            continue

                        seen_external_pids.add(pid)
                        input_hint = self._extract_input_hint(command)
                        external_jobs.append(
                            {
                                "job_id": f"external-{pid}",
                                "tool": tool_name,
                                "status": "running",
                                "priority": 0,
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
                            }
                        )

        tracked_pids_expanded = set(tracked_pids)
        if ppid_map:
            for process_pid in ppid_map:
                cur = process_pid
                visited: set[int] = set()
                while cur in ppid_map and cur not in visited:
                    visited.add(cur)
                    cur = ppid_map[cur]
                    if cur in tracked_pids:
                        tracked_pids_expanded.add(process_pid)
                        break

        external_jobs.extend(
            self._scan_lockfile_external_jobs(
                tracked_pids=tracked_pids_expanded,
                seen_external_pids=seen_external_pids,
            )
        )

        external_jobs.sort(key=lambda job: int(job.get("elapsed_seconds", 0)), reverse=True)
        return external_jobs
