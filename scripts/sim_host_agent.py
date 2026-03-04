#!/usr/bin/env python3
"""Host-side simulation agent.

Responsibilities:
- Discover external simulation jobs from run.lock/run_state.json.
- Expose minimal HTTP API for status/cancel.
- Enforce token auth and deny arbitrary PID cancellation.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import yaml


def _tool_env_suffix(tool: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in tool.upper()).strip("_")


def _resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    if not text.strip():
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


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


def _proc_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    parts = [item for item in raw.split(b"\x00") if item]
    return " ".join(item.decode("utf-8", errors="ignore") for item in parts).strip()


def _proc_rss_mb(pid: int) -> int:
    status_path = Path(f"/proc/{pid}/status")
    try:
        lines = status_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return 0
    for line in lines:
        if not line.startswith("VmRSS:"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            kb = int(parts[1])
        except ValueError:
            continue
        return max(0, kb // 1024)
    return 0


def _proc_cpu_percent(pid: int) -> float:
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        text = stat_path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return 0.0
    right = text.rfind(")")
    if right < 0 or right + 2 >= len(text):
        return 0.0
    rest = text[right + 2 :].split()
    if len(rest) < 20:
        return 0.0
    try:
        utime = int(rest[11])
        stime = int(rest[12])
        start_ticks = int(rest[19])
    except ValueError:
        return 0.0

    try:
        uptime_raw = Path("/proc/uptime").read_text(encoding="utf-8", errors="ignore").strip()
        uptime_seconds = float(uptime_raw.split()[0])
    except (OSError, ValueError, IndexError):
        return 0.0

    clk_tck = os.sysconf("SC_CLK_TCK")
    if clk_tck <= 0:
        return 0.0

    proc_seconds = float(utime + stime) / float(clk_tck)
    elapsed = max(0.001, uptime_seconds - (float(start_ticks) / float(clk_tck)))
    return round(max(0.0, (proc_seconds / elapsed) * 100.0), 2)


def _load_tool_defaults(config_path: Path) -> dict[str, tuple[int, int]]:
    defaults: dict[str, tuple[int, int]] = {}
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except OSError:
        return defaults
    if not isinstance(raw, dict):
        return defaults
    sim_queue = raw.get("sim_queue")
    if not isinstance(sim_queue, dict):
        return defaults
    tools = sim_queue.get("tools")
    if not isinstance(tools, dict):
        return defaults
    for tool_name, tool_cfg in tools.items():
        if not isinstance(tool_name, str) or not isinstance(tool_cfg, dict):
            continue
        cores = int(tool_cfg.get("default_cores", 0) or 0)
        memory_mb = int(tool_cfg.get("default_memory_mb", 0) or 0)
        defaults[tool_name] = (max(0, cores), max(0, memory_mb))
    return defaults


def _scan_roots(tools: dict[str, tuple[int, int]]) -> list[tuple[str, Path]]:
    roots: list[tuple[str, Path]] = []
    for tool_name in tools:
        env_key = f"SIM_INPUT_DIR_{_tool_env_suffix(tool_name)}"
        raw = os.environ.get(env_key, "").strip()
        if not raw:
            continue
        path = _resolve_path(raw)
        if path.is_dir():
            roots.append((tool_name, path))

    global_root = os.environ.get("SIM_INPUT_DIR", "").strip()
    if global_root:
        path = _resolve_path(global_root)
        if path.is_dir():
            fallback_tool = "orca_auto" if "orca_auto" in tools else next(iter(tools), "external")
            roots.append((fallback_tool, path))

    host_kb = os.environ.get("HOST_KB_DIR", "").strip()
    if host_kb:
        path = (_resolve_path(host_kb) / "orca_runs").resolve()
        if path.is_dir():
            fallback_tool = "orca_auto" if "orca_auto" in tools else next(iter(tools), "external")
            roots.append((fallback_tool, path))

    fallback_tool = "orca_auto" if "orca_auto" in tools else next(iter(tools), "external")
    for raw in ("/app/kb/orca_runs", "kb/orca_runs"):
        path = _resolve_path(raw)
        if path.is_dir():
            roots.append((fallback_tool, path))

    deduped: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for tool_name, root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append((tool_name, resolved))
    return deduped


def _elapsed_seconds(started_at_raw: str | None, lock_path: Path) -> int:
    started_at = _parse_iso_datetime(started_at_raw)
    if started_at is not None:
        return max(0, int((datetime.now(timezone.utc) - started_at).total_seconds()))
    try:
        mtime = lock_path.stat().st_mtime
    except OSError:
        return 0
    return max(0, int(time.time() - mtime))


def discover_external_jobs(config_path: Path) -> list[dict[str, Any]]:
    tool_defaults = _load_tool_defaults(config_path)
    roots = _scan_roots(tool_defaults)
    jobs: list[dict[str, Any]] = []
    seen_pids: set[int] = set()

    for tool_name, root in roots:
        for lock_path in root.glob("*/run.lock"):
            lock_dir = lock_path.parent
            lock_data = _read_json_file(lock_path)
            state_data = _read_json_file(lock_dir / "run_state.json")

            state_status = str(state_data.get("status") or "").strip().lower()
            if state_status and state_status not in {"running", "retrying"}:
                continue

            pid_raw = lock_data.get("pid")
            try:
                pid = int(pid_raw)
            except (TypeError, ValueError):
                continue
            if pid <= 0 or pid in seen_pids:
                continue
            if not _is_pid_alive(pid):
                continue
            seen_pids.add(pid)

            started_at_raw = (
                str(state_data.get("started_at"))
                if state_data.get("started_at")
                else str(lock_data.get("started_at") or "")
            )
            reaction_dir = str(state_data.get("reaction_dir") or lock_dir)
            cmdline = _proc_cmdline(pid)
            rss_mb = _proc_rss_mb(pid)
            cpu_percent = _proc_cpu_percent(pid)
            default_cores, default_memory = tool_defaults.get(tool_name, (0, 0))
            # Keep memory_mb as configured allocation baseline and expose measured RSS separately.
            memory_mb = default_memory
            resource_source = "config_default"

            jobs.append(
                {
                    "job_id": f"external-{pid}",
                    "tool": tool_name,
                    "status": state_status or "running",
                    "priority": 0,
                    "cores": default_cores,
                    "memory_mb": memory_mb,
                    "memory_rss_mb": rss_mb,
                    "cpu_percent": cpu_percent,
                    "input_file": reaction_dir,
                    "output_file": None,
                    "submitted_by": 0,
                    "submitted_at": None,
                    "started_at": started_at_raw or None,
                    "completed_at": None,
                    "pid": pid,
                    "elapsed_seconds": _elapsed_seconds(started_at_raw, lock_path),
                    "cli_command": cmdline or f"pid:{pid}",
                    "retry_count": 0,
                    "max_retries": 0,
                    "label": "external",
                    "external": True,
                    "source": "agent",
                    "resource_source": resource_source,
                }
            )

    jobs.sort(key=lambda item: int(item.get("elapsed_seconds", 0)), reverse=True)
    return jobs


def cancel_pid(pid: int, grace_seconds: float) -> tuple[bool, str]:
    if pid <= 0:
        return False, "invalid_pid"
    if not _is_pid_alive(pid):
        return False, "pid_not_alive"

    try:
        os.kill(pid, signal.SIGTERM)
    except (PermissionError, OSError) as exc:
        return False, f"sigterm_failed:{exc}"

    deadline = time.monotonic() + max(0.1, grace_seconds)
    while time.monotonic() < deadline:
        if not _is_pid_alive(pid):
            return True, "terminated"
        time.sleep(0.2)

    try:
        os.kill(pid, signal.SIGKILL)
    except (PermissionError, OSError) as exc:
        return False, f"sigkill_failed:{exc}"

    for _ in range(10):
        if not _is_pid_alive(pid):
            return True, "killed"
        time.sleep(0.1)
    return (not _is_pid_alive(pid)), "unknown"


class _AgentServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        token: str,
        config_path: Path,
        grace_seconds: float,
    ) -> None:
        super().__init__(server_address, _AgentHandler)
        self.token = token
        self.config_path = config_path
        self.grace_seconds = grace_seconds


class _AgentHandler(BaseHTTPRequestHandler):
    server: _AgentServer

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"[sim_host_agent] {self.address_string()} - {fmt % args}\n")

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        expected = self.server.token
        if not expected:
            return False
        auth = self.headers.get("Authorization", "").strip()
        if not auth.startswith("Bearer "):
            return False
        token = auth[len("Bearer ") :].strip()
        return token == expected

    def _read_json_body(self) -> dict[str, Any]:
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return {}
        if size <= 0 or size > 16_384:
            return {}
        raw = self.rfile.read(size)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return

        if self.path != "/v1/sim/external/jobs":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        if not self._authorized():
            self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return

        jobs = discover_external_jobs(self.server.config_path)
        self._send_json(HTTPStatus.OK, {"jobs": jobs, "count": len(jobs)})

    def do_POST(self) -> None:
        if self.path != "/v1/sim/external/cancel":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        if not self._authorized():
            self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return

        payload = self._read_json_body()
        pid: int | None = None
        if "pid" in payload:
            try:
                pid = int(payload.get("pid"))
            except (TypeError, ValueError):
                pid = None
        elif "job_id" in payload:
            job_id = str(payload.get("job_id") or "")
            if job_id.startswith("external-"):
                try:
                    pid = int(job_id.split("-", 1)[1])
                except (TypeError, ValueError):
                    pid = None

        if pid is None or pid <= 0:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_pid"})
            return

        jobs = discover_external_jobs(self.server.config_path)
        if not any(int(job.get("pid", -1)) == pid for job in jobs):
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"error": "job_not_found", "pid": pid, "cancelled": False},
            )
            return

        cancelled, detail = cancel_pid(pid, self.server.grace_seconds)
        status = HTTPStatus.OK if cancelled else HTTPStatus.CONFLICT
        self._send_json(
            status,
            {"cancelled": cancelled, "pid": pid, "detail": detail},
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="External simulation host agent")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--token-env", default="SIM_EXTERNAL_AGENT_TOKEN")
    parser.add_argument("--grace-seconds", type=float, default=10.0)
    parser.add_argument("--allow-empty-token", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    token = os.environ.get(args.token_env, "").strip()
    if not token and not args.allow_empty_token:
        print(
            f"오류: {args.token_env} 환경변수가 비어 있습니다. "
            "호스트 에이전트는 토큰 인증이 필요합니다.",
            file=sys.stderr,
        )
        sys.exit(1)

    config_path = _resolve_path(args.config)
    server = _AgentServer(
        (args.host, args.port),
        token=token,
        config_path=config_path,
        grace_seconds=max(0.1, args.grace_seconds),
    )
    print(
        f"sim_host_agent listening on {args.host}:{args.port} "
        f"(config={config_path})",
        file=sys.stderr,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
