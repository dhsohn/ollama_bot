"""ORCA 시뮬레이션 진행 상태 모니터.

auto_scheduler의 callable 액션으로 등록되어
orca_auto의 run_state.json 파일을 스캔하고 텔레그램 알림용 리포트를 반환한다.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.logging_setup import get_logger

logger = get_logger("orca_monitor")

_ACTIVE_STATUSES = frozenset({"created", "running", "retrying"})


async def _run_blocking(func, *args):
    """Python 3.8 호환 비동기-블로킹 브릿지."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args))


def _find_state_files(base: Path) -> list[Path]:
    """run_state.json 파일 목록을 정렬해 반환한다."""
    return sorted(base.rglob("run_state.json"))


def _load_state_file(path: Path) -> dict[str, Any] | None:
    """run_state.json을 안전하게 로드한다. 실패 시 None."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "run_id" in data:
            return data
        return None
    except Exception as exc:
        logger.warning("state_file_read_failed", path=str(path), error=str(exc))
        return None


def _elapsed_human(started_at: str | None) -> str:
    """ISO timestamp로부터 경과 시간을 '2d 3h 15m' 형태로 반환한다."""
    if not started_at:
        return "?"
    try:
        start = datetime.fromisoformat(started_at)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        total_sec = int((datetime.now(timezone.utc) - start).total_seconds())
        if total_sec < 0:
            return "0m"
        days, rem = divmod(total_sec, 86400)
        hours, rem = divmod(rem, 3600)
        minutes, _ = divmod(rem, 60)
        parts: list[str] = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        parts.append(f"{minutes}m")
        return " ".join(parts)
    except Exception:
        return "?"


def _dir_label(reaction_dir: str | None) -> str:
    """경로에서 마지막 디렉토리명만 추출한다."""
    if not reaction_dir:
        return "unknown"
    return Path(reaction_dir).name


async def generate_orca_progress_report(orca_runs_dir: str = "/orca_runs") -> str:
    """orca_runs_dir 하위의 run_state.json을 스캔하여 진행 리포트를 생성한다."""
    base = Path(orca_runs_dir)
    if not base.exists():
        return f"[ORCA Monitor] 디렉토리를 찾을 수 없습니다: {orca_runs_dir}"

    state_files = await _run_blocking(_find_state_files, base)
    if not state_files:
        return f"[ORCA Monitor] run_state.json 파일이 없습니다: {orca_runs_dir}"

    states: list[dict[str, Any]] = []
    load_errors = 0
    loaded_states = await asyncio.gather(
        *(_run_blocking(_load_state_file, sf) for sf in state_files)
    )
    for data in loaded_states:
        if data:
            states.append(data)
        else:
            load_errors += 1

    if not states:
        return f"[ORCA Monitor] 유효한 상태 파일이 없습니다 (오류: {load_errors}건)"

    # 상태별 분류
    by_status: dict[str, list[dict[str, Any]]] = {}
    for s in states:
        by_status.setdefault(s.get("status", "unknown"), []).append(s)

    total = len(states)
    completed = len(by_status.get("completed", []))
    failed = len(by_status.get("failed", []))
    running = len(by_status.get("running", []))
    retrying = len(by_status.get("retrying", []))
    created = len(by_status.get("created", []))
    completion_pct = completed / total * 100 if total else 0.0

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [
        "=== ORCA Simulation Progress ===",
        f"Report time: {now_str}",
        "",
        f"Total jobs: {total}",
        f"  Completed: {completed}  ({completion_pct:.1f}%)",
        f"  Failed:    {failed}",
        f"  Running:   {running}",
        f"  Retrying:  {retrying}",
        f"  Created:   {created}",
    ]
    if load_errors:
        lines.append(f"  (Parse errors: {load_errors})")
    lines.append("")

    # 진행 중 작업 상세
    active = [s for s in states if s.get("status") in _ACTIVE_STATUSES]
    if active:
        lines.append("--- Active Jobs ---")
        for s in active:
            label = _dir_label(s.get("reaction_dir"))
            status = s.get("status", "?")
            attempts = s.get("attempts", [])
            n = len(attempts) if isinstance(attempts, list) else 0
            max_r = s.get("max_retries", "?")
            elapsed = _elapsed_human(s.get("started_at"))
            lines.append(f"  [{status}] {label}  attempts: {n}/{max_r}  elapsed: {elapsed}")
            if attempts and isinstance(attempts, list):
                last = attempts[-1]
                a_status = last.get("analyzer_status", "")
                if a_status:
                    reason = last.get("analyzer_reason", "")
                    detail = f"    last: {a_status}"
                    if reason:
                        detail += f" - {reason[:80]}"
                    lines.append(detail)
        lines.append("")

    # 최근 완료 (최대 5건)
    completed_states = by_status.get("completed", [])
    if completed_states:
        recent = sorted(
            completed_states,
            key=lambda s: s.get("updated_at", ""),
            reverse=True,
        )[:5]
        lines.append("--- Recent Completions (last 5) ---")
        for s in recent:
            label = _dir_label(s.get("reaction_dir"))
            n = len(s.get("attempts", []))
            elapsed = _elapsed_human(s.get("started_at"))
            lines.append(f"  {label}  attempts: {n}  elapsed: {elapsed}")
        lines.append("")

    # 실패 작업
    failed_states = by_status.get("failed", [])
    if failed_states:
        lines.append("--- Failed Jobs ---")
        for s in failed_states:
            label = _dir_label(s.get("reaction_dir"))
            n = len(s.get("attempts", []))
            lines.append(f"  {label}  attempts: {n}")
            final = s.get("final_result")
            if isinstance(final, dict):
                reason = final.get("reason", final.get("analyzer_status", ""))
                if reason:
                    lines.append(f"    reason: {reason[:120]}")
        lines.append("")

    return "\n".join(lines)
