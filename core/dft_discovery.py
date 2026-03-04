"""DFT 인덱싱 대상 파일 탐색 유틸."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_ORCA_EXTENSIONS = {".out", ".log"}


def discover_orca_targets(
    kb_path: Path,
    *,
    max_bytes: int,
    logger: Any | None = None,
    include_legacy_when_report_exists: bool = False,
    recent_completed_window_minutes: int | None = None,
) -> list[Path]:
    """인덱싱 대상 ORCA 출력 파일 목록을 반환한다.

    규칙:
    1) `orca_runs`: run_state 상태별 정책
       - running/failed: run_state 폴더 내 최신 .out 추적
       - completed: final_result.last_out_path 추적
    2) `orca_outputs`: final_result.last_out_path만 추적
    3) 그 외 경로: 기존 metadata 기반 일반 로직 사용
    4) include_legacy_when_report_exists=True이면 metadata 결과 + legacy 스캔 결과를 합친다.
    """
    policy = _infer_kb_policy(kb_path)
    if policy == "orca_runs":
        metadata_targets, metadata_found = _discover_orca_runs_targets(
            kb_path=kb_path,
            max_bytes=max_bytes,
            logger=logger,
        )
    elif policy == "orca_outputs":
        metadata_targets, metadata_found = _discover_orca_outputs_targets(
            kb_path=kb_path,
            max_bytes=max_bytes,
            logger=logger,
            recent_completed_window_minutes=recent_completed_window_minutes,
        )
    else:
        metadata_targets, metadata_found = _discover_from_run_metadata(
            kb_path=kb_path,
            max_bytes=max_bytes,
            logger=logger,
        )
    if metadata_found and not include_legacy_when_report_exists:
        return metadata_targets

    legacy_targets = _discover_legacy_targets(kb_path=kb_path, max_bytes=max_bytes)
    if not metadata_found:
        return legacy_targets

    merged: dict[str, Path] = {str(p): p for p in metadata_targets}
    for target in legacy_targets:
        merged[str(target)] = target
    return sorted(merged.values(), key=lambda p: str(p))


def _infer_kb_policy(kb_path: Path) -> str:
    parts = {part.lower() for part in kb_path.parts}
    if "orca_runs" in parts:
        return "orca_runs"
    if "orca_outputs" in parts:
        return "orca_outputs"
    return "generic"


def _discover_orca_outputs_targets(
    *,
    kb_path: Path,
    max_bytes: int,
    logger: Any | None,
    recent_completed_window_minutes: int | None,
) -> tuple[list[Path], bool]:
    found = False
    targets: dict[str, Path] = {}
    now_utc = datetime.now(timezone.utc)
    for metadata_name in ("run_report.json", "run_state.json"):
        for metadata_path in kb_path.rglob(metadata_name):
            found = True
            data = _load_report_json(metadata_path, logger)
            if not isinstance(data, dict):
                continue
            resolved = _resolve_from_last_out_path(
                metadata_path=metadata_path,
                data=data,
            )
            if resolved is None:
                continue
            if not _is_recent_completed_output(
                data=data,
                output_path=resolved,
                now_utc=now_utc,
                recent_completed_window_minutes=recent_completed_window_minutes,
            ):
                continue
            _add_if_valid_target(resolved=resolved, max_bytes=max_bytes, targets=targets)
    return (sorted(targets.values(), key=lambda p: str(p)), found)


def _discover_orca_runs_targets(
    *,
    kb_path: Path,
    max_bytes: int,
    logger: Any | None,
) -> tuple[list[Path], bool]:
    found = False
    targets: dict[str, Path] = {}

    for state_path in kb_path.rglob("run_state.json"):
        found = True
        data = _load_report_json(state_path, logger)
        if not isinstance(data, dict):
            continue

        status = str(data.get("status", "")).strip().lower()
        reaction_dir = _resolve_reaction_dir(state_path=state_path, data=data)
        resolved: Path | None = None

        if status == "completed":
            resolved = _resolve_from_last_out_path(
                metadata_path=state_path,
                data=data,
            )
        elif status in {"running", "failed"}:
            resolved = _find_latest_out_in_dir(reaction_dir)
        else:
            resolved = _resolve_from_last_out_path(
                metadata_path=state_path,
                data=data,
            )
            if resolved is None:
                resolved = _find_latest_out_in_dir(reaction_dir)

        if resolved is None:
            continue
        _add_if_valid_target(resolved=resolved, max_bytes=max_bytes, targets=targets)

    return (sorted(targets.values(), key=lambda p: str(p)), found)


def _discover_from_run_metadata(
    *,
    kb_path: Path,
    max_bytes: int,
    logger: Any | None,
) -> tuple[list[Path], bool]:
    report_found = False
    state_found = False
    targets: dict[str, Path] = {}

    for report_path in kb_path.rglob("run_report.json"):
        report_found = True
        data = _load_report_json(report_path, logger)
        if not isinstance(data, dict):
            continue

        resolved = _resolve_from_last_out_path(
            metadata_path=report_path,
            data=data,
        )
        if resolved is None:
            continue
        _add_if_valid_target(resolved=resolved, max_bytes=max_bytes, targets=targets)

    for state_path in kb_path.rglob("run_state.json"):
        state_found = True
        data = _load_report_json(state_path, logger)
        if not isinstance(data, dict):
            continue

        resolved = _resolve_from_last_out_path(
            metadata_path=state_path,
            data=data,
        )
        if resolved is None:
            resolved = _infer_from_run_state(state_path=state_path, data=data)
        if resolved is None:
            continue
        _add_if_valid_target(resolved=resolved, max_bytes=max_bytes, targets=targets)

    found = report_found or state_found
    return (sorted(targets.values(), key=lambda p: str(p)), found)


def _resolve_reaction_dir(*, state_path: Path, data: dict[str, Any]) -> Path:
    reaction_dir_raw = data.get("reaction_dir")
    if isinstance(reaction_dir_raw, str) and reaction_dir_raw.strip():
        reaction_dir = Path(reaction_dir_raw.strip())
        if reaction_dir.is_absolute():
            return reaction_dir
        return (state_path.parent / reaction_dir).resolve()
    return state_path.parent


def _find_latest_out_in_dir(directory: Path) -> Path | None:
    if not directory.is_dir():
        return None
    latest: tuple[float, Path] | None = None
    for candidate in directory.glob("*.out"):
        if not candidate.is_file():
            continue
        try:
            mtime = candidate.stat().st_mtime
        except OSError:
            continue
        if latest is None or mtime > latest[0]:
            latest = (mtime, candidate)
    return latest[1] if latest is not None else None


def _is_recent_completed_output(
    *,
    data: dict[str, Any],
    output_path: Path,
    now_utc: datetime,
    recent_completed_window_minutes: int | None,
) -> bool:
    if recent_completed_window_minutes is None:
        return True
    if recent_completed_window_minutes < 0:
        return True

    final_result = data.get("final_result")
    if not isinstance(final_result, dict):
        return False

    top_status = str(data.get("status", "")).strip().lower()
    final_status = str(final_result.get("status", "")).strip().lower()
    if "completed" not in {top_status, final_status}:
        return False

    window = timedelta(minutes=recent_completed_window_minutes)
    allowed_future_skew = timedelta(minutes=5)
    completed_at_raw = final_result.get("completed_at")
    completed_at = _parse_iso_datetime_utc(completed_at_raw)
    if completed_at is not None:
        age = now_utc - completed_at
        return (-allowed_future_skew) <= age <= window

    try:
        mtime_dt = datetime.fromtimestamp(output_path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return False
    age = now_utc - mtime_dt
    return (-allowed_future_skew) <= age <= window


def _parse_iso_datetime_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _discover_legacy_targets(*, kb_path: Path, max_bytes: int) -> list[Path]:
    targets: list[Path] = []
    for ext in _ORCA_EXTENSIONS:
        for fpath in kb_path.rglob(f"*{ext}"):
            if not fpath.is_file():
                continue
            try:
                if fpath.stat().st_size > max_bytes:
                    continue
            except OSError:
                continue
            targets.append(fpath)
    return sorted(targets, key=lambda p: str(p))


def _resolve_from_last_out_path(*, metadata_path: Path, data: dict[str, Any]) -> Path | None:
    final_result = data.get("final_result")
    if not isinstance(final_result, dict):
        return None
    last_out = final_result.get("last_out_path")
    if not isinstance(last_out, str) or not last_out.strip():
        return None
    return _resolve_last_out_path(
        report_path=metadata_path,
        last_out_path=last_out.strip(),
    )


def _infer_from_run_state(*, state_path: Path, data: dict[str, Any]) -> Path | None:
    selected_inp = data.get("selected_inp")
    if not isinstance(selected_inp, str) or not selected_inp.strip():
        return None

    inp_path = Path(selected_inp.strip())
    stem = inp_path.stem
    if not stem:
        return None

    reaction_dir_raw = data.get("reaction_dir")
    if isinstance(reaction_dir_raw, str) and reaction_dir_raw.strip():
        reaction_dir = Path(reaction_dir_raw.strip())
        if not reaction_dir.is_absolute():
            reaction_dir = (state_path.parent / reaction_dir).resolve()
    else:
        reaction_dir = state_path.parent

    for ext in (".out", ".log"):
        candidate = reaction_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def _add_if_valid_target(
    *,
    resolved: Path,
    max_bytes: int,
    targets: dict[str, Path],
) -> None:
    if resolved.suffix.lower() not in _ORCA_EXTENSIONS:
        return
    try:
        if resolved.stat().st_size > max_bytes:
            return
    except OSError:
        return
    targets[str(resolved)] = resolved


def _resolve_last_out_path(*, report_path: Path, last_out_path: str) -> Path | None:
    raw = Path(last_out_path)
    candidates: list[Path] = []

    # 일반 케이스: report에 기록된 절대 경로가 현재 런타임에서도 유효함
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(report_path.parent / raw)

    # 경로 루트가 다른 환경(/home -> /app 등) 대응: 같은 디렉토리의 파일명으로 재해석
    if raw.name:
        candidates.append(report_path.parent / raw.name)

    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def _load_report_json(report_path: Path, logger: Any | None) -> dict[str, Any] | None:
    try:
        with open(report_path, encoding="utf-8") as f:
            loaded = json.load(f)
        return loaded if isinstance(loaded, dict) else None
    except Exception as exc:
        _log_warning(
            logger,
            "dft_run_report_parse_failed",
            path=str(report_path),
            error=str(exc),
        )
        return None


def _log_warning(logger: Any | None, event: str, **kwargs: Any) -> None:
    if logger is None:
        return
    try:
        logger.warning(event, **kwargs)
    except Exception:
        # 로거 인터페이스 차이(예: mock) 대비
        try:
            logger.warning(f"{event}: {kwargs}")
        except Exception:
            pass
