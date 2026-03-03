"""DFT 인덱싱 대상 파일 탐색 유틸."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_ORCA_EXTENSIONS = {".out", ".log"}


def discover_orca_targets(
    kb_path: Path,
    *,
    max_bytes: int,
    logger: Any | None = None,
) -> list[Path]:
    """인덱싱 대상 ORCA 출력 파일 목록을 반환한다.

    규칙:
    1) kb_path 하위에 run_report.json이 하나라도 있으면
       final_result.last_out_path만 대상으로 사용한다.
    2) run_report.json이 없으면 기존 방식대로 *.out/*.log를 스캔한다.
    """
    report_targets, report_found = _discover_from_run_reports(
        kb_path=kb_path,
        max_bytes=max_bytes,
        logger=logger,
    )
    if report_found:
        return report_targets

    legacy_targets: list[Path] = []
    for ext in _ORCA_EXTENSIONS:
        for fpath in kb_path.rglob(f"*{ext}"):
            if not fpath.is_file():
                continue
            try:
                if fpath.stat().st_size > max_bytes:
                    continue
            except OSError:
                continue
            legacy_targets.append(fpath)
    return sorted(legacy_targets, key=lambda p: str(p))


def _discover_from_run_reports(
    *,
    kb_path: Path,
    max_bytes: int,
    logger: Any | None,
) -> tuple[list[Path], bool]:
    report_found = False
    targets: dict[str, Path] = {}

    for report_path in kb_path.rglob("run_report.json"):
        report_found = True
        data = _load_report_json(report_path, logger)
        if not isinstance(data, dict):
            continue

        final_result = data.get("final_result")
        if not isinstance(final_result, dict):
            continue

        last_out = final_result.get("last_out_path")
        if not isinstance(last_out, str) or not last_out.strip():
            continue

        resolved = _resolve_last_out_path(
            report_path=report_path,
            last_out_path=last_out.strip(),
        )
        if resolved is None:
            _log_warning(
                logger,
                "dft_run_report_last_out_missing",
                path=str(report_path),
                last_out_path=last_out,
            )
            continue

        if resolved.suffix.lower() not in _ORCA_EXTENSIONS:
            continue

        try:
            if resolved.stat().st_size > max_bytes:
                continue
        except OSError:
            continue

        targets[str(resolved)] = resolved

    return (sorted(targets.values(), key=lambda p: str(p)), report_found)


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
