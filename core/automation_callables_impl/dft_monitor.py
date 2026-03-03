"""DFT 계산 파일 변경 감지 및 자동 인덱싱.

kb_dirs를 주기적으로 스캔하여 새로 완료된 ORCA 계산을 감지하고,
DFT 인덱스에 등록 후 텔레그램으로 알림을 전송한다.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from core.dft_discovery import discover_orca_targets
from core.orca_parser import parse_orca_output


def build_dft_monitor_callable(
    dft_index: Any,
    kb_dirs: list[str],
    logger: Any,
    state_file: str | None = None,
):
    """DFT 모니터 callable을 빌드한다.

    Args:
        dft_index: DFTIndex 인스턴스
        kb_dirs: 모니터링할 디렉토리 목록
        logger: 로거

    Returns:
        dft_monitor async callable
    """
    # 파일별 마지막 처리 mtime 캐시 (옵션: 디스크 영속화)
    _last_mtimes: dict[str, float] = _load_state(state_file, logger) if state_file else {}
    _baseline_seeded = bool(_last_mtimes)

    async def dft_monitor(
        max_file_size_mb: int = 64,
        model: str | None = None,
        model_role: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """kb_dirs에서 새로/변경된 ORCA 파일을 감지하여 인덱싱한다.

        Returns:
            새 계산이 있으면 알림 텍스트, 없으면 빈 문자열
        """
        nonlocal _baseline_seeded
        max_bytes = max_file_size_mb * 1024 * 1024
        new_results: list[dict[str, str]] = []
        scanned_mtimes: dict[str, float] = {}
        state_dirty = False

        for kb_dir in kb_dirs:
            kb_path = Path(kb_dir)
            if not kb_path.is_dir():
                continue

            for fpath in discover_orca_targets(kb_path, max_bytes=max_bytes, logger=logger):
                spath = str(fpath)
                current_mtime = os.path.getmtime(spath)
                scanned_mtimes[spath] = current_mtime

                # 첫 실행(기존 상태파일 없음)에서는 baseline만 채우고 알림하지 않는다.
                if not _baseline_seeded:
                    continue

                last_mtime = _last_mtimes.get(spath)
                if last_mtime is not None and current_mtime <= last_mtime:
                    continue

                # 새 파일 또는 변경된 파일 발견
                try:
                    result = parse_orca_output(spath)
                    success = await dft_index.upsert_single(spath)
                    if success:
                        _last_mtimes[spath] = current_mtime
                        state_dirty = True

                        # 알림 정보 수집
                        energy_str = (
                            f"E = {result.energy_hartree:.6f} Eh"
                            if result.energy_hartree is not None
                            else "E = N/A"
                        )
                        method_basis = result.method
                        if result.basis_set:
                            method_basis += f"/{result.basis_set}"

                        status_emoji = {
                            "completed": "OK",
                            "failed": "FAIL",
                            "running": "RUNNING",
                        }.get(result.status, result.status)

                        notes: list[str] = []
                        if result.opt_converged is False:
                            notes.append("NOT CONVERGED")
                        if result.has_imaginary_freq:
                            notes.append("imaginary freq")

                        note_str = f" ({', '.join(notes)})" if notes else ""

                        new_results.append({
                            "formula": result.formula or "unknown",
                            "method_basis": method_basis or "unknown",
                            "energy": energy_str,
                            "status": status_emoji,
                            "calc_type": result.calc_type,
                            "path": _short_path(spath),
                            "note": note_str,
                        })

                        logger.info(
                            "dft_monitor_new_calc",
                            path=spath,
                            formula=result.formula,
                            method=result.method,
                            status=result.status,
                        )
                except Exception as exc:
                    logger.warning(
                        "dft_monitor_parse_error",
                        path=spath,
                        error=str(exc),
                    )

        # 첫 실행 baseline 저장
        if not _baseline_seeded:
            _last_mtimes.clear()
            _last_mtimes.update(scanned_mtimes)
            _baseline_seeded = True
            if state_file:
                _save_state(state_file, _last_mtimes, logger)
            logger.info("dft_monitor_baseline_seeded", file_count=len(_last_mtimes))
            return ""

        # 삭제된 파일은 캐시에서도 제거
        stale_paths = set(_last_mtimes) - set(scanned_mtimes)
        if stale_paths:
            for stale in stale_paths:
                _last_mtimes.pop(stale, None)
            state_dirty = True

        if state_dirty and state_file:
            _save_state(state_file, _last_mtimes, logger)

        if not new_results:
            return ""

        # 알림 메시지 포맷
        lines = [f"DFT Monitor: {len(new_results)}건의 새 계산 감지"]
        for r in new_results:
            lines.append("")  # 계산 사이 빈 줄
            lines.append(
                f"  {r['formula']} | {r['calc_type']} | "
                f"{r['method_basis']} | {r['energy']} | "
                f"{r['status']}{r['note']}"
            )
            lines.append(f"    -> {r['path']}")

        return "\n".join(lines)

    return dft_monitor


def _short_path(path: str) -> str:
    """긴 경로를 마지막 3개 세그먼트로 축약한다."""
    parts = path.replace("\\", "/").split("/")
    if len(parts) <= 3:
        return path
    return "/".join(parts[-3:])


def _load_state(state_file: str | None, logger: Any) -> dict[str, float]:
    """디스크에 저장된 dft_monitor 상태를 로드한다."""
    if not state_file:
        return {}
    try:
        with open(state_file, encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return {}
        state: dict[str, float] = {}
        for k, v in raw.items():
            if isinstance(k, str):
                try:
                    state[k] = float(v)
                except (TypeError, ValueError):
                    continue
        return state
    except FileNotFoundError:
        return {}
    except Exception as exc:
        logger.warning("dft_monitor_state_load_failed", path=state_file, error=str(exc))
        return {}


def _save_state(state_file: str | None, mtimes: dict[str, float], logger: Any) -> None:
    """dft_monitor 상태를 원자적으로 저장한다."""
    if not state_file:
        return
    try:
        path = Path(state_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(mtimes, f, ensure_ascii=False)
        tmp_path.replace(path)
    except Exception as exc:
        logger.warning("dft_monitor_state_save_failed", path=state_file, error=str(exc))
