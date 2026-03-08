"""DFT 계산 파일 변경 감지 및 자동 인덱싱.

kb_dirs를 주기적으로 스캔하여 새로 완료된 ORCA 계산을 감지하고,
DFT 인덱스에 등록 후 텔레그램으로 알림을 전송한다.
"""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from contextlib import suppress
from html import escape as _h
from pathlib import Path
from typing import Any

from core.crest_discovery import discover_crest_targets
from core.crest_parser import CrestResult, parse_crest_output
from core.dft_discovery import DiscoveredTarget, discover_orca_targets
from core.orca_parser import parse_orca_output

GetExternalDirs = Callable[[], Awaitable[list[str]]]


def build_dft_monitor_callable(
    dft_index: Any,
    kb_dirs: list[str],
    logger: Any,
    state_file: str | None = None,
    get_external_dirs: GetExternalDirs | None = None,
):
    """DFT 모니터 callable을 빌드한다.

    Args:
        dft_index: DFTIndex 인스턴스
        kb_dirs: 모니터링할 디렉토리 목록
        logger: 로거
        state_file: 상태 파일 경로
        get_external_dirs: 외부 시뮬레이션 작업 디렉토리 목록을 반환하는 콜백

    Returns:
        dft_monitor async callable
    """
    # 파일별 마지막 처리 mtime 캐시 (옵션: 디스크 영속화)
    _last_mtimes: dict[str, float] = _load_state(state_file, logger) if state_file else {}
    _baseline_seeded = bool(_last_mtimes)

    async def dft_monitor(
        max_file_size_mb: int = 64,
        recent_completed_window_minutes: int = 60,
        timeout: int | None = None,
    ) -> str:
        """kb_dirs에서 새로/변경된 ORCA 파일을 감지하여 인덱싱한다.

        Returns:
            새 계산이 있으면 알림 텍스트, 없으면 빈 문자열
        """
        nonlocal _baseline_seeded
        max_bytes = max_file_size_mb * 1024 * 1024
        new_results: list[dict[str, str]] = []
        scanned_mtimes: dict[str, float] = {}
        processed_this_scan: set[str] = set()
        orca_scanned_paths: set[str] = set()
        state_dirty = False
        missing_kb_dirs: list[str] = []
        # 외부 시뮬레이션 디렉토리 수집
        external_dirs: list[str] = []
        if get_external_dirs is not None:
            try:
                external_dirs = await get_external_dirs()
            except Exception as exc:
                logger.warning("dft_monitor_external_dirs_failed", error=str(exc))

        # kb_dirs + 외부 디렉토리 합쳐서 스캔 (중복 제거)
        all_dirs = list(kb_dirs)
        seen_resolved: set[str] = set()
        for d in all_dirs:
            with suppress(OSError):
                seen_resolved.add(str(Path(d).resolve()))
        for d in external_dirs:
            try:
                resolved = str(Path(d).resolve())
            except OSError:
                continue
            if resolved not in seen_resolved:
                seen_resolved.add(resolved)
                all_dirs.append(d)

        if not all_dirs:
            logger.warning("dft_monitor_no_kb_dirs_configured")
            return ""

        for kb_dir in all_dirs:
            kb_path = Path(kb_dir)
            if not kb_path.is_dir():
                missing_kb_dirs.append(kb_dir)
                continue

            for target in discover_orca_targets(
                kb_path,
                max_bytes=max_bytes,
                logger=logger,
                recent_completed_window_minutes=recent_completed_window_minutes,
            ):
                spath = str(target.path)
                canonical_spath = _canonical_path_key(spath)
                current_mtime = os.path.getmtime(spath)
                scanned_mtimes[canonical_spath] = current_mtime
                orca_scanned_paths.add(spath)

                # 첫 실행(기존 상태파일 없음)에서는 baseline만 채우고 알림하지 않는다.
                if not _baseline_seeded:
                    continue

                last_mtime = _last_mtimes.get(canonical_spath)
                if last_mtime is not None and current_mtime <= last_mtime:
                    continue

                # 같은 스캔에서 이미 처리한 파일 (겹치는 디렉토리 대비)
                if canonical_spath in processed_this_scan:
                    continue
                processed_this_scan.add(canonical_spath)

                # 새 파일 또는 변경된 파일 발견
                try:
                    result = parse_orca_output(spath)

                    # running 판별: run_state.json status 또는 파서 결과
                    is_running = (
                        result.status == "running"
                        or target.run_state_status == "running"
                    )

                    if is_running:
                        # running 계산은 인덱스에 upsert하지 않음
                        _last_mtimes[canonical_spath] = current_mtime
                        state_dirty = True
                    else:
                        success = await dft_index.upsert_single(spath)
                        if not success:
                            continue
                        _last_mtimes[canonical_spath] = current_mtime
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

                    effective_status = "running" if is_running else result.status
                    status_emoji = {
                        "completed": "✅",
                        "failed": "🔴",
                        "running": "🔄",
                    }.get(effective_status, effective_status)

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
                        "path": _short_path(canonical_spath),
                        "note": note_str,
                    })

                    logger.info(
                        "dft_monitor_new_calc",
                        path=spath,
                        formula=result.formula,
                        method=result.method,
                        status=effective_status,
                    )
                except Exception as exc:
                    logger.warning(
                        "dft_monitor_parse_error",
                        path=spath,
                        error=str(exc),
                    )

            # CREST 스캔 (ORCA와 중복 제외)
            for fpath in discover_crest_targets(
                kb_path,
                max_bytes=max_bytes,
                logger=logger,
                exclude_paths=orca_scanned_paths,
            ):
                spath = str(fpath)
                canonical_spath = _canonical_path_key(spath)
                try:
                    current_mtime = os.path.getmtime(spath)
                except OSError:
                    continue
                scanned_mtimes[canonical_spath] = current_mtime

                if not _baseline_seeded:
                    continue

                last_mtime = _last_mtimes.get(canonical_spath)
                if last_mtime is not None and current_mtime <= last_mtime:
                    continue

                if canonical_spath in processed_this_scan:
                    continue
                processed_this_scan.add(canonical_spath)

                try:
                    cresult = parse_crest_output(spath)
                    cresult.source_path = canonical_spath
                    _last_mtimes[canonical_spath] = current_mtime
                    state_dirty = True

                    if cresult.status == "running":
                        text = _format_crest_running(cresult)
                    elif cresult.status == "failed":
                        text = _format_crest_failed(cresult)
                    else:
                        text = _format_crest_completed(cresult)

                    if text:
                        new_results.append({"_progress_text": text})

                    logger.info(
                        "dft_monitor_crest_detected",
                        path=spath,
                        status=cresult.status,
                        n_conformers=cresult.n_conformers,
                    )
                except Exception as exc:
                    logger.warning(
                        "dft_monitor_crest_parse_error",
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
            logger.info(
                "dft_monitor_scan_complete",
                scanned_files=len(scanned_mtimes),
                new_results=0,
                missing_kb_dirs=len(missing_kb_dirs),
                baseline_seeded=True,
            )
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
            logger.info(
                "dft_monitor_scan_complete",
                scanned_files=len(scanned_mtimes),
                new_results=0,
                stale_removed=len(stale_paths),
                missing_kb_dirs=len(missing_kb_dirs),
                baseline_seeded=True,
            )
            return ""

        lines: list[str] = []

        if new_results:
            lines.append(
                f"🧪 <b>DFT Monitor: {len(new_results)}건의 계산 업데이트</b>"
            )
            table_rows: list[str] = []
            table_rows.append("분자       계산  메서드/기저        에너지              상태")
            table_rows.append("─" * 58)
            for r in new_results:
                note_line = ""
                if r["note"]:
                    note_line = f"\n  ⚠️ {_h(r['note'].strip(' ()'))}"
                table_rows.append(
                    f"{_h(r['formula']):<10s} {_h(r['calc_type']):<4s}  "
                    f"{_h(r['method_basis']):<17s} {_h(r['energy']):<19s} "
                    f"{r['status']}"
                    f"{note_line}"
                    f"\n  📂 {_h(r['path'])}"
                )
            lines.append(f"<pre>{chr(10).join(table_rows)}</pre>")

        logger.info(
            "dft_monitor_scan_complete",
            scanned_files=len(scanned_mtimes),
            new_results=len(new_results),
            stale_removed=len(stale_paths),
            missing_kb_dirs=len(missing_kb_dirs),
            baseline_seeded=True,
        )

        return "\n".join(lines)

    return dft_monitor


def _format_crest_running(result: CrestResult) -> str:
    """진행 중인 CREST 계산 알림 텍스트 (HTML)."""
    method = _h(result.method)
    n_atoms_str = f" | 원자 수: {result.n_atoms}" if result.n_atoms else ""
    conformers_str = (
        f"{result.n_conformers}개 구조 발견"
        if result.n_conformers
        else "구조 탐색 중"
    )
    energy_str = (
        f"\n현재 최저 에너지: {result.best_energy_hartree:.6f} Eh"
        if result.best_energy_hartree is not None
        else ""
    )
    return (
        f"🔄 <b>CREST 실행 중: {conformers_str}</b>\n"
        f"메서드: {method}{n_atoms_str}"
        f"{energy_str}\n"
        f"📂 {_h(_short_path(result.source_path))}"
    )


def _format_crest_completed(result: CrestResult) -> str:
    """완료된 CREST 계산 알림 텍스트 (HTML)."""
    method = _h(result.method)
    n_atoms_str = f" | 원자 수: {result.n_atoms}" if result.n_atoms else ""
    conformers_str = (
        f"{result.n_conformers}개 배좌이성질체"
        if result.n_conformers
        else "?"
    )
    energy_str = (
        f"\n최저 에너지: {result.best_energy_hartree:.6f} Eh"
        if result.best_energy_hartree is not None
        else ""
    )
    return (
        f"✅ <b>CREST 완료: {conformers_str} 발견</b>\n"
        f"메서드: {method}{n_atoms_str}"
        f"{energy_str}\n"
        f"📂 {_h(_short_path(result.source_path))}"
    )


def _format_crest_failed(result: CrestResult) -> str:
    """실패한 CREST 계산 알림 텍스트 (HTML)."""
    error = _h(result.error_message or "알 수 없는 오류")
    return (
        f"🔴 <b>CREST 실패</b>\n"
        f"오류: {error}\n"
        f"📂 {_h(_short_path(result.source_path))}"
    )


def _short_path(path: str) -> str:
    """긴 경로를 마지막 3개 세그먼트로 축약한다."""
    parts = path.replace("\\", "/").split("/")
    if len(parts) <= 3:
        return path
    return "/".join(parts[-3:])


def _canonical_path_key(path: str | Path) -> str:
    """같은 파일의 경로 alias를 하나의 canonical key로 정규화한다."""
    try:
        return str(Path(path).expanduser().resolve(strict=False))
    except (OSError, RuntimeError, TypeError, ValueError):
        return str(Path(path).expanduser().absolute())


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
                    normalized_key = _canonical_path_key(k)
                    value = float(v)
                except (TypeError, ValueError):
                    continue
                previous = state.get(normalized_key)
                if previous is None or value > previous:
                    state[normalized_key] = value
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


def _format_error(exc: BaseException) -> str:
    """예외 메시지가 비어있을 때 클래스명을 보존한다."""
    message = str(exc).strip()
    return message if message else exc.__class__.__name__
