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
from core.dft_discovery import discover_orca_targets
from core.orca_parser import OptProgress, parse_opt_progress, parse_orca_output

_RUNNING_PROGRESS_CALC_TYPES = ("opt", "ts", "neb", "irc")



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

            for fpath in discover_orca_targets(
                kb_path,
                max_bytes=max_bytes,
                logger=logger,
                recent_completed_window_minutes=recent_completed_window_minutes,
            ):
                spath = str(fpath)
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

                    # 진행 중인 계산(OPT/TS/NEB/IRC)은 progress + AI 코멘트 알림 생성
                    if result.status == "running" and _is_progress_comment_target(result.calc_type):
                        progress_text = ""
                        try:
                            opt_prog = parse_opt_progress(spath)
                            if opt_prog.steps:
                                opt_prog.source_path = canonical_spath
                                progress_text = _format_opt_progress(opt_prog)
                                if progress_text:
                                    comment = _rule_based_comment(opt_prog)
                                    if comment:
                                        progress_text += f'\n\n💬 "<i>{_h(comment)}</i>"'
                        except Exception as exc:
                            logger.warning(
                                "dft_monitor_progress_parse_error",
                                path=spath,
                                error=str(exc),
                            )

                        # OPT step 파싱 불가(또는 비-OPT 타입) 시 요약 스냅샷
                        if not progress_text:
                            progress_text = _format_running_snapshot(result, canonical_spath)

                        if progress_text:
                            new_results.append({
                                "_progress_text": progress_text,
                            })

                        # running 계산은 인덱스에 upsert하지 않음
                        _last_mtimes[canonical_spath] = current_mtime
                        state_dirty = True
                        continue

                    success = await dft_index.upsert_single(spath)
                    if success:
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

                        status_emoji = {
                            "completed": "✅",
                            "failed": "🔴",
                            "running": "🔄",
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
                            "path": _short_path(canonical_spath),
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

        # 완료된 계산과 진행 중인 계산 분리
        completed = [r for r in new_results if "_progress_text" not in r]
        running = [r for r in new_results if "_progress_text" in r]

        lines: list[str] = []

        if completed:
            lines.append(
                f"🧪 <b>DFT Monitor: {len(completed)}건의 계산 업데이트</b>"
            )
            table_rows: list[str] = []
            table_rows.append("분자       계산  메서드/기저        에너지              상태")
            table_rows.append("─" * 58)
            for r in completed:
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

        if running:
            if lines:
                lines.append("")
                lines.append("─" * 40)
            lines.append(
                f"\n🔄 <b>RUNNING Progress: {len(running)}건의 진행 중인 계산</b>"
            )
            for r in running:
                lines.append("")
                lines.append(r["_progress_text"])

        logger.info(
            "dft_monitor_scan_complete",
            scanned_files=len(scanned_mtimes),
            new_results=len(new_results),
            completed=len(completed),
            running=len(running),
            stale_removed=len(stale_paths),
            missing_kb_dirs=len(missing_kb_dirs),
            baseline_seeded=True,
        )

        return "\n".join(lines)

    return dft_monitor


def _is_progress_comment_target(calc_type: str) -> bool:
    """진행 알림 + AI 코멘트 대상 계산 유형인지 판별한다."""
    normalized = (calc_type or "").lower()
    return any(token in normalized for token in _RUNNING_PROGRESS_CALC_TYPES)


def _format_opt_progress(progress: OptProgress) -> str:
    """최적화 진행 현황을 텔레그램 알림 텍스트로 포맷한다 (HTML)."""
    if not progress.steps:
        return ""

    method_basis = _h(progress.method)
    if progress.basis_set:
        method_basis += f"/{_h(progress.basis_set)}"

    if progress.is_running:
        status = "🔄 RUNNING"
    elif progress.is_converged:
        status = "✅ CONVERGED"
    else:
        status = "⚠️ NOT YET CONVERGED"

    header = (
        f"📈 <b>OPT Progress: {_h(progress.formula or 'unknown')} | "
        f"{_h(progress.calc_type)} | {method_basis}</b>\n"
        f"Status: {status} | Steps: {len(progress.steps)}\n"
        f"📂 {_h(_short_path(progress.source_path))}"
    )

    # 최근 5 스텝 상세 표시
    MAX_DETAIL = 5
    recent = progress.steps[-MAX_DETAIL:]
    older = progress.steps[:-MAX_DETAIL] if len(progress.steps) > MAX_DETAIL else []

    lines = [header, ""]

    # 이전 스텝 요약 (있으면)
    if older:
        e_min = min(s.energy_hartree for s in older)
        e_max = max(s.energy_hartree for s in older)
        lines.append(
            f"Steps 1-{older[-1].cycle}: "
            f"E range [{e_min:.6f}, {e_max:.6f}] Eh"
        )
        lines.append("")

    # 최근 스텝 테이블 (monospace)
    pre_lines: list[str] = []
    pre_lines.append("Step │ Energy (Eh)     │ dE         │ MaxGrad    │ Conv")
    pre_lines.append("─────┼─────────────────┼────────────┼────────────┼─────")
    for s in recent:
        de_str = f"{s.energy_change:.2e}" if s.energy_change is not None else "   -"
        mg_str = f"{s.max_gradient:.2e}" if s.max_gradient is not None else "   -"
        n_conv = sum(1 for v in s.converged_flags.values() if v)
        n_total = len(s.converged_flags)
        conv_str = f"{n_conv}/{n_total}" if n_total > 0 else " -"
        pre_lines.append(
            f"{s.cycle:4d} │ {s.energy_hartree:15.8f} │ {de_str:>10s} │ {mg_str:>10s} │ {conv_str}"
        )
    lines.append(f"<pre>{chr(10).join(pre_lines)}</pre>")

    # dE 트렌드 (마지막 2-3개 값)
    de_values = [
        s.energy_change for s in progress.steps
        if s.energy_change is not None
    ]
    if len(de_values) >= 2:
        trend = de_values[-3:] if len(de_values) >= 3 else de_values[-2:]
        trend_str = " → ".join(f"{v:.2e}" for v in trend)
        lines.append(f"\ndE trend: {trend_str}")

    return "\n".join(lines)


def _format_running_snapshot(result: Any, source_path: str) -> str:
    """OPT step 파싱이 어려운 running 계산의 간단 요약 텍스트를 만든다 (HTML)."""
    method_basis = _h(result.method or "unknown")
    if result.basis_set:
        method_basis += f"/{_h(result.basis_set)}"
    energy_str = (
        f"E = {result.energy_hartree:.6f} Eh"
        if result.energy_hartree is not None
        else "E = N/A"
    )
    return (
        f"🔄 <b>RUNNING: {_h(result.formula or 'unknown')} | "
        f"{_h(result.calc_type or 'unknown')} | {method_basis}</b>\n"
        f"Status: 🔄 RUNNING | {energy_str}\n"
        f"📂 {_h(_short_path(source_path))}"
    )


def _rule_based_comment(progress: OptProgress) -> str:
    """최적화 진행 데이터를 규칙 기반으로 분석하여 한줄 코멘트를 생성한다."""
    if not progress.steps:
        return ""

    last = progress.steps[-1]
    n_steps = len(progress.steps)
    parts: list[str] = []

    # 수렴 플래그 분석
    if last.converged_flags:
        n_conv = sum(1 for v in last.converged_flags.values() if v)
        n_total = len(last.converged_flags)
        if n_conv == n_total:
            return f"{n_steps} 스텝 만에 모든 수렴 조건 충족 — 수렴 완료"
        parts.append(f"수렴 {n_conv}/{n_total}")

    # 에너지 변화 트렌드 분석 (최근 3 스텝)
    recent_de = [
        s.energy_change for s in progress.steps[-3:]
        if s.energy_change is not None
    ]
    if len(recent_de) >= 2:
        all_negative = all(de < 0 for de in recent_de)
        all_decreasing_abs = all(
            abs(recent_de[i]) <= abs(recent_de[i - 1])
            for i in range(1, len(recent_de))
        )
        has_positive = any(de > 0 for de in recent_de)
        oscillating = len(recent_de) >= 3 and any(
            (recent_de[i] > 0) != (recent_de[i - 1] > 0)
            for i in range(1, len(recent_de))
        )

        if oscillating:
            parts.append("⚠ 에너지 진동 중")
        elif has_positive:
            parts.append("⚠ 에너지 상승 감지")
        elif all_negative and all_decreasing_abs:
            parts.append("에너지 안정적 감소")
        elif all_negative:
            parts.append("에너지 감소 중")

    # dE 크기 기반 수렴 근접도
    if last.energy_change is not None:
        abs_de = abs(last.energy_change)
        if abs_de < 5e-6:
            parts.append("dE 수렴 임계값 이내")
        elif abs_de < 5e-5:
            parts.append("dE 수렴 근접")

    # MaxGrad 크기 기반 수렴 근접도
    if last.max_gradient is not None:
        if last.max_gradient < 3e-4:
            parts.append("MaxGrad 수렴 임계값 이내")
        elif last.max_gradient < 3e-3:
            parts.append("MaxGrad 수렴 근접")

    # 스텝 수 경고
    if n_steps >= 100:
        parts.append(f"⚠ {n_steps} 스텝 경과 — 수렴 지연")
    elif n_steps >= 50:
        parts.append(f"{n_steps} 스텝 경과")

    if not parts:
        return ""

    return " | ".join(parts)


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
