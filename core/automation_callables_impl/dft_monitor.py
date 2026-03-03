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
from core.orca_parser import OptProgress, parse_opt_progress, parse_orca_output


def build_dft_monitor_callable(
    dft_index: Any,
    kb_dirs: list[str],
    logger: Any,
    state_file: str | None = None,
    engine: Any | None = None,
):
    """DFT 모니터 callable을 빌드한다.

    Args:
        dft_index: DFTIndex 인스턴스
        kb_dirs: 모니터링할 디렉토리 목록
        logger: 로거
        state_file: 상태 파일 경로
        engine: Engine 인스턴스 (LLM 한줄 해석용, 없으면 해석 생략)

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

                    # 진행 중인 최적화 계산은 progress 알림 생성
                    if result.status == "running" and "opt" in result.calc_type:
                        try:
                            opt_prog = parse_opt_progress(spath)
                            if opt_prog.steps:
                                progress_text = _format_opt_progress(opt_prog)
                                if progress_text:
                                    # LLM 한줄 해석 시도
                                    ai_comment = await _get_ai_comment(
                                        opt_prog, engine, model, model_role,
                                        temperature, max_tokens, logger,
                                    )
                                    if ai_comment:
                                        progress_text += f"\n\nAI: {ai_comment}"
                                    new_results.append({
                                        "_progress_text": progress_text,
                                    })
                        except Exception as exc:
                            logger.warning(
                                "dft_monitor_progress_parse_error",
                                path=spath,
                                error=str(exc),
                            )
                        # running 계산은 인덱스에 upsert하지 않음
                        _last_mtimes[spath] = current_mtime
                        state_dirty = True
                        continue

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

        # 완료된 계산과 진행 중인 계산 분리
        completed = [r for r in new_results if "_progress_text" not in r]
        running = [r for r in new_results if "_progress_text" in r]

        lines: list[str] = []

        if completed:
            lines.append(f"DFT Monitor: {len(completed)}건의 새 계산 감지")
            for r in completed:
                lines.append("")
                lines.append(
                    f"  {r['formula']} | {r['calc_type']} | "
                    f"{r['method_basis']} | {r['energy']} | "
                    f"{r['status']}{r['note']}"
                )
                lines.append(f"    -> {r['path']}")

        if running:
            if lines:
                lines.append("")
                lines.append("---")
            lines.append(f"\nOPT Progress: {len(running)}건의 진행 중인 최적화")
            for r in running:
                lines.append("")
                lines.append(r["_progress_text"])

        return "\n".join(lines)

    return dft_monitor


def _format_opt_progress(progress: OptProgress) -> str:
    """최적화 진행 현황을 텔레그램 알림 텍스트로 포맷한다."""
    if not progress.steps:
        return ""

    method_basis = progress.method
    if progress.basis_set:
        method_basis += f"/{progress.basis_set}"

    status = "RUNNING" if progress.is_running else (
        "CONVERGED" if progress.is_converged else "NOT YET CONVERGED"
    )
    header = (
        f"OPT Progress: {progress.formula or 'unknown'} | "
        f"{progress.calc_type} | {method_basis}\n"
        f"Status: {status} | Steps: {len(progress.steps)}\n"
        f"  -> {_short_path(progress.source_path)}"
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

    # 최근 스텝 테이블
    lines.append("Step | Energy (Eh)     | dE         | MaxGrad    | Conv")
    lines.append("-----|-----------------|------------|------------|-----")
    for s in recent:
        de_str = f"{s.energy_change:.2e}" if s.energy_change is not None else "   -"
        mg_str = f"{s.max_gradient:.2e}" if s.max_gradient is not None else "   -"
        n_conv = sum(1 for v in s.converged_flags.values() if v)
        n_total = len(s.converged_flags)
        conv_str = f"{n_conv}/{n_total}" if n_total > 0 else " -"
        lines.append(
            f"{s.cycle:4d} | {s.energy_hartree:15.8f} | {de_str:>10s} | {mg_str:>10s} | {conv_str}"
        )

    # dE 트렌드 (마지막 2-3개 값)
    de_values = [
        s.energy_change for s in progress.steps
        if s.energy_change is not None
    ]
    if len(de_values) >= 2:
        trend = de_values[-3:] if len(de_values) >= 3 else de_values[-2:]
        trend_str = " -> ".join(f"{v:.2e}" for v in trend)
        lines.append(f"\ndE trend: {trend_str}")

    return "\n".join(lines)


async def _get_ai_comment(
    progress: OptProgress,
    engine: Any | None,
    model: str | None,
    model_role: str | None,
    temperature: float | None,
    max_tokens: int | None,
    logger: Any,
) -> str:
    """LLM에 최적화 진행 상황 한줄 해석을 요청한다.

    engine이 None이거나 호출 실패 시 빈 문자열을 반환한다 (graceful degradation).
    """
    if engine is None or not progress.steps:
        return ""

    prompt = _build_analysis_prompt(progress)
    try:
        raw = await engine.process_prompt(
            prompt=prompt,
            model_override=model,
            model_role=model_role,
            temperature=temperature if temperature is not None else 0.3,
            max_tokens=max_tokens if max_tokens is not None else 150,
        )
        # 첫 줄만 사용 (한 문장 코멘트)
        comment = raw.strip().split("\n")[0].strip()
        return comment
    except Exception as exc:
        logger.warning("dft_monitor_ai_comment_failed", error=str(exc))
        return ""


def _build_analysis_prompt(progress: OptProgress) -> str:
    """최적화 진행 데이터를 LLM 프롬프트로 변환한다."""
    data_lines: list[str] = []
    for s in progress.steps:
        de_str = f"{s.energy_change:.2e}" if s.energy_change is not None else "N/A"
        mg_str = f"{s.max_gradient:.2e}" if s.max_gradient is not None else "N/A"
        n_conv = sum(1 for v in s.converged_flags.values() if v)
        n_total = len(s.converged_flags)
        conv_str = f"{n_conv}/{n_total}" if n_total > 0 else "N/A"
        data_lines.append(
            f"Step {s.cycle}: E={s.energy_hartree:.8f} Eh, "
            f"dE={de_str}, MaxGrad={mg_str}, Conv={conv_str}"
        )

    return (
        "아래 ORCA 구조 최적화 진행 데이터를 분석하여 한국어로 한 문장 코멘트를 작성하세요.\n"
        "수렴 패턴(monotonic/oscillating/plateau/diverging), "
        "예상 남은 스텝, 주의사항을 포함하세요.\n"
        "한 문장만 출력하세요.\n\n"
        f"분자: {progress.formula or 'unknown'}\n"
        f"메서드: {progress.method}/{progress.basis_set}\n"
        f"총 {len(progress.steps)} 스텝\n\n"
        + "\n".join(data_lines)
    )


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
