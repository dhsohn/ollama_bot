"""DFT 계산 파일 변경 감지 및 자동 인덱싱.

kb_dirs를 주기적으로 스캔하여 새로 완료된 ORCA 계산을 감지하고,
DFT 인덱스에 등록 후 텔레그램으로 알림을 전송한다.
"""

from __future__ import annotations

import json
import os
import re
from html import escape as _h
from pathlib import Path
from typing import Any

from core.dft_discovery import discover_orca_targets
from core.orca_parser import OptProgress, parse_opt_progress, parse_orca_output

_RUNNING_PROGRESS_CALC_TYPES = ("opt", "ts", "neb", "irc")

_DFT_SYSTEM_PROMPT = (
    "ORCA DFT 계산 모니터링 전문가. "
    "데이터를 보고 한국어 300자 이내로 코멘트하라. "
    "설명·사고과정 없이 코멘트만."
)

# 사고 과정 유출을 감지하는 패턴들 — 줄 시작 기준
_THINKING_PREFIXES = re.compile(
    r"^("
    r"(분석|해석|판단|검토|확인|관찰|결론|요약|정리|평가)\s*([:：]|결과|해보|하면)"
    r"|let me|okay|alright|well|so,?\s"
    r"|the user|they want|they say|they ask"
    r"|we need|we have|we must|we should|we can|we don'?t"
    r"|i need|i should|i will|i'?ll|i'?m going"
    r"|actually|probably|basically|essentially|note:"
    r"|this (seems?|means?|is|looks?|requires?)"
    r"|but we|but the|but i|since |because "
    r"|here'?s|there'?s|now |means "
    r"|먼저|우선|일단|그런데|따라서|그러므로|결론적으로"
    r"|首先|让我|好的|那么|因此|所以|然后|接下来|需要|根据"
    r"|分析一下|总结|综上|用户|我们|看起来|这[是意表说]"
    r"|第[一二三四五六七八九十]\s*[步,、:：]"
    r"|step\s*\d+\s*[:\-]"
    r"|1[\.\)]\s"
    r")",
    re.IGNORECASE,
)

# 줄 내부 어디서든 사고 과정 유출을 감지하는 패턴 (영어·중국어 메타 추론)
_META_REASONING_RE = re.compile(
    r"("
    # 영어
    r"the user (says?|wants?|asks?|provides?|mentions?)"
    r"|they want|they say|they ask"
    r"|we need to|we have to|we must|we should|we don'?t know"
    r"|comment in korean|comment about|produce a comment"
    r"|data includes|given data|limited data"
    r"|include .{0,30}(summary|check\s*point|포인트)"
    r"|not? mention"
    r"|from (?:the )?data"
    # 중국어
    r"|用户(说|要求|想要|提到|提供|询问)"
    r"|我们需要|需要生成|需要写|需要包含|根据数据|根据给定"
    r"|这[意表说]味着|看起来|似乎|大概|可能需要"
    r"|韩[语文]评论|写一[个句]|生成评论|输出评论"
    r"|没有提到|数据中没有|从数据[中来看]"
    r")",
    re.IGNORECASE,
)



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
        recent_completed_window_minutes: int = 60,
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
        missing_kb_dirs: list[str] = []

        if not kb_dirs:
            logger.warning("dft_monitor_no_kb_dirs_configured")
            return ""

        for kb_dir in kb_dirs:
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

                    # 진행 중인 계산(OPT/TS/NEB/IRC)은 progress + AI 코멘트 알림 생성
                    if result.status == "running" and _is_progress_comment_target(result.calc_type):
                        progress_text = ""
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
                                        progress_text += f'\n\n💬 "<i>{_h(ai_comment)}</i>"'
                        except Exception as exc:
                            logger.warning(
                                "dft_monitor_progress_parse_error",
                                path=spath,
                                error=str(exc),
                            )

                        # OPT step 파싱 불가(또는 비-OPT 타입) 시 요약 스냅샷 + AI 코멘트
                        if not progress_text:
                            progress_text = _format_running_snapshot(result, spath)
                            ai_comment = await _get_ai_comment_for_running(
                                result, spath, engine, model, model_role,
                                temperature, max_tokens, logger,
                            )
                            if ai_comment:
                                progress_text += f'\n\n💬 "<i>{_h(ai_comment)}</i>"'

                        if progress_text:
                            new_results.append({
                                "_progress_text": progress_text,
                            })

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
                f"🧪 <b>DFT Monitor: {len(completed)}건의 새 계산 감지</b>"
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


def _is_thinking_line(line: str) -> bool:
    """줄이 사고 과정/메타 추론인지 판정한다."""
    if _THINKING_PREFIXES.match(line):
        return True
    if _META_REASONING_RE.search(line):
        return True
    return False


def _extract_comment(raw: str) -> str:
    """LLM 응답에서 사고 과정을 제거하고 코멘트를 추출한다."""
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    if not lines:
        return ""

    kept = [line for line in lines if not _is_thinking_line(line)]
    return "\n".join(kept)


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
            system_prompt_override=_DFT_SYSTEM_PROMPT,
        )
        return _extract_comment(raw)
    except Exception as exc:
        logger.warning("dft_monitor_ai_comment_failed", error=str(exc))
        return ""


async def _get_ai_comment_for_running(
    result: Any,
    source_path: str,
    engine: Any | None,
    model: str | None,
    model_role: str | None,
    temperature: float | None,
    max_tokens: int | None,
    logger: Any,
) -> str:
    """OPT step이 부족한 running 계산에 대한 한줄 해석을 요청한다."""
    if engine is None:
        return ""

    prompt = _build_running_analysis_prompt(result, source_path)
    try:
        raw = await engine.process_prompt(
            prompt=prompt,
            model_override=model,
            model_role=model_role,
            temperature=temperature if temperature is not None else 0.3,
            max_tokens=max_tokens if max_tokens is not None else 150,
            system_prompt_override=_DFT_SYSTEM_PROMPT,
        )
        return _extract_comment(raw)
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

    # 소형 모델 토큰 절약: 최근 5 스텝만 전달
    recent_data = data_lines[-5:] if len(data_lines) > 5 else data_lines

    return (
        f"분자: {progress.formula or 'unknown'}, "
        f"메서드: {progress.method}/{progress.basis_set}, "
        f"총 {len(progress.steps)} 스텝\n"
        + "\n".join(recent_data)
        + "\n\n수렴 상태와 주의사항을 한국어 300자 이내로 코멘트하라."
    )


def _build_running_analysis_prompt(result: Any, source_path: str) -> str:
    """OPT step이 없는 running 계산용 LLM 프롬프트."""
    method_basis = result.method or "unknown"
    if result.basis_set:
        method_basis += f"/{result.basis_set}"
    energy_str = (
        f"{result.energy_hartree:.6f} Eh"
        if result.energy_hartree is not None
        else "N/A"
    )
    return (
        f"분자: {result.formula or 'unknown'}, "
        f"계산: {result.calc_type or 'unknown'}, "
        f"메서드: {method_basis}, "
        f"에너지: {energy_str}\n\n"
        f"현재 상태를 한국어 300자 이내로 코멘트하라."
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
