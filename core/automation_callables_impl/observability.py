"""로그/헬스체크 자동화 callable 구현."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from core.engine import Engine
from core.memory import MemoryManager

from .common import (
    SEVERITY_ICONS,
    TRIAGE_SCHEMA,
    count_recent_errors_async,
)


def build_error_log_triage_callable(
    engine: Engine,
    data_dir: str,
    logger: Any,
):
    async def error_log_triage(
        hours_back: int = 6,
        max_errors: int = 50,
        model: str | None = None,
        model_role: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """애플리케이션 로그에서 에러/경고를 분석하고 트리아지 리포트를 생성한다."""
        if hours_back <= 0:
            raise ValueError("hours_back must be > 0")
        if max_errors <= 0:
            raise ValueError("max_errors must be > 0")

        log_path = Path(data_dir) / "logs"

        if not log_path.exists():
            logger.warning("error_log_triage_no_log_dir", path=str(log_path))
            return ""

        error_entries, total_errors, total_warnings = await count_recent_errors_async(
            log_path, hours_back, max_entries=max_errors,
        )

        if not error_entries:
            return ""

        groups: dict[str, list[dict]] = {}
        for entry in error_entries:
            event_name = entry.get("event", "unknown")
            groups.setdefault(event_name, []).append(entry)

        report_lines: list[str] = []
        for event_name, entries in groups.items():
            sample = {k: v for k, v in entries[0].items() if k != "timestamp"}
            report_lines.append(
                f"### 이벤트: {event_name}\n"
                f"- 발생 횟수: {len(entries)}회\n"
                f"- 샘플: {json.dumps(sample, ensure_ascii=False)}"
            )

        prompt = (
            "오류/경고 로그 그룹을 분석하세요. 그룹당 1개 조치만 제시.\n"
            '출력: [{"event":"이벤트명","severity":"urgent|warning|low",'
            '"cause":"원인(1문장)","action":"조치(1문장)","recurring":true|false}]\n\n'
            + "\n\n".join(report_lines)
        )

        analysis_raw = await engine.process_prompt(
            prompt=prompt,
            response_format=TRIAGE_SCHEMA,
            max_tokens=max_tokens if max_tokens is not None else 768,
            temperature=temperature if temperature is not None else 0.3,
            model_override=model,
            model_role=model_role,
        )

        try:
            items = json.loads(analysis_raw)
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "error_log_triage_json_parse_failed",
                response_preview=analysis_raw[:200],
            )
            analysis = analysis_raw
        else:
            if isinstance(items, dict):
                items = [items]
            if not isinstance(items, list):
                logger.warning(
                    "error_log_triage_unexpected_type",
                    got_type=type(items).__name__,
                )
                analysis = analysis_raw
            else:
                parts: list[str] = []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    icon = SEVERITY_ICONS.get(item.get("severity", ""), "\u26aa")
                    recurring = item.get("recurring")
                    recurring_text = (
                        "예"
                        if recurring is True
                        else "아니오"
                        if recurring is False
                        else "?"
                    )
                    parts.append(
                        f"### {item.get('event', '?')}\n"
                        f"- 심각도: {icon} {item.get('severity', '?')}\n"
                        f"- 추정 원인: {item.get('cause', '?')}\n"
                        f"- 권장 조치: {item.get('action', '?')}\n"
                        f"- 반복 패턴: {recurring_text}"
                    )
                analysis = "\n\n".join(parts) if parts else analysis_raw

        header = (
            f"🔍 오류 로그 분석 (최근 {hours_back}시간)\n"
            f"- 분석 샘플: {total_errors + total_warnings}건 "
            f"(오류 {total_errors}건 | 경고 {total_warnings}건)\n"
            f"- 그룹: {len(groups)}개\n\n"
        )
        return header + analysis.strip()

    return error_log_triage


def build_health_check_callable(
    engine: Engine,
    memory: MemoryManager,
    data_dir: str,
    logger: Any,
):
    async def health_check(
        disk_warn_pct: int = 85,
        error_hours_back: int = 1,
        model: str | None = None,
        model_role: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """시스템 상태를 점검한다. 이상 발견 시에만 보고서를 반환한다."""
        _ = (model, model_role, temperature, max_tokens)
        if not (1 <= disk_warn_pct <= 99):
            raise ValueError("disk_warn_pct must be between 1 and 99")
        if error_hours_back <= 0:
            raise ValueError("error_hours_back must be > 0")

        has_issue = False
        lines: list[str] = ["🏥 시스템 상태 점검\n"]

        try:
            status = await engine.get_status()
            ollama_info = status.get("ollama", {})
            if ollama_info.get("status") == "ok":
                models_count = ollama_info.get("models_count", 0)
                default_available = ollama_info.get(
                    "default_model_available", False,
                )
                if not default_available:
                    has_issue = True
                    lines.append(
                        f"⚠️ Ollama: 기본 모델 사용 불가 "
                        f"(모델 {models_count}개)"
                    )
                else:
                    lines.append(
                        f"✅ Ollama: 정상 "
                        f"(모델 {models_count}개, 기본 모델 사용 가능)"
                    )
            else:
                has_issue = True
                error_msg = ollama_info.get("error", "알 수 없는 오류")
                lines.append(f"🔴 Ollama: 오류 ({error_msg})")
        except Exception as exc:
            has_issue = True
            lines.append(f"🔴 Ollama: 연결 실패 ({exc})")

        try:
            ok = await memory.ping()
            if ok:
                lines.append("✅ 데이터베이스: 정상")
            else:
                has_issue = True
                lines.append("🔴 데이터베이스: 응답 이상")
        except Exception as exc:
            has_issue = True
            lines.append(f"🔴 데이터베이스: 오류 ({exc})")

        try:
            usage = shutil.disk_usage(data_dir)
            used_pct = int((usage.used / usage.total) * 100)
            if used_pct >= disk_warn_pct:
                has_issue = True
                lines.append(
                    f"⚠️ 디스크: {used_pct}% 사용 "
                    f"(경고 임계값 {disk_warn_pct}%)"
                )
            else:
                lines.append(f"✅ 디스크: {used_pct}% 사용")
        except Exception as exc:
            has_issue = True
            lines.append(f"🔴 디스크: 확인 실패 ({exc})")

        log_path = Path(data_dir) / "logs"
        _, error_count, _ = await count_recent_errors_async(
            log_path, error_hours_back, max_entries=0,
        )
        if error_count > 0:
            has_issue = True
            lines.append(
                f"⚠️ 오류 로그: 최근 {error_hours_back}시간 "
                f"오류 {error_count}건"
            )
        else:
            lines.append(
                f"✅ 오류 로그: 최근 {error_hours_back}시간 오류 0건"
            )

        if not has_issue:
            logger.info("health_check_all_ok")
            return ""

        return "\n".join(lines)

    return health_check
