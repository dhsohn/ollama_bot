"""헬스체크 자동화 callable 구현."""

from __future__ import annotations

import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any

from core.engine import Engine
from core.memory import MemoryManager

from .common import (
    SEVERITY_ICONS,
    TRIAGE_SCHEMA,
    count_recent_errors_async,
    get_log_level,
    parse_json_array,
    truncate,
)


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
            llm_info = status.get("llm", {})
            if llm_info.get("status") == "ok":
                models_count = llm_info.get("models_count", 0)
                default_available = llm_info.get(
                    "default_model_available", False,
                )
                if not default_available:
                    has_issue = True
                    lines.append(
                        f"⚠️ LLM: 기본 모델 사용 불가 "
                        f"(모델 {models_count}개)"
                    )
                else:
                    lines.append(
                        f"✅ LLM: 정상 "
                        f"(모델 {models_count}개, 기본 모델 사용 가능)"
                    )
            else:
                has_issue = True
                error_msg = llm_info.get("error", "알 수 없는 오류")
                lines.append(f"🔴 LLM: 오류 ({error_msg})")
        except Exception as exc:
            has_issue = True
            lines.append(f"🔴 LLM: 연결 실패 ({exc})")

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


def build_log_triage_callable(
    engine: Engine,
    data_dir: str,
    logger: Any,
):
    def _normalize_text(value: Any, *, max_chars: int = 160) -> str:
        if value is None:
            return ""
        text = " ".join(str(value).strip().split())
        return truncate(text, max_chars)

    def _format_entry_excerpt(entry: dict[str, Any]) -> str:
        parts: list[str] = []
        for key in ("error", "reason", "path", "name", "component"):
            value = _normalize_text(entry.get(key))
            if value:
                parts.append(f"{key}={value}")
        if not parts:
            for key, value in entry.items():
                if key in {"event", "timestamp", "log_level", "level"}:
                    continue
                rendered = _normalize_text(value)
                if rendered:
                    parts.append(f"{key}={rendered}")
                if len(parts) >= 3:
                    break
        return " | ".join(parts[:3]) or "(추가 필드 없음)"

    def _build_event_groups(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        groups: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for entry in entries:
            event_name = _normalize_text(entry.get("event") or "unknown_event", max_chars=100)
            level = get_log_level(entry)
            timestamp = _normalize_text(entry.get("timestamp"), max_chars=40)
            excerpt = _format_entry_excerpt(entry)

            group = groups.get(event_name)
            if group is None:
                group = {
                    "event": event_name,
                    "count": 0,
                    "error_count": 0,
                    "warning_count": 0,
                    "latest_timestamp": timestamp,
                    "samples": [],
                }
                groups[event_name] = group

            group["count"] += 1
            if level == "error":
                group["error_count"] += 1
            elif level == "warning":
                group["warning_count"] += 1

            if timestamp and not group["latest_timestamp"]:
                group["latest_timestamp"] = timestamp
            if excerpt and excerpt not in group["samples"] and len(group["samples"]) < 2:
                group["samples"].append(excerpt)

        return sorted(
            groups.values(),
            key=lambda item: (item["error_count"], item["warning_count"], item["count"]),
            reverse=True,
        )

    def _build_triage_prompt(
        *,
        hours_back: int,
        error_count: int,
        warning_count: int,
        groups: list[dict[str, Any]],
        max_findings: int,
    ) -> str:
        group_lines: list[str] = []
        for idx, group in enumerate(groups[:max_findings * 2], start=1):
            group_lines.append(
                f"{idx}. event={group['event']}\n"
                f"   occurrences={group['count']} "
                f"(error={group['error_count']}, warning={group['warning_count']})\n"
                f"   latest={group['latest_timestamp'] or 'unknown'}"
            )
            for sample_idx, sample in enumerate(group["samples"], start=1):
                group_lines.append(f"   sample_{sample_idx}={sample}")

        return (
            "최근 애플리케이션 로그를 운영 관점에서 triage하세요.\n"
            f"- 분석 범위: 최근 {hours_back}시간\n"
            f"- 전체 집계: error={error_count}, warning={warning_count}\n"
            f"- 최대 {max_findings}개만 반환하세요.\n"
            "- routine 노이즈나 단순 후속 조치가 필요 없는 이벤트는 제외해도 됩니다.\n"
            "- severity 기준: 서비스 중단/데이터 손상/보안 이슈는 urgent, "
            "반복 실패나 성능 저하는 warning, 단발성/영향 낮음은 low.\n"
            "- cause는 로그 근거 기반의 추정만 작성하세요.\n"
            "- action은 사람이 바로 수행할 수 있는 짧은 운영 조치로 쓰세요.\n"
            "- recurring은 동일 이벤트가 2회 이상이면 true를 우선 고려하세요.\n\n"
            "이벤트 요약:\n"
            + "\n".join(group_lines)
        )

    def _format_triage_report(
        *,
        hours_back: int,
        error_count: int,
        warning_count: int,
        analyzed_events: int,
        findings: list[dict[str, Any]],
    ) -> str:
        lines = [
            "🧾 로그 triage 결과",
            f"- 범위: 최근 {hours_back}시간",
            f"- error: {error_count}건",
            f"- warning: {warning_count}건",
            f"- 분석 이벤트: {analyzed_events}개",
            "",
        ]
        for idx, item in enumerate(findings, start=1):
            severity = str(item.get("severity", "low")).strip().lower()
            icon = SEVERITY_ICONS.get(severity, SEVERITY_ICONS["low"])
            recurring = "반복" if item.get("recurring") else "단발"
            event = _normalize_text(item.get("event") or "unknown_event", max_chars=120)
            cause = _normalize_text(item.get("cause") or "원인 정보 부족", max_chars=240)
            action = _normalize_text(item.get("action") or "원문 로그를 확인하세요.", max_chars=240)
            lines.extend([
                f"{idx}. {icon} {event} ({recurring})",
                f"   - 원인: {cause}",
                f"   - 조치: {action}",
            ])
        return "\n".join(lines)

    def _fallback_triage_report(
        *,
        hours_back: int,
        error_count: int,
        warning_count: int,
        groups: list[dict[str, Any]],
    ) -> str:
        lines = [
            "🧾 로그 triage 결과",
            f"- 범위: 최근 {hours_back}시간",
            f"- error: {error_count}건",
            f"- warning: {warning_count}건",
            "- 구조화 분석에 실패해 이벤트 요약으로 대체했습니다.",
            "",
        ]
        for idx, group in enumerate(groups[:5], start=1):
            recurring = "반복" if group["count"] >= 2 else "단발"
            lines.extend([
                f"{idx}. {group['event']} ({recurring}, {group['count']}건)",
                f"   - 최근 시각: {group['latest_timestamp'] or 'unknown'}",
                f"   - 예시: {group['samples'][0] if group['samples'] else '(추가 필드 없음)'}",
                "   - 조치: 동일 event의 원문 로그를 열어 상세 원인을 확인하세요.",
            ])
        return "\n".join(lines)

    async def log_triage(
        hours_back: int = 6,
        max_entries: int = 80,
        max_findings: int = 5,
        model: str | None = None,
        model_role: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
    ) -> str:
        """최근 error/warning 로그를 요약해 운영자가 바로 대응할 수 있게 정리한다."""
        if hours_back <= 0:
            raise ValueError("hours_back must be > 0")
        if max_entries <= 0:
            raise ValueError("max_entries must be > 0")
        if max_findings <= 0:
            raise ValueError("max_findings must be > 0")

        log_path = Path(data_dir) / "logs"
        _, error_count, warning_count = await count_recent_errors_async(
            log_path, hours_back, max_entries=0,
        )
        total_count = error_count + warning_count
        if total_count == 0:
            logger.info("log_triage_no_findings", hours_back=hours_back)
            return ""

        sampled_entries, _, _ = await count_recent_errors_async(
            log_path, hours_back, max_entries=max_entries,
        )
        if not sampled_entries:
            logger.warning("log_triage_counts_without_entries", hours_back=hours_back)
            return ""

        groups = _build_event_groups(sampled_entries)
        if not groups:
            logger.warning("log_triage_no_groups", hours_back=hours_back)
            return ""

        prompt = _build_triage_prompt(
            hours_back=hours_back,
            error_count=error_count,
            warning_count=warning_count,
            groups=groups,
            max_findings=max_findings,
        )

        try:
            raw = await engine.process_prompt(
                prompt=prompt,
                response_format=TRIAGE_SCHEMA,
                max_tokens=max_tokens if max_tokens is not None else 768,
                temperature=temperature if temperature is not None else 0.2,
                model_override=model,
                model_role=model_role,
                timeout=timeout,
            )
            items = parse_json_array(raw)
            if items is None:
                raise ValueError("invalid JSON array response")
        except Exception as exc:
            logger.warning(
                "log_triage_llm_failed",
                error=str(exc),
            )
            return _fallback_triage_report(
                hours_back=hours_back,
                error_count=error_count,
                warning_count=warning_count,
                groups=groups,
            )

        findings: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            severity = str(item.get("severity", "")).strip().lower()
            if severity not in SEVERITY_ICONS:
                continue
            findings.append({
                "event": item.get("event"),
                "severity": severity,
                "cause": item.get("cause"),
                "action": item.get("action"),
                "recurring": bool(item.get("recurring", False)),
            })
            if len(findings) >= max_findings:
                break

        if not findings:
            logger.warning("log_triage_empty_findings")
            return _fallback_triage_report(
                hours_back=hours_back,
                error_count=error_count,
                warning_count=warning_count,
                groups=groups,
            )

        return _format_triage_report(
            hours_back=hours_back,
            error_count=error_count,
            warning_count=warning_count,
            analyzed_events=len(groups),
            findings=findings,
        )

    return log_triage
