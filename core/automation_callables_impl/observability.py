"""헬스체크 자동화 callable 구현."""

from __future__ import annotations

import asyncio
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any

from core.engine import Engine
from core.engine_context import normalize_language
from core.memory import MemoryManager

from .common import (
    SEVERITY_ICONS,
    TRIAGE_SCHEMA,
    count_recent_errors_async,
    get_log_level,
    parse_json_array,
    resolve_llm_timeout,
    truncate,
)


def build_health_check_callable(
    engine: Engine,
    memory: MemoryManager,
    data_dir: str,
    logger: Any,
    default_language: str = "ko",
):
    def _resolve_lang() -> str:
        config = getattr(engine, "_config", None)
        bot = getattr(config, "bot", None)
        value = getattr(bot, "language", default_language)
        if not isinstance(value, str):
            value = default_language
        normalized = normalize_language(value)
        return normalized if normalized in {"ko", "en"} else "ko"

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

        lang = _resolve_lang()

        def _text(ko: str, en: str) -> str:
            return en if lang == "en" else ko

        has_issue = False
        lines: list[str] = [_text("🏥 시스템 상태 점검\n", "🏥 System Health Check\n")]

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
                        _text(
                            f"⚠️ LLM: 기본 모델 사용 불가 (모델 {models_count}개)",
                            f"⚠️ LLM: default model unavailable ({models_count} models)",
                        )
                    )
                else:
                    lines.append(
                        _text(
                            f"✅ LLM: 정상 (모델 {models_count}개, 기본 모델 사용 가능)",
                            f"✅ LLM: ok ({models_count} models, default model available)",
                        )
                    )
            else:
                has_issue = True
                error_msg = llm_info.get("error", _text("알 수 없는 오류", "unknown error"))
                lines.append(_text(f"🔴 LLM: 오류 ({error_msg})", f"🔴 LLM: error ({error_msg})"))
        except Exception as exc:
            has_issue = True
            lines.append(_text(f"🔴 LLM: 연결 실패 ({exc})", f"🔴 LLM: connection failed ({exc})"))

        try:
            ok = await memory.ping()
            if ok:
                lines.append(_text("✅ 데이터베이스: 정상", "✅ Database: ok"))
            else:
                has_issue = True
                lines.append(_text("🔴 데이터베이스: 응답 이상", "🔴 Database: unhealthy response"))
        except Exception as exc:
            has_issue = True
            lines.append(_text(f"🔴 데이터베이스: 오류 ({exc})", f"🔴 Database: error ({exc})"))

        try:
            usage = shutil.disk_usage(data_dir)
            used_pct = int((usage.used / usage.total) * 100)
            if used_pct >= disk_warn_pct:
                has_issue = True
                lines.append(
                    _text(
                        f"⚠️ 디스크: {used_pct}% 사용 (경고 임계값 {disk_warn_pct}%)",
                        f"⚠️ Disk: {used_pct}% used (warn threshold {disk_warn_pct}%)",
                    )
                )
            else:
                lines.append(_text(f"✅ 디스크: {used_pct}% 사용", f"✅ Disk: {used_pct}% used"))
        except Exception as exc:
            has_issue = True
            lines.append(_text(f"🔴 디스크: 확인 실패 ({exc})", f"🔴 Disk: check failed ({exc})"))

        log_path = Path(data_dir) / "logs"
        _, error_count, _ = await count_recent_errors_async(
            log_path, error_hours_back, max_entries=0,
        )
        if error_count > 0:
            has_issue = True
            lines.append(
                _text(
                    f"⚠️ 오류 로그: 최근 {error_hours_back}시간 오류 {error_count}건",
                    f"⚠️ Error logs: {error_count} in the last {error_hours_back} hour(s)",
                )
            )
        else:
            lines.append(
                _text(
                    f"✅ 오류 로그: 최근 {error_hours_back}시간 오류 0건",
                    f"✅ Error logs: 0 in the last {error_hours_back} hour(s)",
                )
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
    default_language: str = "ko",
):
    triage_exclude_names = {"log_triage"}
    triage_exclude_event_prefixes = ("log_triage",)

    def _resolve_lang() -> str:
        config = getattr(engine, "_config", None)
        bot = getattr(config, "bot", None)
        value = getattr(bot, "language", default_language)
        if not isinstance(value, str):
            value = default_language
        normalized = normalize_language(value)
        return normalized if normalized in {"ko", "en"} else "ko"

    def _text(lang: str, ko: str, en: str) -> str:
        return en if lang == "en" else ko

    def _reserve_llm_timeout(total_timeout: int | None) -> int | None:
        if total_timeout is None:
            return None
        timeout_value = max(1, int(total_timeout))
        reserve = min(15, max(1, timeout_value // 5))
        return max(1, timeout_value - reserve)

    def _normalize_text(value: Any, *, max_chars: int = 160) -> str:
        if value is None:
            return ""
        text = " ".join(str(value).strip().split())
        return truncate(text, max_chars)

    def _format_entry_excerpt(entry: dict[str, Any]) -> str:
        lang = _resolve_lang()
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
        return " | ".join(parts[:3]) or _text(lang, "(추가 필드 없음)", "(no extra fields)")

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
        lang = _resolve_lang()
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
            _text(
                lang,
                (
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
                ),
                (
                    "Triage recent application logs from an operations perspective.\n"
                    f"- Scope: last {hours_back} hour(s)\n"
                    f"- Totals: error={error_count}, warning={warning_count}\n"
                    f"- Return at most {max_findings} items.\n"
                    "- You may exclude routine noise or events that do not need follow-up.\n"
                    "- Severity guide: service outage, data loss, or security issues are urgent; "
                    "recurring failures or performance degradation are warning; isolated or low-impact issues are low.\n"
                    "- Write cause as a grounded inference based on the logs.\n"
                    "- Write action as a short operational step a human can take immediately.\n"
                    "- Prefer recurring=true when the same event happened more than once.\n\n"
                    "Event summary:\n"
                ),
            )
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
        lang = _resolve_lang()
        lines = [
            _text(lang, "🧾 로그 triage 결과", "🧾 Log Triage Report"),
            _text(lang, f"- 범위: 최근 {hours_back}시간", f"- Scope: last {hours_back} hour(s)"),
            _text(lang, f"- error: {error_count}건", f"- error: {error_count}"),
            _text(lang, f"- warning: {warning_count}건", f"- warning: {warning_count}"),
            _text(lang, f"- 분석 이벤트: {analyzed_events}개", f"- Analyzed events: {analyzed_events}"),
            "",
        ]
        for idx, item in enumerate(findings, start=1):
            severity = str(item.get("severity", "low")).strip().lower()
            icon = SEVERITY_ICONS.get(severity, SEVERITY_ICONS["low"])
            recurring = _text(lang, "반복", "recurring") if item.get("recurring") else _text(lang, "단발", "one-off")
            event = _normalize_text(item.get("event") or "unknown_event", max_chars=120)
            cause = _normalize_text(
                item.get("cause") or _text(lang, "원인 정보 부족", "insufficient evidence in logs"),
                max_chars=240,
            )
            action = _normalize_text(
                item.get("action") or _text(lang, "원문 로그를 확인하세요.", "check the raw logs"),
                max_chars=240,
            )
            lines.extend([
                f"{idx}. {icon} {event} ({recurring})",
                _text(lang, f"   - 원인: {cause}", f"   - Cause: {cause}"),
                _text(lang, f"   - 조치: {action}", f"   - Action: {action}"),
            ])
        return "\n".join(lines)

    def _fallback_triage_report(
        *,
        hours_back: int,
        error_count: int,
        warning_count: int,
        groups: list[dict[str, Any]],
    ) -> str:
        lang = _resolve_lang()
        lines = [
            _text(lang, "🧾 로그 triage 결과", "🧾 Log Triage Report"),
            _text(lang, f"- 범위: 최근 {hours_back}시간", f"- Scope: last {hours_back} hour(s)"),
            _text(lang, f"- error: {error_count}건", f"- error: {error_count}"),
            _text(lang, f"- warning: {warning_count}건", f"- warning: {warning_count}"),
            _text(
                lang,
                "- 구조화 분석에 실패해 이벤트 요약으로 대체했습니다.",
                "- Structured analysis failed, so this was replaced with an event summary.",
            ),
            "",
        ]
        for idx, group in enumerate(groups[:5], start=1):
            recurring = _text(lang, "반복", "recurring") if group["count"] >= 2 else _text(lang, "단발", "one-off")
            lines.extend([
                _text(
                    lang,
                    f"{idx}. {group['event']} ({recurring}, {group['count']}건)",
                    f"{idx}. {group['event']} ({recurring}, {group['count']})",
                ),
                _text(
                    lang,
                    f"   - 최근 시각: {group['latest_timestamp'] or 'unknown'}",
                    f"   - Latest: {group['latest_timestamp'] or 'unknown'}",
                ),
                _text(
                    lang,
                    f"   - 예시: {group['samples'][0] if group['samples'] else '(추가 필드 없음)'}",
                    f"   - Sample: {group['samples'][0] if group['samples'] else '(no extra fields)'}",
                ),
                _text(
                    lang,
                    "   - 조치: 동일 event의 원문 로그를 열어 상세 원인을 확인하세요.",
                    "   - Action: open the raw logs for this event and inspect the details.",
                ),
            ])
        return "\n".join(lines)

    def _no_actionable_triage_report(
        *,
        hours_back: int,
        error_count: int,
        warning_count: int,
        groups: list[dict[str, Any]],
    ) -> str:
        lang = _resolve_lang()
        lines = [
            _text(lang, "🧾 로그 triage 결과", "🧾 Log Triage Report"),
            _text(lang, f"- 범위: 최근 {hours_back}시간", f"- Scope: last {hours_back} hour(s)"),
            _text(lang, f"- error: {error_count}건", f"- error: {error_count}"),
            _text(lang, f"- warning: {warning_count}건", f"- warning: {warning_count}"),
            _text(lang, f"- 분석 이벤트: {len(groups)}개", f"- Analyzed events: {len(groups)}"),
            _text(
                lang,
                "- 중요 이슈 없음: 구조화 분석 결과 즉시 대응이 필요한 항목은 없었습니다.",
                "- No actionable issues: the structured analysis found nothing that needs immediate action.",
            ),
            "",
        ]
        for idx, group in enumerate(groups[:5], start=1):
            recurring = _text(lang, "반복", "recurring") if group["count"] >= 2 else _text(lang, "단발", "one-off")
            lines.extend([
                _text(
                    lang,
                    f"{idx}. {group['event']} ({recurring}, {group['count']}건)",
                    f"{idx}. {group['event']} ({recurring}, {group['count']})",
                ),
                _text(
                    lang,
                    f"   - 최근 시각: {group['latest_timestamp'] or 'unknown'}",
                    f"   - Latest: {group['latest_timestamp'] or 'unknown'}",
                ),
                _text(
                    lang,
                    f"   - 예시: {group['samples'][0] if group['samples'] else '(추가 필드 없음)'}",
                    f"   - Sample: {group['samples'][0] if group['samples'] else '(no extra fields)'}",
                ),
                _text(
                    lang,
                    "   - 메모: 참고용 이벤트입니다. 필요시 원문 로그를 확인하세요.",
                    "   - Note: this is a reference event. Check the raw logs if needed.",
                ),
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
        llm_timeout: int | None = None,
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
            log_path,
            hours_back,
            max_entries=0,
            exclude_names=triage_exclude_names,
            exclude_event_prefixes=triage_exclude_event_prefixes,
        )
        total_count = error_count + warning_count
        if total_count == 0:
            logger.info("log_triage_no_findings", hours_back=hours_back)
            return ""

        sampled_entries, _, _ = await count_recent_errors_async(
            log_path,
            hours_back,
            max_entries=max_entries,
            exclude_names=triage_exclude_names,
            exclude_event_prefixes=triage_exclude_event_prefixes,
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
        explicit_llm_timeout, timeout_is_hard = resolve_llm_timeout(
            timeout=None,
            llm_timeout=llm_timeout,
        )
        effective_llm_timeout = explicit_llm_timeout or _reserve_llm_timeout(timeout)

        try:
            raw = await asyncio.wait_for(
                engine.process_prompt(
                    prompt=prompt,
                    response_format=TRIAGE_SCHEMA,
                    max_tokens=max_tokens if max_tokens is not None else 768,
                    temperature=temperature if temperature is not None else 0.2,
                    model_override=model,
                    model_role=model_role,
                    timeout=effective_llm_timeout,
                    timeout_is_hard=timeout_is_hard,
                ),
                timeout=effective_llm_timeout,
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

        if not items:
            logger.info(
                "log_triage_no_actionable_findings",
                analyzed_events=len(groups),
                error_count=error_count,
                warning_count=warning_count,
            )
            return _no_actionable_triage_report(
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
            logger.warning("log_triage_invalid_findings")
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
