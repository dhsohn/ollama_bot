"""내장 자동화 callable 등록.

daily_summary처럼 코드 기반 처리가 필요한 자동화를 등록한다.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, time, timedelta, timezone
from datetime import tzinfo
from pathlib import Path

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:
    ZoneInfo = None

    class ZoneInfoNotFoundError(Exception):
        """zoneinfo 미지원 환경에서의 대체 예외."""

from core.engine import Engine
from core.logging_setup import get_logger
from core.memory import MemoryManager

_ROLE_LABELS = {
    "user": "사용자",
    "assistant": "봇",
    "system": "시스템",
}


def register_builtin_callables(
    scheduler,
    engine: Engine,
    memory: MemoryManager,
    allowed_users: list[int],
    data_dir: str = "data",
) -> None:
    """내장 자동화 callable을 스케줄러에 등록한다."""
    logger = get_logger("automation_callables")

    def _safe_timezone(name: str) -> tzinfo:
        if ZoneInfo is not None:
            try:
                return ZoneInfo(name)
            except ZoneInfoNotFoundError:
                logger.warning(
                    "invalid_timezone_fallback",
                    timezone=name,
                    fallback="UTC",
                )
                return ZoneInfo("UTC")

        # Python 3.8 등 zoneinfo 미지원 환경용 최소 폴백
        if name == "UTC":
            return timezone.utc
        if name == "Asia/Seoul":
            return timezone(timedelta(hours=9), name="Asia/Seoul")
        logger.warning(
            "timezone_unsupported_fallback",
            timezone=name,
            fallback="UTC",
        )
        return timezone.utc

    def _truncate(text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return text
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."

    # ── daily_summary ──

    async def daily_summary(
        days_ago: int = 1,
        timezone_name: str = "UTC",
        max_messages_per_user: int = 200,
        max_chars_per_message: int = 500,
    ) -> str:
        """어제(기본) 대화 기록을 조회해 일일 요약을 생성한다."""
        if days_ago < 0:
            raise ValueError("days_ago must be >= 0")
        if max_messages_per_user <= 0:
            raise ValueError("max_messages_per_user must be > 0")
        if max_chars_per_message <= 0:
            raise ValueError("max_chars_per_message must be > 0")

        tz = _safe_timezone(timezone_name)
        local_now = datetime.now(tz)
        target_day = (local_now - timedelta(days=days_ago)).date()
        start_local = datetime.combine(target_day, time.min, tzinfo=tz)
        end_local = start_local + timedelta(days=1)
        start_utc = start_local.astimezone(timezone.utc)
        end_utc = end_local.astimezone(timezone.utc)

        sections: list[str] = []
        found_any = False
        for user_id in allowed_users:
            messages = await memory.get_conversation_in_range(
                chat_id=user_id,
                start_at=start_utc,
                end_at=end_utc,
                limit=max_messages_per_user,
            )
            if not messages:
                continue

            found_any = True
            transcript_lines: list[str] = []
            for msg in messages:
                role = _ROLE_LABELS.get(msg["role"], msg["role"])
                content = _truncate(msg["content"].strip(), max_chars_per_message)
                transcript_lines.append(
                    f"[{msg['timestamp']}] {role}: {content}"
                )

            prompt = (
                f"아래는 {target_day.isoformat()} 하루 동안의 대화 로그입니다.\n"
                "핵심 내용만 한국어로 요약하세요.\n"
                "출력 형식:\n"
                "1) 핵심 주제 (3개 이내)\n"
                "2) 결정/합의 사항\n"
                "3) 남은 작업(TODO)\n"
                "4) 특이사항(없으면 '없음')\n\n"
                "대화 로그:\n"
                + "\n".join(transcript_lines)
            )

            summary = await engine.process_prompt(prompt=prompt, chat_id=user_id)
            sections.append(
                f"## Chat ID {user_id}\n"
                f"- 대화 수: {len(messages)}\n\n"
                f"{summary.strip()}"
            )

        if not found_any:
            return (
                f"📭 {target_day.isoformat()} 요약할 대화 기록이 없습니다.\n"
                "대화가 있는 날에 다시 실행해 주세요."
            )

        if len(sections) == 1:
            return f"🗓️ {target_day.isoformat()} 일일 요약\n\n{sections[0]}"

        return (
            f"🗓️ {target_day.isoformat()} 일일 요약\n\n"
            + "\n\n---\n\n".join(sections)
        )

    # ── extract_preferences ──

    def _parse_json_array(text: str) -> list[dict] | None:
        """LLM 응답에서 JSON 배열을 파싱한다. 코드 펜스를 제거하고 폴백 시도."""
        # 코드 펜스 제거
        cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")

        try:
            result = json.loads(cleaned)
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

        # 폴백: 첫 번째 [ ~ 마지막 ] 범위 추출
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                result = json.loads(cleaned[start : end + 1])
                if isinstance(result, list):
                    return result
            except (json.JSONDecodeError, ValueError):
                pass

        return None

    async def extract_preferences(
        days_ago: int = 1,
        timezone_name: str = "UTC",
        max_messages_per_user: int = 300,
        max_chars_per_message: int = 500,
    ) -> str:
        """대화에서 사용자 선호도/고정 정보를 추출하여 장기 메모리에 저장한다."""
        tz = _safe_timezone(timezone_name)
        local_now = datetime.now(tz)
        target_day = (local_now - timedelta(days=days_ago)).date()
        start_local = datetime.combine(target_day, time.min, tzinfo=tz)
        end_local = start_local + timedelta(days=1)
        start_utc = start_local.astimezone(timezone.utc)
        end_utc = end_local.astimezone(timezone.utc)

        sections: list[str] = []
        found_any = False

        for user_id in allowed_users:
            messages = await memory.get_conversation_in_range(
                chat_id=user_id,
                start_at=start_utc,
                end_at=end_utc,
                limit=max_messages_per_user,
            )
            if not messages:
                continue

            found_any = True
            transcript_lines: list[str] = []
            for msg in messages:
                role = _ROLE_LABELS.get(msg["role"], msg["role"])
                content = _truncate(msg["content"].strip(), max_chars_per_message)
                transcript_lines.append(f"{role}: {content}")

            prompt = (
                "아래는 사용자와의 최근 대화 기록입니다.\n"
                "대화에서 드러나는 사용자의 고정 정보와 선호도를 추출하세요.\n\n"
                "추출 대상:\n"
                "- 선호 언어/응답 스타일 (예: 반말/존댓말, 간결/상세)\n"
                "- 직업/역할/전문 분야\n"
                "- 근무 시간대/타임존\n"
                "- 자주 사용하는 도구/기술\n"
                "- 반복적으로 언급하는 관심사\n"
                "- 기타 고정 정보 (이름, 호칭 등)\n\n"
                '출력 형식 (JSON 배열, 항목이 없으면 빈 배열 []):\n'
                '[{"key": "preferred_language", "value": "한국어, 반말 선호"}, ...]\n\n'
                "중요: 확실하지 않은 추측은 제외하세요. "
                "명확하게 드러난 정보만 포함하세요.\n"
                "새로운 정보가 없으면 빈 배열 []을 반환하세요.\n\n"
                "대화 기록:\n"
                + "\n".join(transcript_lines)
            )

            response = await engine.process_prompt(prompt=prompt, chat_id=user_id)
            items = _parse_json_array(response)

            if items is None:
                logger.warning(
                    "preference_extraction_json_parse_failed",
                    chat_id=user_id,
                    response_preview=response[:200],
                )
                sections.append(
                    f"## Chat ID {user_id}\n- JSON 파싱 실패 (추출 건너뜀)"
                )
                continue

            stored_count = 0
            pref_details: list[str] = []
            for item in items:
                key = item.get("key")
                value = item.get("value")
                if not key or not value:
                    continue
                await memory.store_memory(
                    chat_id=user_id,
                    key=str(key),
                    value=str(value),
                    category="preferences",
                )
                stored_count += 1
                pref_details.append(f"  - {key}: {value}")

            sections.append(
                f"## Chat ID {user_id}\n"
                f"- 추출된 항목: {stored_count}개\n"
                + ("\n".join(pref_details) if pref_details else "  (새로운 선호도 없음)")
            )

        if not found_any:
            return "📭 선호도를 추출할 대화 기록이 없습니다."

        return "🧠 선호도 추출 완료\n\n" + "\n\n".join(sections)

    # ── error_log_triage ──

    async def error_log_triage(
        hours_back: int = 6,
        max_errors: int = 50,
    ) -> str:
        """애플리케이션 로그에서 에러/경고를 분석하고 트리아지 리포트를 생성한다."""
        log_path = Path(data_dir) / "logs"

        if not log_path.exists():
            logger.warning("error_log_triage_no_log_dir", path=str(log_path))
            return ""

        log_files = sorted(log_path.glob("app.log*"))
        if not log_files:
            return ""

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)

        error_entries: list[dict] = []
        for lf in log_files:
            try:
                text = lf.read_text(encoding="utf-8")
            except OSError:
                continue
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    entry = json.loads(stripped)
                except (json.JSONDecodeError, ValueError):
                    continue

                level = entry.get("log_level", "").lower()
                if level not in ("error", "warning"):
                    continue

                # 타임스탬프 필터링
                ts_str = entry.get("timestamp", "")
                if ts_str:
                    try:
                        ts = datetime.fromisoformat(ts_str)
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        if ts < cutoff:
                            continue
                    except (ValueError, TypeError):
                        pass  # 타임스탬프 파싱 실패 시 포함

                error_entries.append(entry)

                if len(error_entries) >= max_errors:
                    break
            if len(error_entries) >= max_errors:
                break

        if not error_entries:
            # 에러 없으면 빈 문자열 → _deliver_output 스킵 (텔레그램 알림 없음)
            return ""

        # 이벤트별 그룹화
        groups: dict[str, list[dict]] = defaultdict(list)
        for entry in error_entries:
            event_name = entry.get("event", "unknown")
            groups[event_name].append(entry)

        report_lines: list[str] = []
        for event_name, entries in groups.items():
            sample = {k: v for k, v in entries[0].items() if k != "timestamp"}
            report_lines.append(
                f"### 이벤트: {event_name}\n"
                f"- 발생 횟수: {len(entries)}회\n"
                f"- 샘플: {json.dumps(sample, ensure_ascii=False)}"
            )

        prompt = (
            "아래는 최근 애플리케이션 오류/경고 로그를 이벤트별로 그룹화한 결과입니다.\n"
            "각 그룹에 대해 다음을 분석하세요:\n\n"
            "1) 심각도 (🔴 긴급 / 🟡 주의 / 🟢 낮음)\n"
            "2) 추정 원인\n"
            "3) 권장 조치\n"
            "4) 반복 패턴 여부\n\n"
            "그룹화된 로그:\n\n"
            + "\n\n".join(report_lines)
        )

        analysis = await engine.process_prompt(prompt=prompt)

        total_errors = sum(
            1
            for entry in error_entries
            if str(entry.get("log_level", "")).lower() == "error"
        )
        total_warnings = sum(
            1
            for entry in error_entries
            if str(entry.get("log_level", "")).lower() == "warning"
        )

        header = (
            f"🔍 오류 로그 분석 (최근 {hours_back}시간)\n"
            f"- 오류: {total_errors}건 | 경고: {total_warnings}건\n"
            f"- 그룹: {len(groups)}개\n\n"
        )
        return header + analysis.strip()

    # ── callable 등록 ──
    scheduler.register_callable("daily_summary", daily_summary)
    scheduler.register_callable("extract_preferences", extract_preferences)
    scheduler.register_callable("error_log_triage", error_log_triage)
