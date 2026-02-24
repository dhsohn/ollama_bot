"""요약/선호도 추출 자동화 callable 구현."""

from __future__ import annotations

import json
from datetime import datetime, time, timedelta, timezone
from typing import Any

from core.engine import Engine
from core.memory import MemoryManager

from .common import (
    DAILY_SUMMARY_SCHEMA,
    PREFERENCES_SCHEMA,
    ROLE_LABELS,
    parse_json_array,
    safe_timezone,
    truncate,
)


def build_daily_summary_callable(
    engine: Engine,
    memory: MemoryManager,
    allowed_users: list[int],
    logger: Any,
):
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

        tz = safe_timezone(timezone_name, logger)
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
                role = ROLE_LABELS.get(msg["role"], msg["role"])
                content = truncate(msg["content"].strip(), max_chars_per_message)
                transcript_lines.append(f"{role}: {content}")

            prompt = (
                f"{target_day.isoformat()} 대화 로그를 분석하세요.\n"
                '출력: {"topics":["주제(최대3)"],"decisions":["결정(최대5)"],'
                '"todos":["할일(최대5)"],"notes":"특이사항 또는 null"}\n\n'
                "대화:\n"
                + "\n".join(transcript_lines)
            )

            raw = await engine.process_prompt(
                prompt=prompt,
                chat_id=user_id,
                format=DAILY_SUMMARY_SCHEMA,
                max_tokens=512,
                temperature=0.5,
            )

            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                logger.warning(
                    "daily_summary_json_parse_failed",
                    chat_id=user_id,
                    response_preview=raw[:200],
                )
                summary = raw
            else:
                if not isinstance(data, dict):
                    logger.warning(
                        "daily_summary_unexpected_type",
                        chat_id=user_id,
                        got_type=type(data).__name__,
                    )
                    summary = raw
                else:
                    topics = data.get("topics")
                    decisions = data.get("decisions")
                    todos = data.get("todos")
                    notes = data.get("notes")
                    lists_valid = (
                        isinstance(topics, list)
                        and all(isinstance(item, str) for item in topics)
                        and isinstance(decisions, list)
                        and all(isinstance(item, str) for item in decisions)
                        and isinstance(todos, list)
                        and all(isinstance(item, str) for item in todos)
                    )
                    notes_valid = notes is None or isinstance(notes, str)
                    if not lists_valid or not notes_valid:
                        logger.warning(
                            "daily_summary_invalid_field_types",
                            chat_id=user_id,
                            topics_type=type(topics).__name__,
                            decisions_type=type(decisions).__name__,
                            todos_type=type(todos).__name__,
                            notes_type=type(notes).__name__,
                        )
                        summary = raw
                    else:
                        lines: list[str] = []
                        if topics:
                            lines.append("1) 핵심 주제")
                            for topic in topics[:3]:
                                lines.append(f"   - {topic}")
                        if decisions:
                            lines.append("2) 결정/합의 사항")
                            for decision in decisions[:5]:
                                lines.append(f"   - {decision}")
                        if todos:
                            lines.append("3) 남은 작업(TODO)")
                            for todo in todos[:5]:
                                lines.append(f"   - {todo}")
                        lines.append(f"4) 특이사항: {notes if notes else '없음'}")
                        summary = "\n".join(lines)

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

    return daily_summary


def build_extract_preferences_callable(
    engine: Engine,
    memory: MemoryManager,
    allowed_users: list[int],
    logger: Any,
):
    async def extract_preferences(
        days_ago: int = 1,
        timezone_name: str = "UTC",
        max_messages_per_user: int = 300,
        max_chars_per_message: int = 500,
    ) -> str:
        """대화에서 사용자 선호도/고정 정보를 추출하여 장기 메모리에 저장한다."""
        tz = safe_timezone(timezone_name, logger)
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
                role = ROLE_LABELS.get(msg["role"], msg["role"])
                content = truncate(msg["content"].strip(), max_chars_per_message)
                transcript_lines.append(f"{role}: {content}")

            prompt = (
                "대화에서 사용자의 고정 정보/선호도를 추출하세요. "
                "확실한 정보만, 최대 10개까지 포함하세요. 없으면 빈 배열.\n"
                '출력: [{"key":"<항목명>","value":"<값>"}]\n\n'
                "대화:\n"
                + "\n".join(transcript_lines)
            )

            response = await engine.process_prompt(
                prompt=prompt,
                chat_id=user_id,
                format=PREFERENCES_SCHEMA,
                max_tokens=512,
                temperature=0.3,
            )
            items = parse_json_array(response)

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
            for item in items[:10]:
                if not isinstance(item, dict):
                    continue
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

    return extract_preferences
