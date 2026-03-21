"""메모리 정리 자동화 callable 구현."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

from core.engine import Engine
from core.memory import MemoryManager

from .common import (
    MEMORY_HYGIENE_SCHEMA,
    SQLITE_TIMESTAMP_FORMAT,
    STALE_EVALUATION_SCHEMA,
    parse_json_array,
    resolve_llm_timeout,
)


def build_memory_hygiene_callable(
    engine: Engine,
    memory: MemoryManager,
    allowed_users: list[int],
    logger: Any,
):
    async def memory_hygiene(
        stale_days: int = 90,
        max_llm_calls: int = 3,
        model: str | None = None,
        model_role: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
        llm_timeout: int | None = None,
    ) -> str:
        """메모리 품질을 점검하고 정리한다."""
        if stale_days < 1:
            raise ValueError("stale_days must be >= 1")
        if max_llm_calls < 0:
            raise ValueError("max_llm_calls must be >= 0")

        effective_llm_timeout, timeout_is_hard = resolve_llm_timeout(
            timeout=timeout,
            llm_timeout=llm_timeout,
        )
        llm_calls_remaining = max_llm_calls
        sections: list[str] = []
        any_work_done = False

        for user_id in allowed_users:
            all_memories = await memory.recall_memory(user_id)
            if not all_memories:
                continue

            duplicates_removed = 0
            stale_removed = 0
            conflicts_resolved = 0

            value_groups: dict[str, list[dict]] = defaultdict(list)
            for mem in all_memories:
                value_groups[mem["value"]].append(mem)

            keys_to_delete: set[str] = set()
            for _, group in value_groups.items():
                if len(group) <= 1:
                    continue
                sorted_group = sorted(
                    group, key=lambda m: m["updated_at"], reverse=True,
                )
                for dup in sorted_group[1:]:
                    if dup["key"] not in keys_to_delete:
                        await memory.delete_memory(user_id, dup["key"])
                        keys_to_delete.add(dup["key"])
                        duplicates_removed += 1

            remaining_memories = [
                m for m in all_memories if m["key"] not in keys_to_delete
            ]

            if llm_calls_remaining > 0 and len(remaining_memories) > 1:
                mem_summary = "\n".join(
                    f"- key={m['key']}, value={m['value']}, "
                    f"updated_at={m['updated_at']}"
                    for m in remaining_memories
                )
                prompt = (
                    "아래 메모리 항목을 분석하세요.\n"
                    "1) 의미상 중복(같은 정보를 다른 키로 저장)을 찾으세요.\n"
                    "2) 서로 충돌하는 항목(같은 주제에 다른 값)을 찾으세요.\n"
                    "두 경우 모두 최신 항목(updated_at 기준)을 유지하세요.\n"
                    '출력: [{"keep_key":"유지할키",'
                    '"delete_key":"삭제할키",'
                    '"reason":"duplicate 또는 conflict"}]\n'
                    "없으면 빈 배열.\n\n"
                    f"메모리:\n{mem_summary}"
                )

                try:
                    raw = await engine.process_prompt(
                        prompt=prompt,
                        response_format=MEMORY_HYGIENE_SCHEMA,
                        max_tokens=max_tokens if max_tokens is not None else 512,
                        temperature=temperature if temperature is not None else 0.2,
                        model_override=model,
                        model_role=model_role,
                        timeout=effective_llm_timeout,
                        timeout_is_hard=timeout_is_hard,
                    )
                    llm_calls_remaining -= 1
                    items = parse_json_array(raw)
                    if items is None:
                        raise ValueError("invalid JSON array response")

                    remaining_keys = {
                        m["key"] for m in remaining_memories
                    }
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        delete_key = item.get("delete_key", "")
                        keep_key = item.get("keep_key", "")
                        reason = item.get("reason", "")
                        if (
                            delete_key in remaining_keys
                            and keep_key in remaining_keys
                            and delete_key != keep_key
                            and delete_key not in keys_to_delete
                        ):
                            await memory.delete_memory(
                                user_id, delete_key,
                            )
                            keys_to_delete.add(delete_key)
                            remaining_keys.discard(delete_key)
                            if reason == "conflict":
                                conflicts_resolved += 1
                            else:
                                duplicates_removed += 1
                except ValueError:
                    logger.warning(
                        "memory_hygiene_semantic_parse_failed",
                        chat_id=user_id,
                    )
                except Exception as exc:
                    logger.warning(
                        "memory_hygiene_semantic_failed",
                        chat_id=user_id,
                        error=str(exc),
                    )

                remaining_memories = [
                    m for m in remaining_memories
                    if m["key"] not in keys_to_delete
                ]

            cutoff_str = (
                datetime.now(UTC) - timedelta(days=stale_days)
            ).strftime(SQLITE_TIMESTAMP_FORMAT)

            stale_candidates = [
                m for m in remaining_memories
                if m["updated_at"] <= cutoff_str
            ]

            if llm_calls_remaining > 0 and stale_candidates:
                stale_summary = "\n".join(
                    f"- key={m['key']}, value={m['value']}, "
                    f"updated_at={m['updated_at']}"
                    for m in stale_candidates
                )
                prompt = (
                    f"아래 메모리 항목은 {stale_days}일 이상 "
                    "업데이트되지 않았습니다. "
                    "각 항목이 여전히 유효한 정보인지 판단하세요.\n"
                    '출력: [{"key":"키","stale":true/false,'
                    '"reason":"판단근거"}]\n'
                    "직업, 이름 등 변하지 않는 정보는 stale=false.\n"
                    "일시적 선호나 시간에 민감한 정보는 stale=true.\n\n"
                    f"항목:\n{stale_summary}"
                )

                try:
                    raw = await engine.process_prompt(
                        prompt=prompt,
                        response_format=STALE_EVALUATION_SCHEMA,
                        max_tokens=max_tokens if max_tokens is not None else 512,
                        temperature=temperature if temperature is not None else 0.2,
                        model_override=model,
                        model_role=model_role,
                        timeout=effective_llm_timeout,
                        timeout_is_hard=timeout_is_hard,
                    )
                    llm_calls_remaining -= 1
                    items = parse_json_array(raw)
                    if items is None:
                        raise ValueError("invalid JSON array response")

                    candidate_keys = {
                        m["key"] for m in stale_candidates
                    }
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        key = item.get("key", "")
                        is_stale = item.get("stale", False)
                        if (
                            is_stale is True
                            and key
                            and key in candidate_keys
                            and key not in keys_to_delete
                        ):
                            await memory.delete_memory(user_id, key)
                            keys_to_delete.add(key)
                            stale_removed += 1
                except ValueError:
                    logger.warning(
                        "memory_hygiene_stale_parse_failed",
                        chat_id=user_id,
                    )
                except Exception as exc:
                    logger.warning(
                        "memory_hygiene_stale_failed",
                        chat_id=user_id,
                        error=str(exc),
                    )

            total_changes = (
                duplicates_removed + stale_removed + conflicts_resolved
            )
            if total_changes > 0:
                any_work_done = True

            final_memories = await memory.recall_memory(user_id)
            remaining_count = len(final_memories)

            sections.append(
                f"## Chat ID {user_id}\n"
                f"- 중복 제거: {duplicates_removed}건\n"
                f"- 오래된 항목 정리: {stale_removed}건\n"
                f"- 충돌 해소: {conflicts_resolved}건\n"
                f"- 남은 메모리: {remaining_count}건"
            )

        if not sections:
            return ""

        if not any_work_done:
            logger.info("memory_hygiene_no_changes")
            return ""

        return "🧹 메모리 정리 완료\n\n" + "\n\n".join(sections)

    return memory_hygiene
