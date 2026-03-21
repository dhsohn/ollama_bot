"""메모리 통합(압축) 자동화 callable 구현."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from core.engine import Engine
from core.memory import MemoryManager

from .common import (
    CONSOLIDATION_MERGE_SCHEMA,
    parse_json_array,
    resolve_llm_timeout,
    truncate,
)


def build_memory_consolidation_callable(
    engine: Engine,
    memory: MemoryManager,
    allowed_users: list[int],
    logger: Any,
):
    async def memory_consolidation(
        min_entries_per_category: int = 5,
        max_llm_calls: int = 3,
        max_entries_per_merge: int = 8,
        model: str | None = None,
        model_role: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
        llm_timeout: int | None = None,
    ) -> str:
        """같은 카테고리의 관련 메모리 항목을 LLM으로 통합하여 압축한다."""
        if min_entries_per_category < 2:
            raise ValueError("min_entries_per_category must be >= 2")
        if max_llm_calls < 0:
            raise ValueError("max_llm_calls must be >= 0")
        if max_entries_per_merge < 2:
            raise ValueError("max_entries_per_merge must be >= 2")

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

            by_category: dict[str, list[dict]] = defaultdict(list)
            for mem in all_memories:
                by_category[mem["category"]].append(mem)

            groups_consolidated = 0
            entries_removed = 0
            entries_created = 0

            for category, entries in sorted(by_category.items()):
                if len(entries) < min_entries_per_category:
                    continue
                if llm_calls_remaining <= 0:
                    break

                entry_summary = "\n".join(
                    f"- key={m['key']}, "
                    f"value={truncate(m['value'], 200)}"
                    for m in entries[:max_entries_per_merge]
                )

                prompt = (
                    f"카테고리 '{category}'에 있는 아래 메모리 항목을 "
                    "분석하세요.\n"
                    "서로 관련된 항목들을 그룹으로 묶어 하나로 통합할 수 "
                    "있는 그룹을 찾으세요.\n"
                    "각 그룹에 대해 통합된 결과의 key와 value를 "
                    "제안하세요.\n"
                    "중요한 정보가 누락되지 않도록 모든 핵심 사실을 "
                    "value에 포함하세요.\n"
                    "통합할 그룹이 없으면 빈 배열을 반환하세요.\n"
                    '출력: [{"merge_keys":["통합할키1","통합할키2",...],'
                    '"new_key":"통합결과키",'
                    '"new_value":"통합결과값(모든핵심정보포함)"}]\n\n'
                    f"항목:\n{entry_summary}"
                )

                try:
                    raw = await engine.process_prompt(
                        prompt=prompt,
                        response_format=CONSOLIDATION_MERGE_SCHEMA,
                        max_tokens=max_tokens if max_tokens is not None else 768,
                        temperature=temperature if temperature is not None else 0.2,
                        model_override=model,
                        model_role=model_role,
                        timeout=effective_llm_timeout,
                        timeout_is_hard=timeout_is_hard,
                    )
                    llm_calls_remaining -= 1
                except Exception as exc:
                    logger.warning(
                        "memory_consolidation_llm_failed",
                        chat_id=user_id,
                        category=category,
                        error=str(exc),
                    )
                    continue

                items = parse_json_array(raw)
                if items is None:
                    logger.warning(
                        "memory_consolidation_parse_failed",
                        chat_id=user_id,
                        category=category,
                        response_preview=truncate(raw, 200),
                    )
                    continue

                existing_keys = {m["key"] for m in entries}
                processed_keys: set[str] = set()

                for item in items:
                    if not isinstance(item, dict):
                        continue
                    merge_keys = item.get("merge_keys", [])
                    new_key = item.get("new_key", "")
                    new_value = item.get("new_value", "")

                    if (
                        not isinstance(merge_keys, list)
                        or not isinstance(new_key, str)
                        or not isinstance(new_value, str)
                        or not new_key
                        or not new_value
                    ):
                        continue

                    # 중복 키를 제거해 실제 병합 대상(고유 키) 기준으로 검증한다.
                    valid_keys: list[str] = []
                    seen_keys: set[str] = set()
                    for key in merge_keys:
                        if not isinstance(key, str):
                            continue
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        if key in existing_keys and key not in processed_keys:
                            valid_keys.append(key)

                    if len(valid_keys) < 2:
                        continue

                    # new_key가 기존에 존재하면서 병합 대상이 아닌 경우 skip
                    if (
                        new_key in existing_keys
                        and new_key not in valid_keys
                    ):
                        continue

                    # 부분 실패 시 유실을 줄이기 위해 upsert를 먼저 수행한다.
                    new_key_is_new = new_key not in existing_keys
                    await memory.store_memory(
                        user_id, new_key, new_value, category,
                    )
                    existing_keys.add(new_key)
                    if new_key_is_new:
                        entries_created += 1

                    delete_targets = [key for key in valid_keys if key != new_key]
                    for old_key in delete_targets:
                        deleted = await memory.delete_memory(user_id, old_key)
                        if deleted:
                            existing_keys.discard(old_key)
                            entries_removed += 1

                    groups_consolidated += 1
                    processed_keys.update(valid_keys)
                    processed_keys.add(new_key)

            net_reduction = entries_removed - entries_created
            if groups_consolidated > 0 or entries_removed > 0 or entries_created > 0:
                any_work_done = True

            final_memories = await memory.recall_memory(user_id)
            remaining_count = len(final_memories)

            sections.append(
                f"## Chat ID {user_id}\n"
                f"- 통합 그룹: {groups_consolidated}건\n"
                f"- 제거된 항목: {entries_removed}건\n"
                f"- 생성된 항목: {entries_created}건\n"
                f"- 순감소: {net_reduction}건\n"
                f"- 남은 메모리: {remaining_count}건"
            )

        if not sections:
            return ""

        if not any_work_done:
            logger.info("memory_consolidation_no_changes")
            return ""

        return "🗜️ 메모리 통합 완료\n\n" + "\n\n".join(sections)

    return memory_consolidation
